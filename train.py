"""tinypan: contrastive goal-conditioned RL, end to end, in one file.

Method (see docstring sections below for the matching paper section):
  Stage 1 (Sec 3.4) -- contrastive value learning, no actions, no rewards:
      V_theta(s, g) = <f_theta(s), h_theta(g)> / sqrt(d)
      fit by symmetric InfoNCE against hindsight-relabeled goals drawn with
      a geometric horizon (Sec 3.2-3.4).
  Stage 2 (Sec 3.5) -- advantage-weighted action decoding, theta FROZEN:
      A(s, a, g) = V_theta(s', g) - V_theta(s, g)
      pi_psi fit by advantage-weighted regression against recorded actions.

theta is frozen at exactly the boundary between the two stages: stage 2
reads it only as a numerical weight, no gradient crosses that line.

Usage:
    python collect.py                              # build a reward-free buffer
    python train.py --data data/point_mass_easy.npz
"""
import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.training.train_state import TrainState
from tqdm import trange

from collect import EXCLUDE_OBS_KEYS, flatten_obs, load_env


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    gamma: float = 0.95          # discount, implemented via the horizon sampler (Sec 3.4)
    dim: int = 64                # embedding dim d
    hidden: int = 256
    depth: int = 2
    batch_size: int = 512        # paper's B
    stage1_steps: int = 20_000
    stage2_steps: int = 20_000
    lr: float = 3e-4
    beta: float = 1.0            # AWR temperature; pass float('inf') for plain BC
    w_max: float = 20.0          # advantage-weight clip
    held_out_frac: float = 0.05
    log_every: int = 500
    eval_episodes: int = 20
    eval_horizon: int = 200
    eval_tail: int = 20           # trailing window checked for sustained proximity (see evaluate())
    eval_tail_frac: float = 0.8   # success = >= this fraction of eval_tail steps within threshold
    success_threshold: float = 0.05
    seed: int = 0


# ---------------------------------------------------------------------------
# Data: fixed-length episode buffer + geometric-horizon hindsight sampler
# ---------------------------------------------------------------------------
def _sample_anchor_and_goal(rng, num_episodes, L, batch_size, gamma):
    """Anchor (e, t) and hindsight goal index (e, goal_t) for a batch.

    goal_t = t + Delta, Delta ~ Geom(1 - gamma) (support {1, 2, ...}).
    Draws that fall past the episode end are REJECTED and resampled, not
    clipped -- clipping would concentrate positives on terminal states
    (Sec 3.4).
    """
    e = rng.integers(0, num_episodes, size=batch_size)
    t = rng.integers(0, L, size=batch_size)
    delta = rng.geometric(1 - gamma, size=batch_size)
    goal_t = t + delta
    invalid = goal_t > L
    while invalid.any():
        delta[invalid] = rng.geometric(1 - gamma, size=invalid.sum())
        goal_t = t + delta
        invalid = goal_t > L
    return e, t, goal_t


def sample_stage1_batch(obs, rng, batch_size, gamma):
    """(s, g) pairs: this is hindsight relabelling -- g is just whatever
    state the behaviour policy happened to reach (Sec 3.4)."""
    e, t, goal_t = _sample_anchor_and_goal(rng, obs.shape[0], obs.shape[1] - 1, batch_size, gamma)
    return obs[e, t], obs[e, goal_t]


def sample_stage2_batch(obs, act, rng, batch_size, gamma):
    """(s, a, s', g) transitions, g relabeled with the same scheme as stage 1."""
    e, t, goal_t = _sample_anchor_and_goal(rng, obs.shape[0], act.shape[1], batch_size, gamma)
    return obs[e, t], act[e, t], obs[e, t + 1], obs[e, goal_t]


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    hidden: int
    depth: int
    out: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(self.out)(x)


class Encoder(nn.Module):
    """Untied state/goal encoders f_theta, h_theta sharing a trunk (Sec 3.3).
    Tying them would force V symmetric; the trunk is shared purely to cut
    parameters."""
    hidden: int
    depth: int
    dim: int

    def setup(self):
        self.trunk = MLP(self.hidden, self.depth, self.hidden)
        self.f_head = nn.Dense(self.dim)
        self.h_head = nn.Dense(self.dim)

    def encode_state(self, s):
        return self.f_head(nn.relu(self.trunk(s)))

    def encode_goal(self, g):
        return self.h_head(nn.relu(self.trunk(g)))

    def __call__(self, s, g):  # only used to init params for both heads at once
        return self.encode_state(s), self.encode_goal(g)


class Policy(nn.Module):
    hidden: int
    depth: int
    act_dim: int

    @nn.compact
    def __call__(self, s, g):
        x = jnp.concatenate([s, g], axis=-1)
        return nn.tanh(MLP(self.hidden, self.depth, self.act_dim)(x))  # unit range, rescaled outside


def scale_action(a_unit, act_min, act_max):
    return act_min + (a_unit + 1.0) * 0.5 * (act_max - act_min)


# ---------------------------------------------------------------------------
# Stage 1: contrastive value learning (Sec 3.4)
# ---------------------------------------------------------------------------
def score_matrix(encoder, params, s, g):
    """M[i, j] = V_theta(s_i, g_j), the rank-d factorisation of Sec 3.3."""
    f = encoder.apply(params, s, method=Encoder.encode_state)  # (B, d)
    h = encoder.apply(params, g, method=Encoder.encode_goal)   # (B, d)
    return f @ h.T / jnp.sqrt(f.shape[-1])


def infonce_loss(encoder, params, s, g):
    """Eq. (3.4): symmetric InfoNCE. Off-diagonal entries of each row/column
    are negatives -- goals belonging to other anchors in the batch."""
    M = score_matrix(encoder, params, s, g)
    labels = jnp.arange(M.shape[0])
    loss_row = optax.softmax_cross_entropy_with_integer_labels(M, labels).mean()
    loss_col = optax.softmax_cross_entropy_with_integer_labels(M.T, labels).mean()
    return 0.5 * (loss_row + loss_col)


def make_stage1_step(encoder):
    @jax.jit
    def step(state, s, g):
        loss, grads = jax.value_and_grad(lambda p: infonce_loss(encoder, p, s, g))(state.params)
        return state.apply_gradients(grads=grads), loss
    return step


# ---------------------------------------------------------------------------
# Stage 2: advantage-weighted action decoding (Sec 3.5), theta frozen
# ---------------------------------------------------------------------------
def value(encoder, params, s, g):
    f = encoder.apply(params, s, method=Encoder.encode_state)
    h = encoder.apply(params, g, method=Encoder.encode_goal)
    return jnp.sum(f * h, axis=-1) / jnp.sqrt(f.shape[-1])


def awr_loss(encoder, policy, policy_params, encoder_params, s, a, s_next, g, beta, w_max):
    advantage = jax.lax.stop_gradient(
        value(encoder, encoder_params, s_next, g) - value(encoder, encoder_params, s, g)
    )  # log-occupancy ratio; the additive normalising constant of V cancels
    weight = jnp.minimum(jnp.exp(advantage / beta), w_max)
    a_pred = policy.apply(policy_params, s, g)
    mse = jnp.sum((a_pred - a) ** 2, axis=-1)
    return jnp.mean(weight * mse)


def make_stage2_step(encoder, policy, beta, w_max):
    @jax.jit
    def step(state, encoder_params, s, a, s_next, g):
        loss_fn = lambda p: awr_loss(encoder, policy, p, encoder_params, s, a, s_next, g, beta, w_max)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss
    return step


# ---------------------------------------------------------------------------
# Sec 3.6: correctness gate -- visualize V_theta(., g) on point_mass
# ---------------------------------------------------------------------------
def plot_value_heatmap(encoder, params, goal_obs, pos_range, save_path, resolution=60, walls=()):
    xs = np.linspace(*pos_range, resolution)
    ys = np.linspace(*pos_range, resolution)
    grid_pos = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    grid_obs = np.concatenate([grid_pos, np.zeros_like(grid_pos)], axis=-1).astype(np.float32)
    goal_batch = np.tile(goal_obs[None], (grid_obs.shape[0], 1)).astype(np.float32)

    v_s_to_g = np.array(value(encoder, params, jnp.array(grid_obs), jnp.array(goal_batch)))
    v_g_to_s = np.array(value(encoder, params, jnp.array(goal_batch), jnp.array(grid_obs)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, v, title in zip(axes, [v_s_to_g, v_g_to_s],
                             ["V(s, g_fixed)", "V(g_fixed, s)  [asymmetry check]"]):
        im = ax.imshow(v.reshape(resolution, resolution), origin="lower",
                        extent=[*pos_range, *pos_range])
        for (wx, wy, ww, wh) in walls:  # overlay actual maze wall geometry for reference
            ax.add_patch(plt.Rectangle((wx, wy), ww, wh, facecolor="white", edgecolor="black", linewidth=1))
        ax.scatter([goal_obs[0]], [goal_obs[1]], c="red", marker="*", s=120)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[gate] saved value heatmap -> {save_path} -- eyeball this before trusting stage 2")


# ---------------------------------------------------------------------------
# Sec 3.6: goal-conditioned rollout evaluation
# ---------------------------------------------------------------------------
def evaluate(env, policy, policy_params, goals, horizon, tail, tail_frac, threshold, act_min, act_max, exclude=()):
    """Success = distance to goal stays under threshold for at least
    `tail_frac` of the trailing `tail` steps -- SUSTAINED proximity, not a
    single frame. This matters because the two naive alternatives are both
    wrong in opposite directions:
      - last-frame-only: a converged, oscillating policy (good conditioning)
        can be mid-swing on the literal final frame and get a false negative.
      - "within threshold at ANY point in the tail": satisfied by one lucky
        pass-through the goal region, a false positive that inflates the
        reported rate without the policy ever having actually converged.
    Requiring most of the window to be close rules out the lucky pass-through
    while still tolerating the small oscillation a converged policy shows.
    Domain-agnostic on purpose (no assumption about which obs dims are
    "position"); pick --success-threshold per domain since obs scales differ.
    """
    successes = 0
    for goal in goals:
        ts = env.reset()
        dists = []
        for _ in range(horizon):
            s = flatten_obs(ts.observation, exclude)
            a_unit = policy.apply(policy_params, jnp.array(s)[None], jnp.array(goal)[None])[0]
            ts = env.step(np.array(scale_action(a_unit, act_min, act_max)))
            dists.append(np.linalg.norm(flatten_obs(ts.observation, exclude) - goal))
        window = np.array(dists[-tail:])
        successes += (window < threshold).mean() >= tail_frac
    return successes / len(goals)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(cfg: Config, data_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)

    npz = np.load(data_path)
    obs, act = npz["obs"], npz["act"]
    act_min, act_max = jnp.array(npz["action_min"]), jnp.array(npz["action_max"])
    n_holdout = max(1, int(obs.shape[0] * cfg.held_out_frac))
    train_obs, train_act = obs[:-n_holdout], act[:-n_holdout]
    holdout_obs = obs[-n_holdout:]
    obs_dim, act_dim = obs.shape[-1], act.shape[-1]
    print(f"buffer: {obs.shape[0]} episodes ({obs.shape[0] - n_holdout} train / {n_holdout} held out), "
          f"obs_dim={obs_dim} act_dim={act_dim}")

    # ---- Stage 1 ----
    encoder = Encoder(hidden=cfg.hidden, depth=cfg.depth, dim=cfg.dim)
    key, sub = jax.random.split(key)
    enc_params = encoder.init(sub, jnp.zeros((1, obs_dim)), jnp.zeros((1, obs_dim)))
    enc_state = TrainState.create(apply_fn=encoder.apply, params=enc_params, tx=optax.adam(cfg.lr))
    stage1_step = make_stage1_step(encoder)

    pbar = trange(cfg.stage1_steps, desc="stage1 (contrastive value)")
    for i in pbar:
        s, g = sample_stage1_batch(train_obs, rng, cfg.batch_size, cfg.gamma)
        enc_state, loss = stage1_step(enc_state, jnp.array(s), jnp.array(g))
        if i % cfg.log_every == 0:
            pbar.set_postfix(infonce_loss=float(loss))

    with open(out / "encoder_params.pkl", "wb") as f:
        pickle.dump(enc_state.params, f)

    # correctness gate (Sec 3.6) -- only meaningful for domains with a plain
    # 2D position in the obs (point_mass family); skip elsewhere rather than
    # assume a layout that isn't there.
    if str(npz["domain"]) in ("point_mass", "point_mass_maze"):
        goal_obs = holdout_obs[0, -1]
        pos = train_obs[..., :2]
        pos_range = (float(pos.min()) * 1.1, float(pos.max()) * 1.1)
        # (x, y, width, height) in plot coords, matching _point_mass_maze_env's wall geoms
        walls = [(-.11, -.3, .02, .4), (.09, -.1, .02, .4)] if str(npz["domain"]) == "point_mass_maze" else ()
        plot_value_heatmap(encoder, enc_state.params, goal_obs, pos_range, out / "value_heatmap.png", walls=walls)
    else:
        print(f"[gate] skipped value heatmap -- no 2D position assumption for domain={npz['domain']}")

    # ---- Stage 2 ----
    policy = Policy(hidden=cfg.hidden, depth=cfg.depth, act_dim=act_dim)
    key, sub = jax.random.split(key)
    policy_params = policy.init(sub, jnp.zeros((1, obs_dim)), jnp.zeros((1, obs_dim)))
    policy_state = TrainState.create(apply_fn=policy.apply, params=policy_params, tx=optax.adam(cfg.lr))
    stage2_step = make_stage2_step(encoder, policy, cfg.beta, cfg.w_max)
    frozen_enc_params = enc_state.params  # theta frozen from here on; no gradient crosses this line

    pbar = trange(cfg.stage2_steps, desc="stage2 (AWR policy)")
    for i in pbar:
        s, a, s_next, g = sample_stage2_batch(train_obs, train_act, rng, cfg.batch_size, cfg.gamma)
        policy_state, loss = stage2_step(policy_state, frozen_enc_params,
                                          jnp.array(s), jnp.array(a), jnp.array(s_next), jnp.array(g))
        if i % cfg.log_every == 0:
            pbar.set_postfix(awr_loss=float(loss))

    with open(out / "policy_params.pkl", "wb") as f:
        pickle.dump(policy_state.params, f)

    # ---- Eval ----
    env = load_env(str(npz["domain"]), str(npz["task"]))
    goal_idx = rng.integers(0, holdout_obs.shape[1], size=cfg.eval_episodes)
    goals = holdout_obs[rng.integers(0, holdout_obs.shape[0], size=cfg.eval_episodes), goal_idx]
    success_rate = evaluate(env, policy, policy_state.params, goals, cfg.eval_horizon,
                             cfg.eval_tail, cfg.eval_tail_frac, cfg.success_threshold, act_min, act_max,
                             EXCLUDE_OBS_KEYS.get(str(npz["domain"]), ()))
    print(f"success rate ({cfg.eval_episodes} held-out goals, beta={cfg.beta}): {success_rate:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/point_mass_easy.npz")
    p.add_argument("--out-dir", default="runs/default")
    for field, default in vars(Config()).items():
        p.add_argument(f"--{field.replace('_', '-')}", type=type(default), default=default)
    args = p.parse_args()
    cfg = Config(**{k: v for k, v in vars(args).items() if k not in ("data", "out_dir")})
    main(cfg, args.data, args.out_dir)
