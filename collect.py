"""Reward-free exploration data collection for tinypan.

Stands in for an ExORL replay buffer: rolls a random policy out on a
dm_control task and stores raw (observation, action) trajectories. No reward
is ever read from the environment -- see Sec 3.1, "Environment rewards are
discarded and never used."

Every episode is collected for a fixed number of steps L, so the buffer is
dense arrays:
    obs:  (num_episodes, L + 1, obs_dim)  -- states s_0 .. s_L (task obs, used for training)
    act:  (num_episodes, L,     act_dim)  -- actions a_0 .. a_{L-1}, where
                                              act[e, t] is taken from obs[e, t]
                                              and leads to obs[e, t + 1].
    qpos/qvel: (num_episodes, L + 1, nq/nv) -- raw physics state at every step.
        Not used for training -- task obs is often a derived/partial view of
        physics state (e.g. walker's "orientations" aren't raw joint angles),
        so it can't be inverted back into a renderable scene. qpos/qvel can:
        render.py teleports a second Physics instance to a goal's qpos/qvel
        to show what the agent was actually conditioned on.
Fixed-length episodes keep the hindsight sampler in train.py a few lines of
vectorized indexing instead of an offset table.
"""
import argparse

import numpy as np
from dm_control import suite


# Some domains' task observation includes a feature computed relative to a
# per-episode RANDOMIZED task parameter -- e.g. reacher's "to_target" is
# (fingertip - that episode's randomly placed built-in target), so the same
# physical arm pose gets a different "to_target" in different episodes. That
# breaks the whole premise of goal-conditioning (state must be a portable
# description comparable across episodes), so such keys are excluded from
# what we treat as "state" -- verified for reacher that position == qpos, so
# dropping to_target leaves a clean physical-state representation.
EXCLUDE_OBS_KEYS = {"reacher": {"to_target"}}


def flatten_obs(obs_dict, exclude=()) -> np.ndarray:
    return np.concatenate([np.ravel(v) for k, v in sorted(obs_dict.items()) if k not in exclude]).astype(np.float32)


class RND:
    """Random Network Distillation novelty scorer, used to greedily steer
    exploration instead of acting i.i.d. random -- the fix for domains like
    walker where random torque just jitters in place (Sec: action_repeat
    alone wasn't enough). A fixed random target net + a predictor trained
    online to match it on visited states: novelty = prediction error, which
    is high on states rarely visited and decays as the predictor learns
    them, pushing the greedy candidate search toward less-visited states.
    One-step lookahead: at each real step, branch a few candidate actions
    from the current physics state (dm_control lets us save/restore physics
    state and step it directly, bypassing the env's episode bookkeeping),
    score the resulting candidate observations, and commit to the most
    novel one for real."""

    def __init__(self, obs_dim, seed, hidden=128, dim=32, lr=1e-3):
        import jax
        import jax.numpy as jnp
        import optax
        from train import MLP
        self._jnp = jnp
        target, predictor = MLP(hidden, 2, dim), MLP(hidden, 2, dim)
        key = jax.random.PRNGKey(seed)
        tkey, pkey = jax.random.split(key)
        self.target_params = target.init(tkey, jnp.zeros((1, obs_dim)))
        self.pred_params = predictor.init(pkey, jnp.zeros((1, obs_dim)))
        self.opt = optax.adam(lr)
        self.opt_state = self.opt.init(self.pred_params)

        @jax.jit
        def novelty(pred_params, obs_batch):
            t = target.apply(self.target_params, obs_batch)
            p = predictor.apply(pred_params, obs_batch)
            return jnp.sum((p - t) ** 2, axis=-1)

        @jax.jit
        def update(pred_params, opt_state, obs_batch):
            def loss_fn(params):
                t = jax.lax.stop_gradient(target.apply(self.target_params, obs_batch))
                p = predictor.apply(params, obs_batch)
                return jnp.mean(jnp.sum((p - t) ** 2, axis=-1))
            grads = jax.grad(loss_fn)(pred_params)
            updates, opt_state = self.opt.update(grads, opt_state)
            return optax.apply_updates(pred_params, updates), opt_state

        self._novelty, self._update = novelty, update

    def score(self, obs_batch):
        return np.array(self._novelty(self.pred_params, self._jnp.array(obs_batch, dtype=self._jnp.float32)))

    def learn(self, obs_batch):
        self.pred_params, self.opt_state = self._update(self.pred_params, self.opt_state,
                                                          self._jnp.array(obs_batch, dtype=self._jnp.float32))


def _point_mass_maze_env(seed):
    """point_mass with two offset walls forming an S-shaped corridor -- the
    correctness gate in Sec 3.6 needs REAL walls: two states close in
    Euclidean position but on opposite sides of a wall should show up as far
    apart in the learned V_theta, and reaching one from the other requires
    the long way around, which is exactly what a Euclidean-distance-based
    (i.e. broken) value function could never represent.

    Built by patching dm_control's own point_mass.xml rather than hand-
    authoring a model: contacts are DISABLED by default in that file (the
    point mass is normally confined only by joint limits, never actually
    collides with anything, so the existing "walls" around the arena are
    pure decoration) -- re-enable contacts and inject two wall geoms, then
    reuse dm_control's own Physics/Task/Environment classes unmodified.
    """
    from dm_control.rl import control
    from dm_control.suite import common, point_mass

    raw = common.read_model("point_mass.xml")
    xml = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    xml = xml.replace('<flag contact="disable"/>', "")
    walls = (
        '<geom name="maze_wall_a" type="box" pos="-.1 -.1 .02" size=".01 .2 .02" material="decoration"/>\n'
        '<geom name="maze_wall_b" type="box" pos=".1 .1 .02" size=".01 .2 .02" material="decoration"/>\n'
    )
    xml = xml.replace("</worldbody>", walls + "</worldbody>")

    physics = point_mass.Physics.from_xml_string(xml, common.ASSETS)
    task = point_mass.PointMass(randomize_gains=False, random=seed)
    return control.Environment(physics, task, time_limit=point_mass._DEFAULT_TIME_LIMIT)


def load_env(domain: str, task: str, seed=None):
    """Every env construction in this codebase goes through here so the maze
    special-case lives in one place."""
    if domain == "point_mass_maze":
        return _point_mass_maze_env(seed)
    kwargs = {"random": seed} if seed is not None else {}
    return suite.load(domain_name=domain, task_name=task, task_kwargs=kwargs)


def collect(domain: str, task: str, num_episodes: int, episode_length: int, seed: int,
            action_repeat: int = 1, exploration: str = "random", rnd_candidates: int = 8):
    """action_repeat > 1 holds each sampled action for several steps instead
    of resampling i.i.d. every step. Pure i.i.d. random torque mostly jitters
    torque-controlled joints around equilibrium without committing to a
    direction (unlike a free-floating point mass, where i.i.d. forces still
    integrate into real displacement) -- holding actions helps but isn't
    enough on its own for domains like walker.

    exploration="rnd" replaces i.i.d. random action selection with greedy
    one-step-lookahead RND (see the RND class): still fully reward-free,
    still stored the same way, just a smarter action-selection rule."""
    env = load_env(domain, task, seed)
    action_spec = env.action_spec()
    exclude = EXCLUDE_OBS_KEYS.get(domain, ())
    obs_dim = flatten_obs(env.reset().observation, exclude).shape[0]
    act_dim = action_spec.shape[0]
    n_sub_steps = int(round(env.control_timestep() / env.physics.timestep()))

    nq, nv = env.physics.model.nq, env.physics.model.nv

    rng = np.random.default_rng(seed)
    obs = np.zeros((num_episodes, episode_length + 1, obs_dim), dtype=np.float32)
    act = np.zeros((num_episodes, episode_length, act_dim), dtype=np.float32)
    qpos = np.zeros((num_episodes, episode_length + 1, nq), dtype=np.float32)
    qvel = np.zeros((num_episodes, episode_length + 1, nv), dtype=np.float32)
    rnd = RND(obs_dim, seed) if exploration == "rnd" else None

    def record(e, t, ts):
        obs[e, t] = flatten_obs(ts.observation, exclude)
        qpos[e, t] = env.physics.data.qpos
        qvel[e, t] = env.physics.data.qvel

    def sample_action():
        if rnd is None:
            return rng.uniform(action_spec.minimum, action_spec.maximum).astype(np.float32)
        candidates = rng.uniform(action_spec.minimum, action_spec.maximum,
                                  size=(rnd_candidates, act_dim)).astype(np.float32)
        state0 = env.physics.get_state().copy()
        cand_obs = []
        for cand in candidates:
            env.physics.set_state(state0)
            env.physics.set_control(cand)
            for _ in range(n_sub_steps):
                env.physics.step()
            cand_obs.append(flatten_obs(env.task.get_observation(env.physics), exclude))
        env.physics.set_state(state0)
        env.physics.forward()
        return candidates[int(np.argmax(rnd.score(np.stack(cand_obs))))]

    for e in range(num_episodes):
        ts = env.reset()
        record(e, 0, ts)
        a = sample_action()
        for t in range(episode_length):
            if t % action_repeat == 0:
                a = sample_action()
            ts = env.step(a)  # reward from ts.reward is intentionally never used
            act[e, t] = a
            if rnd is not None:
                rnd.learn(flatten_obs(ts.observation, exclude)[None])  # distill target on the state just visited
            if ts.last():  # ran into the task's own time limit early -- restart
                ts = env.reset()
                a = sample_action()
            record(e, t + 1, ts)

    return obs, act, qpos, qvel, dict(domain=domain, task=task, action_min=action_spec.minimum,
                                       action_max=action_spec.maximum)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="point_mass")
    p.add_argument("--task", default="easy")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--episode-length", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-repeat", type=int, default=1, help="hold each sampled action for k steps")
    p.add_argument("--exploration", choices=["random", "rnd"], default="random")
    p.add_argument("--rnd-candidates", type=int, default=8)
    p.add_argument("--out", default="data/point_mass_easy.npz")
    args = p.parse_args()

    obs, act, qpos, qvel, meta = collect(args.domain, args.task, args.episodes, args.episode_length,
                                          args.seed, args.action_repeat, args.exploration, args.rnd_candidates)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, obs=obs, act=act, qpos=qpos, qvel=qvel, **meta)
    print(f"saved {obs.shape[0]} episodes x {act.shape[1]} steps -> {args.out}  "
          f"(obs_dim={obs.shape[-1]}, act_dim={act.shape[-1]}, nq={qpos.shape[-1]}, nv={qvel.shape[-1]})")
