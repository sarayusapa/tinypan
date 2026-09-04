# tinypan

A from-scratch JAX implementation of **contrastive goal-conditioned RL** on DeepMind Control: learn a value function from reward-free exploration data via contrastive representation learning, then extract a goal-conditioned policy from it by advantage-weighted regression. No environment reward is ever read, at any point.

The whole method is deliberately kept in as few files as possible, laid out so the code maps directly onto the method description section-by-section rather than being scattered across a framework:

- **`collect.py`** — reward-free exploration data collection (stands in for an ExORL replay buffer)
- **`train.py`** — the entire method: sampler, networks, stage 1 (contrastive value learning), stage 2 (advantage-weighted policy), evaluation
- **`render.py`** — rollout video rendering with the actual goal state visualized side-by-side

This is a step on the way to running the same kind of goal-conditioned learning on Minecraft; DMC is the validation ground first.

## Method

**Stage 1 — contrastive value learning** (no actions, no reward). For a state `s` and a hindsight-relabeled goal `g` (just whatever state the trajectory reached some geometrically-sampled number of steps later, `Δ ~ Geom(1-γ)`, rejected — not clipped — at episode boundaries), train two encoders on a shared trunk with **untied** heads:

```
V_theta(s, g) = <f_theta(s), h_theta(g)> / sqrt(d)
```

fit with a symmetric InfoNCE loss over a batch score matrix (other goals in the batch are the negatives). The heads are untied on purpose — tying them forces `V = FF^T`, which is symmetric by construction, but real reachability isn't (a walker falls more easily than it rises).

**Stage 2 — advantage-weighted action decoding**, with θ **frozen**:

```
A(s, a, g) = V_theta(s', g) - V_theta(s, g)          # frozen theta, stop-gradient
weight     = min(exp(A / beta), w_max)
loss       = weight * || pi_psi(s, g) - a ||^2
```

No gradient ever crosses from stage 2 back into θ — it's read only as a numerical weight. `beta -> inf` collapses exactly to goal-conditioned behavior cloning (division by infinity zeroes the advantage term, weight -> 1), which is the natural baseline to compare against.

**Correctness gate**: before trusting stage 2 at all, `train.py` plots `V_theta(., g)` for a fixed goal `g` over the full 2D position grid (point_mass domains only — the check needs a plain 2D position in the observation). A sane run peaks at the goal and decays outward; comparing `V(s,g)` against `V(g,s)` checks the encoder is actually capable of asymmetry rather than collapsing to something like `-||s-g||`.

## Setup

```bash
pip install -e .                    # or: pip install jax flax optax dm_control numpy matplotlib tqdm imageio imageio-ffmpeg pillow
```

On a CUDA machine, install the GPU wheel instead of plain `jax`:

```bash
pip install -U "jax[cuda12]"
```

GPU memory hygiene on a shared box — JAX preallocates ~75-90% of the card by default regardless of how little the model actually needs. Always set:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Usage

```bash
# 1. collect a reward-free exploration buffer
python collect.py --domain point_mass --task easy --episodes 2000 --episode-length 200 \
    --out data/point_mass_easy.npz

# 2. run stage 1 + gate + stage 2 + eval
python train.py --data data/point_mass_easy.npz --out-dir runs/point_mass_easy

# 3. render rollout videos (agent | goal side-by-side, live distance overlay)
python render.py --data data/point_mass_easy.npz --run runs/point_mass_easy --episodes 3
```

Useful flags: `collect.py --exploration rnd` swaps i.i.d. random actions for a minimal from-scratch Random Network Distillation exploration policy (greedy 1-step-lookahead novelty-seeking — see [Limitations](#limitations-and-honest-results) for what this does and doesn't fix); `collect.py --action-repeat k` holds each sampled action for `k` steps. `train.py --beta inf` trains the BC baseline. `train.py --success-threshold` needs per-domain tuning since observation scales differ.

## Results and honest limitations

Only **point_mass** (a free-floating point mass on an open plane) has genuinely validated goal-conditioned behavior end to end. Every other domain fails, for three *different*, specifically diagnosed reasons rather than one mystery bug — the core algorithm (sampler, InfoNCE, AWR) hasn't needed a fix since point_mass first validated it; every subsequent failure has been about the quality of the exploration data, not the method.

| domain / task | random exploration | RND exploration | notes |
|---|---|---|---|
| `point_mass` / `easy` | **0.35** (sustained metric) | — | genuinely reaches and holds goals; see demos below |
| `walker` / `walk` | 0.00 | 0.00 | random torque on a legged robot just jitters near equilibrium instead of committing to a direction; RND gave a marginal coverage bump (mean height 0.27→0.38) but nowhere near standing height (~1.3-1.6) |
| `reacher` / `easy` | 0.00 | 0.00 | two distinct, fixed-then-still-failing bugs found: (1) the task's `to_target` observation is relative to that episode's own randomly-placed built-in target, not a portable state feature across episodes — excluded via `EXCLUDE_OBS_KEYS` in `collect.py`; (2) the shoulder joint is unlimited/continuous (values observed past 2π), so raw-radian L2 distance doesn't wrap and can call two identical poses "far apart" — **not yet fixed**, would need a cos/sin transform on that joint |
| `finger` / `spin` | 0.00 | 0.00 | clean observation (no contamination bug found), genuine manipulation/dexterity exploration gap |
| `pendulum` / `swingup` | 0.05 | 0.00 | **not a coverage problem** — random exploration already covers the full angular range. The gap is *sustained balance*: swinging through upright is easy (gravity/momentum), holding position there needs active control that's rare in a random buffer. RND made this *worse* (0.05→0.00), consistent with novelty-seeking being architecturally opposed to lingering in a stable state |

### Evaluation metric

Success is **not** "final frame within threshold" (fragile — a converged, happily-oscillating policy can be mid-swing on the literal last frame) and **not** "within threshold at any point in a trailing window" (gameable — a single lucky pass-through the goal region scores as success without the policy ever converging). It's **fraction of the trailing window within threshold ≥ 80%**: sustained proximity, tolerant of the small oscillation a genuinely converged policy shows, immune to a one-off coincidence.

### RND exploration — what it is and isn't

`collect.py --exploration rnd` implements real Random Network Distillation from scratch: a fixed random target network, a predictor trained online to match it, novelty = prediction error, greedy 1-step-lookahead action selection over a few candidate actions using branched physics rollouts (dm_control's `physics.get_state()`/`set_state()`). This is a genuine, correctly-implemented curiosity signal — verified against `dm_control`'s own APIs before use — but it is **myopic**: no temporal credit assignment across steps. The original RND paper trains a full RL policy (PPO) against the novelty reward, which can learn to spend several unrewarding-looking steps setting up a future novel state (e.g. crouching before standing). Ours can't discover that. That's the honest reason it didn't fix walker/finger — not that curiosity-driven exploration doesn't work, but that the lightweight version of it we built here isn't strong enough on its own for domains that need multi-step strategy, only for domains random exploration already almost covers.

### The maze

The paper's actual correctness-gate test needs real walls (two states close in Euclidean position but on opposite sides of a wall should register as *far* in `V_theta`, and reaching one from the other should look asymmetric under exchange). `point_mass_maze` is built in `collect.py` (`load_env`, domain `"point_mass_maze"`) by patching dm_control's own `point_mass.xml`: contacts are **disabled by default** in that file (the point mass is normally confined only by joint limits, never actually collides with the decorative "walls" around the arena), so re-enabling contacts and injecting two offset wall geoms creates a real S-shaped corridor. Verified directly against the physics (1000-step sustained push into a wall shows zero leakage; a full up-then-across traversal correctly passes through one wall's gap and is correctly blocked by the other). Collection/training on it was in progress when the remote box went down mid-session — pending a rerun.

## Architecture

```
collect.py: dm_control rollout --> data/<domain>_<task>.npz
    obs (E,L+1,od)  act (E,L,ad)  qpos/qvel (E,L+1,·)  action_min/max

train.py:
    sample_stage1_batch / sample_stage2_batch
        anchor (e,t) --Δ~Geom(1-γ)--> goal (e,t+Δ)   [reject if t+Δ > L]

    STAGE 1 (contrastive value learning)
        s --[shared MLP trunk, 256x2]--> f_head --> f(s)  (64,)
        g --[shared MLP trunk, 256x2]--> h_head --> h(g)  (64,)
        M = f . h^T / sqrt(64)      (B x B score matrix)
        symmetric InfoNCE (rows + cols, labels = diagonal)
        --> theta = {trunk, f_head, h_head}

    ================ theta FROZEN HERE -- no gradient crosses ================

    [gate] point_mass family only: heatmap V(s,g_fixed) vs V(g_fixed,s) --> .png

    STAGE 2 (advantage-weighted action decoding)
        s,a,s',g --> A = V_theta(s',g) - V_theta(s,g)   [stop_gradient]
        weight = min(exp(A/beta), w_max)
        s,g --[Policy MLP 256x2 + tanh]--> action_hat
        loss = weight * || action_hat - a ||^2
        --> psi (policy_params.pkl)

    eval(): rollout(pi_psi, goal) --> distances[t]
            success = fraction of last `tail` steps under threshold >= 0.8

render.py: loads policy_params.pkl + qpos/qvel from the buffer
    live rollout (policy) -----> side-by-side composite frame
    goal_env teleported to  ---> "dist to goal" text overlay
    goal's qpos/qvel (static)     runs/<name>/videos/*.mp4
```

## Demos

`assets/` holds a few small rendered clips (agent and its actual goal state side-by-side, live distance-to-goal overlaid):

- `point_mass_sustained.mp4` — a clean success: converges and holds
- `point_mass_oscillator.mp4` — a borderline case right at the sustained-metric edge: overshoot-and-correct behavior, close but not fully sustained
- `walker_fail.mp4` — the representative failure mode: random-exploration data never taught the policy to stand
- `value_heatmap.png` — the stage-1 correctness gate on point_mass: `V_theta(s,g)` peaks at the goal and decays outward

## Development notes

Developed against a remote RTX 4090 over SSH (all training happens there, not locally — `dm_control`/`mujoco` need a Python version with prebuilt wheels available; a plain `python3 -m venv` on some platforms will try to build `labmaze` from source via bazel and fail, a conda env with Python 3.11 avoids that). Headless rendering uses `MUJOCO_GL=egl` (no X11 display needed). Independent collection/training runs across domains are launched in parallel once the GPU is free rather than queued sequentially.
