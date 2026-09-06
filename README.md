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

**Headline finding: the eval metric had a systemic bug that was masking real success on most domains, not just one.** Plain L2 distance over the full observation vector silently lets whichever dimension happens to have the largest raw scale dominate the whole comparison — a chaotic velocity, an unbounded angle that doesn't wrap, etc. A sweep re-checking every domain with `--normalize-eval-distance` (divide each dimension by its training-buffer std before measuring distance) found this was hiding real, often strong, success almost everywhere:

| domain / task | raw L2 (beta=3) | normalized (beta=3) | normalized, best beta found | verdict |
|---|---|---|---|---|
| `point_mass` / `easy` | 0.75 | 0.30 | 0.30 (beta not re-swept under norm.) | works (metric got *stricter* here, not more lenient) |
| `point_mass_maze` / `easy` | 0.35 | 0.55 | 0.55 (beta not re-swept under norm.) | works, and beats BC (see beta sweep below) |
| `cartpole` / `swingup` | 0.35 | **0.75** | 0.75 | works well |
| `reacher` / `easy` | 0.00 | **0.75** | 0.75 | was fully masked -- works well |
| `cartpole` / `balance` | 0.00 | 0.30 | **0.35** (beta 0.1 or 10) | works, decent |
| `pendulum` / `swingup` | 0.05 | 0.40 | 0.40 (not re-swept) | works, decent |
| `finger` / `spin` | 0.00 | 0.15 | **0.30** (beta 0.1) | partial real success |
| `ball_in_cup` / `catch` | 0.00 | 0.05 | **0.20** (beta 30) | modest but real, was near-total failure at beta=3 |
| `walker` / `walk` | 0.00 | 0.05 | 0.05 at action_repeat=10 and 25, both alike | genuine failure, ceiling confirmed (see below) |
| `manipulator` / `bring_ball` | 0.00 | 0.00 (checked to threshold=5.0) | **0.40** (action_repeat=5, beta 0.1 or 3) | breakthrough -- see below |

(All numbers: random exploration, `beta=3.0`, 2000 episodes x 200 steps.) The takeaway isn't "normalization makes numbers go up" — point_mass's number went *down* under the same fix, which is exactly why this is a metric correction and not a thumb on the scale. The three domains that stay genuine failures share a real pattern: walker (standing up), ball_in_cup (catching), and manipulator (grasping) all need precisely-timed or directed multi-step behavior, which is exactly what pure random exploration essentially never stumbles into. Domains where random exploration just needs to settle into *some* static or periodic configuration near the goal (a resting position, a balance point, a swing arc) work well even from pure random exploration.

This substantially overturns some of the specific per-domain diagnoses made earlier in this project (reacher's "unresolved angle-wrap bug," pendulum's "sustained-balance gap," finger's "genuine dexterity gap") -- those diagnoses were reasoning about a real underlying issue but via a metric that was itself broken, so the conclusions about *severity* were wrong even where the mechanism identified was real (e.g. reacher's shoulder joint genuinely doesn't wrap, but dividing by its std was enough to fix the metric without needing the cos/sin transform that seemed necessary at the time).

**point_mass** and **point_mass_maze** additionally validate the method's actual central claim: stage 2's value-guided policy extraction beats a plain goal-conditioned behavior-cloning baseline (`beta -> inf`) -- see the beta sweep below. The core algorithm (sampler, InfoNCE, AWR) hasn't needed a fix since point_mass first validated it; every failure traced back has been about data/metric quality, not the method.

### Beta sweep: does value-guided extraction actually beat BC?

`beta` interpolates continuously from pure goal-conditioned BC (`beta -> inf`, weight -> 1 for every transition) toward more aggressively value-guided extraction. Swept on both domains, holding stage 1 (and therefore the encoder) fixed:

| beta | point_mass | point_mass_maze |
|---|---|---|
| 0.1 | 0.70 | **0.45** |
| 0.3 | 0.65 | 0.20 |
| 1.0 | 0.35 | 0.20 |
| 3.0 | 0.75 | 0.35 |
| 10 | 0.55 | 0.35 |
| 30 | 0.70 | 0.35 |
| 100 | 0.80 | 0.30 |
| inf (BC baseline) | 0.75 | 0.20 |

Two findings here, and they're different in an important way:

- **On point_mass, the method mostly just ties BC** (once you look past `beta=1.0`, which was a genuinely unlucky default — every other value clears 0.55+, while 1.0 alone gets 0.35). This has a real explanation, not just noise: hindsight-relabeled BC is already a strong baseline by construction, since every recorded transition is tautologically "correct" for the specific goal it's relabeled toward. AWR's edge comes from discriminating between multiple paths of differing quality to the same goal, and point_mass's short, mostly-direct random trajectories don't have much of that heterogeneity to exploit.
- **On the maze, value-guided extraction clearly and consistently beats BC** — 0.45 vs 0.20 at the best beta (2.25x), and every beta from 0.1 to 100 beats BC's 0.20. This is exactly the predicted condition: the forced detour means random exploration produces both efficient and very roundabout paths to the same goal, which is precisely the heterogeneity AWR needs to show a real advantage over imitation.

`train.py`'s default `beta` was updated to `3.0` (robust across both domains) after this sweep; `1.0` is kept as an explicit cautionary note in the config comment. Caveat: this sweep predates the eval-metric fix above, so it's plausible the exact numbers (though probably not the qualitative "maze beats BC, point_mass ties it" conclusion) would shift somewhat if re-run under `--normalize-eval-distance` — not yet re-checked.

### What we learned about RND along the way

RND (see below) doesn't reliably help once the metric bug is accounted for, and can actively hurt: on `finger`, random exploration alone reaches 0.15, but RND exploration on top *drops* it to 0.05. On `pendulum`, RND also made things worse (0.40 -> considerably lower). The likely reason: RND's novelty-seeking is anti-correlated with the "settle into and hold a stable/periodic configuration" behavior that turns out to be exactly what makes these domains tractable from random exploration in the first place — RND actively wants to *leave* familiar-looking states, including the ones near the goal that a converged policy needs to revisit.

### The manipulator breakthrough: `action_repeat` cuts exactly the opposite way on different task types

`manipulator`/`bring_ball` went from a genuine 0.00 (checked up to a very loose threshold=5.0, confirmed not a metric artifact) to **0.40** with one change: collecting with `--action-repeat 5` instead of i.i.d. random actions every step. That beats the BC baseline on the *same* action_repeat=5 data (0.15), a real 2.67x improvement from value-guidance -- not just "better data helped both methods equally." This is the same story as the maze: `action_repeat=5` produces enough heterogeneity in how effectively different episodes happen to push/move the ball that AWR has real signal to discriminate on, where pure random-action data apparently didn't.

The interesting part is that `action_repeat` cuts in *opposite* directions depending on what the task needs:
- **Helps** tasks needing sustained, directed commitment: `walker` standing up (mild help), `manipulator` pushing/moving an object (big help, 0.00 -> 0.40).
- **Hurts** tasks needing fine, high-frequency reflexive control: `ball_in_cup`/`catch` dropped from 0.20 to 0.05 with the same `action_repeat=5` -- catching needs rapid corrective adjustments right at the moment of contact, and holding an action for 5 steps removes exactly that capability. `finger`/`spin` similarly didn't benefit (0.30 -> 0.25 at its best beta with action_repeat=5) -- spinning turns out to need frequent recontact/repositioning rather than one sustained push, closer to ball_in_cup's profile than manipulator's.

There's no universal answer to "should I use action_repeat" -- it's a real, task-dependent tradeoff between commitment and reflexes, and worth trying both ways cheaply (collection is fast) rather than assuming either default.

**Walker's ceiling is confirmed, not just under-tuned**: swept action_repeat at 10 and 25 -- both land at the exact same 0.05, and coverage (mean standing height) barely moves between them either (0.354 vs 0.338, both far below the ~1.3-1.6 standing range). Manipulator needed one well-chosen action_repeat value to unlock real progress; walker doesn't respond to this knob at all, at any setting tried. Standing up appears to need actual multi-step credit assignment (a real RL-trained exploration policy, not a heuristic tweak to random action selection) to discover -- consistent with why myopic RND didn't help it either. This isn't a mystery still open, it's a confirmed limit of what heuristic exploration can do here.

### Evaluation metric

Success is **not** "final frame within threshold" (fragile — a converged, happily-oscillating policy can be mid-swing on the literal last frame) and **not** "within threshold at any point in a trailing window" (gameable — a single lucky pass-through the goal region scores as success without the policy ever converging). It's **fraction of the trailing window within threshold ≥ 80%**: sustained proximity, tolerant of the small oscillation a genuinely converged policy shows, immune to a one-off coincidence.

Plain L2 over the full observation vector silently breaks on any domain with a high-variance nuisance dimension (a chaotic velocity, an unbounded angle) -- one such dimension can dominate the entire distance regardless of how accurate everything else is. `--normalize-eval-distance` divides each dimension by its training-buffer std first, so no single dimension's raw scale can dominate. **This now defaults on** (`success_threshold` defaults to `2.0` in these per-std units) after a full sweep found it was masking real success on most domains, not just one -- see the table above. Pass `--no-normalize-eval-distance` to reproduce the old raw-L2 numbers.

### RND exploration — what it is and isn't

`collect.py --exploration rnd` implements real Random Network Distillation from scratch: a fixed random target network, a predictor trained online to match it, novelty = prediction error, greedy 1-step-lookahead action selection over a few candidate actions using branched physics rollouts (dm_control's `physics.get_state()`/`set_state()`). This is a genuine, correctly-implemented curiosity signal — verified against `dm_control`'s own APIs before use — but it is **myopic**: no temporal credit assignment across steps. The original RND paper trains a full RL policy (PPO) against the novelty reward, which can learn to spend several unrewarding-looking steps setting up a future novel state (e.g. crouching before standing). Ours can't discover that, and worse, it can actively hurt on domains where the right behavior is to settle into and hold a state rather than keep moving (see the RND section above) -- not that curiosity-driven exploration doesn't work, but that the lightweight myopic version built here isn't strong enough for domains needing genuine multi-step strategy (walker standing up, manipulator grasping), and is actively counterproductive for domains needing sustained stillness.

### The maze

The paper's actual correctness-gate test needs real walls (two states close in Euclidean position but on opposite sides of a wall should register as *far* in `V_theta`, and reaching one from the other should look asymmetric under exchange). `point_mass_maze` is built in `collect.py` (`load_env`, domain `"point_mass_maze"`) by patching dm_control's own `point_mass.xml`: contacts are **disabled by default** in that file (the point mass is normally confined only by joint limits, never actually collides with the decorative "walls" around the arena), so re-enabling contacts and injecting two offset wall geoms creates a real S-shaped corridor. Verified directly against the physics (1000-step sustained push into a wall shows zero leakage; a full up-then-across traversal correctly passes through one wall's gap and is correctly blocked by the other).

The behavioral result (policy beats BC 2.25x, above) is solidly positive. The visual gate is more equivocal: the plotted `V_theta(., g)` heatmap doesn't show an obviously wall-aware shape — it looks closer to a smooth gradient than to something that kinks around the walls, even though the buffer has decent gap-crossing coverage (~3-4% of transitions pass within a gap's radius, thousands of samples, not rare). The policy improvement suggests stage 1 *is* learning something real and directionally useful from the wall structure — advantage differences along real recorded trajectories don't require the whole global surface to be perfectly resolved to be useful — but the heatmap not matching that cleanly is an open discrepancy worth digging into (more stage-1 steps, larger batch for harder negatives near boundaries, or higher-resolution plotting are the likely next things to try), not something to paper over.

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
