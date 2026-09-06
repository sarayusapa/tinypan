# Findings

Detail behind the results table in [README.md](README.md): why each number is what it is, and how the less obvious parts of the codebase work.

## Evaluation

Success means the distance to the goal stays under a threshold for at least 80% of the last 20 steps, not just the final frame (a converged, oscillating policy can be mid-swing on the last frame) and not just any point in the window (a single lucky pass-through the goal region would count otherwise).

The distance itself is per-dimension standardized: each observation dimension is divided by its training-buffer standard deviation before measuring distance, so one high-variance dimension (a chaotic velocity, an unbounded angle) cannot dominate the whole comparison. `--success-threshold` defaults to `2.0` in these units.

Every number in this project uses a 20-goal eval sample except `push_t`, which was checked at 100 (see below). 20 is fine for a clear result but thin for a close call; the closest numbers in the results table (cartpole/balance's edge over BC, point_mass's tie) have not been individually re-checked at a larger sample.

## Value-guided extraction versus plain imitation

BC (`beta = inf`) trains on the identical buffer as the value-guided policy. The only difference is stage 2's weighting: at `beta = inf`, every transition gets weight 1 regardless of its advantage, so BC is what imitation looks like without the value function judging which transitions were good moves.

| domain / task | value-guided | BC | result |
|---|---|---|---|
| `reacher` / `easy` | 0.75 | 0.20 | 3.75x |
| `point_mass_maze` / `easy` | 0.55 | 0.20 | 2.25x |
| `manipulator` / `bring_ball` | 0.40 | 0.15 | 2.67x |
| `finger` / `spin` | 0.30 | 0.15 | 2x |
| `cartpole` / `balance` | 0.35 | 0.30 | 1.17x |
| `pendulum` / `swingup` | 0.40 | 0.40 | ties |
| `ball_in_cup` / `catch` | 0.20 | 0.20 | ties |
| `point_mass` / `easy` | 0.30 | 0.35 | BC ahead |
| `push_t` / `push` | 0.02-0.03 | 0.00-0.03 | both fail |

Value-guided extraction wins where the exploration buffer has real variety in path quality to the same goal (the maze's forced detour, manipulator's and finger's varied outcomes) and ties BC where it does not (point_mass's short, direct trajectories; pendulum and ball_in_cup's fairly uniform attempts). `push_t` fails for both methods, so there is no variety to exploit either way.

## Beta has no universal value

`beta` interpolates from plain BC (`inf`) toward more selective value-guided filtering. Default is `3.0`, chosen as a robust middle ground across `point_mass` and the maze:

| beta | point_mass | point_mass_maze |
|---|---|---|
| 0.1 | 0.70 | 0.45 |
| 0.3 | 0.65 | 0.20 |
| 3.0 | 0.75 | 0.35 |
| 10 | 0.55 | 0.35 |
| 30 | 0.70 | 0.35 |
| 100 | 0.80 | 0.30 |
| inf (BC) | 0.75 | 0.20 |

Per-domain tuning finds real headroom beyond the default:

| domain | beta=3.0 | best found |
|---|---|---|
| `finger` / `spin` | 0.15 | 0.30 (beta=0.1) |
| `ball_in_cup` / `catch` | 0.05 | 0.20 (beta=30) |
| `cartpole` / `balance` | 0.30 | 0.35 (beta=0.1 or 10) |

## Action repeat is task-dependent, not a universal setting

Holding a sampled action for k steps instead of resampling every step helps tasks that need sustained, directed commitment and hurts tasks that need fast reflexive correction.

Helps: `manipulator` / `bring_ball` (0.00 to 0.40 with `action_repeat=5`, still beats BC on the same data at 0.15). `walker` gets a small coverage bump (mean standing height 0.27 to 0.35) but never enough to stand.

Hurts: `ball_in_cup` / `catch` (0.20 to 0.05 with `action_repeat=5`) and `finger` / `spin` (0.30 to 0.25), both of which need frequent recorrection rather than one sustained push.

There is no way to know in advance which regime a task is in. Collection is cheap enough to just try both.

## Domains that do not work, and why

**`walker` / `walk`**, 0.05 across every exploration strategy tried (random, `action_repeat=10`, `action_repeat=25`, RND), with coverage (mean standing height, max reachable 1.3-1.6) barely moving between them: 0.27, 0.35, 0.34, 0.38. Standing up needs credit assignment across several unrewarding-looking steps (crouch, push off) before a payoff, which no myopic exploration strategy can represent. Fixing this needs real RL-trained exploration (a PPO-trained policy against a novelty reward, with value bootstrapping), not a parameter sweep.

**`push_t` / `push`**, 0.02-0.03 for both value-guided and BC, checked at a 100-goal sample after an initial 20-goal run (0.10 vs 0.05) turned out to be a single lucky rollout. Also checked against 4x more training data (2000 to 8000 episodes with `action_repeat=5`): no change (0.03 to 0.02). Pushing a block to a precise target pose needs sustained, accurately-aimed contact along a specific face, which random exploration essentially never produces on purpose, at any data volume tried. No demo GIF: at this success rate, any rollout picked would either be cherry-picked or show the same failure as everything else.

## RND

`collect.py --exploration rnd` implements Random Network Distillation: a fixed random target network, a predictor trained online to match it, novelty defined as prediction error, and greedy one-step-lookahead action selection over a handful of candidates using branched physics rollouts. It is myopic (no credit assignment across steps, unlike the original RND paper's PPO-trained approach), which is why it does not fix walker or manipulator. It can also make things worse on domains where the right behavior is to hold a state rather than keep moving: `finger` drops from 0.15 to 0.05 with RND, and `pendulum` gets worse too, since novelty-seeking is the opposite of staying near a state already visited, including the states near a goal a converged policy needs to hold.

## Environment notes

**`point_mass_maze`** is built by patching dm_control's own `point_mass.xml`: contacts are disabled by default there (the point mass is normally confined only by joint limits), so re-enabling contacts and adding two offset wall geoms creates a real S-shaped corridor, checked directly against the physics (sustained pushes into a wall show zero leakage, gap traversal is correctly blocked or allowed depending on approach angle). The value heatmap for this domain does not show an obviously wall-aware shape despite the strong behavioral result (2.25x over BC); this is an open discrepancy, not resolved here.

**`push_t`** runs on `gym-pusht` (pymunk physics, Gymnasium API), a different environment entirely from dm_control. `collect.py` adapts it through a small shim (`_PushTEnv`, `_PushTPhysics`) that presents the same interface shape as dm_control (`TimeStep`-like objects, `action_spec()`, a `Physics` object with `get_state()`/`set_state()`/`render()`/`data.qpos`/`data.qvel`), so `train.py` and `render.py` needed no changes. `qpos`/`qvel` here are `[agent_xy, block_xy, block_angle]` and their rates of change, read from the two pymunk bodies directly. RND is not supported for this domain since its branching lookahead would need to replicate gym-pusht's internal control loop. Requires `pymunk<7` (the collision-handler API `gym-pusht` 0.1.6 calls was removed in pymunk 7).

**`EXCLUDE_OBS_KEYS`** in `collect.py` drops observation features that are computed relative to a per-episode randomized task parameter rather than being a portable description of physical state: `reacher`'s `to_target` and `manipulator`'s `target_pos` both vary across episodes even when the physical pose does not, so they are excluded from what counts as state.

**`cartpole` / `swingup`** is not in the results table. Random exploration never gets the pole upright across the entire 400,000-transition buffer (`cos(theta)` never exceeds -0.15, 99.2% of states stay within about 25 degrees of hanging straight down), so any success measured there reflects reaching near-bottom resting poses, not swinging up.
