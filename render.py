"""Renders a trained policy's rollout side-by-side with the actual goal state
(teleported from raw qpos/qvel, since task obs can't be inverted back into a
scene for most domains), with the live distance-to-goal overlaid.

Usage:
    python render.py --data data/point_mass_easy.npz --run runs/point_mass_easy
"""
import argparse
import os
os.environ.setdefault("MUJOCO_GL", "egl")  # headless GPU rendering, no X11 display

import pickle
from pathlib import Path

import imageio
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from collect import EXCLUDE_OBS_KEYS, flatten_obs, load_env
from train import Config, Policy, scale_action

SIZE = (240, 320)  # (height, width)

# Several tasks' own original reward target (reacher's "target", point_mass's
# "target", manipulator's "target_ball") is unrelated to our goal-conditioning
# (we condition on a full state, not that marker) but still gets rendered,
# showing a distracting, irrelevant ball at a random position. Hide it.
_STRAY_TARGET_NAMES = ("target", "target_ball")


def hide_stray_targets(env):
    for name in _STRAY_TARGET_NAMES:
        try:
            env.physics.named.model.geom_rgba[name] = [0, 0, 0, 0]
        except (KeyError, AttributeError):
            pass  # push_t's physics shim has no .named; other domains just don't have this geom


def label(frame, lines, color=(255, 255, 0)):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, SIZE[1], 14 * len(lines) + 4], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((2, 2 + 14 * i), line, fill=color)
    return np.array(img)


def main(data_path, run_dir, episodes, horizon, tail, threshold, fps, out_dir, fmt="mp4", stride=1):
    npz = np.load(data_path)
    obs, act = npz["obs"], npz["act"]
    qpos, qvel = npz["qpos"], npz["qvel"]
    act_min, act_max = jnp.array(npz["action_min"]), jnp.array(npz["action_max"])
    domain, task = str(npz["domain"]), str(npz["task"])
    exclude = EXCLUDE_OBS_KEYS.get(domain, ())
    act_dim = act.shape[-1]

    cfg = Config()  # must match the hidden/depth used at train time (defaults, unless overridden)
    policy = Policy(hidden=cfg.hidden, depth=cfg.depth, act_dim=act_dim)
    with open(Path(run_dir) / "policy_params.pkl", "rb") as f:
        policy_params = pickle.load(f)

    n_holdout = max(1, int(obs.shape[0] * cfg.held_out_frac))
    holdout_obs, holdout_qpos, holdout_qvel = obs[-n_holdout:], qpos[-n_holdout:], qvel[-n_holdout:]

    env = load_env(domain, task)
    goal_env = load_env(domain, task)  # separate physics instance, teleported to render the goal
    hide_stray_targets(env)
    hide_stray_targets(goal_env)
    fps = fps or int(round(1.0 / env.control_timestep()))
    rng = np.random.default_rng(0)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for ep in range(episodes):
        e_idx, t_idx = rng.integers(0, holdout_obs.shape[0]), rng.integers(0, holdout_obs.shape[1])
        goal, g_qpos, g_qvel = holdout_obs[e_idx, t_idx], holdout_qpos[e_idx, t_idx], holdout_qvel[e_idx, t_idx]

        goal_env.reset()
        goal_env.physics.data.qpos[:] = g_qpos
        goal_env.physics.data.qvel[:] = g_qvel
        goal_env.physics.forward()
        goal_frame = label(goal_env.physics.render(*SIZE, camera_id=0), ["GOAL (fixed target)"])

        ts = env.reset()
        frames, dists = [], []
        for t in range(horizon):
            s = flatten_obs(ts.observation, exclude)
            dist = float(np.linalg.norm(s - goal))
            dists.append(dist)
            agent_frame = label(env.physics.render(*SIZE, camera_id=0),
                                 [f"AGENT  step {t}/{horizon}", f"dist to goal: {dist:.3f}"])
            frames.append(np.concatenate([agent_frame, goal_frame], axis=1))
            a_unit = policy.apply(policy_params, jnp.array(s)[None], jnp.array(goal)[None])[0]
            ts = env.step(np.array(scale_action(a_unit, act_min, act_max)))

        window = np.array(dists[-tail:])
        frac_close = (window < threshold).mean() if threshold else float("nan")
        path = out / f"{domain}_{task}_ep{ep}.{fmt}"
        save_frames = frames[::stride]  # gif has no compression, so subsample to keep file size sane
        imageio.mimsave(path, save_frames, fps=max(1, fps // stride))
        print(f"saved {path}  (final dist={dists[-1]:.3f}, "
              f"fraction of last {tail} steps < {threshold}: {frac_close:.2f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--tail", type=int, default=20)
    p.add_argument("--success-threshold", type=float, default=0.05)
    p.add_argument("--fps", type=int, default=0, help="0 = infer from control timestep")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    p.add_argument("--stride", type=int, default=1, help="keep every Nth frame (gif has no compression)")
    args = p.parse_args()
    main(args.data, args.run, args.episodes, args.horizon, args.tail, args.success_threshold, args.fps,
         args.out_dir or f"{args.run}/videos", args.format, args.stride)
