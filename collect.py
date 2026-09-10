"""Reward-free exploration data collection. See README.md for the method.

Buffer layout (fixed-length episodes, so the sampler in train.py is plain
vectorized indexing instead of an offset table):
    obs:  (num_episodes, L + 1, obs_dim)
    act:  (num_episodes, L,     act_dim)   act[e, t] takes obs[e, t] to obs[e, t + 1]
    qpos/qvel: (num_episodes, L + 1, nq/nv) raw physics state, for render.py's
        goal-teleport (task obs is often a derived/partial view of physics
        state and can't be inverted back into a scene).
"""
import argparse

import numpy as np
from dm_control import suite


# to_target/target_pos are computed relative to that episode's own randomly
# placed built-in target, not a portable description of state across episodes.
EXCLUDE_OBS_KEYS = {
    "reacher": {"to_target"},
    "manipulator": {"target_pos"},
}


def flatten_obs(obs_dict, exclude=()) -> np.ndarray:
    if isinstance(obs_dict, dict):
        return np.concatenate([np.ravel(v) for k, v in sorted(obs_dict.items()) if k not in exclude]).astype(np.float32)
    return np.asarray(obs_dict, dtype=np.float32)  # push_t's obs is already flat


class RND:
    """Random Network Distillation novelty scorer for greedy exploration.
    Fixed random target net + online predictor; novelty = prediction error.
    One-step lookahead: branch a few candidate actions from the current
    physics state, score the resulting states, commit to the most novel."""

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
    """point_mass with two offset walls forming an S-shaped corridor. Contacts
    are disabled by default in dm_control's point_mass.xml (it's normally
    confined only by joint limits), so the usual arena boundary is decorative;
    this re-enables contacts and patches in real walls."""
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


class _GymTimeStep:
    """Enough of dm_control's TimeStep for the rest of the codebase to not
    care whether it's talking to dm_control or a gymnasium env."""
    def __init__(self, observation, done):
        self.observation = observation
        self._done = done

    def last(self):
        return self._done


class _BoxSpec:
    """dm_control's action_spec() shape, backed by a gymnasium Box."""
    def __init__(self, box):
        self.shape = box.shape
        self.minimum = box.low
        self.maximum = box.high


class _PushTPhysics:
    """dm_control's Physics interface, backed by push_t's two pymunk bodies.
    qpos/qvel are [agent_xy, block_xy, block_angle] and their derivatives."""
    def __init__(self, raw_env):
        self._raw = raw_env
        self.data = argparse.Namespace(qpos=None, qvel=None)
        self.model = argparse.Namespace(nq=5, nv=5)
        self.sync()

    def sync(self):
        u = self._raw
        self.data.qpos = np.array([*u.agent.position, *u.block.position, u.block.angle], dtype=np.float64)
        self.data.qvel = np.array([*u.agent.velocity, *u.block.velocity, u.block.angular_velocity], dtype=np.float64)

    def get_state(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).copy()

    def set_state(self, state):
        self.data.qpos[:], self.data.qvel[:] = state[:5], state[5:]
        self.forward()

    def forward(self):
        u = self._raw
        qpos, qvel = self.data.qpos, self.data.qvel
        u.agent.position, u.block.position, u.block.angle = tuple(qpos[0:2]), tuple(qpos[2:4]), float(qpos[4])
        u.agent.velocity, u.block.velocity, u.block.angular_velocity = tuple(qvel[0:2]), tuple(qvel[2:4]), float(qvel[4])

    def render(self, height, width, camera_id=0):
        from PIL import Image
        return np.array(Image.fromarray(self._raw.render()).resize((width, height)))

    def timestep(self):
        return self._raw.dt


class _PushTEnv:
    """Adapts gym-pusht's Gymnasium API to the dm_control-shaped interface
    the rest of the codebase expects. Random exploration only: RND's branched
    lookahead would need to replicate push_t's internal control loop."""
    def __init__(self, gym_env):
        self._env = gym_env
        self.physics = _PushTPhysics(gym_env.unwrapped)

    def action_spec(self):
        return _BoxSpec(self._env.action_space)

    def reset(self):
        obs, _ = self._env.reset()
        self.physics.sync()
        return _GymTimeStep(obs, False)

    def step(self, action):
        obs, _, terminated, truncated, _ = self._env.step(action)
        self.physics.sync()
        return _GymTimeStep(obs, terminated or truncated)

    def control_timestep(self):
        return 1.0 / self._env.unwrapped.control_hz


def _push_t_env(seed):
    import gymnasium as gym
    import gym_pusht  # noqa: F401 -- registers gym_pusht/PushT-v0
    gym_env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    gym_env.reset(seed=seed)  # gymnasium seeds via reset(); later resets continue the stream
    return _PushTEnv(gym_env)


def load_env(domain: str, task: str, seed=None):
    if domain == "point_mass_maze":
        return _point_mass_maze_env(seed)
    if domain == "push_t":
        return _push_t_env(seed)
    kwargs = {"random": seed} if seed is not None else {}
    return suite.load(domain_name=domain, task_name=task, task_kwargs=kwargs)


def collect(domain: str, task: str, num_episodes: int, episode_length: int, seed: int,
            action_repeat: int = 1, exploration: str = "random", rnd_candidates: int = 8):
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
            ts = env.step(a)
            act[e, t] = a
            if rnd is not None:
                rnd.learn(flatten_obs(ts.observation, exclude)[None])
            if ts.last():
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
