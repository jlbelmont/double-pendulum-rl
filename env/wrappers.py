"""
env/wrappers.py  All environment wrappers for the double/triple pendulum.

Each wrapper adds one layer of behavior:
  CurriculumWrapper          -- staged initialization (swing-up -> balance)
  DoublePendulumRewardWrapper-- two-phase reward shaping
  DomainRandomizationWrapper -- randomizes physics for sim-to-real robustness
  SensorNoiseWrapper         -- simulates real encoder pipeline
  SinCosObsWrapper           -- replaces raw angles with sin/cos pairs

Wrap in this order (inside out):
  CartPoleNLink
    -> CurriculumWrapper
    -> DoublePendulumRewardWrapper
    -> DomainRandomizationWrapper
    -> SensorNoiseWrapper         (optional)
    -> SinCosObsWrapper
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


# ---------------------------------------------------------------------------
# CurriculumWrapper
# ---------------------------------------------------------------------------

class CurriculumWrapper:
    """Two-stage curriculum: swing-up (Stage 1) -> balance (Stage 2).

    Stage 1: pendulums start hanging (theta ~ pi). Agent learns swing-up.
    Stage 2: pendulums start upright (theta ~ 0). Agent learns to balance.

    The stage transition occurs after stage_1_steps total environment steps.

    Parameters
    ----------
    env : CartPoleNLink
        The base environment.
    stage_1_steps : int
        How many total steps to spend in Stage 1 before switching.
    stage_1_init_angle_noise : float
        Noise added to initial angles in Stage 1 (radians).
    stage_2_init_angle_noise : float
        Noise added to initial angles in Stage 2 (radians).
    episode_length : int
        Max steps per episode in both stages.
    """

    def __init__(
        self,
        env: Any,
        stage_1_steps: int = 300000,
        stage_1_init_angle_noise: float = 0.2,
        stage_2_init_angle_noise: float = 0.2,
        episode_length: int = 1000,
    ) -> None:
        self.env = env
        self.n_links = getattr(env, 'n_links', 2)
        self._stage_1_steps = stage_1_steps
        self._s1_noise = stage_1_init_angle_noise
        self._s2_noise = stage_2_init_angle_noise
        self._episode_length = episode_length
        self.current_stage = 1
        self.cumulative_steps = 0
        self._rng = env._rng

    @property
    def obs_space(self) -> Any:
        return self.env.obs_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self.env.reset(seed=seed)
        if self.current_stage == 1:
            self._init_hanging()
        else:
            self._init_upright()
        obs = self.env._get_obs()
        return obs, {"stage": self.current_stage}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, _, terminated, truncated, info = self.env.step(action)
        self.cumulative_steps += 1

        # Stage transition by step count
        if self.current_stage == 1 and self.cumulative_steps >= self._stage_1_steps:
            self.current_stage = 2

        # Termination: cart out of bounds
        x = float(self.env._state[0])
        angles = self.env._state[1:1 + self.n_links]
        if self.current_stage == 2 and np.any(np.abs(angles) > 2.0):
            terminated = True
        if abs(x) > 2.0:
            terminated = True
        if self.env._step_count >= self._episode_length:
            truncated = True

        info["stage"] = self.current_stage
        return obs, 0.0, terminated, truncated, info

    def check_stage_transition(self) -> bool:
        if self.current_stage == 1 and self.cumulative_steps >= self._stage_1_steps:
            self.current_stage = 2
            return True
        return False

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _init_hanging(self) -> None:
        angles = np.full(self.n_links, np.pi) + self._rng.normal(0, self._s1_noise, self.n_links)
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        self.env._state[0] = self._rng.uniform(-0.1, 0.1)
        for i in range(self.n_links):
            self.env._state[1 + i] = angles[i]
        self.env._state[1 + self.n_links] = self._rng.uniform(-0.05, 0.05)
        for i in range(self.n_links):
            self.env._state[2 + self.n_links + i] = self._rng.uniform(-0.1, 0.1)

    def _init_upright(self) -> None:
        angles = self._rng.normal(0, self._s2_noise, self.n_links)
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        self.env._state[0] = self._rng.uniform(-0.05, 0.05)
        for i in range(self.n_links):
            self.env._state[1 + i] = angles[i]
        self.env._state[1 + self.n_links] = self._rng.uniform(-0.02, 0.02)
        for i in range(self.n_links):
            self.env._state[2 + self.n_links + i] = self._rng.uniform(-0.05, 0.05)


# ---------------------------------------------------------------------------
# DoublePendulumRewardWrapper
# ---------------------------------------------------------------------------

class DoublePendulumRewardWrapper:
    """Physics-informed two-phase reward.

    Stage 1 (swing-up):
      Rewards progress toward upright. Allows full rotation.
    Stage 2 (balance):
      Alive bonus minus quadratic angle/velocity/effort penalties.
      Terminates if angles exceed balance_termination_angle.

    Parameters
    ----------
    env : CurriculumWrapper
        Must expose current_stage attribute.
    n_links : int
        Number of pendulum links.
    upright_bonus_threshold : float
        Angle (rad) within which an upright bonus is awarded in Stage 1.
    balance_termination_angle : float
        Angle (rad) beyond which Stage 2 terminates.
    """

    def __init__(
        self,
        env: Any,
        n_links: int = 2,
        upright_bonus_threshold: float = 0.3,
        balance_termination_angle: float = 0.8,
    ) -> None:
        self.env = env
        self._n = n_links
        self._upright_thresh = upright_bonus_threshold
        self._balance_term = balance_termination_angle

    @property
    def obs_space(self) -> Any:
        return self.env.obs_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        return self.env.reset(seed=seed)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, _, terminated, truncated, info = self.env.step(action)

        # Unwrap to get base state
        base = self.env
        while hasattr(base, 'env') and not hasattr(base, '_state'):
            base = base.env
        state = base._state
        x = state[0]
        thetas = state[1:1 + self._n]
        dthetas = state[2 + self._n:2 + 2 * self._n]
        u = float(np.asarray(action).flat[0])

        # Detect stage
        stage = getattr(self.env, 'current_stage', 2)

        if stage == 1:
            reward, extra_term = self._swing_up_reward(thetas, dthetas, u)
        else:
            reward, extra_term = self._balance_reward(thetas, dthetas, x, u)

        terminated = terminated or extra_term
        info['reward_stage'] = stage
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _swing_up_reward(
        self, thetas: np.ndarray, dthetas: np.ndarray, u: float
    ) -> tuple[float, bool]:
        angle_cost = 0.1 * np.sum((np.cos(thetas) - 1.0) ** 2)
        velocity_cost = 0.01 * np.sum(dthetas ** 2)
        effort_cost = 0.001 * u ** 2
        reward = -angle_cost - velocity_cost - effort_cost
        if np.all(np.abs(thetas) < self._upright_thresh):
            reward += 10.0
        return reward, False

    def _balance_reward(
        self, thetas: np.ndarray, dthetas: np.ndarray, x: float, u: float
    ) -> tuple[float, bool]:
        alive = 1.0
        angle_cost = 0.5 * np.sum(thetas ** 2)
        velocity_cost = 0.1 * np.sum(dthetas ** 2)
        position_cost = 0.1 * x ** 2
        effort_cost = 0.01 * u ** 2
        reward = alive - angle_cost - velocity_cost - position_cost - effort_cost
        extra_term = bool(np.any(np.abs(thetas) > self._balance_term))
        return reward, extra_term


# ---------------------------------------------------------------------------
# DomainRandomizationWrapper
# ---------------------------------------------------------------------------

class DomainRandomizationWrapper:
    """Randomizes physical parameters each reset for sim-to-real robustness.

    After warmup_steps total steps, each episode randomly perturbs:
      cart_mass, link_masses, link_lengths, gravity, force_scale, dt.

    The idea is to train on a distribution of systems so the policy
    transfers to the real hardware despite small modeling errors.

    Parameters
    ----------
    env : Any
        Base environment (CartPoleNLink or wrapped).
    cart_mass_range : float
        Max fractional deviation for cart mass (0.10 = +/-10%).
    link_mass_range : float
        Max fractional deviation for link masses.
    link_length_range : float
        Max fractional deviation for link lengths.
    gravity_range : float
        Max fractional deviation for gravity.
    force_scale_range : float
        Max fractional deviation for force scaling (motor variation).
    dt_jitter_range : float
        Max fractional deviation for timestep (scheduling noise).
    enabled : bool
        Set False to disable all randomization.
    warmup_steps : int
        No randomization for the first N total steps.
    rng_seed : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        env: Any,
        cart_mass_range: float = 0.10,
        link_mass_range: float = 0.10,
        link_length_range: float = 0.05,
        gravity_range: float = 0.01,
        force_scale_range: float = 0.15,
        dt_jitter_range: float = 0.10,
        enabled: bool = True,
        warmup_steps: int = 500000,
        rng_seed: int | None = None,
    ) -> None:
        self.env = env
        self._ranges = dict(
            cart=cart_mass_range, mass=link_mass_range, length=link_length_range,
            gravity=gravity_range, force=force_scale_range, dt=dt_jitter_range,
        )
        self._enabled = enabled
        self._warmup_steps = warmup_steps
        self._rng = np.random.default_rng(rng_seed)
        self._total_steps = 0
        self._force_scale = 1.0

        # Store nominal values
        self._nom_cart = float(env.cart_mass)
        self._nom_masses = np.array(env.masses, dtype=np.float64).copy()
        self._nom_lengths = np.array(env.lengths, dtype=np.float64).copy()
        self._nom_g = float(env.g)
        self._nom_dt = float(env.dt)

    @property
    def obs_space(self) -> Any:
        return self.env.obs_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    @property
    def _warmup_active(self) -> bool:
        return self._total_steps < self._warmup_steps

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if self._enabled and not self._warmup_active:
            self._randomize()
        else:
            self._restore()
        return self.env.reset(seed=seed)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        scaled = action * self._force_scale if self._enabled and not self._warmup_active else action
        if self._enabled and not self._warmup_active:
            jitter = self._rng.uniform(1 - self._ranges['dt'], 1 + self._ranges['dt'])
            self.env.dt = self._nom_dt * jitter
        result = self.env.step(scaled)
        self._total_steps += 1
        return result

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _randomize(self) -> None:
        def r(lo, hi):
            return self._rng.uniform(lo, hi)
        rng = self._ranges
        self.env.cart_mass = self._nom_cart * r(1 - rng['cart'], 1 + rng['cart'])
        self.env.masses = self._nom_masses * self._rng.uniform(
            1 - rng['mass'], 1 + rng['mass'], size=self._nom_masses.shape
        )
        self.env.lengths = self._nom_lengths * self._rng.uniform(
            1 - rng['length'], 1 + rng['length'], size=self._nom_lengths.shape
        )
        self.env.g = self._nom_g * r(1 - rng['gravity'], 1 + rng['gravity'])
        self._force_scale = r(1 - rng['force'], 1 + rng['force'])

    def _restore(self) -> None:
        self.env.cart_mass = self._nom_cart
        self.env.masses = self._nom_masses.copy()
        self.env.lengths = self._nom_lengths.copy()
        self.env.g = self._nom_g
        self.env.dt = self._nom_dt
        self._force_scale = 1.0


# ---------------------------------------------------------------------------
# SensorNoiseWrapper
# ---------------------------------------------------------------------------

class SensorNoiseWrapper:
    """Simulates the real hardware sensor pipeline.

    Models:
      1. 14-bit rotary encoders (angle quantization)
      2. Stepper-based position (position quantization)
      3. Angular velocities via finite difference (not measured directly)
      4. Optional observation delay

    Parameters
    ----------
    env : Any
        Environment to wrap.
    encoder_bits : int
        Encoder resolution (14 = 16383 counts/revolution).
    position_resolution : float
        Cart position quantization step (meters).
    velocity_from_finite_diff : bool
        Replace velocity observations with finite-difference estimates.
    observation_delay_steps : int
        Number of steps of delay to simulate (0 = no delay).
    n_links : int
        Number of pendulum links.
    """

    def __init__(
        self,
        env: Any,
        encoder_bits: int = 14,
        position_resolution: float = 0.0001,
        velocity_from_finite_diff: bool = True,
        observation_delay_steps: int = 0,
        n_links: int | None = None,
    ) -> None:
        self.env = env
        self._counts = (2 ** encoder_bits) - 1
        self._angle_q = (2.0 * np.pi) / self._counts
        self._pos_q = position_resolution
        self._fd = velocity_from_finite_diff
        self._delay = observation_delay_steps
        self._n = n_links or getattr(env, 'n_links', 2)
        self._prev: np.ndarray | None = None
        self._buf: list[np.ndarray] = []

    @property
    def obs_space(self) -> Any:
        return self.env.obs_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed)
        self._prev = None
        self._buf.clear()
        noisy = self._add_noise(obs)
        self._prev = noisy.copy()
        return noisy, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        noisy = self._add_noise(obs)
        self._prev = noisy.copy()
        if self._delay > 0:
            self._buf.append(noisy)
            out = self._buf.pop(0) if len(self._buf) > self._delay else self._buf[0]
            return out, reward, terminated, truncated, info
        return noisy, reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _add_noise(self, obs: np.ndarray) -> np.ndarray:
        dt = getattr(self.env, 'dt', 0.005)
        noisy = obs.copy()
        noisy[0] = np.round(noisy[0] / self._pos_q) * self._pos_q
        for i in range(self._n):
            idx = 2 + 2 * i
            noisy[idx] = np.round(noisy[idx] / self._angle_q) * self._angle_q
        if self._fd and self._prev is not None:
            noisy[1] = (noisy[0] - self._prev[0]) / dt
            for i in range(self._n):
                idx = 2 + 2 * i
                noisy[idx + 1] = (noisy[idx] - self._prev[idx]) / dt
        return noisy


# ---------------------------------------------------------------------------
# SinCosObsWrapper
# ---------------------------------------------------------------------------

class SinCosObsWrapper:
    """Replaces raw angle observations with (sin, cos) pairs.

    This eliminates the discontinuity at +/-pi which confuses neural networks.

    Input obs:  [x, x_dot, theta_1, theta_1_dot, theta_2, theta_2_dot, ...]
                 shape: (2 + 2*n_links,)
    Output obs: [x, x_dot, sin(t1), cos(t1), dt1, sin(t2), cos(t2), dt2, ...]
                 shape: (2 + 3*n_links,)

    The agent sees 8D observations for n_links=2 (double pendulum)
    and 11D observations for n_links=3 (triple pendulum).
    """

    def __init__(self, env: Any, n_links: int | None = None) -> None:
        self.env = env
        self.n_links = n_links or getattr(env, 'n_links', 2)
        self._obs_dim = 2 + 3 * self.n_links
        self._obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )

    @property
    def obs_space(self) -> gym.spaces.Box:
        return self._obs_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        return self._transform(obs), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._transform(obs), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _transform(self, obs: np.ndarray) -> np.ndarray:
        out = np.empty(self._obs_dim, dtype=np.float32)
        out[0] = obs[0]  # x
        out[1] = obs[1]  # x_dot
        for i in range(self.n_links):
            theta = obs[2 + 2 * i]
            dtheta = obs[2 + 2 * i + 1]
            base = 2 + 3 * i
            out[base] = np.sin(theta)
            out[base + 1] = np.cos(theta)
            out[base + 2] = dtheta
        return out
