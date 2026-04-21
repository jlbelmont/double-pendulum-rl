"""
env/cartpole_nlink.py  N-link inverted pendulum on a cart.

Rigid-body Lagrangian dynamics with RK4 integration.
Supports 1 to 5 links. Use n_links=2 for the double pendulum,
n_links=3 for the triple pendulum.

Physical parameters (set in conf/hardware.yaml):
  cart_mass    : mass of the cart (kg)
  link_lengths : pivot-to-pivot total length of each link (m)
  link_masses  : mass of each link (kg)
  link_cms     : center-of-mass distance from pivot for each link (m)
  link_inertias: moment of inertia about each link's own CM (kg*m^2)

State vector (internal):
  [x, theta_1, ..., theta_n, x_dot, theta_1_dot, ..., theta_n_dot]

Observation vector (returned by step/reset):
  [x, x_dot, theta_1, theta_1_dot, theta_2, theta_2_dot, ...]

References
----------
- Bogdanov 2004, "Optimal Control of a Double Inverted Pendulum on a Cart"
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces


class CartPoleNLink:
    """N-link inverted pendulum on a cart with rigid-body Lagrangian dynamics.

    Parameters
    ----------
    n_links : int
        Number of pendulum links (2 = double pendulum, 3 = triple pendulum).
    link_lengths : list[float]
        Pivot-to-pivot total length of each link (meters).
    link_masses : list[float]
        Mass of each link (kg).
    link_cms : list[float] or None
        Center-of-mass distance from the pivot for each link (meters).
        If None, defaults to link_lengths (point mass at tip).
    link_inertias : list[float] or None
        Moment of inertia about each link's own center of mass (kg*m^2).
        If None, defaults to zero (point mass assumption).
    cart_mass : float
        Mass of the cart (kg).
    force_mag : float
        Maximum force applied to the cart (Newtons).
    angle_threshold : float
        Episode ends if any angle exceeds this (radians). Use 6.28 for
        full rotation (swing-up), use 0.4 for balance-only tasks.
    x_threshold : float
        Episode ends if cart position exceeds this (meters).
    max_steps : int
        Maximum steps per episode.
    dt : float
        Simulation timestep (seconds). Use 0.005 for 200 Hz.
    discrete : bool
        If True, action space is Discrete(2) {left, right}.
        If False, action space is Box (continuous force).
    gravity : float
        Gravitational acceleration (m/s^2).
    """

    def __init__(
        self,
        n_links: int = 2,
        link_lengths: list[float] | None = None,
        link_masses: list[float] | None = None,
        link_cms: list[float] | None = None,
        link_inertias: list[float] | None = None,
        cart_mass: float = 1.0,
        force_mag: float = 10.0,
        angle_threshold: float = 6.28,
        x_threshold: float = 0.4191,
        max_steps: int = 1000,
        dt: float = 0.005,
        discrete: bool = False,
        gravity: float = 9.81,
    ) -> None:
        assert 1 <= n_links <= 5, f"n_links must be 1-5, got {n_links}"
        self.n_links = n_links
        self.n_dof = 1 + n_links

        if link_lengths is None:
            link_lengths = [0.5] * n_links
        if link_masses is None:
            link_masses = [0.1] * n_links

        self.lengths = np.array(link_lengths, dtype=np.float64)
        self.masses = np.array(link_masses, dtype=np.float64)
        self.cms = np.array(
            link_cms if link_cms is not None else self.lengths.tolist(),
            dtype=np.float64,
        )
        self.inertias = np.array(
            link_inertias if link_inertias is not None else [0.0] * n_links,
            dtype=np.float64,
        )

        self.cart_mass = float(cart_mass)
        self.force_mag = float(force_mag)
        self.angle_threshold = float(angle_threshold)
        self.x_threshold = float(x_threshold)
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.discrete = bool(discrete)
        self.g = float(gravity)

        # Gymnasium spaces
        obs_dim = 2 + 2 * n_links
        self.obs_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        if self.discrete:
            self.action_space = spaces.Discrete(2)
        else:
            self.action_space = spaces.Box(
                low=np.array([-self.force_mag], dtype=np.float32),
                high=np.array([self.force_mag], dtype=np.float32),
                dtype=np.float32,
            )

        self._state: np.ndarray | None = None
        self._step_count = 0
        self._episode_count = 0
        self._rng = np.random.default_rng()

    def seed(self, s: int) -> None:
        self._rng = np.random.default_rng(s)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed(seed)
        self._step_count = 0
        self._episode_count += 1
        q = self._rng.uniform(-0.05, 0.05, size=self.n_dof)
        dq = self._rng.uniform(-0.05, 0.05, size=self.n_dof)
        self._state = np.concatenate([q, dq])
        return self._get_obs(), {"episode": self._episode_count}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._state is not None, "Call reset() before step()"

        if self.discrete:
            force = self.force_mag if int(action) == 1 else -self.force_mag
        else:
            force = float(np.clip(action, -self.force_mag, self.force_mag))

        self._state = self._rk4_step(self._state, force)
        self._step_count += 1

        q = self._state[: self.n_dof]
        x, angles = q[0], q[1:]

        terminated = bool(
            np.any(np.abs(angles) > self.angle_threshold)
            or np.abs(x) > self.x_threshold
        )
        truncated = self._step_count >= self.max_steps
        reward = 0.0 if terminated else 1.0

        return self._get_obs(), reward, terminated, truncated, {
            "x": float(x),
            "angles": angles.tolist(),
            "step": self._step_count,
        }

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Rigid-body Lagrangian physics
    # ------------------------------------------------------------------

    def _rigid_body_params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute rigid-body derived quantities.

        Returns
        -------
        tail_mass : (n,) -- mass of all links above link i
        l_eff     : (n,) -- masses[i]*cms[i] + lengths[i]*tail_mass[i]
        I_pivot   : (n,) -- inertias[i] + masses[i]*cms[i]^2
        """
        n = self.n_links
        tail_mass = np.zeros(n, dtype=np.float64)
        for i in range(n - 2, -1, -1):
            tail_mass[i] = self.masses[i + 1] + tail_mass[i + 1]
        l_eff = self.masses * self.cms + self.lengths * tail_mass
        I_pivot = self.inertias + self.masses * self.cms ** 2
        return tail_mass, l_eff, I_pivot

    def _mass_matrix(self, theta: np.ndarray) -> np.ndarray:
        n, ndof = self.n_links, self.n_dof
        M = np.zeros((ndof, ndof), dtype=np.float64)
        tail_mass, l_eff, I_pivot = self._rigid_body_params()

        M[0, 0] = self.cart_mass + np.sum(self.masses)
        for j in range(n):
            val = l_eff[j] * np.cos(theta[j])
            M[0, j + 1] = val
            M[j + 1, 0] = val
        for i in range(n):
            for j in range(n):
                if i == j:
                    M[i + 1, i + 1] = I_pivot[i] + self.lengths[i] ** 2 * tail_mass[i]
                else:
                    big, sml = max(i, j), min(i, j)
                    M[i + 1, j + 1] = (
                        self.lengths[sml] * l_eff[big] * np.cos(theta[i] - theta[j])
                    )
        return M

    def _forces(self, theta: np.ndarray, dtheta: np.ndarray, force: float) -> np.ndarray:
        n, ndof = self.n_links, self.n_dof
        f = np.zeros(ndof, dtype=np.float64)
        _, l_eff, _ = self._rigid_body_params()

        f[0] = force
        for j in range(n):
            f[0] += l_eff[j] * np.sin(theta[j]) * dtheta[j] ** 2

        for i in range(n):
            f[i + 1] = -l_eff[i] * self.g * np.sin(theta[i])
            for j in range(n):
                if j == i:
                    continue
                big, sml = max(i, j), min(i, j)
                f[i + 1] += (
                    self.lengths[sml] * l_eff[big]
                    * np.sin(theta[i] - theta[j]) * dtheta[j] ** 2
                )
        return f

    def _dynamics(self, state: np.ndarray, force: float) -> np.ndarray:
        ndof = self.n_dof
        q, dq = state[:ndof], state[ndof:]
        theta, dtheta = q[1:], dq[1:]
        M = self._mass_matrix(theta)
        f = self._forces(theta, dtheta, force)
        ddq = np.linalg.solve(M, f)
        return np.concatenate([dq, ddq])

    def _rk4_step(self, state: np.ndarray, force: float) -> np.ndarray:
        dt = self.dt
        k1 = self._dynamics(state, force)
        k2 = self._dynamics(state + 0.5 * dt * k1, force)
        k3 = self._dynamics(state + 0.5 * dt * k2, force)
        k4 = self._dynamics(state + dt * k3, force)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _get_obs(self) -> np.ndarray:
        """Re-order state to observation format.

        Internal: [x, theta_1, ..., theta_n, x_dot, theta_1_dot, ..., theta_n_dot]
        Obs:      [x, x_dot, theta_1, theta_1_dot, theta_2, theta_2_dot, ...]
        """
        ndof = self.n_dof
        q, dq = self._state[:ndof], self._state[ndof:]
        obs = np.empty(2 + 2 * self.n_links, dtype=np.float32)
        obs[0] = q[0]
        obs[1] = dq[0]
        for i in range(self.n_links):
            obs[2 + 2 * i] = q[1 + i]
            obs[2 + 2 * i + 1] = dq[1 + i]
        return obs
