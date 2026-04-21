"""
agent/sac.py  Soft Actor-Critic (SAC) -- fully self-contained.

Haarnoja et al. 2018 (https://arxiv.org/abs/1801.01290)
Automatic entropy tuning: Haarnoja et al. 2018b.

This file has zero dependencies outside of numpy, torch, and Python stdlib.
Everything needed (networks, replay buffer, utilities) is defined here.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Transition dataclass
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    terminated: bool
    truncated: bool = False

    @property
    def done(self) -> bool:
        return self.terminated


# ---------------------------------------------------------------------------
# Neural network building blocks
# ---------------------------------------------------------------------------

def _mlp(
    in_dim: int,
    out_dim: int,
    hidden: Sequence[int] = (256, 256),
) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class _Actor(nn.Module):
    """Squashed Gaussian actor for SAC (tanh-bounded continuous actions)."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, action_dim: int, hidden: Sequence[int]) -> None:
        super().__init__()
        self.trunk = _mlp(obs_dim, hidden[-1], list(hidden[:-1]))
        self.mu = nn.Linear(hidden[-1], action_dim)
        self.log_std = nn.Linear(hidden[-1], action_dim)
        nn.init.constant_(self.log_std.weight, 0.0)
        nn.init.constant_(self.log_std.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        raw = mu if deterministic else mu + std * torch.randn_like(mu)
        action = torch.tanh(raw)
        log_prob = (
            torch.distributions.Normal(mu, std).log_prob(raw)
            - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(dim=-1)
        return action, log_prob


class _SoftQ(nn.Module):
    """Q(s, a) for continuous actions."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: Sequence[int]) -> None:
        super().__init__()
        self.net = _mlp(obs_dim + action_dim, 1, hidden)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class _ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def add(self, t: Transition) -> None:
        self._buf.append(t)

    def sample(self, n: int, device: torch.device) -> dict[str, torch.Tensor]:
        batch = random.sample(self._buf, n)
        f32 = torch.float32
        obs = torch.tensor(np.array([t.obs for t in batch]), dtype=f32, device=device)
        nobs = torch.tensor(np.array([t.next_obs for t in batch]), dtype=f32, device=device)
        acts = torch.tensor(np.array([t.action for t in batch]), dtype=f32, device=device)
        rews = torch.tensor([t.reward for t in batch], dtype=f32, device=device)
        dones = torch.tensor([float(t.done) for t in batch], dtype=f32, device=device)
        return {"obs": obs, "next_obs": nobs, "actions": acts, "rewards": rews, "dones": dones}

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------

class SACAgent:
    """Soft Actor-Critic for continuous action spaces.

    Parameters
    ----------
    obs_dim : int
        Observation dimension (8 for double pendulum with SinCos wrapper).
    action_dim : int
        Action dimension (1 for single-axis cart force).
    hidden_dims : tuple[int, ...]
        MLP hidden layer widths for all networks.
    actor_lr : float
        Actor learning rate.
    critic_lr : float
        Critic (Q-network) learning rate.
    alpha_lr : float
        Entropy temperature learning rate.
    gamma : float
        Discount factor.
    tau : float
        Soft target update coefficient (small = slow update).
    init_alpha : float
        Initial entropy temperature.
    replay_capacity : int
        Maximum number of transitions stored.
    batch_size : int
        Transitions sampled per update step.
    warmup_steps : int
        Steps before the first network update.
    device : str
        'auto', 'cpu', 'cuda', or 'mps'.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        init_alpha: float = 0.2,
        replay_capacity: int = 1_000_000,
        batch_size: int = 256,
        warmup_steps: int = 5000,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if device == "auto":
            if torch.cuda.is_available():
                self._dev = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self._dev = torch.device("mps")
            else:
                self._dev = torch.device("cpu")
        else:
            self._dev = torch.device(device)

        self._action_dim = action_dim
        self._gamma = gamma
        self._tau = tau
        self._batch_size = batch_size
        self._warmup = warmup_steps

        self._actor = _Actor(obs_dim, action_dim, hidden_dims).to(self._dev)
        self._q1 = _SoftQ(obs_dim, action_dim, hidden_dims).to(self._dev)
        self._q2 = _SoftQ(obs_dim, action_dim, hidden_dims).to(self._dev)
        self._q1t = _SoftQ(obs_dim, action_dim, hidden_dims).to(self._dev)
        self._q2t = _SoftQ(obs_dim, action_dim, hidden_dims).to(self._dev)
        self._q1t.load_state_dict(self._q1.state_dict())
        self._q2t.load_state_dict(self._q2.state_dict())

        self._actor_opt = torch.optim.Adam(self._actor.parameters(), lr=actor_lr)
        self._critic_opt = torch.optim.Adam(
            list(self._q1.parameters()) + list(self._q2.parameters()), lr=critic_lr
        )
        self._target_entropy = float(-action_dim)
        self._log_alpha = torch.tensor(math.log(init_alpha), requires_grad=True, device=self._dev)
        self._alpha_opt = torch.optim.Adam([self._log_alpha], lr=alpha_lr)

        self._buf = _ReplayBuffer(replay_capacity)
        self._steps = 0
        self._updates = 0
        self.training = True

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self._dev).unsqueeze(0)
        with torch.no_grad():
            action, _ = self._actor.act(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def store(self, transition: Transition) -> None:
        self._buf.add(transition)
        self._steps += 1

    def update(self) -> dict[str, float]:
        if len(self._buf) < max(self._warmup, self._batch_size):
            return {}

        data = self._buf.sample(self._batch_size, self._dev)
        obs, nobs = data["obs"], data["next_obs"]
        acts, rews, dones = data["actions"], data["rewards"], data["dones"]
        alpha = self._log_alpha.exp().detach()

        # Critic update
        with torch.no_grad():
            na, nlp = self._actor.act(nobs)
            q_next = torch.min(self._q1t(nobs, na), self._q2t(nobs, na)) - alpha * nlp
            q_tgt = rews + self._gamma * (1 - dones) * q_next

        q1p, q2p = self._q1(obs, acts), self._q2(obs, acts)
        critic_loss = F.mse_loss(q1p, q_tgt) + F.mse_loss(q2p, q_tgt)
        self._critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self._q1.parameters()) + list(self._q2.parameters()), 10.0
        )
        self._critic_opt.step()

        # Actor update
        pi, lp = self._actor.act(obs)
        actor_loss = (alpha * lp - torch.min(self._q1(obs, pi), self._q2(obs, pi))).mean()
        self._actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self._actor.parameters(), 10.0)
        self._actor_opt.step()

        # Temperature update
        alpha_loss = -(self._log_alpha * (lp.detach() + self._target_entropy)).mean()
        self._alpha_opt.zero_grad()
        alpha_loss.backward()
        self._alpha_opt.step()

        # Soft target update
        for p, pt in zip(self._q1.parameters(), self._q1t.parameters()):
            pt.data.mul_(1 - self._tau).add_(self._tau * p.data)
        for p, pt in zip(self._q2.parameters(), self._q2t.parameters()):
            pt.data.mul_(1 - self._tau).add_(self._tau * p.data)

        self._updates += 1
        return {
            "loss/critic": float(critic_loss.detach()),
            "loss/actor": float(actor_loss.detach()),
            "agent/alpha": float(alpha),
            "agent/entropy": float(-lp.mean().detach()),
        }

    def eval(self) -> None:
        self.training = False
        self._actor.eval()

    def train(self) -> None:
        self.training = True
        self._actor.train()

    def save(self, path: str) -> None:
        torch.save({
            "actor": self._actor.state_dict(),
            "q1": self._q1.state_dict(),
            "q2": self._q2.state_dict(),
            "q1t": self._q1t.state_dict(),
            "q2t": self._q2t.state_dict(),
            "actor_opt": self._actor_opt.state_dict(),
            "critic_opt": self._critic_opt.state_dict(),
            "log_alpha": self._log_alpha.item(),
            "steps": self._steps,
            "updates": self._updates,
        }, path)
        print(f"[OK] Saved: {path}")

    def load(self, path: str) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location=self._dev)
        self._actor.load_state_dict(state["actor"])
        self._q1.load_state_dict(state["q1"])
        self._q2.load_state_dict(state["q2"])
        self._q1t.load_state_dict(state["q1t"])
        self._q2t.load_state_dict(state["q2t"])
        self._actor_opt.load_state_dict(state["actor_opt"])
        self._critic_opt.load_state_dict(state["critic_opt"])
        self._log_alpha = torch.tensor(
            state["log_alpha"], requires_grad=True, device=self._dev
        )
        alpha_lr = self._alpha_opt.defaults["lr"]
        self._alpha_opt = torch.optim.Adam([self._log_alpha], lr=alpha_lr)
        self._steps = state.get("steps", 0)
        self._updates = state.get("updates", 0)
        print(f"[OK] Loaded: {path}  (steps={self._steps})")

    def export_torchscript(self, path: str, obs_dim: int) -> None:
        """Export actor to TorchScript for deployment on Raspberry Pi."""
        self._actor.eval()
        dummy = torch.randn(1, obs_dim)
        traced = torch.jit.trace(self._actor, dummy)
        torch.jit.save(traced, path)
        print(f"[OK] TorchScript exported: {path}")
        self._actor.train()
