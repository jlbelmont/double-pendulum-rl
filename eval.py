#!/usr/bin/env python3
"""
eval.py  Evaluate a trained policy before deploying to hardware.

Runs 5 test conditions and prints a robustness table:
  1. Nominal (clean dynamics, no noise)
  2. +/-10% parameter variation
  3. +/-20% parameter variation
  4. Real sensor noise (14-bit encoder, finite-diff velocities)
  5. Sensor noise + 1-step latency

Usage
-----
    python3 eval.py --checkpoint results/run_20240101/final_checkpoint.pt
    python3 eval.py --checkpoint results/run_20240101/final_checkpoint.pt --n-episodes 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from agent.sac import SACAgent
from env.cartpole_nlink import CartPoleNLink
from env.wrappers import DomainRandomizationWrapper, SensorNoiseWrapper, SinCosObsWrapper


def make_eval_env(hw: dict, dr_range: float, sensor_noise: bool,
                  delay: int, n_links: int, seed: int) -> SinCosObsWrapper:
    base = CartPoleNLink(
        n_links=n_links,
        link_lengths=[hw[f"link_{i+1}_total_length"] for i in range(n_links)],
        link_masses=[hw[f"link_{i+1}_mass"] for i in range(n_links)],
        link_cms=[hw[f"link_{i+1}_cm_distance"] for i in range(n_links)],
        link_inertias=[hw[f"link_{i+1}_inertia_cm"] for i in range(n_links)],
        cart_mass=hw["cart_mass"],
        force_mag=hw["force_limit"],
        angle_threshold=6.28,
        x_threshold=hw["x_threshold"],
        max_steps=1000,
        dt=hw["dt"],
        discrete=False,
    )
    env: object = base
    if dr_range > 0:
        env = DomainRandomizationWrapper(
            env, cart_mass_range=dr_range, link_mass_range=dr_range,
            link_length_range=dr_range * 0.5, gravity_range=0.01,
            force_scale_range=dr_range * 1.5, dt_jitter_range=0.0,
            enabled=True, warmup_steps=0, rng_seed=seed,
        )
    if sensor_noise:
        env = SensorNoiseWrapper(
            env, encoder_bits=14, velocity_from_finite_diff=True,
            observation_delay_steps=delay, n_links=n_links,
        )
    return SinCosObsWrapper(env, n_links=n_links)


def run_episode(agent: SACAgent, env: SinCosObsWrapper, hw: dict, n_links: int) -> dict:
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    base = env
    while hasattr(base, 'env'):
        base = base.env

    angles_list = [[] for _ in range(n_links)]
    balance_steps = 0
    swing_up_step = None

    for step in range(1000):
        action = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        thetas = [float(base._state[1 + i]) for i in range(n_links)]
        for i, t in enumerate(thetas):
            angles_list[i].append(abs(t))
        if all(abs(t) < 0.1 for t in thetas):
            balance_steps += 1
            if swing_up_step is None:
                swing_up_step = step
        if terminated or truncated:
            break

    return {
        "reward": total_reward,
        "steps": steps,
        "survived": steps >= 1000,
        "swing_up_step": swing_up_step,
        "max_angles": [max(a) for a in angles_list],
    }


def summarize(episodes: list[dict]) -> dict:
    n = len(episodes)
    survived = sum(1 for e in episodes if e["survived"])
    swings = [e["swing_up_step"] for e in episodes if e["swing_up_step"] is not None]
    return {
        "n": n,
        "survival_rate": survived / n,
        "mean_reward": float(np.mean([e["reward"] for e in episodes])),
        "swing_up_rate": len(swings) / n,
        "mean_swing_up_step": float(np.mean(swings)) if swings else None,
        "mean_steps": float(np.mean([e["steps"] for e in episodes])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-links", type=int, default=2)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--hw-config", default="conf/hardware.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.hw_config) as f:
        hw = yaml.safe_load(f)

    obs_dim = 2 + 3 * args.n_links
    agent = SACAgent(obs_dim=obs_dim, action_dim=1)
    agent.load(args.checkpoint)
    agent.eval()

    conditions = [
        ("Nominal",                    0.00, False, 0),
        ("Param variation +/-10%",     0.10, False, 0),
        ("Param variation +/-20%",     0.20, False, 0),
        ("Sensor noise",               0.10, True,  0),
        ("Sensor noise + 1-step delay",0.10, True,  1),
    ]

    results = {}
    for name, dr, noise, delay in conditions:
        print(f"[OK] {name} ...")
        env = make_eval_env(hw, dr, noise, delay, args.n_links, seed=42)
        episodes = [run_episode(agent, env, hw, args.n_links) for _ in range(args.n_episodes)]
        results[name] = summarize(episodes)
        env.close()

    print(f"\n{'='*80}")
    print("ROBUSTNESS EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"{'Condition':<30} {'Survival':>10} {'Avg Reward':>12} {'SwingUp%':>10}")
    print(f"{'-'*80}")
    for name, r in results.items():
        surv = f"{r['survival_rate']*100:.1f}%"
        rew = f"{r['mean_reward']:.1f}"
        su = f"{r['swing_up_rate']*100:.1f}%" if r['swing_up_rate'] > 0 else "N/A"
        print(f"{name:<30} {surv:>10} {rew:>12} {su:>10}")
    print(f"{'='*80}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[OK] Results saved: {args.output}")


if __name__ == "__main__":
    main()
