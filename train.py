#!/usr/bin/env python3
"""
train.py  Train a SAC agent on the double (or triple) pendulum.

Usage
-----
# Double pendulum (default):
    python3 train.py

# Resume from a checkpoint:
    python3 train.py --resume results/run_20240101_120000/checkpoint_ep5000.pt

# Triple pendulum (after updating conf/hardware.yaml):
    python3 train.py --n-links 3

# Override total steps:
    python3 train.py --total-steps 3000000
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Make sure the repo root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from agent.sac import SACAgent, Transition
from env.cartpole_nlink import CartPoleNLink
from env.wrappers import (
    CurriculumWrapper,
    DoublePendulumRewardWrapper,
    DomainRandomizationWrapper,
    SensorNoiseWrapper,
    SinCosObsWrapper,
)


# ---------------------------------------------------------------------------
# Pause / resume via Ctrl-C or SIGUSR1
# ---------------------------------------------------------------------------
_PAUSE = False

def _handle_signal(signum: int, frame: object) -> None:
    global _PAUSE
    _PAUSE = True
    name = "SIGUSR1" if signum == signal.SIGUSR1 else "SIGINT"
    print(f"\n[OK] {name} received -- pausing after this episode")

signal.signal(signal.SIGINT, _handle_signal)
try:
    signal.signal(signal.SIGUSR1, _handle_signal)
except AttributeError:
    pass  # Windows does not have SIGUSR1


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(hw: dict, tr: dict, n_links: int) -> SinCosObsWrapper:
    """Build the full wrapped environment.

    Wrapper stack (inside to outside):
      CartPoleNLink
        -> CurriculumWrapper
        -> DoublePendulumRewardWrapper
        -> DomainRandomizationWrapper
        -> SensorNoiseWrapper  (if enabled)
        -> SinCosObsWrapper
    """
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
        max_steps=tr["episode_length"],
        dt=hw["dt"],
        discrete=False,
        gravity=9.81,
    )

    curriculum = CurriculumWrapper(
        base,
        stage_1_steps=tr["stage_1_steps"],
        stage_1_init_angle_noise=tr["stage_1_noise"],
        stage_2_init_angle_noise=tr["stage_2_noise"],
        episode_length=tr["episode_length"],
    )

    reward_env = DoublePendulumRewardWrapper(curriculum, n_links=n_links)

    dr = DomainRandomizationWrapper(
        reward_env,
        cart_mass_range=tr["dr_cart_mass"],
        link_mass_range=tr["dr_link_mass"],
        link_length_range=tr["dr_link_length"],
        gravity_range=tr["dr_gravity"],
        force_scale_range=tr["dr_force_scale"],
        dt_jitter_range=tr["dr_dt_jitter"],
        enabled=tr["dr_enabled"],
        warmup_steps=tr["dr_warmup_steps"],
        rng_seed=tr["seed"],
    )

    env = dr
    if tr.get("sensor_noise_enabled", False):
        env = SensorNoiseWrapper(
            dr,
            encoder_bits=tr.get("encoder_bits", 14),
            velocity_from_finite_diff=tr.get("velocity_from_finite_diff", True),
            observation_delay_steps=tr.get("observation_delay_steps", 0),
            n_links=n_links,
        )

    return SinCosObsWrapper(env, n_links=n_links), curriculum, dr


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC on pendulum")
    parser.add_argument("--n-links", type=int, default=2,
                        help="Number of pendulum links (2=double, 3=triple)")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="Override total training steps from config")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--hw-config", default="conf/hardware.yaml")
    parser.add_argument("--tr-config", default="conf/training.yaml")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    with open(args.hw_config) as f:
        hw = yaml.safe_load(f)
    with open(args.tr_config) as f:
        tr = yaml.safe_load(f)

    if args.total_steps:
        tr["total_steps"] = args.total_steps

    # Set seeds
    torch.manual_seed(tr["seed"])
    np.random.seed(tr["seed"])

    # Output directory
    run_id = time.strftime("%Y%m%d_%H%M%S")
    if args.resume:
        out = Path(args.resume).parent
    else:
        out = Path(args.output_dir) / f"run_{run_id}"
    out.mkdir(parents=True, exist_ok=True)

    print("[OK] Double/Triple Pendulum SAC Training")
    print(f"[OK] n_links = {args.n_links}")
    print(f"[OK] Output:  {out}")
    print(f"[OK] PID:     {os.getpid()}  (send SIGUSR1 or Ctrl-C to pause)")

    # Build environment
    env, curriculum, dr_wrapper = make_env(hw, tr, args.n_links)
    obs_dim = env.obs_space.shape[0]
    print(f"[OK] obs_dim = {obs_dim}  (2 + 3*n_links via SinCos wrapper)")

    # Build agent
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=1,
        hidden_dims=tuple(tr["hidden_dims"]),
        actor_lr=tr["actor_lr"],
        critic_lr=tr["critic_lr"],
        alpha_lr=tr["alpha_lr"],
        gamma=tr["gamma"],
        tau=tr["tau"],
        init_alpha=tr["init_alpha"],
        replay_capacity=tr["replay_capacity"],
        batch_size=tr["batch_size"],
        warmup_steps=tr["warmup_steps"],
        device=tr["device"],
        seed=tr["seed"],
    )

    # Resume from checkpoint
    total_steps = 0
    episode_rewards: list[float] = []
    if args.resume:
        agent.load(args.resume)
        total_steps = agent._steps
        print(f"[OK] Resumed at step {total_steps}")

    print(f"\n{'='*60}")
    print(f"[OK] Training for {tr['total_steps']:,} steps")
    print(f"{'='*60}\n")

    while total_steps < tr["total_steps"]:
        if _PAUSE:
            ckpt = out / f"checkpoint_paused_step{total_steps}.pt"
            agent.save(str(ckpt))
            print(f"[OK] Paused. Resume with: python3 train.py --resume {ckpt}")
            break

        obs, _ = env.reset()
        ep_reward = 0.0
        ep_steps = 0

        while True:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_steps += 1
            total_steps += 1

            agent.store(Transition(
                obs=obs, action=action, reward=reward,
                next_obs=next_obs, terminated=terminated, truncated=truncated,
            ))
            agent.update()
            obs = next_obs

            if terminated or truncated:
                break

        episode_rewards.append(ep_reward)
        ep_num = len(episode_rewards)

        # Check curriculum stage
        curriculum.check_stage_transition()

        # Log progress
        if ep_num % tr["log_interval"] == 0:
            recent = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
            avg = np.mean(recent)
            dr_on = "ON" if dr_wrapper._total_steps >= dr_wrapper._warmup_steps else "OFF"
            print(
                f"[OK] Ep {ep_num:5d} | Steps {total_steps:8,} | "
                f"Stage {curriculum.current_stage} | "
                f"R {ep_reward:8.1f} | Avg(100) {avg:8.1f} | DR {dr_on}"
            )

        # Checkpoint
        if ep_num % tr["checkpoint_interval"] == 0:
            ckpt = out / f"checkpoint_ep{ep_num}.pt"
            agent.save(str(ckpt))

    # Final save
    final = out / "final_checkpoint.pt"
    agent.save(str(final))

    avg_final = float(np.mean(episode_rewards[-100:])) if len(episode_rewards) >= 100 else float(np.mean(episode_rewards))
    metrics = {
        "total_steps": total_steps,
        "total_episodes": len(episode_rewards),
        "final_avg_reward_100ep": avg_final,
        "best_reward": float(max(episode_rewards)),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"[OK] Training complete")
    print(f"[OK] Final 100-ep avg reward: {avg_final:.1f}")
    print(f"[OK] Checkpoint: {final}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
