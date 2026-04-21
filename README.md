# Double Pendulum RL

Train and deploy a reinforcement learning controller for a double (or triple)
inverted pendulum on a cart.  The agent uses Soft Actor-Critic (SAC) and runs
at 200 Hz on a Raspberry Pi.

## What this repo contains

```
double_pendulum_rl/
  env/               Physics simulation and training wrappers
  agent/             SAC implementation (self-contained, no dependencies)
  conf/              Hardware and training config files
  deploy/            Export script -- produces files ready to copy to the Pi
  docs/              Hardware setup, calibration, and triple-pendulum guide
  train.py           Start training
  eval.py            Test a trained policy before deploying
```

## Requirements

- Python 3.10 or newer
- A machine with at least 8 GB RAM (16 GB recommended for training)
- GPU optional but not required

```bash
pip install -r requirements.txt
```

## Step 1 -- Configure your hardware

Edit `conf/hardware.yaml` with the measured values for your physical system.
The default values match the reference hardware (see `docs/hardware_setup.md`).

Key parameters:

| Parameter | Default | What it means |
|-----------|---------|---------------|
| `cart_mass` | 0.909 kg | Mass of cart + motor |
| `link_1_mass` | 0.238 kg | Mass of first pendulum link |
| `link_2_mass` | 0.148 kg | Mass of second pendulum link |
| `x_threshold` | 0.4191 m | Half the usable track length |
| `force_limit` | 5.0 N | Maximum force the motor can apply |
| `encoder_correction_1/2` | 4967 / 7485 | Calibration offsets (measure on your hardware) |

## Step 2 -- Train

```bash
python3 train.py
```

Training takes 2-4 hours on a modern CPU. Progress is printed every 25
episodes. Checkpoints are saved to `results/run_YYYYMMDD_HHMMSS/`.

To resume an interrupted run:

```bash
python3 train.py --resume results/run_YYYYMMDD_HHMMSS/checkpoint_ep5000.pt
```

To train for more steps (useful if the policy is not stable enough):

```bash
python3 train.py --total-steps 3000000
```

### What to watch for

The log prints `Avg(100)` -- the average reward over the last 100 episodes.
A good double-pendulum policy reaches Avg(100) above 800 by the end of
training. If it plateaus below 400, try increasing `total_steps`.

## Step 3 -- Evaluate before deploying

```bash
python3 eval.py --checkpoint results/run_YYYYMMDD_HHMMSS/final_checkpoint.pt
```

This runs 5 test conditions (nominal, parameter variation, sensor noise) and
prints a robustness table.  Aim for at least 80% survival rate under nominal
conditions before putting the policy on the real hardware.

## Step 4 -- Export for the Raspberry Pi

```bash
python3 deploy/export_policy.py \
    --checkpoint results/run_YYYYMMDD_HHMMSS/final_checkpoint.pt \
    --output deploy_output/
```

This creates two files in `deploy_output/`:

- `policy.pt` -- the neural network (TorchScript format, runs without training code)
- `deploy_policy.py` -- the complete control loop script

## Step 5 -- Copy to the Pi and run

```bash
# Copy files
scp -r deploy_output/ pi@<your-pi-ip>:~/pendulum/

# On the Pi
cd ~/pendulum
python3 deploy_policy.py
```

Press `Ctrl-C` to stop. The motor is sent a stop command automatically.

For full wiring diagrams, encoder calibration, and troubleshooting see
`docs/hardware_setup.md`.

## Extending to a triple pendulum

See `docs/extending_to_triple.md`. The short version:

```bash
python3 train.py --n-links 3
python3 eval.py  --checkpoint results/.../final_checkpoint.pt --n-links 3
python3 deploy/export_policy.py --checkpoint results/.../final_checkpoint.pt --n-links 3
```

No code changes are required -- just measure the third link and add it to
`conf/hardware.yaml`.

## How the physics works

The simulation uses Lagrangian mechanics with full rigid-body parameters
(center-of-mass distances and moments of inertia about each link's CM).
The equations of motion are integrated with a 4th-order Runge-Kutta solver
at 200 Hz, matching the real hardware control rate.

The agent learns using Soft Actor-Critic, a model-free off-policy algorithm
that automatically tunes its exploration temperature.

Domain randomization (randomizing masses, link lengths, gravity, and motor
force by small percentages) and sensor noise simulation (14-bit encoder
quantization + finite-difference velocities) are applied during training
so the policy transfers to real hardware without fine-tuning.

## File structure reference

| File | Purpose |
|------|---------|
| `env/cartpole_nlink.py` | Physics engine for N-link pendulum |
| `env/wrappers.py` | Curriculum, reward, randomization, noise, observation wrappers |
| `agent/sac.py` | SAC agent with replay buffer and TorchScript export |
| `conf/hardware.yaml` | Physical system parameters |
| `conf/training.yaml` | Learning hyperparameters |
| `train.py` | Training entry point |
| `eval.py` | Robustness evaluation |
| `deploy/export_policy.py` | Export trained policy for Raspberry Pi |

## License

MIT
