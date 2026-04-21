#!/usr/bin/env python3
"""
deploy/export_policy.py  Export trained policy for Raspberry Pi deployment.

Produces two files in the output directory:
  policy.pt          -- TorchScript model for fast on-device inference
  deploy_policy.py   -- Complete control loop script for the Raspberry Pi

Usage
-----
    python3 deploy/export_policy.py --checkpoint results/run_.../final_checkpoint.pt
    python3 deploy/export_policy.py --checkpoint results/run_.../final_checkpoint.pt --n-links 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.sac import SACAgent


DEPLOY_SCRIPT = '''#!/usr/bin/env python3
"""
deploy_policy.py  Real-time control loop for the double pendulum on Raspberry Pi.

Hardware connections:
  ESP32  --> /dev/ttyUSB0  (encodes angles via 14-bit encoders, 921600 baud)
  Arduino --> /dev/ttyACM0  (drives stepper motor, 115200 baud)

Before running:
  1. pip3 install torch pyserial numpy
  2. Copy policy.pt to the same directory as this script
  3. Update ENCODER_CORRECTION_1 and ENCODER_CORRECTION_2 (see calibration guide)
  4. Run: python3 deploy_policy.py

Safety:
  Press Ctrl-C to stop. The motor sends a stop command on exit.
  Watchdog terminates if any angle exceeds ANGLE_WATCHDOG_RAD.
"""

import struct
import time

import numpy as np
import torch

# --- Hardware constants (from spec sheet) ---
ENCODER_COUNTS    = {encoder_counts}
BAUD_ESP32        = {baud_esp32}
BAUD_ARDUINO      = {baud_arduino}
FORCE_LIMIT_N     = {force_limit}          # Newtons
CART_MASS_KG      = {cart_mass}            # kg
TRACK_HALF_M      = {track_half}           # m
STEPS_PER_METER   = {steps_per_meter:.4f}  # stepper steps / meter
CONTROL_HZ        = {control_hz}
DT                = 1.0 / CONTROL_HZ

# --- Calibration (measure on your hardware) ---
ENCODER_CORRECTION_1 = {corr1}  # counts offset for link 1 encoder
ENCODER_CORRECTION_2 = {corr2}  # counts offset for link 2 encoder

# --- Safety limits ---
ANGLE_WATCHDOG_RAD = 2.0          # Stop if angle exceeds this
CART_SOFT_LIMIT    = TRACK_HALF_M * 0.85  # Start reducing force near track ends

# --- Load policy ---
policy = torch.jit.load("policy.pt", map_location="cpu")
policy.eval()
print("[OK] Policy loaded")


def encoder_to_rad(raw: int, correction: int) -> float:
    """Convert 14-bit encoder count to radians."""
    return -((raw - correction) * 2 * np.pi) / ENCODER_COUNTS


def sincos_obs(x: float, dx: float, t1: float, dt1: float,
               t2: float, dt2: float) -> np.ndarray:
    """Build 8D observation matching training SinCosObsWrapper."""
    return np.array([
        x, dx,
        np.sin(t1), np.cos(t1), dt1,
        np.sin(t2), np.cos(t2), dt2,
    ], dtype=np.float32)


def force_to_timer(force_n: float) -> int:
    """Convert force (N) to Arduino stepper timer TOP value."""
    force_n = np.clip(force_n, -FORCE_LIMIT_N, FORCE_LIMIT_N)
    accel = force_n / CART_MASS_KG
    velocity = np.clip(accel * DT, -5.0, 5.0)
    freq = -(velocity * STEPS_PER_METER)
    if abs(freq) > 2:
        return int((0.5 / freq) / (1.0 / 250000.0))
    return 65535 if freq >= 0 else -65535


def main() -> None:
    import serial

    esp32   = serial.Serial("/dev/ttyUSB0", BAUD_ESP32,   timeout=0.003)
    arduino = serial.Serial("/dev/ttyACM0", BAUD_ARDUINO, timeout=0.003)
    time.sleep(3)
    arduino.reset_input_buffer()
    print("[OK] Serial connections established")

    # Read initial cart position
    arduino.write(struct.pack("<i", 65535))
    raw_pos = bytearray()
    while len(raw_pos) < 2:
        raw_pos.extend(arduino.read(2 - len(raw_pos)))
    prev_x = (struct.unpack(">h", raw_pos)[0] * 0.638175) / 6400

    prev_t1, prev_t2 = 0.0, 0.0
    loop_count = 0

    print(f"[OK] Control loop running at {{CONTROL_HZ}} Hz  |  Ctrl-C to stop")

    try:
        while True:
            t_start = time.perf_counter()

            # Read encoder angles from ESP32
            buf = bytearray()
            while len(buf) < 6:
                buf.extend(esp32.read(6 - len(buf)))
            if len(buf) == 6:
                a1, a2, _ = struct.unpack(">HHH", buf)
                t1 = encoder_to_rad(a1, ENCODER_CORRECTION_1)
                t2 = encoder_to_rad(a2, ENCODER_CORRECTION_2) + t1
            else:
                t1, t2 = prev_t1, prev_t2
            esp32.reset_input_buffer()

            # Finite-difference velocities
            dt1 = (t1 - prev_t1) / DT
            dt2 = (t2 - prev_t2) / DT

            # Read cart position from Arduino
            arduino.reset_output_buffer()
            arduino.write(struct.pack("<i", 65535))
            buf2 = bytearray()
            while len(buf2) < 2:
                buf2.extend(arduino.read(2 - len(buf2)))
            if len(buf2) == 2:
                x = (struct.unpack(">h", buf2)[0] * 0.638175) / 6400
            else:
                x = prev_x
            arduino.reset_input_buffer()
            dx = (x - prev_x) / DT

            # Safety checks
            if abs(t1) > ANGLE_WATCHDOG_RAD or abs(t2) > ANGLE_WATCHDOG_RAD:
                print(f"[FAIL] Angle watchdog: t1={{t1:.2f}} t2={{t2:.2f}}")
                arduino.write(b"\\xff\\xff\\xff\\xff")
                break
            if abs(x) > TRACK_HALF_M:
                print(f"[FAIL] Cart limit: x={{x:.3f}}")
                arduino.write(b"\\xff\\xff\\xff\\xff")
                break

            # Run policy
            obs = sincos_obs(x, dx, t1, dt1, t2, dt2)
            with torch.no_grad():
                # Actor outputs tanh-squashed action in [-1, 1]
                mu, _ = policy(torch.from_numpy(obs).unsqueeze(0))
                raw_action = mu.squeeze()
                action = torch.tanh(raw_action).item()
            force = action * FORCE_LIMIT_N

            # Soft track limit near ends
            if abs(x) > CART_SOFT_LIMIT:
                damp = 1.0 - (abs(x) - CART_SOFT_LIMIT) / (TRACK_HALF_M - CART_SOFT_LIMIT)
                force *= max(0.1, damp)

            # Send motor command
            arduino.write(struct.pack("<i", force_to_timer(force)))
            buf3 = bytearray()
            while len(buf3) < 2:
                buf3.extend(arduino.read(2 - len(buf3)))
            if len(buf3) == 2:
                x = (struct.unpack(">h", buf3)[0] * 0.638175) / 6400

            prev_t1, prev_t2, prev_x = t1, t2, x
            loop_count += 1
            if loop_count % 200 == 0:
                print(f"[OK] t1={{t1:6.3f}} t2={{t2:6.3f}} x={{x:6.3f}} F={{force:5.2f}}N")

            elapsed = time.perf_counter() - t_start
            wait = DT - elapsed
            if wait > 0:
                time.sleep(wait)

    except KeyboardInterrupt:
        print("\\n[OK] Stopped by user")
    finally:
        arduino.write(b"\\xff\\xff\\xff\\xff")
        time.sleep(0.05)
        arduino.close()
        esp32.close()
        print("[OK] Serial closed")


if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to final_checkpoint.pt")
    parser.add_argument("--n-links", type=int, default=2)
    parser.add_argument("--output", default="deploy_output")
    parser.add_argument("--hw-config", default="conf/hardware.yaml")
    args = parser.parse_args()

    with open(args.hw_config) as f:
        hw = yaml.safe_load(f)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    obs_dim = 2 + 3 * args.n_links
    agent = SACAgent(obs_dim=obs_dim, action_dim=1)
    agent.load(args.checkpoint)
    agent.eval()

    # Export TorchScript
    model_path = out / "policy.pt"
    agent.export_torchscript(str(model_path), obs_dim)

    # Generate deploy script with hardware constants filled in
    script = DEPLOY_SCRIPT.format(
        encoder_counts=int(hw.get("encoder_counts", 16383)),
        baud_esp32=int(hw.get("baud_esp32", 921600)),
        baud_arduino=int(hw.get("baud_arduino", 115200)),
        force_limit=float(hw.get("force_limit", 5.0)),
        cart_mass=float(hw.get("cart_mass", 0.909)),
        track_half=float(hw.get("x_threshold", 0.4191)),
        steps_per_meter=float(hw.get("steps_per_meter", 10029.6)),
        control_hz=int(hw.get("control_hz", 200)),
        corr1=int(hw.get("encoder_correction_1", 4967)),
        corr2=int(hw.get("encoder_correction_2", 7485)),
    )
    script_path = out / "deploy_policy.py"
    script_path.write_text(script)
    script_path.chmod(0o755)

    print(f"\n[OK] Export complete: {out}/")
    print(f"[OK]   policy.pt       -- TorchScript model for Pi")
    print(f"[OK]   deploy_policy.py -- control loop script")
    print(f"\n[OK] Next steps:")
    print(f"[OK]   scp -r {out}/ pi@<your-pi-ip>:~/pendulum/")
    print(f"[OK]   ssh pi@<your-pi-ip>")
    print(f"[OK]   cd ~/pendulum && python3 deploy_policy.py")


if __name__ == "__main__":
    main()
