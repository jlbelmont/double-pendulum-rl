# Hardware Setup and Calibration Guide

## What You Need

| Component | Spec |
|-----------|------|
| Raspberry Pi | Pi 4 (2GB+ RAM) or Pi 5 |
| Arduino | Uno or Mega (drives stepper) |
| ESP32 | Reads 14-bit rotary encoders |
| Track | 0.8382 m total length |
| Cart | 0.909 kg with stepper motor |
| Link 1 | 0.25 m, 0.238 kg |
| Link 2 | 0.20 m, 0.148 kg |
| Encoders | 14-bit (16383 counts/revolution) |
| Power supply | 24V for stepper, 5V for Pi/ESP32/Arduino |

## Wiring

```
ESP32 (encoders)
  /dev/ttyUSB0  921600 baud
  Reads 6 bytes per cycle: [enc1_hi, enc1_lo, enc2_hi, enc2_lo, enc3_hi, enc3_lo]
  (enc3 is unused for double pendulum)

Arduino (stepper)
  /dev/ttyACM0  115200 baud
  Receives: 4-byte signed int (timer TOP value)
  Sends back: 2-byte signed short (cart position in steps)
```

## Install Software on Raspberry Pi

```bash
# Python and dependencies
sudo apt update
sudo apt install python3-pip python3-venv

python3 -m venv pendulum_env
source pendulum_env/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pyserial numpy
```

## Deploy the Policy

```bash
# On your computer (after training):
python3 deploy/export_policy.py \
    --checkpoint results/run_YYYYMMDD/final_checkpoint.pt \
    --output deploy_output/

# Copy to Pi:
scp -r deploy_output/ pi@<pi-ip>:~/pendulum/

# On the Pi:
cd ~/pendulum
source ~/pendulum_env/bin/activate
python3 deploy_policy.py
```

## Encoder Calibration

The encoder correction values offset the raw encoder count so that
0 corresponds to the physical upright position.

To calibrate:
1. Manually hold both pendulums exactly vertical (upright).
2. Read the raw encoder values from the ESP32.
3. Set those values as `encoder_correction_1` and `encoder_correction_2`
   in `conf/hardware.yaml`.
4. Re-export the policy: `python3 deploy/export_policy.py ...`

Example reading raw values:
```python
import serial, struct, time
esp32 = serial.Serial("/dev/ttyUSB0", 921600, timeout=0.003)
time.sleep(1)
buf = bytearray()
while len(buf) < 6:
    buf.extend(esp32.read(6 - len(buf)))
a1, a2, _ = struct.unpack(">HHH", buf)
print(f"Link 1 raw: {a1}, Link 2 raw: {a2}")
```

## Safety

The control loop has two hardware watchdogs:

| Condition | Action |
|-----------|--------|
| Any angle > 2.0 rad (~115 deg) | Emergency stop |
| Cart position > track half | Emergency stop |
| Cart near track ends | Force reduced proportionally |

To stop at any time: **press Ctrl-C**. The script always sends a stop
command to the Arduino before closing.

## Troubleshooting

| Problem | Check |
|---------|-------|
| No serial port found | `ls /dev/ttyUSB*` and `ls /dev/ttyACM*` |
| Encoder reads garbage | Check baud rate and cable connections |
| Cart drifts one direction | Recalibrate encoder corrections |
| Policy oscillates but does not balance | Run eval.py to check robustness |
| Pi too slow | Use `torch.set_num_threads(2)`, check control loop timing |
