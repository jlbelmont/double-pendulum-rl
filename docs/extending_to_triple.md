# Extending to a Triple Pendulum

This guide walks you through adding a third link to the simulation and re-training.
No changes to the physics engine are needed -- `CartPoleNLink` already supports any
number of links via the `n_links` parameter.

## What You Need to Measure on the Hardware

Measure the same six numbers for link 3 that you already have for links 1 and 2:

| Parameter | Symbol | How to measure |
|-----------|--------|----------------|
| Total length | `l` | Ruler from pivot to tip |
| Mass | `m` | Kitchen scale (remove from cart first) |
| CM distance from pivot | `l_cm` | Balance the link on a knife edge |
| Moment of inertia about CM | `I_cm` | Bifilar pendulum method (see below) |

### Measuring I_cm (bifilar pendulum method)

1. Suspend the link horizontally from two parallel strings of equal length `L`.
2. Twist it through a small angle and release.
3. Measure the period `T` of the torsional oscillation.
4. Formula: `I_cm = m * r^2 * g * T^2 / (4 * pi^2 * L)`
   where `r` is half the distance between the two strings and `g = 9.81`.

## Update conf/hardware.yaml

Add the link 3 values:

```yaml
# Link 3 (add these lines)
link_3_total_length: 0.XX      # meters
link_3_mass:         0.XXX     # kg
link_3_cm_distance:  0.XXX     # meters from pivot
link_3_inertia_cm:   0.XXXXXXX # kg*m^2 about CM
encoder_correction_3: 0        # calibrate this last
```

Also add an encoder correction entry (calibration procedure is the same as for
links 1 and 2 -- see hardware_setup.md).

## Train the Triple Pendulum

```bash
python3 train.py --n-links 3
```

That is the only change needed to switch from double to triple.
The observation dimension automatically expands from 8 to 11
(x, dx, sin/cos/dot for each of the 3 links).

The triple pendulum is significantly harder to learn. Recommended config changes
in `conf/training.yaml`:

```yaml
total_steps: 5000000          # was 2000000 -- needs 2.5x more
stage_1_steps: 500000         # was 300000
dr_warmup_steps: 1000000      # was 500000
```

## Export and Deploy

Same command as for the double pendulum -- just add `--n-links 3`:

```bash
python3 deploy/export_policy.py \
    --checkpoint results/run_YYYYMMDD/final_checkpoint.pt \
    --n-links 3 \
    --output deploy_output/
```

The generated `deploy_policy.py` will automatically read the third encoder
from the ESP32 byte stream (bytes 4-5, currently labelled `enc3` and marked
unused in the double-pendulum version).

## Wiring the Third Encoder

The ESP32 firmware already sends 6 bytes per cycle:

```
[enc1_hi, enc1_lo, enc2_hi, enc2_lo, enc3_hi, enc3_lo]
```

For the double pendulum, `enc3` is read and discarded. For the triple pendulum,
edit `deploy_policy.py` after export to use it:

```python
# Find this line (double pendulum):
a1, a2, _ = struct.unpack(">HHH", buf)

# Replace with (triple pendulum):
a1, a2, a3 = struct.unpack(">HHH", buf)
t3 = encoder_to_rad(a3, ENCODER_CORRECTION_3) + t2
```

And add `ENCODER_CORRECTION_3` near the top of `deploy_policy.py`.

## Evaluating Before Deploying

```bash
python3 eval.py \
    --checkpoint results/run_YYYYMMDD/final_checkpoint.pt \
    --n-links 3
```

Target survival rates before putting the policy on real hardware:

| Condition | Minimum |
|-----------|---------|
| Nominal | 90% |
| +/-20% param variation | 60% |
| Sensor noise | 70% |
| Sensor noise + 1-step delay | 50% |

If survival rates are below these thresholds, train for more steps before
deploying.
