# Checked IK move

`ik_move.py` solves a Cartesian target, checks a sampled joint-space path against
the bimanual MuJoCo model, and only sends motion commands with `--live`. The default
is the right arm (`can0`), 2 cm upward in **MuJoCo world coordinates** (meters).
The controlled point is `right_gripper_center` or `left_gripper_center`.

## Try the requested move without hardware

From the repository root on the laptop:

```bash
.venv/bin/python mujoco/examples/ik_move.py --offline --arm right --delta 0 0 0.02
.venv/bin/python mujoco/examples/ik_move.py --offline --arm right --delta 0 0 0.02 --preview
```

Offline starts both arms at model zero. Use `--q-right` and `--q-left` followed by
seven model joint angles in radians to supply a different simulated starting pose.
On macOS use `mjpython` for the viewer. The preview animates forward kinematics;
it is not a dynamics or controller-stability test.

## Plan from actual motor feedback

```bash
.venv/bin/python mujoco/examples/ik_move.py --arm right --delta 0 0 0.02 --preview
```

Both CAN interfaces must answer because the other arm participates in collision
checks. This mode reads parameters and state, but never enables, disables, moves,
zeros, or writes motor parameters. Angles are decoded using each motor's measured
PMAX/VMAX/TMAX values. Motor 8 is read but never driven.

An absolute target can be supplied with `--target X Y Z` instead of `--delta`.
`--keep-orientation` additionally constrains the final orientation to about 2°.
Without it, IK prefers the starting orientation and rejects paths exceeding 15°
of gripper rotation. Unreachable or rejected targets never proceed to enabling.

## Run the checked motion

### Small reference offsets after zeroing

J2 sits at a model joint-limit boundary: right J2 must be <=0, left J2 >=0.
A small mechanical zero-reference offset can therefore reject an otherwise
correct physical pose. Errors now print every offending joint, angle and excess.

**Only when both arms physically match the intended model zero pose**, support them
with their controllers stopped and motors disabled, then run:

```bash
.venv/bin/python mujoco/examples/ik_move.py --capture-reference-zero
```

This samples both arms 20 times, rejects motion/enabled motors or corrections over
5 degrees, and records software offsets in the mapping file. It backs up the old
mapping as `ik_mapping.json.before-reference-<timestamp>`. It sends only read
requests: it does not move the arms, change motor zeros, widen model limits or
write firmware settings. It preserves existing sign verification. Do not capture
a different pose to suppress a limit error; correct the physical pose or mapping.
After changing any motor zero, the previous software offsets must be reverified.

### Execute

First stop other motor controllers and support the arms. Verify that the physical
scene, stationary grippers, interface assignments and zero/sign mapping match the
model. The mapping in `ik_mapping.json` records the operator's zero/direction
confirmation from 2026-10-25; **reverify it after changing any zero or assembly**.
It currently assumes the model's closed/default jaw positions. Edit those positions
if the real grippers differ. `ik_mapping.example.json` is deliberately unverified.

```bash
.venv/bin/python mujoco/examples/ik_move.py --arm right --delta 0 0 0.02 --preview --live
```

Close the preview to proceed. The script rereads both arms, rejects a changed
starting pose, requires disabled motors, MIT control mode and a nonzero
communication watchdog on the selected arm, and then enables only J1–J7 of that
arm. It does not configure or persist a watchdog; its timeout units and actual
stop behavior must be checked against the installed motor firmware. A host-side
Python loop cannot replace the motor watchdog or a hardware stop.

The motion uses a quintic ramp capped at 0.08 rad/s and 0.15 rad/s², then holds the
target. **Ctrl+C releases the selected arm. Support it before stopping.** Normal
exceptions, SIGINT, SIGTERM and SIGHUP attempt to disable every joint whose enable
was attempted. A broken CAN bus can prevent disable transmission, and SIGKILL or
power loss cannot run Python cleanup.

Live runs print the worst tracking joint every 0.5 seconds and write a JSONL trace
under `/tmp/openarm-ik-trace-*.jsonl` (exact path printed at startup). The trace
contains every feedback sample, commanded/model/motor angles, speed, torque and
status plus the stop reason. A tracking error now names the offending joint and
its target, actual angle, velocity and torque; preserve the trace before adjusting
gains or thresholds.

From exact model zero, the requested right-arm lift requires about 32° of elbow
rotation and takes approximately 13 seconds; a small Cartesian displacement does
not necessarily mean small joint rotations.

## Checks and limits

- Maximum target displacement: 5 cm; maximum joint displacement: 0.6 rad.
- Joint ranges from MJCF; a 0.02 rad boundary margin where the start permits it.
- At least 101 path samples, with no more than 0.002 rad between samples.
- 3 mm modeled collision clearance, honoring the MJCF's contact exclusions.
- Fresh replies from all eight motors on both buses; motor faults and temperature
  at or above 60°C reject the run.
- During motion: tracking error <=0.08 rad, measured joint speed <=0.5 rad/s,
  measured torque thresholds, loop timing checks, collision checks of measured and
  commanded configurations, and rejection if the other arm or grippers move.
- Fixed conservative MIT gains; no gravity compensation or payload identification.
  An unsupported pose may fail the tracking check. Measured torque checks are not
  an independent hardware torque limit.

These are **model checks, not a physical safety guarantee**. Tables, cables,
payloads, people, calibration errors, approximate collision shapes and excluded
body pairs are not covered by the stock model. Discrete path sampling is not
continuous collision detection. Add environment geometry with `--model` as needed.
Use an attended first test with access to the hardware stop.

## Tests

```bash
.venv/bin/python -m unittest discover -s mujoco/examples -p test_ik_move.py -v
```

Tests are headless and never connect to CAN. They cover reachable IK, real modeled
collision rejection, joint/target limits, smooth trajectory bounds, dry-run
isolation, existing-controller rejection, watchdog rejection, command encoding,
and selected-joint cleanup after a simulated communication failure.

The optional `--output plan.json` saves a report only; reports cannot be replayed
as motion commands, so hardware state is always reread and replanned.
