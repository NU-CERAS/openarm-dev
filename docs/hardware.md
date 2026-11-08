# Hardware commands

These commands drive the seven arm motors of each OpenArm v1.0 arm over Linux SocketCAN. `can0` is the left arm and `can1` is the right arm; each interface carries one arm. They do not scan/command every arbitrary device on a bus or include the gripper.

1. Install with `bash setup/linux.sh --hardware`.
2. Bring each interface up as classic CAN at 1 Mbit/s with a long transmit queue (the default 10-frame queue drops writes with `ENOBUFS`):
   ```sh
   sudo ip link set can0 type can bitrate 1000000 && sudo ip link set can0 txqueuelen 1000 && sudo ip link set can0 up
   ```
   Add `fd off` after the bitrate on FD-capable adapters. `ip -details link show can0` is read-only.
3. Verify motor types/IDs, sign conventions, joint limits, temperature/tracking limits and gains in `robots/v1.0/config/left.json` / `right.json` against the actual arms, or point at another file with `--config can1=PATH`.
4. Support the arm mechanically; keep the physical emergency stop accessible. Commands **move immediately** (there is no preview or confirmation flag) and **disable motors at exit**, including errors and Ctrl-C, so the arm goes limp afterward. They do not provide sustained gravity support or a certified emergency stop. Killing the process or losing host power can bypass Python cleanup; motor/controller watchdog configuration must be established independently.

Examples:

```sh
openarm can-check --real can0 can1
openarm status --real can1
openarm zero --real can1
openarm jog 7 0.1 --real can1 --duration 3
openarm home --real can1 --duration 10
openarm home --sim can0 can1 --real can0 can1 --duration 10   # sim and both real arms together
```

**Gravity compensation.** Each control cycle the CAN backend computes, from the MuJoCo model at the measured joint angles, the torque gravity puts on each joint and adds it to the motor command. `gravity_scale` in the arm config sets how much of that model torque is sent (0 = off, 1 = full; shipped at 0.5), and each joint's `tau_max` caps it in Nm. `status --real ...` shows the value as `gravity_feedforward` without enabling motors. Raise `gravity_scale` toward 1.0 once small moves track well.

**Holding.** Add `--hold` to `home`, `jog` or `pose move` to keep real arms at the target after the move, until Ctrl-C (`--hold`) or for a number of seconds (`--hold 30`). Motors are disabled when the hold ends.

`zero` only records a local software reference while motors are disabled; it does not rewrite persistent motor calibration. Existing raw limits remain in force. Poses are saved per arm and shared between `--sim` and `--real`, so a pose saved in simulation can be replayed on the real arm.

Before any arm moves, every selected arm's full path is checked against joint limits and the configured speed limit. During motion, commands abort on stale feedback, motor faults, temperature, bus health, joint limits and tracking error. When several arms (sim and/or real) move together, a failure or Ctrl-C on any of them stops all of them. There is no lock against other `openarm` commands, ROS controllers or other CAN applications on the same interface; stop those before commanding an arm. These tools provide no collision checking or hard real-time guarantee.
