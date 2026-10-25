"""
Zero arms: moves every joint (J1-J7) of the real arms to position 0 over CAN
(openarm_can) and holds it there until Ctrl-C. No simulation.

Run from the repository root:
    python mujoco/examples/zero_arms.py                     # read-only
    python mujoco/examples/zero_arms.py --live              # moves both arms
    python mujoco/examples/zero_arms.py --iface can1 --live # left arm only

Without --live the motors are never enabled: the script only reads each arm's
joint positions and prints how far it would move them.

The target is not stepped to 0: it starts at the position each motor reports
and ramps to 0 at --speed rad/s, so the MIT spring only ever corrects a small
error. On Ctrl-C the motors are disabled and go limp, so support the arms
before stopping. The grippers are not touched.
"""
import argparse
import time

import numpy as np
import openarm_can as oa

parser = argparse.ArgumentParser()
parser.add_argument("--iface", action="append",
                    help="SocketCAN interface of an arm; repeat for several (default: can0 and can1)")
parser.add_argument("--live", action="store_true", help="Enable the motors and move the real arms")
parser.add_argument("--kp", type=float, default=30.0, help="MIT position gain (Nm/rad)")
parser.add_argument("--kd", type=float, default=1.0, help="MIT damping gain (Nm*s/rad)")
parser.add_argument("--speed", type=float, default=0.3, help="How fast the target ramps to 0 (rad/s)")
args = parser.parse_args()
ifaces = args.iface or ["can0", "can1"]  # right arm, left arm

# Standard OpenArm layout: J1-J2 DM8009, J3-J4 DM4340, J5-J7 DM4310.
# Send IDs 0x01-0x07, replies come back on send ID + 0x10.
MOTOR_TYPES = [oa.MotorType.DM8009, oa.MotorType.DM8009,
               oa.MotorType.DM4340, oa.MotorType.DM4340,
               oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310]
SEND_IDS = list(range(0x01, 0x08))
RECV_IDS = [i + 0x10 for i in SEND_IDS]

CONTROL_PERIOD = 0.01  # send CAN commands at 100 Hz


def check_motors(arm, iface):
    """Return the real joint positions, or raise if any motor is silent or faulted."""
    positions = []
    for i, motor in enumerate(arm.get_arm().get_motors()):
        if not arm.get_arm().get_link_stats(i).ever_responded():
            raise RuntimeError(f"{iface}: motor 0x{SEND_IDS[i]:02X} did not answer")
        if motor.has_error():
            raise RuntimeError(f"{iface}: motor 0x{SEND_IDS[i]:02X} reports "
                               f"{oa.motor_error_to_string(motor.get_error_code())}")
        positions.append(motor.get_position())
    return np.array(positions)


# 1. Connect to every arm and read where it is before touching anything
arms = {}
for iface in ifaces:
    arm = oa.OpenArm(iface, False)  # classic CAN, never CAN-FD
    arm.init_arm_motors(MOTOR_TYPES, SEND_IDS, RECV_IDS)
    arm.set_callback_mode_all(oa.CallbackMode.STATE)
    arms[iface] = arm

for _ in range(5):
    for arm in arms.values():
        arm.refresh_all()
    time.sleep(0.01)
    for arm in arms.values():
        arm.recv_all()

start = {iface: check_motors(arm, iface) for iface, arm in arms.items()}
for iface, q in start.items():
    print(f"{iface} start (rad):", q.round(3))

# All arms reach 0 together, timed by the joint that is furthest away
max_dist = max(np.abs(q).max() for q in start.values())
ramp_s = max(1.0, max_dist / args.speed)

if not args.live:
    print(f"Read-only (no --live): motors stay disabled. "
          f"--live would reach 0 in {ramp_s:.1f} s at {args.speed} rad/s.")
    raise SystemExit

# 2. Enable, then ramp every joint from its start position to 0 and hold
try:
    for arm in arms.values():
        arm.enable_all()
    time.sleep(0.01)
    for arm in arms.values():
        arm.recv_all()
    print(f"LIVE: kp={args.kp} kd={args.kd}, reaching 0 in {ramp_s:.1f} s, then holding. "
          f"Ctrl-C to release.")

    t0 = time.time()
    last_print = 0.0
    while True:
        now = time.time()
        s = min(1.0, (now - t0) / ramp_s)

        for iface, arm in arms.items():
            target = start[iface] * (1.0 - s)
            arm.get_arm().mit_control_all(
                [oa.MITParam(args.kp, args.kd, q, 0.0, 0.0) for q in target])
        time.sleep(CONTROL_PERIOD)
        for arm in arms.values():
            arm.recv_all()
        real = {iface: check_motors(arm, iface) for iface, arm in arms.items()}

        if now - last_print >= 0.5:
            last_print = now
            state = "moving " if s < 1.0 else "holding"
            print(f"t={now - t0:5.1f}s {state}  " +
                  "  ".join(f"{iface} {q.round(3)}" for iface, q in real.items()))
except KeyboardInterrupt:
    pass
finally:
    # 3. Always release the motors, even after a fault
    for arm in arms.values():
        arm.disable_all()
    time.sleep(0.1)
    for arm in arms.values():
        arm.recv_all()
    print("Motors disabled.")
