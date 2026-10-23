"""
Sim-to-real mirror: runs the same simulation as Example1 and makes the real
left arm follow the simulated left_rev1 joint over CAN (openarm_can).

Run from the repository root:
    python "mujoco/Robot Controller Examples/Example2: Sim-to-Real Mirror.py"          # read-only
    python "mujoco/Robot Controller Examples/Example2: Sim-to-Real Mirror.py" --live   # moves the arm

Without --live the motors are never enabled: the script only reads the real
arm's joint positions and prints what it would command.
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import openarm_can as oa

parser = argparse.ArgumentParser()
parser.add_argument("--iface", default="can1", help="SocketCAN interface of the left arm")
parser.add_argument("--live", action="store_true", help="Enable the motors and move the real arm")
parser.add_argument("--kp", type=float, default=30.0, help="MIT position gain (Nm/rad)")
parser.add_argument("--kd", type=float, default=1.0, help="MIT damping gain (Nm*s/rad)")
parser.add_argument("--max-angle", type=float, default=1.5, help="J1 target clamp (+/- rad)")
parser.add_argument("--max-speed", type=float, default=1.0, help="J1 target rate limit (rad/s)")
args = parser.parse_args()

# Standard OpenArm layout: J1-J2 DM8009, J3-J4 DM4340, J5-J7 DM4310.
# Send IDs 0x01-0x07, replies come back on send ID + 0x10.
MOTOR_TYPES = [oa.MotorType.DM8009, oa.MotorType.DM8009,
               oa.MotorType.DM4340, oa.MotorType.DM4340,
               oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310]
SEND_IDS = list(range(0x01, 0x08))
RECV_IDS = [i + 0x10 for i in SEND_IDS]

# Sim joints that the real arm follows, and the sign between sim and motor.
# Flip a sign if the real joint turns the opposite way from the viewer.
MIRRORED = {0: +1.0}  # index into the 7 left-arm joints -> sign

CONTROL_PERIOD = 0.01  # send CAN commands at 100 Hz


def check_motors(arm):
    """Return the real joint positions, or raise if any motor is silent or faulted."""
    positions = []
    for i, motor in enumerate(arm.get_arm().get_motors()):
        if not arm.get_arm().get_link_stats(i).ever_responded():
            raise RuntimeError(f"motor 0x{SEND_IDS[i]:02X} did not answer")
        if motor.has_error():
            raise RuntimeError(f"motor 0x{SEND_IDS[i]:02X} reports "
                               f"{oa.motor_error_to_string(motor.get_error_code())}")
        positions.append(motor.get_position())
    return np.array(positions)


# 1. Load the robot (same model as Example1)
model = mujoco.MjModel.from_xml_path(
    "mujoco/Bimanual Openarm v0.3/openarm_bimanual.mjcf.xml"
)
data = mujoco.MjData(model)
sim_qpos_adr = [model.jnt_qposadr[model.joint(f"left_rev{j}").id] for j in range(1, 8)]
J1_ACTUATOR = model.actuator("left_rev1_ctrl").id

# 2. Connect to the real arm and read where it is before touching anything
arm = oa.OpenArm(args.iface, False)  # classic CAN, never CAN-FD
arm.init_arm_motors(MOTOR_TYPES, SEND_IDS, RECV_IDS)
arm.set_callback_mode_all(oa.CallbackMode.STATE)
for _ in range(5):
    arm.refresh_all()
    time.sleep(0.01)
    arm.recv_all()
start = check_motors(arm)
print("Real arm start (rad):", start.round(3))

# Joints that aren't mirrored hold their start pose; mirrored ones begin
# there too and are rate-limited toward the sim, so nothing jumps.
target = start.copy()

if args.live:
    arm.enable_all()
    time.sleep(0.01)
    arm.recv_all()
    print(f"LIVE: kp={args.kp} kd={args.kd}, J1 within +/-{args.max_angle} rad "
          f"at <= {args.max_speed} rad/s. Close the viewer or Ctrl-C to release.")
else:
    print("Read-only (no --live): motors stay disabled.")

# 3. Start the simulator
try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        next_control = time.time()
        last_print = 0.0

        while viewer.is_running():

            start_t = time.time()

            # 4. Same motor command as Example1, applied to left_rev1 instead of right_rev1
            data.ctrl[:] = 0
            data.ctrl[J1_ACTUATOR] = 4.0 * np.sin(2 * data.time)

            # 5. Advance the physics simulation
            mujoco.mj_step(model, data)

            # 6. Every CONTROL_PERIOD, push the sim joint angle to the real arm
            if start_t >= next_control:
                next_control = start_t + CONTROL_PERIOD
                sim_q = data.qpos[sim_qpos_adr]
                max_step = args.max_speed * CONTROL_PERIOD
                for j, sign in MIRRORED.items():
                    goal = np.clip(sign * sim_q[j], -args.max_angle, args.max_angle)
                    target[j] += np.clip(goal - target[j], -max_step, max_step)

                if args.live:
                    arm.get_arm().mit_control_all(
                        [oa.MITParam(args.kp, args.kd, q, 0.0, 0.0) for q in target])
                else:
                    arm.refresh_all()
                arm.recv_all()
                real_q = check_motors(arm)

                if start_t - last_print >= 0.5:
                    last_print = start_t
                    print(f"t={data.time:6.2f}s  sim J1 {sim_q[0]:+.3f}  "
                          f"target {target[0]:+.3f}  real J1 {real_q[0]:+.3f}")

            # 7. Update the screen
            viewer.sync()

            # 8. Keep the simulation near real-time
            elapsed = time.time() - start_t
            time.sleep(max(0, model.opt.timestep - elapsed))
except KeyboardInterrupt:
    pass
finally:
    # 9. Always release the motors, even after a fault
    arm.disable_all()
    time.sleep(0.1)
    arm.recv_all()
    print("Motors disabled.")
