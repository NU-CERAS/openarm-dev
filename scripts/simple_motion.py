#!/usr/bin/env python3
"""Loop OpenArm through a few poses in MuJoCo, smoothly, for testing.

Start the sim first:
    ros2 launch openarm_mujoco_sim openarm_mujoco.launch.py
Then, in another terminal:
    ros2 run openarm_mujoco_sim simple_motion.py              # both arms, loop forever
    ros2 run openarm_mujoco_sim simple_motion.py --cycles 3   # stop after 3 loops
    ros2 run openarm_mujoco_sim simple_motion.py --arms left --move-time 4
(or run the file directly with python3)

Ctrl+C stops it. The arms hold their last commanded pose.

How it works: the forward_position controllers jump straight to whatever angle
they receive, so this script sends many small steps (50 per second) that blend
from one waypoint to the next with an ease-in/ease-out curve.
"""

import argparse
import math
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

RATE_HZ = 50.0
GRIPPER_OPEN = 0.044   # meters
GRIPPER_CLOSED = 0.0

# Waypoints for the LEFT arm: (name, [joint1..joint7] in radians, gripper opening).
# Left-arm ranges: j1 -3.49..1.40, j2 -3.32..0.17, j3 ±1.57, j4 0..2.44,
#                  j5 ±1.57, j6 ±0.79, j7 ±1.57.  Edit these to change the motion.
WAYPOINTS = [
    ("home",    [0.0,  0.0,  0.0, 1.57, 0.0, 0.0, 0.0], GRIPPER_OPEN),
    ("reach",   [-0.6, -0.3, 0.0, 0.6,  0.0, 0.0, 0.0], GRIPPER_OPEN),
    ("grab",    [-0.6, -0.3, 0.0, 0.6,  0.0, 0.0, 0.0], GRIPPER_CLOSED),
    ("lift",    [-1.2, -0.2, 0.0, 1.2,  0.0, 0.0, 0.0], GRIPPER_CLOSED),
    ("out",     [-0.8, -0.7, 0.5, 1.2,  0.0, 0.0, 0.0], GRIPPER_CLOSED),
    ("release", [-0.8, -0.7, 0.5, 1.2,  0.0, 0.0, 0.0], GRIPPER_OPEN),
]


def mirror_for_right(q):
    """Right arm's joint1/joint2 ranges are the left arm's flipped, so flip their signs."""
    return [-q[0], -q[1]] + list(q[2:])


def ease(s):
    """0 -> 1 with zero speed at both ends (smooth start and stop)."""
    return 0.5 - 0.5 * math.cos(math.pi * s)


def blend(a, b, s):
    return [x + (y - x) * s for x, y in zip(a, b)]


class SimpleMotion(Node):
    def __init__(self, arms, move_time, hold_time, cycles):
        super().__init__("simple_motion")
        self.arms = arms
        self.move_time = move_time
        self.hold_time = hold_time
        self.cycles = cycles

        self.arm_pub = {
            side: self.create_publisher(Float64MultiArray, f"/{side}_forward_position_controller/commands", 10)
            for side in arms
        }
        self.grip_pub = {
            side: self.create_publisher(Float64MultiArray, f"/{side}_gripper_controller/commands", 10)
            for side in arms
        }
        self.joint_positions = {}
        self.create_subscription(JointState, "/joint_states", self.on_joint_states, 10)

        self.start = None        # pose we blend from at the start of each move
        self.index = 0           # waypoint we're moving toward
        self.t = 0.0
        self.cycle = 0
        self.timer = self.create_timer(1.0 / RATE_HZ, self.step)
        self.get_logger().info(f"Waiting for /joint_states (arms: {', '.join(arms)})...")

    def on_joint_states(self, msg):
        self.joint_positions.update(zip(msg.name, msg.position))

    def current_left_frame_pose(self):
        """Read where the arm actually is, expressed as a left-arm waypoint."""
        side = self.arms[0]
        q = [self.joint_positions[f"openarm_{side}_joint{i}"] for i in range(1, 8)]
        if side == "right":
            q = mirror_for_right(q)
        grip = self.joint_positions[f"openarm_{side}_finger_joint1"]
        return q, grip

    def publish(self, q, grip):
        for side in self.arms:
            cmd = q if side == "left" else mirror_for_right(q)
            self.arm_pub[side].publish(Float64MultiArray(data=[float(x) for x in cmd]))
            self.grip_pub[side].publish(Float64MultiArray(data=[float(grip)]))

    def report_error(self, name, target_q):
        worst = 0.0
        for side in self.arms:
            cmd = target_q if side == "left" else mirror_for_right(target_q)
            for i, target in enumerate(cmd, start=1):
                actual = self.joint_positions.get(f"openarm_{side}_joint{i}")
                if actual is not None:
                    worst = max(worst, abs(actual - target))
        self.get_logger().info(f"  reached '{name}'  (max joint error {math.degrees(worst):.2f} deg)")

    def step(self):
        if self.start is None:
            needed = [f"openarm_{self.arms[0]}_joint{i}" for i in range(1, 8)]
            needed.append(f"openarm_{self.arms[0]}_finger_joint1")
            if not all(n in self.joint_positions for n in needed):
                return
            self.start = self.current_left_frame_pose()
            self.get_logger().info(f"Starting. Cycle 1 of {self.cycles or 'infinite'}")

        name, target_q, target_grip = WAYPOINTS[self.index]
        start_q, start_grip = self.start

        s = ease(min(self.t / self.move_time, 1.0))
        self.publish(blend(start_q, target_q, s), start_grip + (target_grip - start_grip) * s)
        self.t += 1.0 / RATE_HZ

        if self.t >= self.move_time + self.hold_time:
            self.report_error(name, target_q)
            self.start = (target_q, target_grip)
            self.t = 0.0
            self.index = (self.index + 1) % len(WAYPOINTS)
            if self.index == 0:
                self.cycle += 1
                self.get_logger().info(f"Cycle {self.cycle} done")
                if self.cycles and self.cycle >= self.cycles:
                    self.get_logger().info("Finished.")
                    self.timer.cancel()
                    raise SystemExit
                self.get_logger().info(f"Cycle {self.cycle + 1} of {self.cycles or 'infinite'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", choices=["left", "right", "both"], default="both")
    parser.add_argument("--move-time", type=float, default=3.0, help="seconds per move (default 3)")
    parser.add_argument("--hold-time", type=float, default=0.5, help="seconds to pause at each pose (default 0.5)")
    parser.add_argument("--cycles", type=int, default=0, help="number of loops, 0 = forever (default)")
    args, ros_args = parser.parse_known_args()

    arms = ["left", "right"] if args.arms == "both" else [args.arms]
    rclpy.init(args=[sys.argv[0]] + ros_args)
    node = SimpleMotion(arms, args.move_time, args.hold_time, args.cycles)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
