import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from .execution import ArmCommand, ArmState, joint_names


class Ros2ArmInterface(Node):
    def __init__(self, arm: str, joint_limits: np.ndarray, *, feedback_timeout: float = 0.2, max_tracking_error: float = 0.25):
        super().__init__(f"openarm_{arm}_skill_interface")
        self.arm = arm
        self.names = joint_names(arm)
        self.grip_name = f"openarm_{arm}_finger_joint1"
        self.feedback_timeout = feedback_timeout
        self.max_tracking_error = max_tracking_error
        self.joint_limits = np.asarray(joint_limits, dtype=float)
        if self.joint_limits.shape != (7, 2) or not np.isfinite(self.joint_limits).all():
            raise ValueError("ROS execution requires seven calibrated lower/upper joint limits")
        if np.any(self.joint_limits[:, 0] >= self.joint_limits[:, 1]):
            raise ValueError("Invalid robot joint limits")
        self.state: ArmState | None = None
        self.received_at = 0.0
        self.arm_pub = self.create_publisher(Float64MultiArray, f"/{arm}_forward_position_controller/commands", 10)
        self.grip_pub = self.create_publisher(Float64MultiArray, f"/{arm}_gripper_controller/commands", 10)
        self.create_subscription(JointState, "/joint_states", self._feedback, 10)

    def _feedback(self, message: JointState) -> None:
        if len(message.position) != len(message.name) or len(message.velocity) != len(message.name):
            raise ValueError("Joint feedback must include position and velocity for every joint")
        indices = {name: index for index, name in enumerate(message.name)}
        required = self.names + [self.grip_name]
        if not all(name in indices for name in required):
            return
        self.state = ArmState(
            np.array([message.position[indices[name]] for name in self.names]),
            np.array([message.velocity[indices[name]] for name in self.names]),
            float(message.position[indices[self.grip_name]]),
            message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
        )
        if not np.isfinite(np.r_[self.state.position, self.state.velocity, self.state.gripper]).all():
            raise ValueError("Non-finite robot feedback")
        self.received_at = time.monotonic()

    def read(self) -> ArmState:
        rclpy.spin_once(self, timeout_sec=0)
        if self.state is None or time.monotonic() - self.received_at > self.feedback_timeout:
            raise RuntimeError("Robot joint feedback is missing or stale")
        return self.state

    def write(self, command: ArmCommand) -> None:
        state = self.read()
        if np.any(command.position < self.joint_limits[:, 0]) or np.any(command.position > self.joint_limits[:, 1]):
            raise ValueError("Arm command exceeds the configured robot joint limits")
        if np.max(np.abs(command.position - state.position)) > self.max_tracking_error:
            raise RuntimeError("Command exceeds the permitted joint tracking error")
        self.arm_pub.publish(Float64MultiArray(data=command.position.tolist()))
        self.grip_pub.publish(Float64MultiArray(data=[float(command.gripper)]))
