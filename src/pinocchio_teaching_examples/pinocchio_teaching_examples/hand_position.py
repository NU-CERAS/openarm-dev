"""Bridge ROS joint-state messages to a Pinocchio kinematics calculation."""
import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from .common import load_arm, hand_position


class HandPosition(Node):
    def __init__(self):
        super().__init__('hand_position')
        self.model, self.data, self.frame_id = load_arm()
        self.publisher = self.create_publisher(PointStamped, 'teaching/hand_position', 10)
        self.subscription = self.create_subscription(
            JointState, 'joint_states', self.receive_joints, 10)

    def receive_joints(self, message):
        if len(message.name) != len(message.position):
            self.get_logger().warning('Ignoring a JointState with mismatched names and positions.')
            return
        positions = dict(zip(message.name, message.position))
        q = pin.neutral(self.model)
        # Match names, never assume that a ROS message uses the URDF's joint order.
        for joint_id in range(1, self.model.njoints):
            name = self.model.names[joint_id]
            if name not in positions or not np.isfinite(positions[name]):
                self.get_logger().warning(f'Ignoring a JointState missing a finite {name} position.')
                return
            q[self.model.joints[joint_id].idx_q] = positions[name]
        xyz = hand_position(self.model, self.data, self.frame_id, q)
        point = PointStamped()
        point.header.stamp = message.header.stamp
        point.header.frame_id = 'base_link'
        point.point.x, point.point.y, point.point.z = map(float, xyz)
        self.publisher.publish(point)


def main(args=None):
    rclpy.init(args=args)
    node = HandPosition()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
