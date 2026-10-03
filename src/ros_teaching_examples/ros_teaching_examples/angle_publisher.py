"""A timer publishes an angle in radians; ROS delivers it to subscribers."""
import math
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64


class AnglePublisher(Node):
    def __init__(self):
        super().__init__('angle_publisher')
        self.publisher = self.create_publisher(Float64, 'teaching/angle', 10)
        self.start = self.get_clock().now()
        self.timer = self.create_timer(0.5, self.publish_angle)

    def publish_angle(self):
        elapsed = (self.get_clock().now() - self.start).nanoseconds / 1e9
        message = Float64()
        message.data = math.sin(elapsed)
        self.publisher.publish(message)
        self.get_logger().info(f'Sent {message.data:.3f} radians')


def main(args=None):
    rclpy.init(args=args)
    node = AnglePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
