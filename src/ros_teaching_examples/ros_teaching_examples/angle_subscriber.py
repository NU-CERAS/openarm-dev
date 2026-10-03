"""The callback runs when a message arrives, independently of the publisher."""
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class AngleSubscriber(Node):
    def __init__(self):
        super().__init__('angle_subscriber')
        self.subscription = self.create_subscription(
            Float64, 'teaching/angle', self.receive_angle, 10)

    def receive_angle(self, message):
        degrees = math.degrees(message.data)
        self.get_logger().info(f'Received {message.data:.3f} rad = {degrees:.1f} degrees')


def main(args=None):
    rclpy.init(args=args)
    node = AngleSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
