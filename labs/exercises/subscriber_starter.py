"""Lab 01 exercise. Start the angle publisher in another terminal first."""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class StudentSubscriber(Node):
    def __init__(self):
        super().__init__('student_subscriber')
        # TODO: assign self.subscription using self.create_subscription.
        # Message type: Float64; topic: 'teaching/angle'; callback: self.receive; depth: 10.
        raise NotImplementedError('Create the subscription and remove this exception.')

    def receive(self, message):
        # TODO: replace this line with a log of the angle converted to degrees.
        self.get_logger().info(f'{message.data:.3f} radians')


if __name__ == '__main__':
    rclpy.init()
    node = StudentSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
