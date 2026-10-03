"""Publish joint positions, accept live parameters, and reset motion by service.

This is a visualization example: it publishes measured-state messages without
controlling hardware or simulating forces.
"""
import math
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger


class JointMotion(Node):
    def __init__(self):
        super().__init__('joint_motion')
        self.declare_parameter('amplitude', 0.7)  # radians
        self.declare_parameter('frequency_hz', 0.2)  # cycles per second
        result = self.validate_parameters([
            self.get_parameter('amplitude'), self.get_parameter('frequency_hz')])
        if not result.successful:
            raise ValueError(result.reason)
        self.add_on_set_parameters_callback(self.validate_parameters)
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.reset_service = self.create_service(Trigger, 'reset_motion', self.reset_motion)
        self.phase = 0.0
        self.last_tick = self.get_clock().now()
        self.timer = self.create_timer(0.05, self.publish_joints)

    def validate_parameters(self, parameters):
        bounds = {'amplitude': (0.0, math.pi), 'frequency_hz': (0.0, 2.0)}
        for parameter in parameters:
            if parameter.name in bounds:
                low, high = bounds[parameter.name]
                value = parameter.value
                if (not isinstance(value, (float, int)) or isinstance(value, bool)
                        or not math.isfinite(value) or not low <= value <= high):
                    return SetParametersResult(successful=False,
                        reason=f'{parameter.name} must be finite and between {low} and {high}')
        return SetParametersResult(successful=True)

    def publish_joints(self):
        now = self.get_clock().now()
        dt = max(0.0, (now - self.last_tick).nanoseconds / 1e9)
        self.last_tick = now
        frequency = self.get_parameter('frequency_hz').value
        self.phase = (self.phase + 2 * math.pi * frequency * dt) % (2 * math.pi)
        angle = self.get_parameter('amplitude').value * math.sin(self.phase)
        message = JointState()
        message.header.stamp = now.to_msg()
        message.name = ['shoulder_joint', 'elbow_joint']
        message.position = [angle, -0.5 * angle]
        self.publisher.publish(message)

    def reset_motion(self, request, response):
        self.phase = 0.0
        self.last_tick = self.get_clock().now()
        response.success = True
        response.message = 'Phase reset; periodic motion resumes on the next timer tick.'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = JointMotion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
