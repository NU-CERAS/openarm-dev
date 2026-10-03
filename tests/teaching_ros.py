"""End-to-end topic, parameter, service, and bridge checks in an isolated domain.

Run: python3 tests/teaching_ros.py (inside a sourced container terminal).
"""
import math
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

# Keep test nodes separate from the normal RViz session.
os.environ['ROS_DOMAIN_ID'] = '77'
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_srvs.srv import Trigger


def wait_until(node, predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if predicate():
            return
    raise AssertionError('Timed out waiting for ROS communication')


def call(node, client, request):
    assert client.wait_for_service(timeout_sec=10), 'Service was not discovered'
    future = client.call_async(request)
    wait_until(node, future.done)
    result = future.result()
    assert result is not None
    return result


def main():
    processes = []
    files = []
    rclpy.init()
    node = Node('teaching_smoke_test')
    try:
        with tempfile.TemporaryDirectory(prefix='teaching-ros-') as directory:
            for package, executable in [
                ('ros_teaching_examples', 'angle_publisher'),
                ('ros_teaching_examples', 'angle_subscriber'),
                ('ros_teaching_examples', 'joint_motion'),
                ('pinocchio_teaching_examples', 'hand_position'),
            ]:
                output = open(Path(directory) / f'{executable}.log', 'w')
                files.append(output)
                processes.append(subprocess.Popen(
                    ['ros2', 'run', package, executable], stdout=output,
                    stderr=subprocess.STDOUT, start_new_session=True))
            angles, joints, points = [], {}, []
            def receive_joints(message):
                stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
                joints[stamp] = dict(zip(message.name, message.position))
            subscriptions = [
                node.create_subscription(Float64, '/teaching/angle', angles.append, 10),
                node.create_subscription(JointState, '/joint_states', receive_joints, 10),
                node.create_subscription(PointStamped, '/teaching/hand_position', points.append, 10),
            ]
            wait_until(node, lambda: len(angles) >= 3 and len(points) >= 3)
            assert all(math.isfinite(message.data) and abs(message.data) <= 1 for message in angles)
            def matching_sample():
                return any((p.header.stamp.sec, p.header.stamp.nanosec) in joints for p in points)
            wait_until(node, matching_sample)
            for point in points:
                stamp = (point.header.stamp.sec, point.header.stamp.nanosec)
                if stamp not in joints:
                    continue
                q = joints[stamp]
                q1, q2 = q['shoulder_joint'], q['elbow_joint']
                expected_x = 0.6 * math.cos(q1) + 0.4 * math.cos(q1 + q2)
                expected_z = -0.6 * math.sin(q1) - 0.4 * math.sin(q1 + q2)
                assert point.header.frame_id == 'base_link'
                assert abs(point.point.x - expected_x) < 1e-10
                assert abs(point.point.y) < 1e-10
                assert abs(point.point.z - expected_z) < 1e-10
            parameters = node.create_client(SetParameters, '/joint_motion/set_parameters')
            def set_parameter(name, value):
                request = SetParameters.Request(parameters=[Parameter(
                    name=name, value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                                    double_value=value))])
                return call(node, parameters, request).results[0]
            assert not set_parameter('amplitude', -1.0).successful
            assert set_parameter('amplitude', 1.0).successful
            assert set_parameter('frequency_hz', 0.0).successful
            reset = node.create_client(Trigger, '/reset_motion')
            assert call(node, reset, Trigger.Request()).success
            points.clear()
            wait_until(node, lambda: any(abs(p.point.x - 1.0) < 1e-10
                                        and abs(p.point.z) < 1e-10 for p in points))
            # Replace the motion source with a reversed-name message. The bridge
            # must still assign shoulder and elbow to the correct model indices.
            os.killpg(processes[2].pid, signal.SIGINT)
            processes[2].wait(timeout=5)
            publisher = node.create_publisher(JointState, '/joint_states', 10)
            wait_until(node, lambda: publisher.get_subscription_count() > 0)
            sample = JointState()
            sample.header.stamp = node.get_clock().now().to_msg()
            sample.name = ['elbow_joint', 'shoulder_joint']
            sample.position = [-0.9, 0.3]
            publisher.publish(sample)
            def reversed_sample():
                return next((p for p in points
                             if p.header.stamp == sample.header.stamp), None)
            wait_until(node, lambda: reversed_sample() is not None)
            point = reversed_sample()
            assert abs(point.point.x - (0.6 * math.cos(0.3) + 0.4 * math.cos(-0.6))) < 1e-10
            assert abs(point.point.z - (-0.6 * math.sin(0.3) - 0.4 * math.sin(-0.6))) < 1e-10
            for output in files:
                output.flush()
            assert 'Received' in (Path(directory) / 'angle_subscriber.log').read_text()
            assert all(processes[index].poll() is None for index in (0, 1, 3))
            print('PASS: angle delivery, subscriber output, joint-name mapping, live hand position,')
            print('      parameter validation/update, and reset service.')
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
        for output in files:
            output.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
