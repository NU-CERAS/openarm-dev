from dataclasses import dataclass
import threading
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARMS = ("left", "right")


def arm_joint_names(arm: str) -> list[str]:
    if arm not in ARMS:
        raise ValueError("Arm must be left or right")
    return [f"openarm_{arm}_joint{index}" for index in range(1, 8)]


def duration(seconds: float) -> Duration:
    nanoseconds = round(float(seconds) * 1_000_000_000)
    if nanoseconds < 0:
        raise ValueError("Duration cannot be negative")
    return Duration(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)


def trajectory_message(arm: str, times, positions, velocities) -> JointTrajectory:
    message = JointTrajectory(joint_names=arm_joint_names(arm))
    for elapsed, position, velocity in zip(times, positions, velocities, strict=True):
        message.points.append(JointTrajectoryPoint(positions=[float(x) for x in position],
                                                   velocities=[float(x) for x in velocity],
                                                   time_from_start=duration(elapsed)))
    trajectory_arrays(arm, message)
    return message


def trajectory_arrays(arm: str, message: JointTrajectory):
    if message.joint_names != arm_joint_names(arm):
        raise ValueError("Trajectory joint names/order do not match the selected arm")
    if message.header.stamp.sec or message.header.stamp.nanosec:
        raise ValueError("Relative trajectories require a zero header timestamp")
    times = np.array([point.time_from_start.sec + point.time_from_start.nanosec * 1e-9 for point in message.points])
    if len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Trajectory times must start at zero and increase strictly")
    positions = np.asarray([point.positions for point in message.points], dtype=float)
    velocities = np.asarray([point.velocities for point in message.points], dtype=float)
    if positions.shape != (len(times), 7) or velocities.shape != positions.shape:
        raise ValueError("Each waypoint requires seven positions and velocities")
    if not all(np.isfinite(values).all() for values in (times, positions, velocities)):
        raise ValueError("Trajectory contains non-finite values")
    return times, positions, velocities


@dataclass(frozen=True)
class JointSnapshot:
    names: tuple[str, ...]
    positions: np.ndarray
    velocities: np.ndarray
    timestamp: float
    received_at: float

    def arm(self, arm: str):
        indices = [self.names.index(name) for name in arm_joint_names(arm)]
        return self.positions[indices].copy(), self.velocities[indices].copy()


class ExecutionCancelled(Exception):
    pass


class RosExecutionClient:
    def __init__(self, namespace="", feedback_timeout=0.25):
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = Node("openarm_execution_client", namespace=namespace, context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.feedback_timeout = feedback_timeout
        self.snapshot = None
        self.sequence = 0
        self.active_goal = None
        self.subscription = self.node.create_subscription(JointState, "joint_states", self.feedback,
                                                          qos_profile_sensor_data)
        self.actions = {arm: ActionClient(self.node, FollowJointTrajectory,
                                         f"{arm}_joint_trajectory_controller/follow_joint_trajectory") for arm in ARMS}

    def feedback(self, message):
        required = [name for arm in ARMS for name in arm_joint_names(arm)]
        if not set(required).issubset(message.name):
            return
        if len(set(message.name)) != len(message.name) or len(message.position) != len(message.name) or len(message.velocity) != len(message.name):
            raise ValueError("JointState requires unique names and complete position/velocity feedback")
        positions, velocities = np.asarray(message.position), np.asarray(message.velocity)
        if not np.isfinite([positions, velocities]).all():
            raise ValueError("JointState contains non-finite feedback")
        self.snapshot = JointSnapshot(tuple(message.name), positions, velocities,
                                      message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
                                      time.monotonic())
        self.sequence += 1

    def wait_ready(self, timeout=10):
        deadline = time.monotonic() + timeout
        while self.snapshot is None or not all(action.server_is_ready() for action in self.actions.values()):
            self.executor.spin_once(timeout_sec=0.02)
            if time.monotonic() > deadline:
                raise TimeoutError("ROS execution actions and both-arm JointState feedback are required")
        self.read()

    def read(self):
        self.executor.spin_once(timeout_sec=0)
        deadline = time.monotonic() + self.feedback_timeout
        while self.snapshot is None or time.monotonic() - self.snapshot.received_at > self.feedback_timeout:
            if time.monotonic() >= deadline:
                raise RuntimeError("ROS joint feedback is missing or stale")
            self.executor.spin_once(timeout_sec=0.02)
        return self.snapshot

    def wait_future(self, future, deadline):
        while not future.done():
            self.executor.spin_once(timeout_sec=0.02)
            if time.monotonic() > deadline:
                raise TimeoutError("ROS action response exceeded its deadline")
        return future.result()

    def execute(self, arm, trajectory, cancel, publish):
        times, _, _ = trajectory_arrays(arm, trajectory)
        self.read()
        handle = self.wait_future(self.actions[arm].send_goal_async(FollowJointTrajectory.Goal(trajectory=trajectory)),
                                  time.monotonic() + 3)
        if not handle.accepted:
            raise RuntimeError("ROS trajectory goal was rejected; inspect the execution node log")
        self.active_goal = handle
        result_future = handle.get_result_async()
        deadline = time.monotonic() + float(times[-1]) + 5
        cancellation = None
        try:
            while not result_future.done():
                if cancel.is_set() and cancellation is None:
                    cancellation = handle.cancel_goal_async()
                self.executor.spin_once(timeout_sec=0.02)
                publish(self.read())
                if time.monotonic() > deadline:
                    raise TimeoutError("ROS trajectory execution exceeded its deadline")
            response = result_future.result()
            if response.status == GoalStatus.STATUS_CANCELED:
                raise ExecutionCancelled()
            if response.status != GoalStatus.STATUS_SUCCEEDED or response.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
                raise RuntimeError(f"ROS trajectory failed: {response.result.error_string}")
            sequence = self.sequence
            deadline = time.monotonic() + 1
            while self.sequence == sequence:
                self.executor.spin_once(timeout_sec=0.02)
                if time.monotonic() > deadline:
                    raise TimeoutError("No fresh JointState arrived after action completion")
            publish(self.read())
            return response.result
        finally:
            try:
                if not result_future.done():
                    self.wait_future(handle.cancel_goal_async(), time.monotonic() + 2)
                    self.wait_future(result_future, time.monotonic() + 2)
            finally:
                self.active_goal = None

    def close(self):
        if self.active_goal is not None:
            raise RuntimeError("Cancel the active ROS goal before closing its client")
        self.executor.shutdown(timeout_sec=2)
        for action in self.actions.values():
            action.destroy()
        self.node.destroy_node()
        self.context.shutdown()


class RosExecutionServer(Node):
    def __init__(self, backend, namespace=""):
        super().__init__("openarm_execution_server", namespace=namespace)
        self.backend = backend
        self.reservation = threading.Lock()
        self.group = ReentrantCallbackGroup()
        self.publisher = self.create_publisher(JointState, "joint_states", 10)
        self.timer = self.create_timer(0.02, lambda: self.publish_state(), callback_group=self.group)
        self.servers = [ActionServer(self, FollowJointTrajectory,
                                    f"{arm}_joint_trajectory_controller/follow_joint_trajectory",
                                    execute_callback=lambda handle, side=arm: self.execute(side, handle),
                                    goal_callback=lambda goal, side=arm: self.accept(side, goal),
                                    cancel_callback=lambda handle: CancelResponse.ACCEPT,
                                    callback_group=self.group) for arm in ARMS]

    def publish_state(self, snapshot=None):
        state = self.backend.read() if snapshot is None else snapshot
        message = JointState(name=list(state.names), position=state.positions.tolist(), velocity=state.velocities.tolist())
        message.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(message)

    def accept(self, arm, goal):
        try:
            if (goal.path_tolerance or goal.goal_tolerance or goal.multi_dof_trajectory.points
                    or goal.goal_time_tolerance.sec or goal.goal_time_tolerance.nanosec):
                raise ValueError("Configure backend tolerances; per-goal overrides and multi-DOF trajectories are unsupported")
            trajectory_arrays(arm, goal.trajectory)
            self.backend.validate(arm, goal.trajectory)
        except ValueError as error:
            self.get_logger().error(str(error))
            return GoalResponse.REJECT
        if not self.reservation.acquire(blocking=False):
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def execute(self, arm, handle):
        result = FollowJointTrajectory.Result()
        started = time.monotonic()

        def publish(snapshot):
            self.publish_state(snapshot)
            position, velocity = snapshot.arm(arm)
            feedback = FollowJointTrajectory.Feedback(joint_names=arm_joint_names(arm))
            feedback.header.stamp = self.get_clock().now().to_msg()
            feedback.actual.positions = position.tolist()
            feedback.actual.velocities = velocity.tolist()
            feedback.actual.time_from_start = duration(time.monotonic() - started)
            handle.publish_feedback(feedback)

        try:
            if handle.is_cancel_requested:
                raise ExecutionCancelled()
            self.backend.execute(arm, handle.request.trajectory, lambda: handle.is_cancel_requested, publish)
            if handle.is_cancel_requested:
                handle.canceled()
            else:
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                handle.succeed()
        except ExecutionCancelled:
            handle.canceled()
        except Exception as error:
            result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
            result.error_string = str(error)
            handle.abort()
        finally:
            self.reservation.release()
        return result

    def destroy_node(self):
        for server in self.servers:
            server.destroy()
        return super().destroy_node()
