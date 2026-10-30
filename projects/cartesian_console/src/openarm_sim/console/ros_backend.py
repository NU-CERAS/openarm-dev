import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from openarm_tools.recording.storage import lock
from openarm_tools.robot.config import load, validate_positions
from openarm_tools.ros_execution import ExecutionCancelled, JointSnapshot, RosExecutionServer, arm_joint_names, trajectory_arrays

from ..execution import JointTrajectory
from .session import Feedback, Preview, RobotWorker, check_start, hardware_configuration


def stop_viewer(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise TimeoutError("Robot feedback viewer did not stop") from None
    if process.returncode not in (0, -signal.SIGINT):
        raise RuntimeError(f"Robot feedback viewer exited with code {process.returncode}")


class RobotExecutionBackend:
    def __init__(self, worker):
        self.worker = worker

    def read(self):
        feedback = self.worker.read()
        names = tuple(name for arm in ("left", "right") for name in arm_joint_names(arm))
        return JointSnapshot(names, np.concatenate([feedback.positions[arm] for arm in ("left", "right")]),
                             np.concatenate([feedback.velocities[arm] for arm in ("left", "right")]),
                             time.time(), feedback.received_at)

    def plan(self, arm, message):
        times, positions, velocities = trajectory_arrays(arm, message)
        trajectory = JointTrajectory(arm, times, positions, velocities, "ros2")
        measured = self.worker.read()
        if np.max(np.abs(positions[0] - measured.positions[arm])) > 0.03:
            raise ValueError("Trajectory start differs from measured joints")
        start_positions = {name: value.copy() for name, value in measured.positions.items()}
        start_positions[arm] = positions[0].copy()
        start = Feedback(start_positions, measured.velocities, measured.received_at)
        cfg = load(robot="v1.0", arm=arm) if self.worker.backend == "demo" else hardware_configuration(self.worker.configuration)[arm][1]
        for position in positions:
            validate_positions(cfg, position.tolist())
        if self.worker.backend == "can":
            sample_times = np.linspace(0, trajectory.duration, max(1000, len(times) * 10))
            for position in trajectory.sample(sample_times):
                validate_positions(cfg, position.tolist())
            if np.max(np.abs(trajectory._spline(sample_times, 1))) > cfg["max_speed"] + 1e-6:
                raise ValueError("Trajectory exceeds configured physical joint speed")
        return Preview(trajectory, None, start)

    def validate(self, arm, message):
        if self.worker.state == "executing":
            raise ValueError("Another robot trajectory is executing")
        self.plan(arm, message)

    def execute(self, arm, message, cancelled, publish):
        plan = self.plan(arm, message)
        check_start(plan, self.worker.read())
        self.worker.execute(plan)
        deadline = time.monotonic() + plan.trajectory.duration + 5
        while self.worker.completed_plan is not plan:
            if cancelled():
                self.worker.cancel()
            publish(self.read())
            if self.worker.state == "cancelled":
                raise ExecutionCancelled()
            if time.monotonic() > deadline:
                self.worker.cancel()
                raise TimeoutError("Robot execution exceeded its deadline")
            time.sleep(0.01)
        publish(self.read())


def main():
    parser = argparse.ArgumentParser(description="Execute standard ROS 2 joint trajectory actions with MuJoCo or CAN")
    parser.add_argument("--backend", choices=["demo", "can"], default="demo")
    parser.add_argument("--hardware-config", type=Path)
    parser.add_argument("--supported", action="store_true")
    parser.add_argument("--show-plant", action="store_true")
    parser.add_argument("--drum-asset", type=Path)
    parser.add_argument("--namespace", default="")
    args, ros_args = parser.parse_known_args()
    if args.backend == "can":
        if not args.supported or args.hardware_config is None or args.show_plant:
            raise ValueError("CAN requires reviewed --hardware-config and --supported, without a plant window")
        hardware_configuration(args.hardware_config)
    with ExitStack() as resources:
        resources.enter_context(lock(f"ros-execution:{args.namespace}"))
        worker = RobotWorker(args.backend, args.hardware_config, args.drum_asset)
        resources.callback(worker.close)
        if not worker.ready.wait(10):
            raise TimeoutError("Robot execution backend did not become ready")
        worker.read()
        rclpy.init(args=ros_args)
        resources.callback(rclpy.try_shutdown)
        node = RosExecutionServer(RobotExecutionBackend(worker), args.namespace)
        resources.callback(node.destroy_node)
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        resources.callback(executor.shutdown)
        if args.show_plant:
            viewer = subprocess.Popen([sys.executable, "-m", "openarm_sim.console.ros_viewer", "--namespace", args.namespace])
            resources.callback(stop_viewer, viewer)
        print(json.dumps({"ros_execution": "ready", "backend": args.backend, "namespace": args.namespace}), flush=True)
        try:
            executor.spin()
        except KeyboardInterrupt:
            worker.cancel()


if __name__ == "__main__":
    main()
