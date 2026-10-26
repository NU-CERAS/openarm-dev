import argparse
import json
import time
from pathlib import Path

import numpy as np

from .execution import ArmCommand, JointTrajectory, MujocoArmInterface, joint_names
from .simulation import ReachSimulation


def load_trajectory(path: Path):
    payload = json.loads(path.read_text())
    if payload["schema"] != "openarm.joint_trajectory.v1" or payload["joint_names"] != joint_names(payload["arm"]):
        raise ValueError("Unsupported trajectory schema or joint order")
    if (payload["position_unit"], payload["time_unit"], payload["gripper_unit"]) != ("rad", "s", "m"):
        raise ValueError("Trajectory units must be radians, seconds, and meters")
    trajectory = JointTrajectory(payload["arm"], payload["times"], payload["positions"], payload["velocities"], payload["source"])
    ArmCommand(trajectory.positions[0], payload["gripper"])
    return trajectory, float(payload["gripper"])


def execute(interface, trajectory: JointTrajectory, gripper: float, advance=None):
    initial = interface.read()
    if np.max(np.abs(initial.position - trajectory.positions[0])) > 0.03:
        raise RuntimeError("Robot starting position differs from trajectory start by more than 0.03 radians")
    clock_start = time.monotonic()
    t = 0.0
    worst_error = 0.0
    while t <= trajectory.duration + 1.0:
        desired = trajectory.sample(t)
        actual = interface.read()
        error = float(np.max(np.abs(desired - actual.position)))
        worst_error = max(worst_error, error)
        if error > 0.25:
            interface.write(ArmCommand(actual.position, actual.gripper))
            raise RuntimeError("Joint trajectory tracking error exceeds 0.25 radians")
        interface.write(ArmCommand(desired, gripper))
        if advance is None:
            t += 0.01
            time.sleep(max(0.0, clock_start + t - time.monotonic()))
        else:
            advance()
            t += 0.01
    final = interface.read()
    final_error = float(np.max(np.abs(final.position - trajectory.positions[-1])))
    if final_error > 0.03:
        raise RuntimeError(f"Final joint error exceeds 0.03 radians: {final_error}")
    return {"max_tracking_error_rad": worst_error, "final_joint_error_rad": final_error}


def main():
    parser = argparse.ArgumentParser(description="Inspect or execute a named-joint OpenArm trajectory")
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--backend", choices=["sim", "ros2"], default="sim")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    trajectory, gripper = load_trajectory(args.trajectory)
    if not args.execute:
        print(json.dumps({"duration_s": trajectory.duration, "joint_names": joint_names(trajectory.arm),
                          "first_command": ArmCommand(trajectory.positions[0], gripper).controller_messages(trajectory.arm),
                          "executed": False}, indent=2))
        return
    if args.backend == "sim":
        with ReachSimulation() as sim:
            interface = MujocoArmInterface(sim, trajectory.arm)

            def advance():
                for _ in range(round(0.01 / sim.model.opt.timestep)):
                    sim.step()

            print(json.dumps(execute(interface, trajectory, gripper, advance)))
    else:
        import rclpy
        from .ros2_interface import Ros2ArmInterface

        rclpy.init()
        with ReachSimulation() as model_reference:
            limits = model_reference.model.jnt_range[model_reference.joint_ids[trajectory.arm]].copy()
        interface = Ros2ArmInterface(trajectory.arm, limits)
        try:
            deadline = time.monotonic() + 3.0
            while interface.state is None and time.monotonic() < deadline:
                rclpy.spin_once(interface, timeout_sec=0.05)
            print(json.dumps(execute(interface, trajectory, gripper)))
        finally:
            interface.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
