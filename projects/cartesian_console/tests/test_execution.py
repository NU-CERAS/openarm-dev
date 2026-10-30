import numpy as np
import yaml

from openarm_sim.execution import ArmCommand, MujocoArmInterface, joint_names
from openarm_sim.simulation import ROOT, ReachSimulation


def test_commands_match_existing_ros2_controllers():
    controllers = yaml.safe_load((ROOT / "src/ros/openarm_mujoco_sim/config/controllers.yaml").read_text())
    with ReachSimulation() as sim:
        for arm in ("left", "right"):
            interface = MujocoArmInterface(sim, arm)
            state = interface.read()
            command = ArmCommand(state.position, state.gripper)
            interface.write(command)
            assert controllers[f"{arm}_forward_position_controller"]["ros__parameters"]["joints"] == joint_names(arm)
            trajectory_controller = controllers[f"{arm}_joint_trajectory_controller"]["ros__parameters"]
            assert trajectory_controller["joints"] == joint_names(arm)
            assert trajectory_controller["type"] == "joint_trajectory_controller/JointTrajectoryController"
            assert trajectory_controller["command_interfaces"] == ["position"]
            assert trajectory_controller["state_interfaces"] == ["position", "velocity"]
            assert command.controller_messages(arm)[f"/{arm}_gripper_controller/commands"] == [0.044]
            np.testing.assert_allclose(sim._desired[sim.actuator_ids[arm]], state.position)
