import json
import os
import time

import mujoco
import numpy as np
import pytest

from openarm_sim.console.session import MujocoPlant, RobotWorker, SyncSession
from openarm_sim.deployment.adapter import PolicyAdapter, PolicySettings, SimulationPolicyInterface
from openarm_sim.deployment.can import CanPolicyInterface, GripperCalibration
from openarm_sim.execution import ArmCommand
from openarm_sim.paths import PROJECT_ROOT, RUNS
from openarm_sim.pickup import StickPickupEnv
from openarm_sim.residual_drumming import ResidualDrumStrikeEnv
from openarm_tools.robot.config import model_position, motor_position


def test_zero_pose_initializes_both_arms_and_preserves_torso_reference():
    plant = MujocoPlant(realtime=False)
    try:
        for arm in ("left", "right"):
            np.testing.assert_array_equal(plant.read().positions[arm], np.zeros(7))
            assert abs(plant.sim.tcp_position(arm)[2] - 0.08189955) < 1e-7
        assert not plant.sim._collision(plant.sim.data)
        np.testing.assert_allclose(plant.sim.data.site_xpos[plant.sim.model.site("torso_top_reference").id],
                                   [-0.00975, 0, 0.773], atol=1e-12)
    finally:
        plant.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Requires actual Tk events")
def test_console_start_and_arm_selection_use_measured_tcp():
    from openarm_sim.console.app import SyncConsole

    worker = RobotWorker(realtime=False)
    session = None
    console = None
    try:
        assert worker.ready.wait(10)
        session = SyncSession(worker)
        console = SyncConsole(session, RUNS / "tests/deployment/targets.json", width=900, height=550)
        for arm in ("right", "left"):
            console.arm.set(arm)
            expected = session.reference.from_world(session.twin.tcp_position(arm))
            np.testing.assert_allclose([float(value.get()) for value in console.coordinates], expected, atol=0.006)
            np.testing.assert_allclose(session.sync().positions[arm], np.zeros(7), atol=0.002)
            assert session.plan is None
            assert worker.state == "connected"
    finally:
        if console is not None and not console.closed:
            console.close()
        elif session is not None:
            session.close()
        else:
            worker.close()


@pytest.mark.parametrize("mode", ["absolute", "joint_delta", "pickup_residual"])
def test_policy_adapter_executes_physical_pickup_through_position_interface(mode):
    with StickPickupEnv() as env:
        env.reset(seed=0)
        initial_height = env.stick_position()[2]
        interface = SimulationPolicyInterface(env.sim, "right")
        limits = env.sim.model.jnt_range[env.sim.joint_ids["right"]]
        settings = PolicySettings(mode=mode)
        with PolicyAdapter(interface, limits, settings, lambda: env.sim.data.time) as adapter:
            for action in env.reference_pickup_actions():
                nominal = np.r_[env.sim._desired[env.sim.actuator_ids["right"]] + 0.025 * action[:7],
                                (float(action[7]) + 1) * 0.022]
                if mode == "absolute":
                    adapter.step(nominal, env.sim.data.time)
                elif mode == "joint_delta":
                    adapter.step(action, env.sim.data.time)
                else:
                    adapter.step(np.zeros(8), env.sim.data.time, nominal)
                for _ in range(round(0.02 / env.sim.model.opt.timestep)):
                    env.sim.step()
            assert env.grasp_contacts() == 2
            assert env.stick_position()[2] - initial_height > 0.10
            assert not env.unsafe_contacts()
        assert interface.closed


def test_strike_residual_adapter_executes_a_physical_impact():
    with ResidualDrumStrikeEnv(randomization=0.0) as env:
        env.reset(seed=0)
        interface = SimulationPolicyInterface(env.sim, "right")
        action = np.r_[np.zeros(7), -1.0]
        impacts = 0
        with PolicyAdapter(interface, env.sim.model.jnt_range[env.sim.joint_ids["right"]],
                           PolicySettings(mode="strike_residual"), lambda: env.sim.data.time) as adapter:
            for step in range(100):
                adapter.step(action, env.sim.data.time, env.reference.sample((step + 1) * 0.02))
                for _ in range(round(0.02 / env.sim.model.opt.timestep)):
                    env.sim.step()
                    for index, contact in enumerate(env.sim.data.contact):
                        if {int(contact.geom1), int(contact.geom2)} == {env.env.drum_id, env.env.stick_id}:
                            force = np.empty(6)
                            mujoco.mj_contactForce(env.sim.model, env.sim.data, index, force)
                            impacts += int(force[0] > 0)
            assert impacts > 0
            assert env.env.pickup.grasp_contacts() == 2


@pytest.mark.parametrize("failure", ["stale_action", "stale_feedback", "speed", "cadence", "late_cadence"])
def test_invalid_policy_commands_stop_the_actual_sim_interface(failure):
    plant = MujocoPlant(realtime=False)
    interface = SimulationPolicyInterface(plant.sim, "right")
    limits = plant.sim.model.jnt_range[plant.sim.joint_ids["right"]]
    settings = PolicySettings(mode="joint_delta", max_velocity_rad_s=0.3)
    clock = (lambda: plant.sim.data.time + 1) if failure == "stale_feedback" else (lambda: plant.sim.data.time)
    try:
        with pytest.raises(RuntimeError):
            with PolicyAdapter(interface, limits, settings, clock) as adapter:
                action = np.r_[np.zeros(7), 1.0]
                if failure == "stale_action":
                    adapter.step(action, clock() - 1)
                elif failure == "speed":
                    action[0] = 1
                    adapter.step(action, clock())
                elif failure == "cadence":
                    adapter.step(action, clock())
                    adapter.step(action, clock())
                elif failure == "late_cadence":
                    adapter.step(action, clock())
                    for _ in range(round(0.04 / plant.sim.model.opt.timestep)):
                        plant.sim.step()
                    adapter.step(action, clock())
        assert interface.closed
    finally:
        plant.close()


def test_calibrated_motor_coordinates_round_trip_model_angles():
    joint = {"sign": -1, "motor_zero_rad": 0.7}
    assert motor_position(joint, 0.2) == pytest.approx(0.5)
    assert model_position(joint, 0.5) == pytest.approx(0.2)
    assert model_position(joint, 0.7) == 0


def test_gripper_placeholder_is_rejected_before_hardware_connection():
    with pytest.raises(ValueError, match="placeholders"):
        GripperCalibration(json.loads((PROJECT_ROOT / "configs/policy_gripper.example.json").read_text()))
    with pytest.raises(ValueError, match="placeholders"):
        CanPolicyInterface(PROJECT_ROOT / "configs/sync_hardware.example.json",
                           PROJECT_ROOT / "configs/policy_gripper.example.json", "right")


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Requires actual Tk events")
def test_console_startup_preserves_a_measured_nonzero_robot_pose():
    from openarm_sim.console.app import SyncConsole

    worker = RobotWorker(realtime=False)
    previous = session = console = None
    try:
        assert worker.ready.wait(10)
        previous = SyncSession(worker)
        assert previous.prepare_relative("right", [40, -25, -30]) is not None
        previous.execute()
        deadline = time.monotonic() + 20
        while worker.state != "completed":
            worker.read()
            if time.monotonic() > deadline:
                raise TimeoutError("The native robot did not reach its setup pose")
            time.sleep(0.01)
        initial = worker.read().positions["right"].copy()
        session = SyncSession(worker)
        console = SyncConsole(session, RUNS / "tests/deployment/nonzero_targets.json", width=900, height=550)
        np.testing.assert_allclose(session.sync().positions["right"], initial, atol=0.002)
        np.testing.assert_allclose([float(value.get()) for value in console.coordinates], [40, -25, -30], atol=1.0)
        snapshot = json.loads((console.session_directory / "startup.json").read_text())
        assert not snapshot["motion_commanded"]
        np.testing.assert_allclose(snapshot["joint_positions_rad"]["right"], initial, atol=0.002)
    finally:
        if console is not None and not console.closed:
            console.close()
        elif session is not None:
            session.close()
        else:
            worker.close()
        if previous is not None:
            previous.close()
