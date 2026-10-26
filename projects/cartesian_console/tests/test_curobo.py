import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

from openarm_sim.simulation import ReachSimulation
from openarm_sim.paths import RUNS


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("curobo") is None or not (RUNS / "planning/openarm.yml").exists(),
    reason="Requires the optional cuRobo installation and generated robot assets",
)


def test_exported_robot_matches_mujoco_kinematics():
    from yourdfpy import URDF

    robot = URDF.load(str(RUNS / "planning/openarm.urdf"), load_meshes=False)
    rng = np.random.default_rng(42)
    with ReachSimulation() as sim:
        for _ in range(10):
            configuration = {}
            for index in range(sim.model.njnt):
                lower, upper = sim.model.jnt_range[index]
                value = float(rng.uniform(lower, upper))
                sim.data.qpos[sim.model.jnt_qposadr[index]] = value
                configuration[sim.model.joint(index).name] = value
            mujoco.mj_forward(sim.model, sim.data)
            robot.update_cfg(configuration)
            for arm in ("left", "right"):
                matrix = robot.get_transform(f"openarm_{arm}_hand_tcp", "world")
                np.testing.assert_allclose(matrix[:3, 3], sim.tcp_position(arm), atol=1e-8)
                np.testing.assert_allclose(matrix[:3, :3], sim.data.site_xmat[sim.site_ids[arm]].reshape(3, 3), atol=1e-8)


@pytest.mark.parametrize("with_obstacle", [False, True])
def test_curobo_trajectory_executes_through_physics(with_obstacle):
    obstacles = ({"name": "reach_obstacle", "position": [0.4, -0.24, 0.4], "half_size": [0.012, 0.012, 0.08]},) if with_obstacle else ()
    with ReachSimulation(planner="curobo", obstacles=obstacles) as sim:
        target = np.array([0.52, -0.35, 0.44] if with_obstacle else [0.43, -0.24, 0.44])
        if with_obstacle:
            goal = sim._solve("right", target)
            data = mujoco.MjData(sim.model)
            collisions = []
            for fraction in np.linspace(0, 1, 40):
                data.qpos[:] = sim.data.qpos + fraction * (goal - sim.data.qpos)
                mujoco.mj_forward(sim.model, data)
                collisions.append(sim._collision(data))
            assert not collisions[0] and not collisions[-1]
            assert any(collisions[1:-1])
        result = sim.start_reach("right", target)
        assert result.accepted, result.reason
        assert sim.trajectory.source == "curobo"
        for _ in range(15000):
            sim.step()
            if sim.status.state != "running":
                break
        assert sim.status.state == "succeeded", sim.status
        assert np.linalg.norm(sim.tcp_position("right") - target) < 0.01


@pytest.mark.skipif(not (RUNS / "drumset/converted/manifest.json").exists(), reason="Requires the imported drum model")
def test_ik_rejection_preserves_state_and_allows_another_reach():
    manifest = RUNS / "drumset/converted/manifest.json"
    with ReachSimulation(planner="curobo", drum_asset=manifest, worktable=False) as sim:
        rejected_pad = next(pad for pad in sim.drum_asset["pads"] if pad["source_node"] == "Object_33")
        target = np.array(rejected_pad["center"]) + np.array(rejected_pad["normal"]) * 0.14
        positions = sim.data.qpos.copy()
        controls = sim.data.ctrl.copy()
        desired = sim._desired.copy()
        result = sim.start_reach("right", target)
        assert not result.accepted
        assert "IK" in result.reason
        assert "right" in result.reason
        np.testing.assert_array_equal(sim.data.qpos, positions)
        np.testing.assert_array_equal(sim.data.ctrl, controls)
        np.testing.assert_array_equal(sim._desired, desired)
        assert sim.status.state == "idle"
        valid_pad = next(pad for pad in sim.drum_asset["pads"] if pad["source_node"] == "Object_28")
        target = np.array(valid_pad["center"]) + np.array(valid_pad["normal"]) * 0.14
        result = sim.start_reach("left", target)
        assert result.accepted, result.reason
        for _ in range(15000):
            sim.step()
            if sim.status.state != "running":
                break
        assert sim.status.state == "succeeded", sim.status


def test_blocked_goal_returns_a_rejected_plan_without_changing_commands():
    obstacle = {"name": "blocked_goal", "position": [0.52, -0.35, 0.44], "half_size": [0.07, 0.07, 0.07]}
    with ReachSimulation(planner="curobo", obstacles=(obstacle,)) as sim:
        assert not sim._collision(sim.data)
        controls = sim.data.ctrl.copy()
        desired = sim._desired.copy()
        result = sim.start_reach("right", np.array([0.52, -0.35, 0.44]))
        assert not result.accepted
        assert "cuRobo" in result.reason
        assert sim.status.state == "idle"
        np.testing.assert_array_equal(sim.data.ctrl, controls)
        np.testing.assert_array_equal(sim._desired, desired)
