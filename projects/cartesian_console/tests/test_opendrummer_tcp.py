import importlib.util
import time

import mujoco
import numpy as np
import pytest

from openarm_sim.simulation import reach_simulation
from openarm_sim.console.session import RobotWorker, SyncSession
from openarm_sim.paths import RUNS


def test_stick_tip_is_planning_tcp():
    with reach_simulation("opendrummer", planner="mink") as sim:
        for arm in ("left", "right"):
            assert sim.model.site(sim.site_ids[arm]).name == f"{arm}_stick_tip"
            target = sim.tcp_position(arm) + [0, 0, .01]
            solution = sim._solve(arm, target)
            assert solution is not None
            probe = mujoco.MjData(sim.model)
            probe.qpos[:] = solution
            mujoco.mj_forward(sim.model, probe)
            np.testing.assert_allclose(probe.site_xpos[sim.site_ids[arm]], target, atol=.002)


@pytest.mark.skipif(importlib.util.find_spec("yourdfpy") is None, reason="Requires cuRobo export dependencies")
@pytest.mark.parametrize("robot_model", ["v1.0", "opendrummer"])
def test_planning_export_retains_tip_frames(tmp_path, robot_model):
    from yourdfpy import URDF
    from openarm_sim.planning.assets import export_robot

    path = export_robot(tmp_path / "robot", robot_model)
    robot = URDF.load(str(path), load_meshes=False)
    with reach_simulation(robot_model) as sim:
        for seed in range(3):
            rng = np.random.default_rng(seed)
            configuration = {}
            for index in range(sim.model.njnt):
                value = float(rng.uniform(*sim.model.jnt_range[index]))
                sim.data.qpos[sim.model.jnt_qposadr[index]] = value
                configuration[sim.model.joint(index).name] = value
            mujoco.mj_forward(sim.model, sim.data)
            robot.update_cfg(configuration)
            for arm in ("left", "right"):
                actual = robot.get_transform(f"{arm}_planning_tcp", "world")
                np.testing.assert_allclose(actual[:3, 3], sim.tcp_position(arm), atol=1e-8)
                np.testing.assert_allclose(actual[:3, :3], sim.data.site_xmat[sim.site_ids[arm]].reshape(3, 3), atol=1e-8)


@pytest.mark.parametrize("planner", ["mink", "ompl", "curobo"])
def test_two_reaches_use_measured_joints_and_tip(planner):
    if planner == "curobo" and (importlib.util.find_spec("curobo") is None or
                                not (RUNS / "planning/opendrummer/openarm.yml").is_file()):
        pytest.skip("Requires cuRobo and model-specific planning assets")
    worker = RobotWorker(realtime=False, robot_model="opendrummer")
    session = None
    try:
        assert worker.ready.wait(15)
        worker.read()
        session = SyncSession(worker, planner, robot_model="opendrummer")
        for _ in range(2):
            measured = session.sync()
            target = session.twin.tcp_position("right") + [0, 0, .01]
            plan = session.prepare("right", target)
            assert plan is not None, session.rejection_reason
            np.testing.assert_allclose(plan.trajectory.positions[0], measured.positions["right"], atol=.002)
            session.execute()
            deadline = time.monotonic() + 20
            while worker.completed_plan is not plan:
                worker.read()
                assert time.monotonic() < deadline
                time.sleep(.01)
            final = session.sync()
            np.testing.assert_allclose(session.preview.data.qpos[session.preview.qpos_ids["right"]], final.positions["right"])
            assert np.linalg.norm(session.preview.tcp_position("right") - target) < .01
    finally:
        if session is not None:
            session.close()
        else:
            worker.close()
