import json
import importlib.util
import multiprocessing
import os
import time

import numpy as np
import pytest
import mujoco

from openarm_sim.paths import BODY_REFERENCE, PROJECT_ROOT, RUNS
from openarm_sim.simulation import ReachSimulation
from openarm_sim.console.session import Preview, RobotWorker, SyncSession, check_start, hardware_configuration
from openarm_sim.execution import JointTrajectory
from openarm_tools.recording.storage import lock


def wait_for(worker, state, seconds=15):
    deadline = time.monotonic() + seconds
    while worker.state != state:
        worker.read()
        if time.monotonic() > deadline:
            raise TimeoutError(f"Expected {state}, received {worker.state}")
        time.sleep(0.01)


def test_physical_execution_feedback_preview_and_saved_target():
    worker = RobotWorker(realtime=False)
    session = None
    try:
        assert worker.ready.wait(10)
        worker.read()
        session = SyncSession(worker)
        np.testing.assert_allclose(session.reference.origin_m, [-0.00975, 0, 0.773], atol=1e-12)
        np.testing.assert_allclose(session.reference.rotation, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(session.reference.to_world([43.975, -24, -33.3]), [0.43, -0.24, 0.44], atol=1e-12)
        np.testing.assert_allclose(session.reference.from_world([0.43, -0.24, 0.44]), [43.975, -24, -33.3], atol=1e-12)
        assert session.twin.table_height == 0.0
        table = session.twin.model.geom("worktable").id
        assert session.twin.model.geom_pos[table, 2] + session.twin.model.geom_size[table, 2] == 0.0
        assert session.twin.model.geom_pos[session.twin.model.geom("floor").id, 2] == -0.75
        initial = session.sync().positions["right"].copy()
        assert session.prepare("left", [0.43, -0.24, 0.44]) is None
        assert session.plan is None
        assert "chosen arm" in session.rejection_reason
        plan = session.prepare_relative("right", [43.975, -24, -33.3])
        assert plan is not None
        session.show_preview(plan.trajectory.duration)
        np.testing.assert_allclose(session.sync().positions["right"], initial, atol=0.002)
        assert np.linalg.norm(session.preview.tcp_position("right") - plan.target) < 0.01
        session.execute()
        wait_for(worker, "completed")
        feedback = session.sync()
        np.testing.assert_allclose(session.twin.data.site_xpos[session.twin.model.site("torso_top_reference").id], session.reference.origin_m, atol=1e-12)
        np.testing.assert_array_equal(session.twin.data.qpos[session.twin.qpos_ids["right"]], feedback.positions["right"])
        assert np.linalg.norm(session.twin.tcp_position("right") - plan.target) < 0.015
        assert worker.report["final_joint_error_rad"] < 0.03
        with pytest.raises(RuntimeError, match="moved after preview"):
            check_start(plan, feedback)
        output = RUNS / "tests/sync/targets.json"
        session.save(output, "pad_approach")
        arm, target = session.load(output, "pad_approach")
        assert arm == "right"
        np.testing.assert_array_equal(target, plan.target)
        saved = json.loads(output.read_text())["targets"]["pad_approach"]
        assert saved["execution_state"] == "completed"
        assert saved["reference_site"] == "torso_top_reference"
        np.testing.assert_allclose(saved["target_relative_cm"], [43.975, -24, -33.3], atol=1e-12)
        np.testing.assert_allclose(saved["observed_joint_positions_rad"], feedback.positions["right"], atol=0.002)
        session.prepare("right", [0.44, -0.24, 0.44])
        session.save(output, "next_target")
        assert json.loads(output.read_text())["targets"]["next_target"]["execution_state"] == "not_executed"
    finally:
        if session is not None:
            session.close()
        else:
            worker.close()
    assert not worker.thread.is_alive()


def test_cancellation_stops_physical_execution_and_keeps_feedback():
    worker = RobotWorker()
    session = None
    try:
        assert worker.ready.wait(10)
        session = SyncSession(worker)
        session.prepare("right", [0.43, -0.24, 0.44])
        session.execute()
        time.sleep(0.1)
        worker.cancel()
        wait_for(worker, "cancelled")
        assert worker.report is None
        session.sync()
        time.sleep(0.1)
        assert time.monotonic() - worker.read().received_at < 0.1
    finally:
        if session is not None:
            session.close()
        else:
            worker.close()


def test_hardware_placeholders_fail_before_connecting():
    with pytest.raises(ValueError, match="placeholders"):
        hardware_configuration(PROJECT_ROOT / "configs/sync_hardware.example.json")


def test_stale_feedback_rejected_from_stopped_physics_transport():
    worker = RobotWorker()
    session = None
    try:
        assert worker.ready.wait(10)
        session = SyncSession(worker)
        plan = session.prepare("right", [0.43, -0.24, 0.44])
        feedback = worker.read()
        worker.close()
        time.sleep(0.3)
        with pytest.raises(RuntimeError, match="stale"):
            check_start(plan, feedback)
    finally:
        if session is not None:
            session.close()
        else:
            worker.close()


def acquire_in_another_process(identity, results):
    directory = RUNS / "cache/other_client"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(directory)
    try:
        with lock(identity):
            results.put("acquired")
    except RuntimeError:
        results.put("busy")


def test_shared_ownership_lock_works_across_processes_and_cache_settings():
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    identity = "sync-test:actual-process-ownership"
    with lock(identity):
        process = context.Process(target=acquire_in_another_process, args=(identity, results))
        process.start()
        assert results.get(timeout=15) == "busy"
        process.join(timeout=15)
        assert process.exitcode == 0
    process = context.Process(target=acquire_in_another_process, args=(identity, results))
    process.start()
    assert results.get(timeout=15) == "acquired"
    process.join(timeout=15)
    assert process.exitcode == 0
    results.close()


def test_torso_reference_lies_on_the_actual_top_housing_surface():
    with ReachSimulation(tabletop_mount=True) as sim:
        origin = sim.data.site_xpos[sim.model.site(BODY_REFERENCE["site"]).id]
        distance = mujoco.mj_rayMesh(sim.model, sim.data, sim.model.geom(BODY_REFERENCE["surface_geom"]).id,
                                     origin + [0, 0, 0.1], np.array([0.0, 0.0, -1.0]))
        assert abs(distance - 0.1) < 0.0002


@pytest.mark.parametrize("arm,y", [("right", -25), ("left", 25)])
def test_default_torso_relative_coordinates_reach_through_actual_physics(arm, y):
    worker = RobotWorker(realtime=False)
    session = None
    try:
        assert worker.ready.wait(10)
        session = SyncSession(worker)
        plan = session.prepare_relative(arm, [40, y, -30])
        assert plan is not None, session.rejection_reason
        session.execute()
        wait_for(worker, "completed")
        session.sync()
        np.testing.assert_allclose(session.reference.from_world(session.twin.tcp_position(arm)),
                                   [40, y, -30], atol=1.0)
    finally:
        if session is not None:
            session.close()
        else:
            worker.close()


@pytest.mark.parametrize("planner", ["ompl", "mink", pytest.param("curobo", marks=pytest.mark.skipif(
    importlib.util.find_spec("curobo") is None or not (RUNS / "planning/openarm.yml").exists(),
    reason="Requires cuRobo and exported robot assets"))])
def test_successive_reaches_start_from_completed_feedback_and_sync_both_models(planner):
    worker = RobotWorker(realtime=False)
    session = None
    previous = None
    try:
        assert worker.ready.wait(10)
        session = SyncSession(worker, planner=planner)
        for target in ([40, -25, -30], [43, -25, -27], [38, -28, -25]):
            plan = session.prepare_relative("right", target)
            assert plan is not None, session.rejection_reason
            np.testing.assert_allclose(plan.trajectory.positions[0], plan.start.positions["right"], atol=1e-5)
            if previous is not None:
                assert np.linalg.norm(plan.trajectory.positions[0]) > 0.2
                np.testing.assert_allclose(plan.start.positions["right"], previous, atol=0.002)
                wrong_trajectory = JointTrajectory("right", [0, 2], [np.zeros(7), plan.trajectory.positions[-1]],
                                                   np.zeros((2, 7)), planner)
                with pytest.raises(RuntimeError, match="measured planning state"):
                    check_start(Preview(wrong_trajectory, plan.target, plan.start), worker.read())
            session.execute()
            wait_for(worker, "completed")
            measured = session.sync()
            np.testing.assert_allclose(worker.report["final_joint_positions_rad"]["right"], measured.positions["right"], atol=0.002)
            for sim in (session.twin, session.preview):
                for arm in ("right", "left"):
                    np.testing.assert_allclose(sim.data.qpos[sim.qpos_ids[arm]], measured.positions[arm], atol=1e-12)
                    np.testing.assert_allclose(sim._desired[sim.actuator_ids[arm]], measured.positions[arm], atol=1e-12)
            np.testing.assert_allclose(session.reference.from_world(session.twin.tcp_position("right")), target, atol=1.0)
            previous = measured.positions["right"].copy()
    finally:
        if session is not None:
            session.close()
        else:
            worker.close()
