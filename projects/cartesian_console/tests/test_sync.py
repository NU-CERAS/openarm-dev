import json
import multiprocessing
import os
import time

import numpy as np
import pytest

from openarm_sim.paths import PROJECT_ROOT, RUNS
from openarm_sim.console.session import RobotWorker, SyncSession, check_start, hardware_configuration
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
        assert session.twin.table_height == 0.0
        table = session.twin.model.geom("worktable").id
        assert session.twin.model.geom_pos[table, 2] + session.twin.model.geom_size[table, 2] == 0.0
        assert session.twin.model.geom_pos[session.twin.model.geom("floor").id, 2] == -0.75
        initial = session.sync().positions["right"].copy()
        assert session.prepare("left", [0.43, -0.24, 0.44]) is None
        assert session.plan is None
        assert "chosen arm" in session.rejection_reason
        plan = session.prepare("right", [0.43, -0.24, 0.44])
        assert plan is not None
        session.show_preview(plan.trajectory.duration)
        np.testing.assert_allclose(session.sync().positions["right"], initial, atol=0.002)
        assert np.linalg.norm(session.preview.tcp_position("right") - plan.target) < 0.01
        session.execute()
        wait_for(worker, "completed")
        feedback = session.sync()
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
