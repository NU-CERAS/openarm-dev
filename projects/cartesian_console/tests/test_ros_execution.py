import importlib.util
import os
import signal
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import pytest

from openarm_sim.console.session import RobotWorker, SyncSession
from openarm_sim.paths import RUNS


pytestmark = pytest.mark.skipif(importlib.util.find_spec("rclpy") is None, reason="Requires an activated ROS 2 environment")


@pytest.fixture
def ros_backend():
    namespace = f"/test_{uuid.uuid4().hex[:12]}"
    directory = RUNS / "tests/ros" / namespace[1:]
    directory.mkdir(parents=True)
    with (directory / "backend.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-u", "-m", "openarm_sim.console.ros_backend",
                                    "--namespace", namespace], stdout=log, stderr=subprocess.STDOUT)
        try:
            yield namespace, process
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            assert process.wait(timeout=10) == 0, (directory / "backend.log").read_text()


@pytest.fixture
def ros_session(ros_backend):
    namespace, _ = ros_backend
    worker = RobotWorker("ros", ros_namespace=namespace)
    session = None
    try:
        assert worker.ready.wait(12)
        worker.read()
        session = SyncSession(worker)
        yield session
    finally:
        session.close() if session is not None else worker.close()


def completed(session, plan):
    deadline = time.monotonic() + plan.trajectory.duration + 8
    while session.worker.completed_plan is not plan:
        session.worker.read()
        assert time.monotonic() < deadline
        time.sleep(0.01)
    return session.sync()


def test_ros_preview_execution_and_consecutive_measured_starts(ros_session):
    session = ros_session
    for target in ([40, -25, -30], [42, -25, -28]):
        before = session.sync()
        plan = session.prepare_relative("right", target)
        assert plan is not None, session.rejection_reason
        session.show_preview(plan.trajectory.duration)
        time.sleep(0.1)
        np.testing.assert_allclose(session.worker.read().positions["right"], before.positions["right"], atol=0.002)
        np.testing.assert_allclose(plan.trajectory.positions[0], before.positions["right"], atol=0.002)
        session.execute()
        actual = completed(session, plan)
        assert session.worker.report["transport"] == "ros2"
        assert session.worker.report["final_joint_error_rad"] < 0.03
        np.testing.assert_allclose(session.preview.data.qpos[session.preview.qpos_ids["right"]], actual.positions["right"], atol=1e-12)
        assert np.linalg.norm(session.preview.tcp_position("right") - plan.target) < 0.015
    assert np.linalg.norm(plan.trajectory.positions[0]) > 0.1


def test_ros_cancel_holds_measured_pose(ros_session):
    session = ros_session
    before = session.sync().positions["right"].copy()
    plan = session.prepare_relative("right", [40, -25, -30])
    assert plan is not None
    session.execute()
    deadline = time.monotonic() + 5
    while np.linalg.norm(session.worker.read().positions["right"] - before) < 0.03:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    session.worker.cancel()
    deadline = time.monotonic() + 3
    while session.worker.state != "cancelled":
        session.worker.read()
        assert time.monotonic() < deadline
        time.sleep(0.01)
    held = session.sync().positions["right"].copy()
    time.sleep(0.2)
    np.testing.assert_allclose(session.sync().positions["right"], held, atol=0.003)
    assert np.linalg.norm(held) > 0.02


def test_ros_rejects_a_changed_start_without_moving(ros_backend):
    from openarm_tools.ros_execution import RosExecutionClient, trajectory_message

    namespace, _ = ros_backend
    client = RosExecutionClient(namespace)
    try:
        client.wait_ready()
        before = client.read().arm("right")[0]
        message = trajectory_message("right", [0, 2], [before + 0.2, before + 0.21], np.zeros((2, 7)))
        with pytest.raises(RuntimeError, match="goal was rejected"):
            client.execute("right", message, threading.Event(), lambda snapshot: None)
        np.testing.assert_allclose(client.read().arm("right")[0], before, atol=0.002)
    finally:
        client.close()


def test_ros_lost_feedback_fails_closed(ros_backend):
    from openarm_tools.ros_execution import RosExecutionClient

    namespace, process = ros_backend
    client = RosExecutionClient(namespace)
    try:
        client.wait_ready()
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=10) == 0
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            client.executor.spin_once(timeout_sec=0.01)
        with pytest.raises(RuntimeError, match="missing or stale"):
            client.read()
    finally:
        client.close()


def test_ros_other_arm_motion_invalidates_an_existing_preview(ros_session):
    from openarm_tools.ros_execution import RosExecutionClient, trajectory_message

    session = ros_session
    plan = session.prepare_relative("right", [40, -25, -30])
    assert plan is not None
    client = RosExecutionClient(session.worker.ros_namespace)
    try:
        client.wait_ready()
        before = client.read().arm("left")[0]
        target = before.copy()
        target[0] -= 0.1
        message = trajectory_message("left", [0, 2], [before, target], np.zeros((2, 7)))
        client.execute("left", message, threading.Event(), lambda snapshot: None)
        deadline = time.monotonic() + 1
        while abs(session.worker.read().positions["left"][0] - target[0]) > 0.01:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        with pytest.raises(RuntimeError, match="moved after preview"):
            session.execute()
        np.testing.assert_allclose(session.worker.read().positions["right"], plan.start.positions["right"], atol=0.002)
    finally:
        client.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Requires actual Tk events")
def test_ros_console_nonzero_start_drag_execute_and_automatic_sync(ros_session):
    from openarm_sim.console.app import SyncConsole
    from test_gizmo import pump

    session = ros_session
    first = session.prepare_relative("right", [40, -25, -30])
    assert first is not None
    session.execute()
    before = completed(session, first).positions["right"].copy()
    console = SyncConsole(session, RUNS / "tests/ros/gui_targets.json", width=900, height=550)
    try:
        pump(console, lambda: console.canvas.winfo_ismapped() and console.frame is not None
             and console.resize_pending is None and 2 in console.gizmo_handles)
        np.testing.assert_allclose(session.preview.data.qpos[session.preview.qpos_ids["right"]], before, atol=0.002)
        assert np.linalg.norm(before) > 0.1
        tip = console.gizmo_handles[2][1].astype(int)
        console.canvas.event_generate("<ButtonPress-1>", x=int(tip[0]), y=int(tip[1]))
        assert console.drag is not None
        pixel = console.drag.camera.project(console.drag.origin + console.drag.axis * 0.13, console.drag.frame)
        console.canvas.event_generate("<B1-Motion>", x=pixel[0], y=pixel[1])
        console.canvas.event_generate("<ButtonRelease-1>", x=pixel[0], y=pixel[1])
        following = session.plan
        assert following is not None
        np.testing.assert_allclose(following.trajectory.positions[0], before, atol=0.002)
        np.testing.assert_allclose(session.worker.read().positions["right"], before, atol=0.002)
        console.execute_button.invoke()
        pump(console, lambda: session.worker.completed_plan is following and console.last_report is session.worker.report)
        actual = session.sync()
        np.testing.assert_allclose(session.preview.data.qpos[session.preview.qpos_ids["right"]], actual.positions["right"], atol=1e-12)
        np.testing.assert_allclose([float(value.get()) for value in console.coordinates],
                                   session.reference.from_world(session.twin.tcp_position("right")), atol=0.01)
        assert console.execute_button.instate(["disabled"])
    finally:
        console.close()
