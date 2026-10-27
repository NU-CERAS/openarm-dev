import os
import time

import numpy as np
import pytest

from openarm_sim.camera import OverheadCamera
from openarm_sim.console.gizmo import AxisDrag, hit_axis
from openarm_sim.console.session import RobotWorker, SyncSession
from openarm_sim.paths import RUNS
from openarm_sim.simulation import ReachSimulation


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_axis_drag_recovers_centimeters_from_actual_camera_projection(axis):
    with ReachSimulation(tabletop_mount=True) as sim, OverheadCamera(sim.model, 1280, 720, 0.0) as camera:
        sim.set_camera(np.array([1.55, 1.15, 1.25]), np.array([0.25, 0, 0.25]), 48)
        frame = camera.capture(sim.data)
        origin = sim.tcp_position("right")
        direction = np.eye(3)[axis]
        start = camera.project(origin + direction * 0.08, frame, subpixel=True)
        end = camera.project(origin + direction * 0.10, frame, subpixel=True)
        drag = AxisDrag.begin(camera, frame, origin, direction, axis, start)
        assert drag is not None
        target = drag.target(end)
        assert target is not None
        assert abs((target - origin)[axis] - 0.02) < 1e-10
        np.testing.assert_allclose(np.delete(target - origin, axis), [0, 0], atol=1e-12)
        projected = (np.array(camera.project(origin, frame)), np.array(camera.project(origin + direction * 0.10, frame)))
        assert hit_axis(projected[1], {axis: projected}) == axis


def pump(console, predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while not predicate():
        console.root.update()
        if console.error is not None:
            raise console.error
        if time.monotonic() > deadline:
            raise TimeoutError("GUI did not reach the expected state")
        time.sleep(0.01)


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Requires a desktop display for actual Tk events")
@pytest.mark.parametrize("release_axis", [0, 1, 2])
def test_gui_axis_drag_preview_cancel_and_physical_execution(release_axis):
    from openarm_sim.console.app import SyncConsole

    worker = RobotWorker(realtime=False)
    session = None
    console = None
    try:
        assert worker.ready.wait(10)
        session = SyncSession(worker)
        console = SyncConsole(session, RUNS / "tests/gizmo/targets.json", width=900, height=550)
        console.root.update()
        console.root.focus_force()
        pump(console, lambda: console.frame is not None and console.resize_pending is None and len(console.gizmo_handles) == 3)
        original = session.sync().positions["right"].copy()
        for axis in (0, 1, 2):
            pump(console, lambda: console.frame is not None and axis in console.gizmo_handles)
            point = console.gizmo_handles[axis][1].astype(int)
            original_values = [value.get() for value in console.coordinates]
            console.canvas.event_generate("<ButtonPress-1>", x=int(point[0]), y=int(point[1]))
            assert console.drag is not None and console.drag.index == axis
            end = console.drag.camera.project(console.drag.origin + console.drag.axis * 0.12, console.drag.frame)
            console.canvas.event_generate("<B1-Motion>", x=end[0], y=end[1])
            assert console.drag_changed
            assert session.plan is None
            np.testing.assert_allclose(session.sync().positions["right"], original, atol=0.002)
            if axis != release_axis:
                console.root.event_generate("<Escape>")
                assert console.drag is None
                assert [value.get() for value in console.coordinates] == original_values
                console.canvas.event_generate("<ButtonRelease-1>", x=end[0], y=end[1])
            else:
                console.canvas.event_generate("<ButtonRelease-1>", x=end[0], y=end[1])
                assert console.drag is None
                assert session.plan is not None, session.rejection_reason
                assert console.execute_button.instate(["!disabled"])
        target = session.plan.target.copy()
        np.testing.assert_allclose(session.sync().positions["right"], original, atol=0.002)
        console.execute()
        pump(console, lambda: worker.state == "completed")
        session.sync()
        assert np.linalg.norm(session.twin.tcp_position("right") - target) < 0.015
        pump(console, lambda: console.last_report is worker.report and console.preview_started is None)
        assert console.display_mode.get() == "Live"
        previous = session.sync().positions["right"].copy()
        following_target = session.reference.from_world(session.twin.tcp_position("right")) + [1, 0, 2]
        for variable, value in zip(console.coordinates, following_target):
            variable.set(str(value))
        console.preview_button.invoke()
        assert session.plan is not None, session.rejection_reason
        following_plan = session.plan
        np.testing.assert_allclose(following_plan.trajectory.positions[0], previous, atol=0.002)
        assert np.linalg.norm(following_plan.trajectory.positions[0]) > 0.02
        console.execute_button.invoke()
        pump(console, lambda: worker.completed_plan is following_plan and console.last_report is worker.report)
        assert console.preview_started is None
        assert console.display_mode.get() == "Live"
        measured = session.sync()
        np.testing.assert_allclose(session.preview.data.qpos[session.preview.qpos_ids["right"]], measured.positions["right"], atol=1e-12)
        assert np.linalg.norm(session.twin.tcp_position("right") - following_plan.target) < 0.015
    finally:
        if console is not None and not console.closed:
            console.close()
        elif session is not None:
            session.close()
        else:
            worker.close()
