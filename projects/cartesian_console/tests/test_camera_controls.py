import os

import mujoco
import numpy as np
import pytest

from openarm_sim.console.camera_controls import CameraControls
from openarm_sim.console.session import RobotWorker, SyncSession
from openarm_sim.simulation import ReachSimulation


@pytest.mark.parametrize("action", [mujoco.mjtMouse.mjMOUSE_ROTATE_V, mujoco.mjtMouse.mjMOUSE_ROTATE_H,
                                   mujoco.mjtMouse.mjMOUSE_MOVE_V, mujoco.mjtMouse.mjMOUSE_MOVE_H,
                                   mujoco.mjtMouse.mjMOUSE_ZOOM])
def test_native_camera_controls_preserve_robot_and_field_of_view(action):
    with ReachSimulation(tabletop_mount=True) as sim:
        camera_id = sim.model.camera("overhead").id
        position = sim.model.cam_pos[camera_id].copy()
        joints = sim.data.qpos.copy()
        fovy = float(sim.model.cam_fovy[camera_id])
        controls = CameraControls(sim)
        controls.move(action, .10, .05)
        assert np.linalg.norm(sim.model.cam_pos[camera_id] - position) > .001
        assert sim.model.cam_fovy[camera_id] == fovy
        np.testing.assert_array_equal(sim.data.qpos, joints)
        actual = (sim.camera_look_at - sim.model.cam_pos[camera_id]) / controls.camera.distance
        np.testing.assert_allclose(actual, controls.scene.camera[0].forward, atol=1e-6)


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Requires a display for native Tk events")
def test_gui_camera_gestures_preserve_pending_tcp_target(tmp_path):
    from openarm_sim.console.app import SyncConsole

    worker = RobotWorker(realtime=False)
    session = console = None
    try:
        assert worker.ready.wait(10)
        session = SyncSession(worker, planner="mink")
        console = SyncConsole(session, tmp_path / "targets.json", width=900, height=550)
        console.root.update()
        console.root.focus_force()
        target = session.preview.tcp_position("right") + [.01, 0, .01]
        plan = session.prepare("right", target)
        assert plan is not None, session.rejection_reason
        joints = session.preview.data.qpos.copy()
        camera_id = session.preview.model.camera("overhead").id
        for button, state in ((1, 0), (3, 0), (2, 0), (1, 1), (3, 1)):
            position = session.preview.model.cam_pos[camera_id].copy()
            console.canvas.event_generate(f"<ButtonPress-{button}>", x=5, y=5, state=state)
            assert console.camera_drag is not None
            assert console.drag is None
            console.canvas.event_generate("<Motion>", x=55, y=40, state=state | (1 << (7 + button)))
            console.canvas.event_generate(f"<ButtonRelease-{button}>", x=55, y=40, state=state)
            assert console.camera_drag is None
            assert np.linalg.norm(session.preview.model.cam_pos[camera_id] - position) > .001
            assert session.plan is plan
            np.testing.assert_array_equal(session.preview.data.qpos, joints)
        distance = np.linalg.norm(session.preview.model.cam_pos[camera_id] - session.preview.camera_look_at)
        console.canvas.event_generate("<Button-4>", x=5, y=5)
        closer = np.linalg.norm(session.preview.model.cam_pos[camera_id] - session.preview.camera_look_at)
        assert closer < distance
        console.canvas.event_generate("<Button-5>", x=5, y=5)
        assert np.linalg.norm(session.preview.model.cam_pos[camera_id] - session.preview.camera_look_at) > closer
        position = session.preview.model.cam_pos[camera_id].copy()
        console.resize_view(console.width + 20, console.height + 20)
        np.testing.assert_array_equal(session.preview.model.cam_pos[camera_id], position)
        console.view(False)
        np.testing.assert_allclose(session.preview.model.cam_pos[camera_id], console.camera_defaults[0][0])
        assert session.plan is plan
        np.testing.assert_array_equal(session.preview.data.qpos, joints)
        assert worker.state == "connected"
    finally:
        if console is not None:
            console.close()
        elif session is not None:
            session.close()
        else:
            worker.close()
