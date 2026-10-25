import numpy as np
import pytest

from openarm_sim.camera import OverheadCamera
from openarm_sim.simulation import ReachSimulation


def test_camera_observes_table_from_robot_head():
    with ReachSimulation() as sim, OverheadCamera(sim.model, 320, 240) as camera:
        frame = camera.capture(sim.data)
        np.testing.assert_allclose(frame.position, [0.038, 0.0, 0.808], atol=1e-6)
        assert sim.model.cam_fovy[camera.camera_id] == 52
        viewing_direction = -frame.rotation[:, 2]
        np.testing.assert_allclose(viewing_direction, [0.469472, 0, -0.882948], atol=1e-6)
        assert sim.model.vis.quality.offsamples == 4


def test_camera_ray_recovers_table_coordinates():
    with ReachSimulation() as sim, OverheadCamera(sim.model, 640, 480) as camera:
        frame = camera.capture(sim.data)
        pixel = camera.project(np.array([0.52, -0.32, sim.table_height]), frame)
        surface = camera.unproject(*pixel, frame)
        assert surface[2] == pytest.approx(sim.table_height, abs=1e-9)
        assert np.linalg.norm(surface[:2] - [0.52, -0.32]) < 0.004


def test_camera_unprojects_object_surface_above_table():
    # 桌面上的物体必须被点到它自己的表面，而不是被投影到桌面平面。
    box = dict(name="probe_box", position=[0.50, -0.20, 0.33], half_size=[0.04, 0.04, 0.03])
    with ReachSimulation(obstacles=(box,)) as sim, OverheadCamera(sim.model, 640, 480) as camera:
        frame = camera.capture(sim.data)
        top = np.array([0.50, -0.20, 0.36])
        pixel = camera.project(top, frame)
        surface = camera.unproject(*pixel, frame)
        np.testing.assert_allclose(surface, top, atol=0.002)
        # 桌面平面假设会把同一个像素解到 z=table_height，偏差 6 cm。
        assert surface[2] - sim.table_height > 0.04


def test_depth_buffer_resolves_points_without_a_physics_snapshot():
    # 真机换深度相机时没有 MjData 可供求交，深度通道必须能独立工作。
    with ReachSimulation() as sim, OverheadCamera(sim.model, 640, 480, include_depth=True) as camera:
        frame = camera.capture(sim.data)
        assert frame.depth is not None and frame.depth.shape == (frame.rgb.shape[0], frame.rgb.shape[1])
        pixel = camera.project(np.array([0.40, 0.0, sim.table_height]), frame)
        with OverheadCamera(sim.model, 640, 480) as depth_only:
            point = depth_only.unproject(*pixel, frame)
        assert np.linalg.norm(point - [0.40, 0.0, sim.table_height]) < 0.004


@pytest.mark.parametrize("arm", ["right", "left"])
def test_pixel_target_is_reached_by_physics(arm):
    with ReachSimulation() as sim, OverheadCamera(sim.model, 640, 480) as camera:
        sign = -1 if arm == "right" else 1
        frame = camera.capture(sim.data)
        pixel = camera.project(np.array([0.43, sign * 0.24, sim.table_height]), frame)
        target = camera.unproject(*pixel, frame) + [0, 0, 0.14]
        start = sim.tcp_position(arm).copy()
        result = sim.start_reach(arm, target)
        assert result.accepted, result.reason
        for _ in range(10000):
            sim.step()
            if sim.status.state != "running":
                break
        assert sim.status.state == "succeeded", sim.status
        assert np.linalg.norm(sim.tcp_position(arm) - target) < 0.01
        assert np.linalg.norm(start - target) > 0.04
        assert sim.data.time > 0.5
        assert not sim.data.warning.number.any()


def test_unreachable_target_does_not_change_command():
    with ReachSimulation() as sim:
        previous = sim.data.ctrl.copy()
        result = sim.start_reach("right", np.array([3.0, -0.2, 0.5]))
        assert not result.accepted
        np.testing.assert_array_equal(sim.data.ctrl, previous)
        assert sim.status.state == "idle"


def test_consecutive_targets_use_current_physical_state():
    with ReachSimulation() as sim:
        for arm, target in [
            ("right", [0.43, -0.24, 0.44]),
            ("right", [0.48, -0.22, 0.44]),
            ("right", [0.40, -0.18, 0.44]),
            ("left", [0.43, 0.24, 0.44]),
        ]:
            result = sim.start_reach(arm, np.array(target))
            assert result.accepted, result.reason
            for _ in range(10000):
                sim.step()
                if sim.status.state != "running":
                    break
            assert sim.status.state == "succeeded", sim.status
            assert np.linalg.norm(sim.tcp_position(arm) - target) < 0.01


def test_outside_image_is_rejected():
    with ReachSimulation() as sim, OverheadCamera(sim.model, 320, 240) as camera:
        frame = camera.capture(sim.data)
        with pytest.raises(ValueError, match="Pixel"):
            camera.unproject(-1, 50, frame)


def test_selection_keeps_captured_camera_geometry():
    with ReachSimulation() as sim, OverheadCamera(sim.model, 320, 240) as camera:
        frame = camera.capture(sim.data)
        point = camera.unproject(200, 150, frame)
        sim.data.time += 1.0
        np.testing.assert_array_equal(point, camera.unproject(200, 150, frame))
        assert frame.timestamp == 0.0


def test_camera_adjustment_keeps_pixel_geometry_consistent():
    with ReachSimulation() as sim, OverheadCamera(sim.model, 640, 480) as camera:
        first = camera.capture(sim.data)
        pixel = camera.project(np.array([0.52, -0.32, sim.table_height]), first)
        original = camera.unproject(*pixel, first)
        sim.set_camera(np.array([-0.04, 0.04, 1.35]), np.array([0.45, 0, 0.30]), 62.0)
        second = camera.capture(sim.data)
        np.testing.assert_allclose(camera.unproject(*pixel, first), original)
        pixel = camera.project(np.array([0.52, -0.32, sim.table_height]), second)
        assert np.linalg.norm(camera.unproject(*pixel, second) - [0.52, -0.32, sim.table_height]) < 0.004
