import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from openarm_sim.camera import OverheadCamera
from openarm_sim.simulation import ReachSimulation


MANIFEST = Path(".work/drumset/converted/manifest.json")
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="Requires the downloaded and converted SINNIK model")


def test_original_electronic_kit_loads_and_renders():
    data = json.loads(MANIFEST.read_text())
    assert data["author"] == "SINNIK"
    assert data["license"] == "CC-BY-4.0"
    assert len(data["parts"]) == 45
    assert sum(part["faces"] for part in data["parts"]) == 416246
    with ReachSimulation(drum_asset=MANIFEST, worktable=False) as sim:
        assert len(sim.drum_asset["pads"]) == 6
        for _ in range(1000):
            sim.step()
        assert not sim.data.warning.number.any()
        with OverheadCamera(sim.model, 640, 360) as camera:
            frame = camera.capture(sim.data)
            assert frame.rgb.shape == (360, 640, 3)
            assert frame.rgb.std() > 2


def test_all_imported_pad_colliders_produce_physical_contacts():
    from openarm_sim.drum_asset import load_asset

    spec = mujoco.MjSpec()
    spec.option.timestep = 0.001
    asset = load_asset(spec, MANIFEST)
    body = spec.worldbody.add_body(name="contact_probe", pos=[0, 0, 2])
    body.add_freejoint(name="probe_free")
    body.add_geom(name="probe_stick", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=[0, 0, -0.15, 0, 0, 0.15], size=[0.006, 0, 0], mass=0.04)
    model = spec.compile()
    data = mujoco.MjData(model)
    probe_id = model.geom("probe_stick").id
    for pad in asset["pads"]:
        mujoco.mj_resetData(model, data)
        normal = np.array(pad["normal"])
        data.qpos[:3] = np.array(pad["center"]) + normal * 0.17
        data.qpos[3:7] = pad["quaternion_wxyz"]
        data.qvel[:3] = -normal * 0.5
        mujoco.mj_forward(model, data)
        peak_force = 0.0
        pad_id = model.geom(pad["name"]).id
        for _ in range(100):
            mujoco.mj_step(model, data)
            for index, contact in enumerate(data.contact):
                if {contact.geom1, contact.geom2} == {probe_id, pad_id}:
                    force = np.empty(6)
                    mujoco.mj_contactForce(model, data, index, force)
                    peak_force = max(peak_force, float(force[0]))
        assert peak_force > 0, pad["name"]
        assert not data.warning.number.any()


def test_click_selects_the_imported_pad_surface():
    from openarm_sim.drum_asset import pick_pad

    with ReachSimulation(drum_asset=MANIFEST, worktable=False) as sim:
        sim.set_camera(np.array([1.75, 1.5, 1.45]), np.array([0.30, 0, 0.55]), 52)
        with OverheadCamera(sim.model, 1280, 720) as camera:
            frame = camera.capture(sim.data)
            expected = next(pad for pad in sim.drum_asset["pads"] if pad["source_node"] == "Object_28")
            pixel = camera.project(np.array(expected["center"]), frame)
            pad, point = pick_pad(*pixel, frame, camera.width, camera.height, sim.drum_asset["pads"])
            assert pad["name"] == expected["name"]
            assert np.linalg.norm(point - expected["center"]) < 0.005


def test_robot_head_camera_covers_all_pad_disks():
    with ReachSimulation(drum_asset=MANIFEST, worktable=False) as sim:
        with OverheadCamera(sim.model, 1280, 720) as camera:
            frame = camera.capture(sim.data)
            assert frame.position[2] == 1.55
            assert sim.model.cam_fovy[camera.camera_id] == 60
            for pad in sim.drum_asset["pads"]:
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, np.array(pad["quaternion_wxyz"]))
                rotation = rotation.reshape(3, 3)
                for angle in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                    point = np.array(pad["center"]) + pad["radius"] * (
                        np.cos(angle) * rotation[:, 0] + np.sin(angle) * rotation[:, 1]
                    )
                    u, v = camera.project(point, frame)
                    assert 16 <= u < camera.width - 16, (pad["name"], u, v)
                    assert 16 <= v < camera.height - 16, (pad["name"], u, v)
