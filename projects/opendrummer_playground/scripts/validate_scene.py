"""Check composition, surface fits, and static initial target reachability."""
import json

import mujoco
import numpy as np

from model import OUTPUT, HoldController
from scene import build_scene, solve_approach


def segment_hits_triangles(start, end, triangles):
    """Moller-Trumbore segment/triangle intersection; visual sanity check only."""
    direction = end-start
    a = triangles[:, 0]
    e1, e2 = triangles[:, 1]-a, triangles[:, 2]-a
    p = np.cross(np.broadcast_to(direction, e2.shape), e2)
    det = np.einsum('ij,ij->i', e1, p)
    valid = np.abs(det) > 1e-12
    inv = np.zeros_like(det)
    inv[valid] = 1/det[valid]
    tvec = start-a
    u = np.einsum('ij,ij->i', tvec, p)*inv
    q = np.cross(tvec, e1)
    v = q @ direction * inv
    t = np.einsum('ij,ij->i', e2, q)*inv
    return int(np.sum(valid & (u >= 0) & (v >= 0) & (u+v <= 1) & (t > 1e-6) & (t < 1-1e-6)))


def kit_triangles(model, data):
    triangles = []
    for gid in range(model.ngeom):
        if not model.geom(gid).name.startswith('kit_material_'):
            continue
        mid = model.geom_dataid[gid]
        vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
        vertices = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        faces = model.mesh_face[model.mesh_faceadr[mid]:model.mesh_faceadr[mid]+model.mesh_facenum[mid]]
        triangles.append(vertices[faces])
    return np.concatenate(triangles)


def validate():
    (OUTPUT/'scene-validation.json').unlink(missing_ok=True)
    model, robot_report, report = build_scene(collisions=False)
    assert model.nq == model.nv == model.nu == 14
    assert not np.any(model.geom_contype) and not np.any(model.geom_conaffinity)
    assert len(report['surfaces']) == 8
    assert abs(report['kit_bounds_m'][0][2]) < 1e-8
    for surface in report['surfaces']:
        np.testing.assert_allclose(np.linalg.norm(surface['normal']), 1, atol=1e-10)
        assert .1 < surface['radius'] < .35
        if surface['kind'] != 'cymbal':
            assert surface['plane_deviation_m'] < 1e-5
    controller = HoldController(model, robot_report)
    data = mujoco.MjData(model)
    controller.reset(data)
    original = mujoco.MjModel.from_xml_path(str(OUTPUT/'opendrummer.xml'))
    original_data = mujoco.MjData(original)
    HoldController(original, robot_report).reset(original_data)
    translation = np.array(report['robot_translation_m'])
    for index in range(original.njnt):
        name = original.joint(index).name
        np.testing.assert_allclose(data.xanchor[model.joint(name).id]-original_data.xanchor[index], translation, atol=1e-9)
    for side in ('left', 'right'):
        name = f'{side}_stick_tip'
        np.testing.assert_allclose(data.site(name).xpos-original_data.site(name).xpos, translation, atol=1e-9)
    support = model.geom('robot_support_visual').id
    np.testing.assert_allclose(data.geom_xpos[support, 2]-model.geom_size[support, 2], 0, atol=1e-10)
    np.testing.assert_allclose(data.geom_xpos[support, 2]+model.geom_size[support, 2], translation[2], atol=1e-10)
    triangles = kit_triangles(model, data)

    def check_sticks():
        for side in ('left', 'right'):
            sid = model.site(f'{side}_stick_tip').id
            tip = data.site_xpos[sid]
            axis = data.site_xmat[sid].reshape(3, 3)[:, 2]
            assert segment_hits_triangles(tip-.3048*axis, tip, triangles) == 0, f'{side} stick centerline intersects kit'

    check_sticks()
    approaches = []
    for side, target in report['config']['initial_targets'].items():
        approach = solve_approach(model, robot_report, report, side, target)
        assert approach['reachable'], approach
        controller.target = np.array(approach['joint_positions'])
        assert np.all(controller.target >= controller.bounds[:, 0])
        assert np.all(controller.target <= controller.bounds[:, 1])
        controller.reset(data)
        check_sticks()
        for _ in range(round(2/model.opt.timestep)):
            controller.step(data)
            assert np.all(np.abs(data.actuator_force[controller.a]) <= controller.effort+1e-8)
        assert np.max(np.abs(data.qpos[controller.q]-controller.target)) < .005
        assert sum(int(w.number) for w in data.warning) == 0
        assert data.ncon == 0
        approaches.append(approach)
    result = {'status': 'PASS', 'surface_count': len(report['surfaces']), 'initial_approaches': approaches,
              'robot_elevation_m': float(translation[2]), 'support_floor_and_top_alignment_checked': True,
              'ready_and_approach_stick_centerlines_clear': True,
              'clearance_scope': 'Stick centerline/kit triangles only, plus manual visual review. No swept-path, full-body, or finite-radius collision certification.',
              'contacts_enabled': False}
    (OUTPUT/'scene-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    validate()
