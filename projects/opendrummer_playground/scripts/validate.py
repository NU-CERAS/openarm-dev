"""Headless import, kinematics, and gravity-hold acceptance checks."""
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from openarm_drummer.model import ASSET, JOINTS, OUTPUT, SOURCE, HoldController, build, forward_urdf, origin, vector


def check_official_inertials(model, controller):
    """Compare compiled COM/tensors with upstream in world space across poses."""
    official = ET.parse(ASSET / 'provenance/official-v1.urdf').getroot()
    mapping = json.loads((ASSET / 'inertial-provenance.json').read_text())
    robot = ET.parse(SOURCE).getroot()
    zero = forward_urdf(robot, {})
    official_zero = forward_urdf(official, {})

    def tensor(element):
        v = {k: float(x) for k, x in element.attrib.items()}
        return np.array([[v['ixx'], v['ixy'], v['ixz']],
                         [v['ixy'], v['iyy'], v['iyz']],
                         [v['ixz'], v['iyz'], v['izz']]])

    # This also checks the three fixed base assemblies, which MuJoCo fuses away.
    for row in mapping['official_assemblies']:
        actual = robot.find(f"link[@name='{row['target_link']}']/inertial")
        expected = official.find(f"link[@name='{row['official_link']}']/inertial")
        np.testing.assert_allclose(float(actual.find('mass').get('value')), float(expected.find('mass').get('value')), rtol=1e-12)
        A = zero[row['target_link']] @ origin(actual.find('origin'))
        B = official_zero[row['official_link']] @ origin(expected.find('origin'))
        np.testing.assert_allclose(A[:3, 3], B[:3, 3], atol=1e-12)
        np.testing.assert_allclose(A[:3, :3] @ tensor(actual.find('inertia')) @ A[:3, :3].T,
                                   B[:3, :3] @ tensor(expected.find('inertia')) @ B[:3, :3].T, atol=1e-12)

    rng = np.random.default_rng(7)
    poses = [np.zeros(14), *rng.uniform(-0.3, 0.3, (10, 14))]
    signs = json.loads((ASSET / 'joint-limits.json').read_text())['joint_limits']
    data = mujoco.MjData(model)
    max_com_error, max_tensor_error = 0.0, 0.0
    for pose in poses:
        pose[[3, 10]] = np.abs(pose[[3, 10]])
        data.qpos[controller.q] = pose
        mujoco.mj_forward(model, data)
        angles = {f'openarm_{side}_joint{i}': pose[k*7+i-1]*signs[f'{side[0]}j{i}']['axis_sign_relative_to_official']
                  for k, side in enumerate(('left', 'right')) for i in range(1, 8)}
        frames = forward_urdf(official, angles)
        for side in ('left', 'right'):
            for i in range(1, 7):
                name = f'openarm_{side}_link{i}'
                inertial = official.find(f"link[@name='{name}']/inertial")
                T = frames[name] @ origin(inertial.find('origin'))
                bid = model.jnt_bodyid[model.joint(f'{side[0]}j{i}').id]
                R = data.ximat[bid].reshape(3, 3)
                actual_tensor = R @ np.diag(model.body_inertia[bid]) @ R.T
                expected_tensor = T[:3, :3] @ tensor(inertial.find('inertia')) @ T[:3, :3].T
                max_com_error = max(max_com_error, float(np.linalg.norm(data.xipos[bid]-T[:3, 3])))
                max_tensor_error = max(max_tensor_error, float(np.linalg.norm(actual_tensor-expected_tensor)))
                np.testing.assert_allclose(model.body_mass[bid], float(inertial.find('mass').get('value')), rtol=1e-5)
    assert max_com_error < 5e-5, max_com_error
    assert max_tensor_error < 1e-6, max_tensor_error
    return {'official_assemblies_checked': len(mapping['official_assemblies']),
            'official_inertia_poses_checked': len(poses),
            'max_official_com_error_m': max_com_error,
            'max_official_tensor_error_kg_m2': max_tensor_error}


def validate():
    (OUTPUT / 'validation-report.json').unlink(missing_ok=True)
    model, report = build()
    robot = ET.parse(SOURCE).getroot()
    data = mujoco.MjData(model)
    controller = HoldController(model, report)
    official_checks = check_official_inertials(model, controller)
    assert model.nq == model.nv == model.nu == model.njnt == 14
    assert set(model.joint(i).name for i in range(14)) == set(JOINTS)
    assert np.all(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    assert not np.any(model.geom_contype) and not np.any(model.geom_conaffinity)
    assert np.all(model.body_mass[1:] > 0) and np.all(model.body_inertia[1:] > 0)
    assert np.all(2*model.body_inertia[1:].max(axis=1) <= model.body_inertia[1:].sum(axis=1)+1e-12)
    # Independently sum each fixed assembly from source masses to check fusion.
    parents = {j.find('child').get('link'): j for j in robot.findall('joint')}
    masses = dict.fromkeys(JOINTS, 0.0)
    for link in robot.findall('link'):
        mass = link.find('inertial/mass')
        if mass is None:
            continue
        value = float(mass.get('value'))
        cursor = link.get('name')
        while cursor in parents:
            parent_joint = parents[cursor]
            if parent_joint.get('type') == 'revolute':
                masses[parent_joint.get('name')] += value
                break
            cursor = parent_joint.find('parent').get('link')
    for name, mass in masses.items():
        np.testing.assert_allclose(report['moving_body_masses_kg'][name], mass, rtol=1e-5)
    poses = [np.zeros(14)]
    for index in range(14):
        jid = model.joint(JOINTS[index]).id
        np.testing.assert_allclose(model.jnt_range[jid], controller.bounds[index], atol=1e-10)
        for endpoint in controller.bounds[index]:
            pose = np.zeros(14)
            pose[index] = endpoint
            poses.append(pose)
    rng = np.random.default_rng(42)
    poses.extend(rng.uniform(controller.bounds[:, 0], controller.bounds[:, 1], (10, 14)))
    max_position_error, max_axis_error = 0.0, 0.0
    for pose in poses:
        data.qpos[controller.q] = pose
        mujoco.mj_forward(model, data)
        transforms = forward_urdf(robot, dict(zip(JOINTS, pose)))
        for name in JOINTS:
            joint = robot.find(f"joint[@name='{name}']")
            transform = transforms[joint.find('child').get('link')]
            jid = model.joint(name).id
            position_error = np.linalg.norm(data.xanchor[jid]-transform[:3, 3])
            axis = vector(joint.find('axis').get('xyz'))
            expected_axis = transform[:3, :3] @ (axis/np.linalg.norm(axis))
            axis_error = np.linalg.norm(data.xaxis[jid]-expected_axis)
            max_position_error = max(max_position_error, position_error)
            max_axis_error = max(max_axis_error, axis_error)
        for name, tip in report['tips'].items():
            expected = transforms[tip['link']] @ np.array([*tip['local_position'], 1])
            error = np.linalg.norm(data.site_xpos[model.site(name).id]-expected[:3])
            max_position_error = max(max_position_error, error)
    # mj_saveLastXML rounds transforms, so allow micrometer-scale conversion error.
    assert max_position_error < 5e-5, max_position_error
    assert max_axis_error < 5e-5, max_axis_error
    controller.reset(data)
    target = controller.target.copy()
    # Perturb the pose to test recovery, rather than checking equilibrium alone.
    data.qpos[controller.q] += 0.03
    mujoco.mj_forward(model, data)
    peak_speed, peak_torque = 0.0, 0.0
    for _ in range(round(5/model.opt.timestep)):
        controller.step(data)
        peak_speed = max(peak_speed, float(np.max(np.abs(data.qvel))))
        peak_torque = max(peak_torque, float(np.max(np.abs(data.actuator_force))))
        assert np.all(np.abs(data.actuator_force[controller.a]) <= controller.effort+1e-8)
        assert np.all(np.abs(data.qvel[controller.v]) <= controller.velocity)
        assert np.all(data.qpos[controller.q] >= controller.bounds[:, 0]-0.001)
        assert np.all(data.qpos[controller.q] <= controller.bounds[:, 1]+0.001)
    error = float(np.max(np.abs(data.qpos[controller.q]-target)))
    assert error < 0.005, error
    assert np.max(np.abs(data.qvel)) < 0.01
    assert sum(int(w.number) for w in data.warning) == 0
    # Drive each actuator through a small smooth excursion independently.
    max_tracking_error = 0.0
    for index, name in enumerate(JOINTS):
        controller.target = target.copy()
        controller.reset(data)
        for step in range(round(2/model.opt.timestep)):
            goal = target.copy()
            goal[index] += 0.10*np.sin(np.pi*(step+1)*model.opt.timestep/2)**2
            controller.step(data, goal)
            max_tracking_error = max(max_tracking_error, float(np.max(np.abs(data.qpos[controller.q]-goal))))
            assert np.all(np.abs(data.actuator_force[controller.a]) <= controller.effort+1e-8)
            assert np.all(np.abs(data.qvel[controller.v]) <= controller.velocity)
        assert data.qpos[controller.q[index]] < target[index]+0.04, name
        assert sum(int(w.number) for w in data.warning) == 0
    assert max_tracking_error < 0.04, max_tracking_error
    result = {'status': 'PASS', **official_checks, 'kinematic_poses_checked': len(poses),
              'maximum_fk_position_error_m': max_position_error,
              'maximum_fk_axis_error': max_axis_error,
              'gravity_hold_duration_s': 5, 'initial_perturbation_rad': 0.03,
              'final_hold_error_rad': error, 'peak_hold_speed_rad_s': peak_speed,
              'peak_hold_torque_Nm': peak_torque,
              'joint_excursions_checked': 14, 'maximum_tracking_error_rad': max_tracking_error}
    (OUTPUT / 'validation-report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    print(f'Reports and generated model: {OUTPUT}')


if __name__ == '__main__':
    validate()
