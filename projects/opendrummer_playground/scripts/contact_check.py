"""One slow, torque-driven touch and retraction; no rhythmic striking."""
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from openarm_drummer.collisions import CONTACT_CONFIG
from openarm_drummer.model import HoldController, OUTPUT
from openarm_drummer.scene import build_scene, solve_approach


class ContactCheck:
    def __init__(self, model, robot_report, scene_report):
        self.model = model
        self.config = json.loads(CONTACT_CONFIG.read_text())
        for key in ('test_approach_m', 'test_command_penetration_m', 'test_descent_seconds',
                    'test_dwell_seconds', 'test_retract_seconds', 'force_stop_N', 'penetration_stop_m'):
            if not np.isfinite(self.config[key]) or self.config[key] <= 0:
                raise ValueError(f'{key} must be positive and finite')
        self.controller = HoldController(model, robot_report)
        self.side, self.target = self.config['test_arm'], self.config['test_target']
        self.surface = next(s for s in scene_report['surfaces'] if s['name'] == self.target)
        self.sid = model.site(f'{self.side}_stick_tip').id
        self.tip = model.geom(f'{self.side}_stick_tip_collision').id
        self.head = model.geom(f'{self.target}_head_collision').id
        self.heights = np.linspace(self.config['test_approach_m'], -self.config['test_command_penetration_m'], 41)
        self.poses = []
        seed = None
        for height in self.heights:
            result = solve_approach(model, robot_report, scene_report, self.side, self.target, height=float(height), seed=seed)
            if result['position_error_m'] > .0005 or result['direction_error_degrees'] > 1:
                raise RuntimeError(f'Contact path IK failed at height {height}: {result}')
            seed = np.array(result['joint_positions'])
            self.poses.append(seed)
        self.poses = np.array(self.poses)
        if np.max(np.abs(np.diff(self.poses, axis=0))) > .1:
            raise RuntimeError('Discontinuous IK branch in contact path')
        self.controller.target = self.poses[0].copy()
        self.duration = self.config['test_descent_seconds']+self.config['test_dwell_seconds']+self.config['test_retract_seconds']
        self.finished = False
        self.failure = None
        self.rows = []
        self.metrics = {'peak_tip_force_N': 0., 'max_tip_penetration_m': 0.,
                        'normal_impulse_Ns': 0., 'contact_steps': 0, 'unexpected_contacts': [],
                        'peak_joint_speed_rad_s': 0., 'minimum_tip_height_m': float('inf')}

    def reset(self, data):
        self.controller.reset(data)
        # No teleporting is permitted after this initial placement.
        if any(c.dist < -1e-5 for c in data.contact):
            raise RuntimeError('Initial contact-test pose intersects collision geometry')

    def step(self, data):
        if self.finished:
            return
        cfg = self.config
        t = data.time
        if t <= cfg['test_descent_seconds']:
            fraction = t/cfg['test_descent_seconds']
        elif t <= cfg['test_descent_seconds']+cfg['test_dwell_seconds']:
            fraction = 1.
        else:
            fraction = 1-(t-cfg['test_descent_seconds']-cfg['test_dwell_seconds'])/cfg['test_retract_seconds']
        fraction = np.clip(fraction, 0, 1)
        fraction = fraction*fraction*(3-2*fraction)
        position = fraction*(len(self.poses)-1)
        index = min(int(position), len(self.poses)-2)
        blend = position-index
        goal = (1-blend)*self.poses[index]+blend*self.poses[index+1]
        self.controller.step(data, goal)
        force, penetration = 0., 0.
        for i, contact in enumerate(data.contact):
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, wrench)
            normal_force = max(0., float(wrench[0]))
            if {contact.geom1, contact.geom2} == {self.tip, self.head}:
                force += normal_force
                penetration = max(penetration, -float(contact.dist))
            elif normal_force > .05:
                pair = [self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name]
                if pair not in self.metrics['unexpected_contacts']:
                    self.metrics['unexpected_contacts'].append(pair)
                self.failure = f'Unexpected contact: {pair}'
        height = float((data.site_xpos[self.sid]-np.array(self.surface['center'])) @ np.array(self.surface['normal']))
        self.metrics['minimum_tip_height_m'] = min(self.metrics['minimum_tip_height_m'], height)
        self.metrics['peak_tip_force_N'] = max(self.metrics['peak_tip_force_N'], force)
        self.metrics['max_tip_penetration_m'] = max(self.metrics['max_tip_penetration_m'], penetration)
        self.metrics['normal_impulse_Ns'] += force*self.model.opt.timestep
        self.metrics['contact_steps'] += int(force > .01)
        self.metrics['peak_joint_speed_rad_s'] = max(self.metrics['peak_joint_speed_rad_s'], float(np.max(np.abs(data.qvel))))
        if force > cfg['force_stop_N'] or penetration > cfg['penetration_stop_m']:
            self.failure = 'Force or penetration stop threshold exceeded'
        if np.any(np.abs(data.qvel[self.controller.v]) > self.controller.velocity):
            self.failure = 'Joint speed exceeded URDF limit'
        if np.any(np.abs(data.actuator_force[self.controller.a]) > self.controller.effort+1e-8):
            self.failure = 'Actuator force exceeded URDF limit'
        if sum(int(w.number) for w in data.warning):
            self.failure = 'MuJoCo solver warning'
        self.rows.append({'time_s': float(data.time), 'tip_height_m': height,
                          'tip_force_N': force, 'tip_penetration_m': penetration})
        self.finished = bool(self.failure) or data.time >= self.duration
        if self.finished:
            self.metrics['final_tip_height_m'] = height

    def report(self):
        passed = (self.finished and not self.failure and self.metrics['peak_tip_force_N'] > .05
                  and self.metrics['final_tip_height_m'] > self.config['test_approach_m']-.005)
        return {'status': 'PASS' if passed else 'FAIL', 'failure': self.failure,
                'arm': self.side, 'target': self.target, 'timestep_s': self.model.opt.timestep,
                'metrics': self.metrics, 'scope': 'One slow rounded-tip/drumhead contact; approximate contact parameters, no acoustic or membrane calibration.'}

    def save(self):
        (OUTPUT/'contact-test.json').write_text(json.dumps(self.report(), indent=2)+'\n')
        (OUTPUT/'contact-trace.json').write_text(json.dumps(self.rows)+'\n')


def validate():
    (OUTPUT/'contact-validation.json').unlink(missing_ok=True)
    model, robot, scene = build_scene()
    # Positive checks for every playing surface, using a small independent probe.
    xml = ET.parse(OUTPUT/'scene.xml')
    body = ET.SubElement(xml.find('worldbody'), 'body', name='test_probe', pos='0 0 10')
    ET.SubElement(body, 'freejoint', name='test_probe_joint')
    ET.SubElement(body, 'inertial', pos='0 0 0', mass='.01', diaginertia='1e-6 1e-6 1e-6')
    ET.SubElement(body, 'geom', name='test_probe_geom', type='sphere', size='.004', contype='2', conaffinity='1')
    probe = mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(), encoding='unicode'))
    probe_data = mujoco.MjData(probe)
    probe_id = probe.geom('test_probe_geom').id
    qa = probe.jnt_qposadr[probe.joint('test_probe_joint').id]
    for surface in scene['surfaces']:
        probe_data.qpos[qa:qa+3] = np.array(surface['center'])+.003*np.array(surface['normal'])
        mujoco.mj_forward(probe, probe_data)
        head = probe.geom(surface['name']+'_head_collision').id
        assert any({c.geom1, c.geom2} == {probe_id, head} and c.dist < 0 for c in probe_data.contact), surface['name']
    # Ready pose should not be disturbed by artificial proxy intersections.
    controller = HoldController(model, robot)
    resting = mujoco.MjData(model)
    controller.reset(resting)
    assert not any(c.dist < -1e-5 for c in resting.contact)
    for _ in range(round(1/model.opt.timestep)):
        controller.step(resting)
    assert np.max(np.abs(resting.qpos[controller.q]-controller.target)) < .005
    results = []
    for dt in (model.opt.timestep, model.opt.timestep/2):
        model.opt.timestep = dt
        test = ContactCheck(model, robot, scene)
        data = mujoco.MjData(model)
        test.reset(data)
        while not test.finished:
            test.step(data)
        result = test.report()
        print(json.dumps(result, indent=2))
        test.save()
        results.append(result)
        if result['status'] != 'PASS':
            raise RuntimeError('Slow contact test failed; see contact-test.json')
    relative_force_change = abs(results[0]['metrics']['peak_tip_force_N']-results[1]['metrics']['peak_tip_force_N'])/results[0]['metrics']['peak_tip_force_N']
    assert relative_force_change < .25, relative_force_change
    # Negative control: same torque trajectory with contacts disabled must pass
    # through the plane and must not report a successful touch.
    model.opt.timestep = results[0]['timestep_s']
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    ghost = ContactCheck(model, robot, scene)
    ghost_data = mujoco.MjData(model)
    ghost.reset(ghost_data)
    while not ghost.finished:
        ghost.step(ghost_data)
    assert ghost.metrics['peak_tip_force_N'] == 0
    assert ghost.metrics['minimum_tip_height_m'] < -.001
    assert ghost.report()['status'] == 'FAIL'
    (OUTPUT/'contact-validation.json').write_text(json.dumps({'status': 'PASS', 'surface_probe_checks': len(scene['surfaces']),
        'runs': results, 'relative_peak_force_change': relative_force_change,
        'disabled_contact_control': {'force_N': ghost.metrics['peak_tip_force_N'], 'minimum_tip_height_m': ghost.metrics['minimum_tip_height_m']}}, indent=2)+'\n')


if __name__ == '__main__':
    validate()
