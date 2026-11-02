"""Contact-triggered single snare strike and recovery, optionally repeated."""
import json

import mujoco
import numpy as np

from openarm_drummer.model import CONFIG, OUTPUT, HoldController
from openarm_drummer.scene import build_scene, solve_approach

STRIKE_CONFIG = CONFIG.with_name('strike.json')


def smooth(x):
    x = np.clip(x, 0, 1)
    return x*x*(3-2*x), 6*x*(1-x)


class HitDetector:
    """One event per contact episode, rearmed only after sustained separation."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.armed = True
        self.clear_time = 0.
        self.episode = None
        self.events = []

    def update(self, t, force, gap, dt):
        new = None
        if force > self.cfg['release_force_N']:
            self.clear_time = 0.
            if self.episode is None:
                self.episode = {'onset_time_s': t, 'peak_force_N': 0., 'normal_impulse_Ns': 0., 'contact_duration_s': 0.}
            self.episode['peak_force_N'] = max(self.episode['peak_force_N'], force)
            self.episode['normal_impulse_Ns'] += force*dt
            self.episode['contact_duration_s'] += dt
            self.episode['last_contact_time_s'] = t
            if self.armed and force >= self.cfg['hit_force_N'] and self.episode['normal_impulse_Ns'] >= self.cfg['hit_impulse_Ns']:
                self.armed = False
                self.episode['confirmed_time_s'] = t
                self.events.append(self.episode)
                new = self.episode
        elif gap >= self.cfg['rearm_clearance_m']:
            self.clear_time += dt
            if self.clear_time >= self.cfg['rearm_seconds']:
                self.armed = True
                self.episode = None
        else:
            self.clear_time = 0.
        return new


class StrikeSession:
    def __init__(self, model, robot_report, scene_report, count=1, impact_seed=None, configuration=None):
        if type(count) is not int or not 1 <= count <= 100:
            raise ValueError('Strike count must be between 1 and 100')
        self.model = model
        self.config = json.loads(STRIKE_CONFIG.read_text()) if configuration is None else dict(configuration)
        self.cfg = self.config
        for key, value in self.cfg.items():
            if isinstance(value, (int, float)) and (not np.isfinite(value) or (key != 'impact_joint7_rad' and value <= 0)):
                raise ValueError(f'{key} must be finite and positive')
        if self.cfg['ready_height_m'] <= self.cfg['approach_height_m']:
            raise ValueError('Ready height must exceed approach height')
        if not 0 < self.cfg['contact_kp_multiplier'] <= 1:
            raise ValueError('Contact stiffness multiplier must be in (0, 1]')
        self.side, self.target = self.cfg['arm'], self.cfg['target']
        if (self.side, self.target) != ('left', 'snare'):
            raise ValueError('This validated strike routine currently supports left/snare only')
        self.controller = HoldController(model, robot_report)
        self.selected = np.array([6])  # Only J7 softens on contact.
        self.surface = next(s for s in scene_report['surfaces'] if s['name'] == self.target)
        self.center, self.normal = np.array(self.surface['center']), np.array(self.surface['normal'])
        self.sid = model.site(f'{self.side}_stick_tip').id
        self.tip = model.geom(f'{self.side}_stick_tip_collision').id
        self.head = model.geom(f'{self.target}_head_collision').id
        self.probe = mujoco.MjData(model)
        c = self.controller

        def accepts_wrist_arc(pose):
            q7 = pose[6]

            def tip(angle):
                self.probe.qpos[c.q] = pose
                self.probe.qpos[c.q[6]] = angle
                mujoco.mj_forward(model, self.probe)
                return float((self.probe.site_xpos[self.sid]-self.center)@self.normal)

            gradient = (tip(q7+.0001)-tip(q7-.0001))/.0002
            if abs(gradient) < .02:
                return False
            endpoints = []
            for wanted in (self.cfg['ready_height_m'], -self.cfg['command_depth_m']):
                direction = np.sign(wanted/gradient)
                angles = np.linspace(q7, np.clip(q7+direction*1.5, *c.bounds[6]), 301)
                values = np.array([tip(a)-wanted for a in angles])
                crossings = np.flatnonzero(values[:-1]*values[1:] <= 0)
                if not len(crossings):
                    return False
                endpoints.append(angles[crossings[0]])
            for angle in np.linspace(*endpoints, 101):
                height = tip(angle)
                radial = np.linalg.norm(self.probe.site_xpos[self.sid]-self.center-height*self.normal)
                if radial > self.surface['radius']-.01:
                    return False
                for contact in self.probe.contact:
                    virtual = height < 0 and self.head in (contact.geom1, contact.geom2) and any(
                        model.geom(g).name.startswith(self.side+'_stick_') for g in (contact.geom1, contact.geom2))
                    if contact.dist < -1e-5 and not virtual and {contact.geom1, contact.geom2} != {self.tip, self.head}:
                        return False
            return True

        # Solve the impact pose first, then lift only J7. This keeps the impact
        # point centered despite the tip following a circular wrist arc.
        impact = solve_approach(model, robot_report, scene_report, self.side, self.target,
                                height=0., wrist_angle=self.cfg['impact_joint7_rad'], joint_margin=.03, seed=impact_seed,
                                pose_validator=accepts_wrist_arc if impact_seed is not None else None)
        if not impact['reachable']:
            raise RuntimeError('No reachable wrist impact pose')
        self.impact_pose = np.array(impact['joint_positions'])
        def height(angle):
            self.probe.qpos[c.q] = self.impact_pose
            self.probe.qpos[c.q[6]] = angle
            mujoco.mj_forward(model, self.probe)
            return float((self.probe.site_xpos[self.sid]-self.center)@self.normal)
        q7 = self.impact_pose[6]
        gradient = (height(q7+.0001)-height(q7-.0001))/.0002
        if abs(gradient) < .02:
            raise RuntimeError('J7 has insufficient normal motion at impact')
        def pose_at_height(wanted):
            direction = np.sign(wanted/gradient)
            previous, previous_error = q7, height(q7)-wanted
            for angle in np.linspace(q7, np.clip(q7+direction*1.5, c.bounds[6,0], c.bounds[6,1]), 301)[1:]:
                error = height(angle)-wanted
                if error*previous_error <= 0:
                    lo, hi = previous, angle
                    for _ in range(40):
                        mid = (lo+hi)/2
                        if (height(lo)-wanted)*(height(mid)-wanted) <= 0:
                            hi = mid
                        else:
                            lo = mid
                    pose = self.impact_pose.copy()
                    pose[6] = (lo+hi)/2
                    return pose
                previous, previous_error = angle, error
            raise RuntimeError(f'J7 cannot reach tip height {wanted}')
        self.ready = pose_at_height(self.cfg['ready_height_m'])
        self.approach = pose_at_height(self.cfg['approach_height_m'])
        self.bottom = pose_at_height(-self.cfg['command_depth_m'])
        staging = solve_approach(model, robot_report, scene_report, self.side, self.target,
                                 height=self.cfg['ready_height_m'], seed=self.ready)
        if not staging['reachable']:
            raise RuntimeError('No reachable staging pose')
        self.staging = np.array(staging['joint_positions'])
        self.controller.target = self.staging.copy()
        # Certify the commanded wrist arc before running dynamics.
        self.arc = []
        for angle in np.linspace(self.ready[6], self.bottom[6], 101):
            h = height(angle)
            delta = self.probe.site_xpos[self.sid]-self.center
            radius = float(np.linalg.norm(delta-h*self.normal))
            if radius > self.surface['radius']-.01:
                raise RuntimeError('Wrist arc leaves the drum strike area')
            for contact in self.probe.contact:
                virtual_head_contact = h < 0 and self.head in (contact.geom1, contact.geom2) and any(
                    model.geom(g).name.startswith(self.side+'_stick_') for g in (contact.geom1, contact.geom2))
                if contact.dist < -1e-5 and not virtual_head_contact and {contact.geom1, contact.geom2} != {self.tip, self.head}:
                    raise RuntimeError(f'Wrist arc intersects: {model.geom(contact.geom1).name}, {model.geom(contact.geom2).name}, height={h:.4f}')
            self.arc.append({'joint7_rad': float(angle), 'height_m': h, 'radial_offset_m': radius})
        self.count = count
        self.detector = HitDetector(self.cfg)
        self.state = 'ready'
        self.state_start = 0.
        self.cycle = 1
        self.cycle_hit_start = 0
        self.completed = 0
        self.finished = False
        self.failure = None
        self.transitions = [{'state': 'ready', 'time_s': 0., 'cycle': 1}]
        self.rows, self.cycles = [], []
        self.expected_hit_time = None
        self.impact_speed = 0.
        self.metrics = {'peak_force_N': 0., 'max_penetration_m': 0., 'peak_tip_speed_m_s': 0.,
                        'max_tracking_error_rad': 0., 'unexpected_contacts': [], 'max_joint_speed_rad_s': 0.,
                        'max_stick_direction_error_degrees': 0.,
                        'max_held_joint_deviation_rad': 0., 'max_held_joint_command_change_rad': 0.,
                        'max_hit_radial_offset_m': 0.}

    def reset(self, data):
        self.controller.reset(data)
        if any(c.dist < -1e-5 for c in data.contact):
            raise RuntimeError('Ready strike pose intersects collision proxies')

    def transition(self, state, data):
        self.state, self.state_start = state, float(data.time)
        self.transitions.append({'state': state, 'time_s': float(data.time), 'cycle': self.cycle})
        if state == 'downstroke':
            # Nominal crossing time of the commanded tip plane (not a beat clock).
            fraction = (self.impact_pose[6]-self.approach[6])/(self.bottom[6]-self.approach[6])
            lo, hi = 0., 1.
            for _ in range(40):
                mid = (lo+hi)/2
                if smooth(mid)[0] < fraction:
                    lo = mid
                else:
                    hi = mid
            self.expected_hit_time = data.time+(lo+hi)/2*self.cfg['downstroke_seconds']
        if state == 'retract':
            self.retract_start = self.ready.copy()
            self.retract_start[6] = data.qpos[self.controller.q[6]]

    def desired(self, data):
        elapsed = data.time-self.state_start
        cfg = self.cfg
        if self.state in ('ready', 'recovery', 'complete'):
            pose = self.staging if self.state == 'ready' and self.cycle == 1 else self.ready
            return pose.copy(), np.zeros(14)
        if self.state == 'approach':
            duration = cfg['approach_seconds']
            start = self.staging if self.cycle == 1 else self.ready
            end = self.approach
        elif self.state == 'downstroke':
            duration = cfg['downstroke_seconds']
            start, end = self.approach, self.bottom
        elif self.state == 'contact':
            target = self.ready.copy()
            target[6] = data.qpos[self.controller.q[6]]
            return target, np.zeros(14)
        elif self.state == 'retract':
            blend, rate = smooth(elapsed/cfg['retract_seconds'])
            return ((1-blend)*self.retract_start+blend*self.ready,
                    rate/cfg['retract_seconds']*(self.ready-self.retract_start))
        else:
            raise RuntimeError(self.state)
        blend, rate = smooth(elapsed/duration)
        return start+(end-start)*blend, (end-start)*rate/duration

    def abort(self, reason):
        self.failure = reason
        self.finished = True

    def step(self, data, joint7_offset=0.0):
        if self.finished:
            return
        cfg, c = self.cfg, self.controller
        elapsed = data.time-self.state_start
        if self.state == 'ready' and elapsed >= cfg['ready_seconds']:
            self.transition('approach', data)
        elif self.state == 'approach' and elapsed >= cfg['approach_seconds']:
            self.transition('downstroke', data)
        elif self.state == 'contact':
            self.transition('retract', data)
        elif self.state == 'downstroke' and elapsed > cfg['downstroke_seconds']+cfg['contact_timeout_seconds']:
            self.abort('Missed strike: no qualified contact before timeout')
            return
        elif self.state == 'retract' and elapsed >= cfg['retract_seconds']:
            self.transition('recovery', data)
        elif self.state == 'recovery' and elapsed >= cfg['recovery_seconds']:
            hits = len(self.detector.events)-self.cycle_hit_start
            height = float((data.site_xpos[self.sid]-self.center)@self.normal)
            if hits != 1 or height < cfg['ready_height_m']-.005 or np.max(np.abs(data.qvel)) > .08 or not self.detector.armed:
                self.abort(f'Recovery failed: hits={hits}, height={height:.4f}, rearmed={self.detector.armed}')
                return
            self.cycles.append({'cycle': self.cycle, 'hit_count': hits, 'recovered_at_s': float(data.time), 'recovery_height_m': height})
            self.completed += 1
            if self.completed == self.count:
                self.transition('complete', data)
                self.finished = True
                return
            self.cycle += 1
            self.cycle_hit_start = len(self.detector.events)
            self.transition('ready', data)

        if not np.isfinite(joint7_offset) or abs(joint7_offset) > .2:
            raise ValueError('J7 adjustment must be finite and within 0.2 rad')
        target, velocity = self.desired(data)
        target[6] += joint7_offset
        if np.any(target < c.bounds[:, 0]) or np.any(target > c.bounds[:, 1]) or np.any(np.abs(velocity) > c.velocity):
            self.abort('Command exceeds joint position or velocity limits')
            return
        kp = c.kp.astype(float)
        if self.state == 'contact' or (self.state == 'retract' and data.time-self.state_start < cfg['compliance_seconds']):
            kp[self.selected] *= cfg['contact_kp_multiplier']
        torque = kp*(target-data.qpos[c.q])+c.kd*(velocity-data.qvel[c.v])+data.qfrc_bias[c.v]
        data.ctrl[c.a] = np.clip(torque, -c.effort, c.effort)
        jp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jp, None, self.sid)
        tip_velocity = jp@data.qvel
        self.impact_speed = max(0., float(-tip_velocity@self.normal))
        mujoco.mj_step(self.model, data)

        force, penetration = 0., 0.
        for index, contact in enumerate(data.contact):
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, index, wrench)
            value = max(0., float(wrench[0]))
            if {contact.geom1, contact.geom2} == {self.tip, self.head}:
                force += value
                penetration = max(penetration, -float(contact.dist))
            elif value > .05:
                pair = [self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name]
                if pair not in self.metrics['unexpected_contacts']:
                    self.metrics['unexpected_contacts'].append(pair)
                self.abort(f'Unexpected contact: {pair}')
        height = float((data.site_xpos[self.sid]-self.center)@self.normal)
        self.probe.qpos[c.q] = target
        mujoco.mj_forward(self.model, self.probe)
        self.desired_direction = self.probe.site_xmat[self.sid].reshape(3, 3)[:, 2]
        actual_direction = data.site_xmat[self.sid].reshape(3, 3)[:, 2]
        direction_error = float(np.degrees(np.arccos(np.clip(actual_direction@self.desired_direction, -1, 1))))
        hit = self.detector.update(float(data.time), force, height, self.model.opt.timestep)
        if hit is not None:
            radial = float(np.linalg.norm(data.site_xpos[self.sid]-self.center-height*self.normal))
            hit['radial_offset_m'] = radial
            self.metrics['max_hit_radial_offset_m'] = max(self.metrics['max_hit_radial_offset_m'], radial)
            if radial > self.surface['radius']-.01:
                self.abort('Impact outside drum strike area')
            hit.update(cycle=self.cycle, downward_speed_m_s=self.impact_speed,
                       expected_contact_time_s=self.expected_hit_time,
                       timing_error_s=None if self.expected_hit_time is None else hit['onset_time_s']-self.expected_hit_time)
            if self.state != 'downstroke' or len(self.detector.events)-self.cycle_hit_start != 1:
                self.abort('Extra or out-of-phase hit detected')
            else:
                self.transition('contact', data)
        tracking = float(np.max(np.abs(target-data.qpos[c.q])))
        self.metrics['peak_force_N'] = max(self.metrics['peak_force_N'], force)
        self.metrics['max_penetration_m'] = max(self.metrics['max_penetration_m'], penetration)
        self.metrics['peak_tip_speed_m_s'] = max(self.metrics['peak_tip_speed_m_s'], float(np.linalg.norm(tip_velocity)))
        self.metrics['max_tracking_error_rad'] = max(self.metrics['max_tracking_error_rad'], tracking)
        self.metrics['max_joint_speed_rad_s'] = max(self.metrics['max_joint_speed_rad_s'], float(np.max(np.abs(data.qvel))))
        self.metrics['max_stick_direction_error_degrees'] = max(self.metrics['max_stick_direction_error_degrees'], direction_error)
        if self.state in ('downstroke', 'contact', 'retract', 'recovery'):
            held = np.r_[0:6, 7:14]
            command_change = float(np.max(np.abs(target[held]-self.impact_pose[held])))
            deviation = float(np.max(np.abs(data.qpos[c.q[held]]-self.impact_pose[held])))
            self.metrics['max_held_joint_command_change_rad'] = max(self.metrics['max_held_joint_command_change_rad'], command_change)
            self.metrics['max_held_joint_deviation_rad'] = max(self.metrics['max_held_joint_deviation_rad'], deviation)
            if command_change > 1e-10:
                self.abort('Non-wrist joint commanded during strike')
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            self.abort('Non-finite simulation state')
        if force > cfg['force_stop_N'] or penetration > cfg['penetration_stop_m']:
            self.abort('Force/penetration threshold exceeded')
        if tracking > cfg['tracking_stop_rad'] or np.linalg.norm(tip_velocity) > cfg['tip_speed_stop_m_s']:
            self.abort('Tracking/tip-speed threshold exceeded')
        if direction_error > cfg['direction_stop_degrees']:
            self.abort('Stick direction tracking threshold exceeded')
        if np.any(np.abs(data.qvel[c.v]) > c.velocity) or np.any(np.abs(data.actuator_force[c.a]) > c.effort+1e-8):
            self.abort('Measured joint velocity/effort limit exceeded')
        if np.any(data.qpos[c.q] < c.bounds[:, 0]-.001) or np.any(data.qpos[c.q] > c.bounds[:, 1]+.001):
            self.abort('Measured joint position limit exceeded')
        if sum(int(w.number) for w in data.warning):
            self.abort('MuJoCo solver warning')
        self.rows.append({'time_s': float(data.time), 'cycle': self.cycle, 'state': self.state,
                          'commanded_joint_positions_rad': target.tolist(),
                          'actual_joint_positions_rad': data.qpos[c.q].tolist(),
                          'tip_height_m': height, 'normal_force_N': force, 'penetration_m': penetration,
                          'tip_speed_m_s': float(np.linalg.norm(tip_velocity)), 'tracking_error_rad': tracking})

    def report(self):
        return {'status': 'PASS' if self.finished and not self.failure and self.completed == self.count else 'FAIL',
                'failure': self.failure, 'requested_strikes': self.count, 'completed_strikes': self.completed,
                'hit_count': len(self.detector.events), 'timestep_s': self.model.opt.timestep,
                'config': self.cfg, 'metrics': self.metrics, 'hits': self.detector.events,
                'cycles': self.cycles, 'transitions': self.transitions,
                'motion_mode': 'joint7_only', 'wrist_arc': self.arc}

    def save(self):
        (OUTPUT/'strike-report.json').write_text(json.dumps(self.report(), indent=2)+'\n')
        (OUTPUT/'strike-trace.json').write_text(json.dumps(self.rows)+'\n')


def validate():
    (OUTPUT/'strike-validation.json').unlink(missing_ok=True)
    model, robot, scene = build_scene()
    results = []
    for dt, count in ((.001, 10), (.0005, 5)):
        model.opt.timestep = dt
        session = StrikeSession(model, robot, scene, count)
        data = mujoco.MjData(model)
        session.reset(data)
        while not session.finished:
            session.step(data)
        session.save()
        report = session.report()
        print(json.dumps({key: report[key] for key in ('status', 'failure', 'hit_count', 'completed_strikes', 'metrics')}, indent=2))
        if report['status'] != 'PASS':
            raise RuntimeError('Strike validation failed; see strike-report.json')
        assert all(c['hit_count'] == 1 for c in report['cycles'])
        assert report['metrics']['max_held_joint_command_change_rad'] < 1e-10
        assert report['metrics']['max_held_joint_deviation_rad'] < .01
        assert report['metrics']['max_hit_radial_offset_m'] < .005
        assert all(h['contact_duration_s'] < .1 for h in report['hits'])
        assert np.ptp([h['timing_error_s'] for h in report['hits']]) < .01
        assert np.std([h['peak_force_N'] for h in report['hits']])/np.mean([h['peak_force_N'] for h in report['hits']]) < .05
        results.append(report)
    relative_force_change = abs(results[0]['metrics']['peak_force_N']-results[1]['metrics']['peak_force_N'])/results[0]['metrics']['peak_force_N']
    assert relative_force_change < .25
    # No contact means no hit, irrespective of commanded joint motion.
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    ghost = StrikeSession(model, robot, scene)
    data = mujoco.MjData(model)
    ghost.reset(data)
    while not ghost.finished:
        ghost.step(data)
    assert ghost.report()['hit_count'] == 0
    assert ghost.failure.startswith('Missed strike'), ghost.failure
    (OUTPUT/'strike-validation.json').write_text(json.dumps({'status': 'PASS', 'runs': results,
        'relative_peak_force_change': relative_force_change,
        'no_contact_control': {'hit_count': 0, 'failure': ghost.failure}}, indent=2)+'\n')


if __name__ == '__main__':
    validate()
