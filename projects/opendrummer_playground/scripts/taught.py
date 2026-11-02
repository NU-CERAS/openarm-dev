"""Persistent arm/drum poses and contact-driven playback of taught J7 strokes."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from model import CONFIG, OUTPUT, SOURCE, HoldController
from strike import HitDetector, STRIKE_CONFIG, smooth

PRESETS = CONFIG.with_name('taught_poses.json')


def scene_signature(scene):
    files = [SOURCE, CONFIG, CONFIG.with_name('scene.json'), CONFIG.with_name('contact.json')]
    payload = b''.join(p.read_bytes() for p in files)+json.dumps(scene['surfaces'], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


class PoseStore:
    def __init__(self, model, robot, scene, path=PRESETS):
        self.path, self.signature = Path(path), scene_signature(scene)
        self.controller = HoldController(model, robot)
        self.names = {s['name'] for s in scene['surfaces']}
        self.document = {'version': 1, 'presets': {}}
        if self.path.exists():
            self.document = json.loads(self.path.read_text())
            if self.document.get('version') != 1 or not isinstance(self.document.get('presets'), dict):
                raise ValueError('Unsupported taught-pose file')

    def key(self, arm, drum):
        if arm not in ('left', 'right') or drum not in self.names:
            raise ValueError('Choose an arm and drum')
        return f'{arm}/{drum}'

    def validate(self, arm, pose):
        q = np.asarray(pose, dtype=float)
        bounds = self.controller.bounds[:7] if arm == 'left' else self.controller.bounds[7:]
        if q.shape != (7,) or not np.isfinite(q).all() or np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]):
            raise ValueError('Pose must contain seven finite joint angles within limits')
        return q

    def get(self, arm, drum, complete=True):
        item = self.document['presets'].get(self.key(arm, drum))
        if item is None:
            raise ValueError('No saved ready pose for this arm/drum')
        if item.get('arm') != arm or item.get('drum') != drum:
            raise ValueError('Saved preset arm/drum does not match its key')
        if item['scene_signature'] != self.signature:
            raise ValueError('Robot or scene changed. Re-teach this preset before playback.')
        self.validate(arm, item['ready_rad'])
        if complete:
            if 'strike_joint7_rad' not in item:
                raise ValueError('Save the J7 strike angle first')
            self.validate(arm, item['ready_rad'][:6]+[item['strike_joint7_rad']])
            if abs(item['strike_joint7_rad']-item['ready_rad'][6]) < .01:
                raise ValueError('Starting and strike J7 angles must differ')
        return dict(item)

    def write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.document, indent=2, allow_nan=False)+'\n')
        temporary.replace(self.path)

    def save_ready(self, arm, drum, pose):
        q = self.validate(arm, pose)
        self.document['presets'][self.key(arm, drum)] = {
            'arm': arm, 'drum': drum, 'ready_rad': q.tolist(), 'scene_signature': self.signature,
            'downstroke_seconds': json.loads(STRIKE_CONFIG.read_text())['downstroke_seconds'],
            'retract_seconds': json.loads(STRIKE_CONFIG.read_text())['retract_seconds']}
        self.write()

    def save_strike(self, arm, drum, pose):
        item = self.get(arm, drum, complete=False)
        q = self.validate(arm, pose)
        if np.max(np.abs(q[:6]-np.array(item['ready_rad'][:6]))) > 1e-6:
            raise ValueError('Joints 1–6 changed. Save ready again, then jog only J7.')
        if abs(q[6]-item['ready_rad'][6]) < .01:
            raise ValueError('Jog J7 down before saving the strike angle')
        item['strike_joint7_rad'] = float(q[6])
        self.document['presets'][self.key(arm, drum)] = item
        self.write()


class TaughtStrike:
    """Smooth full-arm approach followed by J7-only stroke and rebound."""
    def __init__(self, model, robot, scene, preset, count=1, initial=None, approach_path=None, ready_pose=None, transfer_motion=None):
        if type(count) is not int or not 1 <= count <= 100:
            raise ValueError('Strike count must be 1–100')
        self.model, self.controller = model, HoldController(model, robot)
        self.cfg = json.loads(STRIKE_CONFIG.read_text())
        for name in ('downstroke_seconds', 'retract_seconds'):
            value = float(preset[name])
            if not np.isfinite(value) or value <= 0:
                raise ValueError('Invalid saved stroke duration')
            self.cfg[name] = value
        self.side, self.target = preset['arm'], preset['drum']
        self.selected = np.arange(7) if self.side == 'left' else np.arange(7, 14)
        self.wrist = int(self.selected[-1])
        self.held = np.array([i for i in range(14) if i != self.wrist])
        c = self.controller
        self.initial = c.target.copy() if initial is None else np.array(initial, dtype=float)
        self.ready = self.initial.copy() if ready_pose is None else np.array(ready_pose,dtype=float)
        self.ready[self.selected] = preset['ready_rad']
        if initial is None:
            # A standalone demo starts at its taught ready pose. Moving between
            # different drums requires a separately checked transfer trajectory.
            self.initial = self.ready.copy()
        self.bottom = self.ready.copy()
        self.bottom[self.wrist] = preset['strike_joint7_rad']
        for pose in (self.initial, self.ready, self.bottom):
            if pose.shape != (14,) or not np.isfinite(pose).all() or np.any(pose < c.bounds[:,0]) or np.any(pose > c.bounds[:,1]):
                raise ValueError('Saved pose exceeds joint limits')
        if 1.5*abs(self.bottom[self.wrist]-self.ready[self.wrist])/self.cfg['downstroke_seconds'] > c.velocity[self.wrist]:
            raise ValueError('Saved stroke exceeds J7 velocity limit; reduce travel or increase duration')
        surface = next(s for s in scene['surfaces'] if s['name'] == self.target)
        self.center, self.normal = np.array(surface['center']), np.array(surface['normal'])
        self.radius = surface['radius']
        self.sid = model.site(self.side+'_stick_tip').id
        self.tip = model.geom(self.side+'_stick_tip_collision').id
        self.head = model.geom(self.target+'_head_collision').id
        self.probe = mujoco.MjData(model)
        heights = []
        for mix in np.linspace(0, 1, 151):
            self.probe.qpos[c.q] = self.ready+mix*(self.bottom-self.ready)
            mujoco.mj_forward(model, self.probe)
            delta = self.probe.site_xpos[self.sid]-self.center
            height = float(delta@self.normal)
            heights.append(height)
            if np.linalg.norm(delta-height*self.normal) > self.radius-.005:
                raise ValueError('Saved wrist arc leaves the selected drum strike area')
            for contact in self.probe.contact:
                intended = self.head in (contact.geom1, contact.geom2) and any(
                    model.geom(g).name.startswith(self.side+'_stick_') for g in (contact.geom1,contact.geom2))
                if contact.dist < -1e-5 and not (intended and height < .005):
                    raise ValueError('Saved wrist arc intersects other geometry')
        self.ready_height = heights[0]
        if self.ready_height < .01 or heights[-1] > .003 or heights[-1] >= heights[0]:
            raise ValueError('Save a raised ready pose (≥1 cm), then lower J7 to touch or cross the head')
        if np.max(np.diff(heights)) > .0001:
            raise ValueError('J7 stroke must move toward the head throughout its arc')
        self.count, self.cycle, self.completed = count, 1, 0
        self.state, self.state_start = 'approach', 0.
        self.finished, self.failure = False, None
        self.detector = HitDetector(self.cfg)
        self.cycle_hit_start = 0
        self.metrics = {'peak_force_N': 0., 'max_penetration_m': 0., 'peak_tip_speed_m_s': 0.,
                        'max_held_joint_command_change_rad': 0.}
        self.rows = []
        self.transfer_motion = transfer_motion
        if approach_path is not None and transfer_motion is None:
            self.transfer_motion = json.loads(CONFIG.with_name('demo_motion.json').read_text())
        self.approach_path = np.array(approach_path if approach_path is not None else [self.initial,self.ready],dtype=float)
        if (self.approach_path.ndim!=2 or self.approach_path.shape[1]!=14 or len(self.approach_path)<2 or not np.isfinite(self.approach_path).all()
            or np.any(self.approach_path<c.bounds[:,0]) or np.any(self.approach_path>c.bounds[:,1])
            or not np.allclose(self.approach_path[0],self.initial) or not np.allclose(self.approach_path[-1],self.ready)):
            raise ValueError('Approach path must connect current pose to ready pose')
        if approach_path is not None:
            fraction=float(self.transfer_motion['transfer_speed_fraction'])
            acceleration=np.tile(np.array(self.transfer_motion['transfer_acceleration_rad_s2'],dtype=float),2)
            if not np.isfinite(fraction) or not 0 < fraction <= 1 or acceleration.shape!=(14,) or not np.isfinite(acceleration).all() or np.any(acceleration<=0):
                raise ValueError('Invalid transfer speed fraction or acceleration limits')
            self.transfer_velocity = fraction*c.velocity
            distance=np.abs(np.diff(self.approach_path,axis=0))
            # Cubic smoothstep peaks at 1.5*dq/T, acceleration at 6*dq/T^2.
            # Leave 2% velocity headroom for the physical servo to track the cap.
            velocity_time=np.max(1.5*distance/(.98*self.transfer_velocity),axis=1)
            acceleration_time=np.max(np.sqrt(6*distance/acceleration),axis=1)
            self.approach_durations=np.maximum(.1,np.maximum(velocity_time,acceleration_time))
            self.metrics.update(max_transfer_command_motor_fraction=0.,max_transfer_actual_motor_fraction=0.)
        else:
            self.transfer_velocity=c.velocity
            self.approach_durations=np.array([2.])
        self.approach_ends = np.cumsum(self.approach_durations)

    def start_live(self, data):
        """Continue from the current simulation state, without resetting time or joints."""
        if not np.allclose(data.qpos[self.controller.q],self.initial,atol=.02):
            raise ValueError('Robot moved while planning; retry the drum')
        self.state_start = float(data.time)

    def reset(self, data):
        self.controller.target = self.initial.copy()
        self.controller.reset(data)
        if any(c.dist < -1e-5 for c in data.contact):
            raise ValueError('Starting pose intersects geometry; load the ready pose first')
        self.state_start = float(data.time)

    def transition(self, state, data):
        self.state, self.state_start = state, float(data.time)
        if state == 'retract':
            self.retract_start = self.ready.copy()
            self.retract_start[self.wrist] = data.qpos[self.controller.q[self.wrist]]

    def abort(self, message):
        self.failure, self.finished = message, True

    def step(self, data):
        if self.finished:
            return
        cfg, c = self.cfg, self.controller
        elapsed = data.time-self.state_start
        transfer_active = self.state=='approach' and self.transfer_motion is not None
        if self.state == 'approach' and elapsed >= self.approach_ends[-1]:
            self.transition('ready',data)
        elif self.state == 'ready' and elapsed >= cfg['ready_seconds']:
            self.transition('downstroke',data)
        elif self.state == 'downstroke' and elapsed >= cfg['downstroke_seconds']+cfg['contact_timeout_seconds']:
            self.abort('Missed strike: no qualified contact'); return
        elif self.state == 'retract' and elapsed >= cfg['retract_seconds']:
            self.transition('recovery',data)
        elif self.state == 'recovery' and elapsed >= cfg['recovery_seconds']:
            if len(self.detector.events)-self.cycle_hit_start != 1 or not self.detector.armed or np.max(np.abs(data.qvel[c.v])) > .08:
                self.abort('Recovery failed: extra/missing hit, motion, or insufficient clearance'); return
            self.completed += 1
            if self.completed == self.count:
                self.finished = True; self.state = 'complete'; return
            self.cycle += 1
            self.cycle_hit_start = len(self.detector.events)
            self.transition('ready',data)
        elapsed = data.time-self.state_start
        target, velocity = self.ready.copy(), np.zeros(14)
        if self.state in ('approach','downstroke','retract'):
            if self.state == 'approach':
                index=min(int(np.searchsorted(self.approach_ends,elapsed,side='right')),len(self.approach_durations)-1)
                start,end,duration = self.approach_path[index],self.approach_path[index+1],self.approach_durations[index]
                elapsed -= 0 if index==0 else self.approach_ends[index-1]
            elif self.state == 'downstroke':
                start,end,duration = self.ready,self.bottom,cfg['downstroke_seconds']
            else:
                start,end,duration = self.retract_start,self.ready,cfg['retract_seconds']
            blend,rate = smooth(elapsed/duration)
            target,velocity = start+(end-start)*blend,(end-start)*rate/duration
        if np.any(np.abs(velocity) > c.velocity):
            self.abort('Commanded velocity limit exceeded'); return
        if transfer_active and np.any(np.abs(velocity)>self.transfer_velocity+1e-8):
            self.abort('Transfer command exceeds configured motor-speed fraction'); return
        kp = c.kp.astype(float)
        if self.state == 'retract' and elapsed < cfg['compliance_seconds']:
            kp[self.wrist] *= cfg['contact_kp_multiplier']
        torque = kp*(target-data.qpos[c.q])+c.kd*(velocity-data.qvel[c.v])+data.qfrc_bias[c.v]
        data.ctrl[c.a] = np.clip(torque,-c.effort,c.effort)
        jac = np.zeros((3,self.model.nv))
        mujoco.mj_jacSite(self.model,data,jac,None,self.sid)
        speed = float(np.linalg.norm(jac@data.qvel))
        mujoco.mj_step(self.model,data)
        force,penetration = 0.,0.
        for index,contact in enumerate(data.contact):
            wrench = np.zeros(6); mujoco.mj_contactForce(self.model,data,index,wrench)
            value = max(0.,float(wrench[0]))
            if {contact.geom1,contact.geom2} == {self.tip,self.head}:
                force += value; penetration = max(penetration,-float(contact.dist))
            elif value > .05:
                self.abort('Unexpected contact: '+self.model.geom(contact.geom1).name+' / '+self.model.geom(contact.geom2).name)
        height = float((data.site_xpos[self.sid]-self.center)@self.normal)
        event = self.detector.update(float(data.time),force,height,self.model.opt.timestep)
        if event is not None:
            event.update(arm=self.side,drum=self.target,cycle=self.cycle)
            if self.state != 'downstroke':
                self.abort('Extra or out-of-phase hit')
            else:
                self.transition('retract',data)
        self.metrics['peak_force_N'] = max(self.metrics['peak_force_N'],force)
        self.metrics['max_penetration_m'] = max(self.metrics['max_penetration_m'],penetration)
        self.metrics['peak_tip_speed_m_s'] = max(self.metrics['peak_tip_speed_m_s'],speed)
        if transfer_active:
            command_fraction=float(np.max(np.abs(velocity)/c.velocity))
            actual_fraction=float(np.max(np.abs(data.qvel[c.v])/c.velocity))
            self.metrics['max_transfer_command_motor_fraction']=max(self.metrics['max_transfer_command_motor_fraction'],command_fraction)
            self.metrics['max_transfer_actual_motor_fraction']=max(self.metrics['max_transfer_actual_motor_fraction'],actual_fraction)
            if np.any(np.abs(data.qvel[c.v])>self.transfer_velocity+1e-3):
                self.abort('Measured transfer speed exceeds configured motor-speed fraction')
        if self.state != 'approach':
            self.metrics['max_held_joint_command_change_rad'] = max(self.metrics['max_held_joint_command_change_rad'],float(np.max(np.abs(target[self.held]-self.ready[self.held]))))
        tip_speed_limit = self.transfer_motion['transfer_tip_speed_stop_m_s'] if transfer_active else cfg['tip_speed_stop_m_s']
        if force > cfg['force_stop_N'] or penetration > cfg['penetration_stop_m'] or speed > tip_speed_limit:
            self.abort('Force, penetration, or tip-speed limit exceeded')
        if np.max(np.abs(target-data.qpos[c.q])) > cfg['tracking_stop_rad']:
            self.abort('Joint tracking limit exceeded')
        if not np.isfinite(data.qpos).all() or np.any(data.qpos[c.q]<c.bounds[:,0]-.001) or np.any(data.qpos[c.q]>c.bounds[:,1]+.001) or np.any(np.abs(data.qvel[c.v])>c.velocity) or sum(int(w.number) for w in data.warning):
            self.abort('Joint limit or solver check failed')
        self.rows.append({'time_s':float(data.time),'state':self.state,'target_rad':target.tolist(),'actual_rad':data.qpos[c.q].tolist()})

    def report(self):
        return {'status':'PASS' if self.finished and not self.failure and self.completed==self.count else 'FAIL',
                'failure':self.failure,'completed_strikes':self.completed,'hit_count':len(self.detector.events),
                'arm':self.side,'drum':self.target,'metrics':self.metrics,'hits':self.detector.events,
                'transfer_motion':self.transfer_motion,'transfer_duration_s':float(self.approach_ends[-1])}

    def save(self):
        (OUTPUT/'taught-strike-report.json').write_text(json.dumps(self.report(),indent=2)+'\n')
