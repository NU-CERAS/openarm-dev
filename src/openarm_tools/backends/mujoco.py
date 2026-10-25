"""A persistent local MuJoCo session (not a connection to another running viewer)."""
import json
import numpy as np
import mujoco
from openarm_tools.robot.config import root, validate_positions
from openarm_tools.recording.storage import write_json


class Simulation:
    def __init__(self, cfg, state_path, reset=False):
        self.cfg, self.state_path = cfg, state_path
        self.model = mujoco.MjModel.from_xml_path(str(root() / cfg['model']))
        self.data = mujoco.MjData(self.model)
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.ids = [self.model.joint(j['name']).id for j in cfg['joints']]
        self.qadr = [int(self.model.jnt_qposadr[i]) for i in self.ids]
        self.vadr = [int(self.model.jnt_dofadr[i]) for i in self.ids]
        if state_path.exists() and not reset:
            state = json.loads(state_path.read_text())
            if state['config'] != cfg['_fingerprint']:
                raise ValueError('Simulation configuration changed; run sim --reset.')
            self.data.qpos[:] = state['qpos']
            self.data.qvel[:] = state['qvel']
        mujoco.mj_forward(self.model, self.data)
        # Hold all actuated joints so the unselected arm does not fall freely.
        self.actuated = []
        for aid in range(self.model.nu):
            if self.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
                jid = int(self.model.actuator_trnid[aid, 0])
                self.actuated.append((aid, int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])))
        self.hold = self.data.qpos.copy()

    def read(self):
        return [float(self.data.qpos[i]) for i in self.qadr]

    def status(self):
        return [{'joint': j['name'], 'position': p, 'velocity': float(self.data.qvel[v])}
                for j, p, v in zip(self.cfg['joints'], self.read(), self.vadr)]

    def step(self, target):
        validate_positions(self.cfg, target)
        self.hold[self.qadr] = target
        for aid, qa, va in self.actuated:
            # Existing models use either position servos or direct torque motors.
            if self.model.actuator_biastype[aid] == mujoco.mjtBias.mjBIAS_AFFINE:
                control = self.hold[qa]
            else:
                control = 20 * (self.hold[qa] - self.data.qpos[qa]) - 2 * self.data.qvel[va] + self.data.qfrc_bias[va]
            if self.model.actuator_ctrllimited[aid]:
                control = np.clip(control, *self.model.actuator_ctrlrange[aid])
            self.data.ctrl[aid] = control
        mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError('Non-finite simulation state.')

    def move(self, points):
        start_time = self.data.time
        for t, q in points:
            while self.data.time - start_time < t:
                self.step(q)
        for _ in range(int(0.5 / self.model.opt.timestep)):
            self.step(q)
        error = max(abs(a - b) for a, b in zip(self.read(), q))
        print(f'Maximum final tracking error: {error:.4f} rad')

    def close(self):
        write_json(self.state_path, {'config': self.cfg['_fingerprint'],
                                   'qpos': self.data.qpos.tolist(), 'qvel': self.data.qvel.tolist()})
