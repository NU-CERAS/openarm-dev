"""A persistent local MuJoCo session (not a connection to another running viewer)."""
import hashlib
import json
import numpy as np
import mujoco
from openarm_tools.robot.config import root, validate_positions
from openarm_tools.recording.storage import write_json


class Simulation:
    """One shared model holding every configured arm; cfgs maps an arm key to its config."""

    def __init__(self, cfgs, state_path, reset=False):
        self.cfgs, self.state_path = cfgs, state_path
        models = {cfg['model'] for cfg in cfgs.values()}
        if len(models) != 1:
            raise ValueError('All simulated arms must use the same model.')
        self.fingerprint = hashlib.sha256(''.join(cfgs[k]['_fingerprint'] for k in sorted(cfgs)).encode()).hexdigest()
        self.model = mujoco.MjModel.from_xml_path(str(root() / models.pop()))
        self.data = mujoco.MjData(self.model)
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.qadr, self.vadr = {}, {}
        for key, cfg in cfgs.items():
            ids = [self.model.joint(j['name']).id for j in cfg['joints']]
            self.qadr[key] = [int(self.model.jnt_qposadr[i]) for i in ids]
            self.vadr[key] = [int(self.model.jnt_dofadr[i]) for i in ids]
        if state_path.exists() and not reset:
            state = json.loads(state_path.read_text())
            if state['config'] != self.fingerprint:
                raise ValueError('Simulation configuration changed; run sim --reset.')
            self.data.qpos[:] = state['qpos']
            self.data.qvel[:] = state['qvel']
        mujoco.mj_forward(self.model, self.data)
        # Hold all actuated joints so arms that are not moving do not fall freely.
        self.actuated = []
        for aid in range(self.model.nu):
            if self.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
                jid = int(self.model.actuator_trnid[aid, 0])
                self.actuated.append((aid, int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])))
        self.hold = self.data.qpos.copy()

    def read(self, key):
        return [float(self.data.qpos[i]) for i in self.qadr[key]]

    def status(self, key):
        return [{'joint': j['name'], 'position': p, 'velocity': float(self.data.qvel[v])}
                for j, p, v in zip(self.cfgs[key]['joints'], self.read(key), self.vadr[key])]

    def step(self, targets):
        for key, target in targets.items():
            validate_positions(self.cfgs[key], target)
            self.hold[self.qadr[key]] = target
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

    def move(self, plans, stop=None):
        """Run every arm's trajectory on one shared timeline; plans maps arm key to points."""
        keys = list(plans)
        if len({len(plans[k]) for k in keys}) != 1:
            raise ValueError('Simulated arm trajectories must share one timeline.')
        start_time = self.data.time
        targets = {}
        for samples in zip(*(plans[k] for k in keys)):
            if stop is not None and stop.is_set():
                raise RuntimeError('Stopped.')
            t = samples[0][0]
            targets = {k: q for k, (_, q) in zip(keys, samples)}
            while self.data.time - start_time < t:
                self.step(targets)
        for _ in range(int(0.5 / self.model.opt.timestep)):
            self.step(targets)
        for key, q in targets.items():
            error = max(abs(a - b) for a, b in zip(self.read(key), q))
            print(f'sim {key}: maximum final tracking error {error:.4f} rad')

    def close(self):
        write_json(self.state_path, {'config': self.fingerprint,
                                   'qpos': self.data.qpos.tolist(), 'qvel': self.data.qvel.tolist()})
