from copy import copy

import mujoco
import numpy as np

from openarm_drummer.model import JOINTS


def extrapolate_encoders(positions, velocities, age, max_age=.03):
    positions, velocities = np.asarray(positions), np.asarray(velocities)
    if positions.shape != (14,) or velocities.shape != (14,) or not np.isfinite([positions, velocities]).all():
        raise ValueError("Encoder prediction requires fourteen finite positions and velocities")
    if not np.isfinite(age) or not 0 <= age <= max_age+1e-9:
        raise RuntimeError("Encoder sample exceeds the prediction horizon")
    return positions+velocities*age


class EncoderGravity:
    def __init__(self, model, kp):
        self.model = copy(model)
        self.data = mujoco.MjData(self.model)
        self.q = np.array([self.model.jnt_qposadr[self.model.joint(name).id] for name in JOINTS])
        self.v = np.array([self.model.jnt_dofadr[self.model.joint(name).id] for name in JOINTS])
        self.kp = np.asarray(kp, dtype=float)
        if self.kp.shape != (14,) or not np.isfinite(self.kp).all() or np.any(self.kp <= 0):
            raise ValueError("Nominal position gains must be fourteen positive values")

    def position_target(self, desired, measured):
        values = np.asarray(measured, dtype=float)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError("Gravity computation requires fourteen measured encoders")
        self.data.qpos[self.q] = values
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return np.asarray(desired, dtype=float) + self.data.qfrc_bias[self.v] / self.kp
