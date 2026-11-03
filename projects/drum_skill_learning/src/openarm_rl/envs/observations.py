from dataclasses import dataclass

import numpy as np


@dataclass
class DrumObservation:
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    gripper_position: float
    gripper_velocity: float
    tip_position: np.ndarray
    tip_velocity: np.ndarray
    stick_quaternion_wxyz: np.ndarray
    stick_position: np.ndarray
    tcp_position: np.ndarray
    drum_surface_position: np.ndarray
    time_to_hit: float
    target_speed: float
    previous_action: np.ndarray

    def vector(self) -> np.ndarray:
        shapes = {
            "joint_position": (7,), "joint_velocity": (7,), "tip_position": (3,),
            "tip_velocity": (3,), "stick_quaternion_wxyz": (4,), "stick_position": (3,),
            "tcp_position": (3,), "drum_surface_position": (3,), "previous_action": (8,),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid observation field: {name}")
            setattr(self, name, value)
        if abs(np.linalg.norm(self.stick_quaternion_wxyz) - 1) > 0.01:
            raise ValueError("Stick orientation must be a normalized wxyz quaternion")
        result = np.concatenate([
            self.joint_position, self.joint_velocity, [self.gripper_position, self.gripper_velocity],
            self.tip_position - self.drum_surface_position, self.tip_velocity, self.stick_quaternion_wxyz,
            self.stick_position - self.tcp_position, [self.time_to_hit, self.target_speed], self.previous_action,
        ]).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Observation contains non-finite values")
        return result


def residual_observation(base: np.ndarray, reference_joints, reference_tip, actual_tip, phase: float, stage: int):
    result = np.concatenate([base, reference_joints, np.asarray(reference_tip) - actual_tip, [phase, stage / 2.0]]).astype(np.float32)
    if result.shape != (51,) or not np.isfinite(result).all():
        raise ValueError("Residual strike observation must contain 51 finite values")
    return result
