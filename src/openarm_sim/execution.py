from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicHermiteSpline


def joint_names(arm: str) -> list[str]:
    if arm not in ("left", "right"):
        raise ValueError("arm must be left or right")
    return [f"openarm_{arm}_joint{index}" for index in range(1, 8)]


@dataclass
class ArmState:
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    gripper: float
    timestamp: float


@dataclass
class ArmCommand:
    position: NDArray[np.float64]
    gripper: float

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        if self.position.shape != (7,) or not np.isfinite(self.position).all():
            raise ValueError("Arm command requires seven finite joint positions in radians")
        if not np.isfinite(self.gripper) or not 0 <= self.gripper <= 0.044:
            raise ValueError("Gripper command must be in [0, 0.044] meters")

    def controller_messages(self, arm: str) -> dict[str, list[float]]:
        joint_names(arm)
        return {
            f"/{arm}_forward_position_controller/commands": self.position.tolist(),
            f"/{arm}_gripper_controller/commands": [float(self.gripper)],
        }


@dataclass
class JointTrajectory:
    arm: str
    times: NDArray[np.float64]
    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]
    source: str

    def __post_init__(self):
        joint_names(self.arm)
        self.times = np.asarray(self.times, dtype=float)
        self.positions = np.asarray(self.positions, dtype=float)
        self.velocities = np.asarray(self.velocities, dtype=float)
        if self.times.ndim != 1 or len(self.times) < 2 or self.times[0] != 0 or np.any(np.diff(self.times) <= 0):
            raise ValueError("Trajectory times must start at zero and increase strictly")
        if self.positions.shape != (len(self.times), 7) or self.velocities.shape != self.positions.shape:
            raise ValueError("Trajectory positions and velocities must have shape (N, 7)")
        if not all(np.isfinite(value).all() for value in (self.times, self.positions, self.velocities)):
            raise ValueError("Trajectory values must be finite")
        self._spline = CubicHermiteSpline(self.times, self.positions, self.velocities)

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    def sample(self, time_s: float) -> NDArray[np.float64]:
        return self._spline(np.clip(time_s, 0, self.duration))

    def retime(self, max_velocity: float = 0.6, max_acceleration: float = 1.5):
        if max_velocity <= 0 or max_acceleration <= 0:
            raise ValueError("Trajectory limits must be positive")
        sample_times = np.linspace(0, self.duration, max(1000, len(self.times) * 10))
        speed = float(np.max(np.abs(self._spline(sample_times, 1))))
        acceleration = float(np.max(np.abs(self._spline(sample_times, 2))))
        scale = max(1.0, speed / max_velocity, np.sqrt(acceleration / max_acceleration)) * 1.01
        return JointTrajectory(self.arm, self.times * scale, self.positions, self.velocities / scale, self.source)

    def to_dict(self, gripper: float) -> dict:
        ArmCommand(self.positions[0], gripper)
        return {
            "schema": "openarm.joint_trajectory.v1", "arm": self.arm,
            "joint_names": joint_names(self.arm), "position_unit": "rad",
            "time_unit": "s", "gripper_unit": "m", "gripper": gripper,
            "times": self.times.tolist(), "positions": self.positions.tolist(),
            "velocities": self.velocities.tolist(), "source": self.source,
        }


class MujocoArmInterface:
    def __init__(self, simulation, arm: str):
        joint_names(arm)
        self.sim = simulation
        self.arm = arm
        self.grip_id = self.sim.model.actuator(f"{arm}_finger1_ctrl").id
        self.grip_joint = self.sim.model.joint(f"openarm_{arm}_finger_joint1").id

    def read(self) -> ArmState:
        model, data = self.sim.model, self.sim.data
        return ArmState(
            data.qpos[self.sim.qpos_ids[self.arm]].copy(),
            data.qvel[model.jnt_dofadr[self.sim.joint_ids[self.arm]]].copy(),
            float(data.qpos[model.jnt_qposadr[self.grip_joint]]), float(data.time),
        )

    def write(self, command: ArmCommand) -> None:
        limits = self.sim.model.jnt_range[self.sim.joint_ids[self.arm]]
        if np.any(command.position < limits[:, 0]) or np.any(command.position > limits[:, 1]):
            raise ValueError("Arm command exceeds model joint limits")
        self.sim._desired[self.sim.actuator_ids[self.arm]] = command.position
        self.sim._desired[self.grip_id] = command.gripper


def residual_strike_command(reference, previous_target, action, residual_scale: float = 0.08) -> ArmCommand:
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (8,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
        raise ValueError("Residual action must contain eight finite values in [-1, 1]")
    previous_target = np.asarray(previous_target, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if previous_target.shape != (7,) or reference.shape != (7,):
        raise ValueError("Reference and previous joint target must contain seven positions")
    desired = reference + residual_scale * action[:7]
    position = previous_target + np.clip(desired - previous_target, -0.025, 0.025)
    gripper = 0.0066 * (float(action[7]) + 1) / 2
    return ArmCommand(position, gripper)
