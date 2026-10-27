from dataclasses import dataclass
from typing import Callable, Protocol
import time

import numpy as np

from ..execution import ArmCommand, ArmState, MujocoArmInterface, residual_strike_command


class ArmInterface(Protocol):
    def read(self) -> ArmState: ...

    def write(self, command: ArmCommand) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class PolicySettings:
    mode: str = "absolute"
    period_s: float = 0.02
    feedback_timeout_s: float = 0.1
    action_timeout_s: float = 0.1
    max_tracking_error_rad: float = 0.25
    max_velocity_rad_s: float = 1.25
    joint_step_rad: float = 0.025
    pickup_joint_scale_rad: float = 0.06
    pickup_gripper_scale_m: float = 0.004
    pickup_smoothing: float = 0.8
    strike_scale_rad: float = 0.08

    def __post_init__(self):
        if self.mode not in ("absolute", "joint_delta", "pickup_residual", "strike_residual"):
            raise ValueError("Unsupported policy action mode")
        values = np.array([self.period_s, self.feedback_timeout_s, self.action_timeout_s,
                           self.max_tracking_error_rad, self.max_velocity_rad_s, self.joint_step_rad,
                           self.pickup_joint_scale_rad, self.strike_scale_rad], dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Policy timing, scales and limits must be finite and positive")
        if self.period_s != 0.02 or self.joint_step_rad != 0.025:
            raise ValueError("Current policies require a 20 ms period and 0.025 rad training action scale")
        if not np.isfinite([self.pickup_smoothing, self.pickup_gripper_scale_m]).all():
            raise ValueError("Invalid pickup residual configuration")
        if not 0 <= self.pickup_smoothing < 1 or not 0 <= self.pickup_gripper_scale_m <= 0.005:
            raise ValueError("Invalid pickup residual configuration")


class CommandDecoder:
    def __init__(self, settings: PolicySettings, initial: ArmCommand, limits: np.ndarray):
        self.settings = settings
        self.previous = ArmCommand(initial.position.copy(), initial.gripper)
        self.residual = np.zeros(8)
        self.limits = np.asarray(limits, dtype=float)
        if self.limits.shape != (7, 2) or not np.isfinite(self.limits).all() or np.any(self.limits[:, 0] >= self.limits[:, 1]):
            raise ValueError("Policy execution requires seven finite joint limit pairs")

    def decode(self, action: np.ndarray, reference: np.ndarray | None = None) -> ArmCommand:
        action = np.asarray(action, dtype=float)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError("Policy action must contain eight finite values")
        settings = self.settings
        if settings.mode != "absolute" and np.any(np.abs(action) > 1):
            raise ValueError("Normalized policy actions must be in [-1, 1]")
        if settings.mode == "absolute":
            command = ArmCommand(action[:7], float(action[7]))
        elif settings.mode == "joint_delta":
            command = ArmCommand(self.previous.position + settings.joint_step_rad * action[:7],
                                 float((action[7] + 1) * 0.022))
        elif settings.mode == "strike_residual":
            if reference is None:
                raise ValueError("Strike residuals require the current nominal joint reference")
            command = residual_strike_command(reference, self.previous.position, action, settings.strike_scale_rad)
        else:
            reference = np.asarray(reference, dtype=float)
            if reference.shape != (8,) or not np.isfinite(reference).all():
                raise ValueError("Pickup residuals require the eight-dimensional ACT nominal command")
            residual = settings.pickup_smoothing * self.residual + (1 - settings.pickup_smoothing) * action
            target = reference.copy()
            target[:7] += settings.pickup_joint_scale_rad * residual[:7]
            target[7] = np.clip(target[7] + settings.pickup_gripper_scale_m * residual[7], 0, 0.044)
            command = ArmCommand(target[:7], float(target[7]))
            self.residual = residual
        target = np.clip(command.position, self.limits[:, 0], self.limits[:, 1])
        position = self.previous.position + np.clip(target - self.previous.position, -settings.joint_step_rad, settings.joint_step_rad)
        return ArmCommand(position, command.gripper)

    def commit(self, command: ArmCommand) -> None:
        self.previous = ArmCommand(command.position.copy(), command.gripper)


class PolicyAdapter:
    def __init__(self, interface: ArmInterface, limits: np.ndarray, settings: PolicySettings,
                 clock: Callable[[], float] = time.monotonic):
        self.interface, self.settings, self.clock = interface, settings, clock
        self.limits = np.asarray(limits, dtype=float)
        if self.limits.shape != (7, 2) or not np.isfinite(self.limits).all() or np.any(self.limits[:, 0] >= self.limits[:, 1]):
            raise ValueError("Policy execution requires seven finite joint limit pairs")
        self.decoder: CommandDecoder | None = None
        self.entered = False
        self.last_tick: float | None = None

    def read(self) -> ArmState:
        state = self.interface.read()
        if np.shape(state.position) != (7,) or np.shape(state.velocity) != (7,):
            raise ValueError("Policy feedback requires seven joint positions and velocities")
        if not np.isfinite(np.r_[state.position, state.velocity, state.gripper, state.timestamp]).all():
            raise ValueError("Policy feedback must be finite")
        age = self.clock() - state.timestamp
        if not -1e-6 <= age <= self.settings.feedback_timeout_s:
            raise RuntimeError("Policy feedback is stale or uses a different clock")
        if not -1e-8 <= state.gripper <= 0.044 + 1e-8:
            raise ValueError("Measured gripper coordinate exceeds the model limits")
        if np.any(state.position < self.limits[:, 0]) or np.any(state.position > self.limits[:, 1]):
            raise ValueError("Measured joint position exceeds the model limits")
        return state

    def __enter__(self):
        ready = False
        try:
            state = self.read()
            self.decoder = CommandDecoder(self.settings, ArmCommand(state.position, float(np.clip(state.gripper, 0, 0.044))), self.limits)
            self.entered = True
            ready = True
            return self
        finally:
            if not ready:
                self.interface.close()

    def _feedback_for_tick(self, sampled_at: float) -> tuple[float, ArmState]:
        if not self.entered:
            raise RuntimeError("Use the policy adapter as a context manager")
        now = self.clock()
        if not np.isfinite(sampled_at) or not -1e-6 <= now - sampled_at <= self.settings.action_timeout_s:
            raise RuntimeError("Policy action is stale or uses a different clock")
        if self.last_tick is not None:
            interval = now - self.last_tick
            if interval < self.settings.period_s * 0.95 or interval > self.settings.period_s * 1.5:
                raise RuntimeError("Policy command cadence differs from the configured control period")
        state = self.read()
        if np.max(np.abs(state.position - self.decoder.previous.position)) > self.settings.max_tracking_error_rad:
            raise RuntimeError("Policy tracking error exceeds its configured limit")
        return now, state

    def _write(self, command: ArmCommand, now: float, state: ArmState) -> ArmCommand:
        speed = np.max(np.abs(command.position - self.decoder.previous.position)) / self.settings.period_s
        if speed > self.settings.max_velocity_rad_s + 1e-8:
            raise RuntimeError("Policy command exceeds the configured robot speed; review training execution limits")
        if np.max(np.abs(command.position - state.position)) > self.settings.max_tracking_error_rad:
            raise RuntimeError("Policy target exceeds the configured tracking-error limit")
        self.interface.write(command)
        self.decoder.commit(command)
        self.last_tick = now
        return command

    def step(self, action: np.ndarray, sampled_at: float, reference: np.ndarray | None = None) -> ArmCommand:
        now, state = self._feedback_for_tick(sampled_at)
        return self._write(self.decoder.decode(action, reference), now, state)

    def hold(self, sampled_at: float) -> ArmCommand:
        now, state = self._feedback_for_tick(sampled_at)
        return self._write(self.decoder.previous, now, state)

    def __exit__(self, kind, error, traceback):
        self.entered = False
        self.interface.close()


class SimulationPolicyInterface:
    def __init__(self, simulation, arm: str):
        self.interface = MujocoArmInterface(simulation, arm)
        self.closed = False

    def read(self) -> ArmState:
        if self.closed:
            raise RuntimeError("Simulation policy interface is closed")
        return self.interface.read()

    def write(self, command: ArmCommand) -> None:
        if self.closed:
            raise RuntimeError("Simulation policy interface is closed")
        self.interface.write(command)

    def close(self) -> None:
        if not self.closed:
            state = self.read()
            limits = self.interface.sim.model.jnt_range[self.interface.sim.joint_ids[self.interface.arm]]
            self.interface.write(ArmCommand(np.clip(state.position, limits[:, 0], limits[:, 1]),
                                            float(np.clip(state.gripper, 0, 0.044))))
            self.closed = True
