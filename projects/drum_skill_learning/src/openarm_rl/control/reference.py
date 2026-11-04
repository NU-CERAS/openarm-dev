from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicHermiteSpline


@dataclass
class JointReference:
    times: np.ndarray
    positions: np.ndarray
    wrist: int
    impact_time: float
    contact_multiplier: float = .45
    compliance_seconds: float = .08
    velocities: np.ndarray | None = None
    active_joints: tuple[int, ...] | None = None

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.positions = np.asarray(self.positions, dtype=float)
        if self.times.ndim != 1 or self.positions.shape != (len(self.times), 14) or self.times[0] != 0:
            raise ValueError("Reference requires time-indexed fourteen-joint positions")
        if not np.isfinite(self.times).all() or not np.isfinite(self.positions).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError("Reference values must be finite and times strictly increasing")
        if self.wrist not in (6, 13) or not 0 < self.impact_time < self.duration:
            raise ValueError("Invalid wrist or reference impact time")
        self.active_joints = (self.wrist,) if self.active_joints is None else tuple(self.active_joints)
        arm = set(range(7)) if self.wrist == 6 else set(range(7, 14))
        if not self.active_joints or self.wrist not in self.active_joints or not set(self.active_joints) <= arm:
            raise ValueError("Strike reference active joints must belong to its stick arm")
        held = np.array(sorted(set(range(14))-set(self.active_joints)), dtype=int)
        if not np.allclose(self.positions[:, held], self.positions[0, held], atol=1e-8):
            raise ValueError("Strike reference must hold inactive joints")
        if self.velocities is None:
            self.velocities = np.zeros_like(self.positions)
        self.velocities = np.asarray(self.velocities, dtype=float)
        if self.velocities.shape != self.positions.shape or not np.isfinite(self.velocities).all():
            raise ValueError("Reference velocities must match the position knots")
        if not np.allclose(self.velocities[:, held], 0) or not np.allclose(self.velocities[[0, -1]], 0):
            raise ValueError("Reference must hold other joints and settle at both endpoints")
        self.spline = CubicHermiteSpline(self.times, self.positions, self.velocities)

    @property
    def duration(self):
        return float(self.times[-1])

    def sample(self, elapsed):
        if not np.isfinite(elapsed):
            raise ValueError("Elapsed time must be finite")
        value = np.clip(elapsed, 0, self.duration)
        return self.spline(value), self.spline(value, 1)

    def document(self):
        return {"schema": "openarm.timed_strike.v1" if len(self.active_joints) == 1 else "openarm.timed_strike.v2",
                "active_joints": list(self.active_joints), "times_s": self.times.tolist(),
                "positions_rad": self.positions.tolist(), "wrist": self.wrist, "impact_time_s": self.impact_time,
                "contact_multiplier": self.contact_multiplier, "compliance_seconds": self.compliance_seconds,
                "velocities_rad_s": self.velocities.tolist()}

    def motor_target(self, elapsed, measured, servo):
        if len(self.active_joints) != 1:
            raise ValueError("Full-arm strike references require encoder-only joint command execution")
        desired, _ = self.sample(elapsed)
        motor = servo.position_target(desired, measured)
        if self.times[3] <= elapsed < self.times[4] + self.compliance_seconds:
            motor[self.wrist] = measured[self.wrist] + self.contact_multiplier * (motor[self.wrist] - measured[self.wrist])
        return motor

    def residual_weight(self, elapsed):
        if elapsed < self.times[1] or elapsed >= self.times[5]:
            return 0.0
        if elapsed <= self.times[4]:
            return 1.0
        phase = (elapsed - self.times[4]) / (self.times[5] - self.times[4])
        return float(1 - phase * phase * (3 - 2 * phase))

    @classmethod
    def load(cls, document):
        if document["schema"] not in ("openarm.timed_strike.v1", "openarm.timed_strike.v2"):
            raise ValueError("Unknown timed reference schema")
        return cls(document["times_s"], document["positions_rad"], document["wrist"], document["impact_time_s"],
                   document["contact_multiplier"], document["compliance_seconds"], document.get("velocities_rad_s"),
                   tuple(document["active_joints"]) if "active_joints" in document else None)

    @classmethod
    def from_plan(cls, plan, cfg, wrist):
        times = np.cumsum([0, cfg["ready_seconds"], cfg["approach_seconds"], cfg["downstroke_seconds"],
                           .02, cfg["retract_seconds"], cfg["recovery_seconds"]])
        fraction = (plan["impact_pose"][wrist] - plan["approach"][wrist]) / (plan["bottom"][wrist] - plan["approach"][wrist])
        from scipy.optimize import brentq

        crossing = brentq(lambda x: x * x * (3 - 2 * x) - fraction, 0, 1)
        return cls(times, [plan["ready"], plan["ready"], plan["approach"], plan["bottom"], plan["bottom"], plan["ready"], plan["ready"]],
                   wrist, times[2] + crossing * cfg["downstroke_seconds"], cfg["contact_kp_multiplier"], cfg["compliance_seconds"])


class EncoderHistory:
    frame_dim = 59

    def __init__(self, frames=8, action_dim=1):
        if type(frames) is not int or frames < 1:
            raise ValueError("History frame count must be positive")
        self.frames = frames
        if action_dim not in (1, 2):
            raise ValueError("Encoder history supports one or two control components")
        self.action_dim, self.frame_dim = action_dim, 58 + action_dim
        self.rows = []

    def observe(self, q, dq, reference_q, reference_dq, previous_action, elapsed, speed):
        previous_action = np.asarray(previous_action).reshape(-1)
        if len(previous_action) != self.action_dim:
            raise ValueError("Previous action dimensions differ from the history")
        values = np.concatenate((q, dq, reference_q, reference_dq, previous_action, [elapsed, speed])).astype(np.float32)
        if values.shape != (self.frame_dim,) or not np.isfinite(values).all():
            raise ValueError("Encoder observation requires finite measured joints and reference values")
        if not self.rows:
            self.rows = [values.copy() for _ in range(self.frames)]
        else:
            self.rows = [*self.rows[1:], values]
        return np.concatenate(self.rows)


class SpeedConditionedReference:
    def __init__(self, document):
        if document["schema"] != "openarm.speed_conditioned_strike.v1":
            raise ValueError("Unsupported speed-conditioned reference")
        self.speeds = np.asarray(document["speeds_m_s"], dtype=float)
        self.references = [JointReference.load(row) for row in document["references"]]
        if len(self.references) != len(self.speeds) or len(self.speeds) < 2 or np.any(np.diff(self.speeds) <= 0):
            raise ValueError("Reference speeds must increase strictly")
        first = self.references[0]
        for reference in self.references:
            if (reference.wrist != first.wrist or reference.active_joints != first.active_joints
                    or not np.allclose(reference.positions[0], first.positions[0], atol=1e-8)):
                raise ValueError("Speed references must share the arm and ready pose")

    def at(self, speed):
        if not np.isfinite(speed) or not self.speeds[0] <= speed <= self.speeds[-1]:
            raise ValueError("Requested speed is outside the reference bank")
        index = min(len(self.speeds) - 2, max(0, int(np.searchsorted(self.speeds, speed)) - 1))
        weight = float((speed - self.speeds[index]) / (self.speeds[index + 1] - self.speeds[index]))
        first, second = self.references[index:index + 2]
        blend = lambda a, b: (1 - weight) * a + weight * b
        return JointReference(blend(first.times, second.times), blend(first.positions, second.positions), first.wrist,
                              blend(first.impact_time, second.impact_time), blend(first.contact_multiplier, second.contact_multiplier),
                              blend(first.compliance_seconds, second.compliance_seconds), blend(first.velocities, second.velocities),
                              first.active_joints)


class ResidualCommand:
    def __init__(self, bounds, velocity, acceleration, period=.01, scale=.08, slew=.015):
        self.bounds, self.velocity, self.acceleration = map(np.asarray, (bounds, velocity, acceleration))
        self.period, self.scale, self.slew = period, scale, slew
        self.offset = 0.0
        self.previous = self.previous_velocity = None

    def step(self, reference, action, wrist):
        value = np.asarray(action, dtype=float)
        if value.shape != (1,) or not np.isfinite(value).all() or abs(value[0]) > 1:
            raise ValueError("Residual action must contain one finite value in [-1,1]")
        self.offset += float(np.clip(self.scale * value[0] - self.offset, -self.slew, self.slew))
        desired = np.asarray(reference, dtype=float).copy()
        desired[wrist] += self.offset
        if desired.shape != (14,) or not np.isfinite(desired).all():
            raise ValueError("Joint command requires fourteen finite positions")
        if self.previous is None:
            self.previous = np.asarray(reference, dtype=float).copy()
            self.previous_velocity = np.zeros(14)
        velocity = np.clip((desired - self.previous) / self.period, -self.velocity, self.velocity)
        velocity = np.clip(velocity, self.previous_velocity - self.acceleration * self.period,
                           self.previous_velocity + self.acceleration * self.period)
        desired = np.clip(self.previous + velocity * self.period, self.bounds[:, 0], self.bounds[:, 1])
        self.previous_velocity = (desired - self.previous) / self.period
        self.previous = desired.copy()
        return desired
