import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from ..control.reference import EncoderHistory, JointReference, ResidualCommand, SpeedConditionedReference
from ..control.servo import EncoderGravity
from ..envs.camera_free_strike import REWARD_VERSION
from ..skills.registry import SKILLS


class DrumActor:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "skill.json").read_text())
        m = self.manifest
        if m["schema"] != "openarm.camera_free_skill.v1" or m["reward_version"] != REWARD_VERSION:
            raise ValueError("Unsupported camera-free skill version")
        self.skill = SKILLS[m["target"]]
        self.action_semantics = m.get("action_semantics", "wrist_residual")
        if self.action_semantics not in ("wrist_residual", "absolute_wrist_position_velocity"):
            raise ValueError("Unsupported deployed strike action semantics")
        self.action_dim = 2 if self.action_semantics == "absolute_wrist_position_velocity" else 1
        if m["action_dim"] != self.action_dim:
            raise ValueError("Actor action dimensions differ from its command semantics")
        if (m["arm"], m["tool"]) != (self.skill.arm, self.skill.name) or m["optimization_updates"] <= 0:
            raise ValueError("A trained matching drum/arm actor is required")
        for name, expected in m["sha256"].items():
            if Path(name).name != name:
                raise ValueError("Invalid bundle filename")
            if hashlib.sha256((self.directory / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Bundle checksum differs: {name}")
        self.actor = torch.jit.load(str(self.directory / "actor.pt"), map_location="cpu").eval()
        torch.set_num_threads(1)
        with torch.no_grad():
            for _ in range(3):
                self.actor(torch.zeros((1, m["actor_dim"])))
        self.reference = JointReference.load(json.loads((self.directory / "reference.json").read_text()))
        profile = json.loads((self.directory / "profile.json").read_text())
        self.reference_bank = SpeedConditionedReference(profile["reference_bank"]) if "reference_bank" in profile else None
        self.limits = json.loads((self.directory / "limits.json").read_text())
        self.model = mujoco.MjModel.from_binary_path(str(self.directory / "gravity_model.mjb"))
        self.servo = EncoderGravity(self.model, self.limits["position_gains"])
        if self.reference.wrist != self.skill.wrist or m["actor_dim"] != m["history_frames"] * (58 + self.action_dim):
            raise ValueError("Bundle reference or observation dimensions differ")
        self.running = False

    def begin(self, q, dq, sampled_at, now, target_speed=.6):
        q, dq = np.asarray(q), np.asarray(dq)
        self.validate_feedback(q, dq, sampled_at, now)
        if self.running:
            raise RuntimeError("Skill is already running")
        if not np.isfinite(target_speed) or not .35 <= target_speed <= .85:
            raise ValueError("Requested speed is outside the trained range")
        if self.reference_bank is not None:
            self.reference = self.reference_bank.at(target_speed)
        if np.max(np.abs(q - self.reference.positions[0])) > .03 or np.max(np.abs(dq)) > .08:
            raise ValueError("Move to the fixed ready pose and settle before starting the skill")
        self.history = EncoderHistory(self.manifest["history_frames"], self.action_dim)
        self.command = ResidualCommand(self.limits["bounds_rad"], self.limits["velocity_rad_s"], self.limits["acceleration_rad_s2"],
                                       self.limits["period_s"], self.limits["scale_rad"], self.limits["slew_rad"])
        self.previous_action = np.zeros(self.action_dim, np.float32)
        self.output_velocity = np.zeros(14)
        if self.action_semantics == "absolute_wrist_position_velocity":
            bounds = np.asarray(self.limits["bounds_rad"])
            self.command.previous = np.clip(self.servo.position_target(q, q), bounds[:, 0], bounds[:, 1])
            self.command.previous_velocity = np.zeros(14)
            lower, upper = bounds[self.skill.wrist]
            self.previous_action[0] = float(np.clip(2*(self.command.previous[self.skill.wrist]-lower)/(upper-lower)-1, -1, 1))
        self.started, self.last_tick, self.speed = now, None, target_speed
        self.running = True

    def validate_feedback(self, q, dq, sampled_at, now):
        if q.shape != (14,) or dq.shape != (14,) or not np.isfinite([q, dq]).all():
            raise ValueError("Both arms require fourteen finite encoder positions/velocities")
        if not np.isfinite([sampled_at, now]).all() or not 0 <= now - sampled_at <= self.limits["feedback_timeout_s"]:
            raise RuntimeError("Encoder feedback is stale or has an invalid clock")
        bounds = np.array(self.limits["bounds_rad"])
        if np.any(q < bounds[:, 0] - .001) or np.any(q > bounds[:, 1] + .001):
            raise RuntimeError("Measured joints exceed trained limits")
        if np.any(np.abs(dq) > np.array(self.limits["velocity_rad_s"]) / .8):
            raise RuntimeError("Measured joint speed exceeds model limits")

    def step(self, q, dq, sampled_at, now):
        if not self.running:
            raise RuntimeError("Begin the skill before requesting commands")
        q, dq = np.asarray(q), np.asarray(dq)
        self.validate_feedback(q, dq, sampled_at, now)
        if self.last_tick is not None and not 0 < now - self.last_tick <= .05:
            raise RuntimeError(f"Policy control deadline missed: period={now-self.last_tick:.3f}s, elapsed={now-self.started:.3f}s")
        elapsed = now - self.started
        if elapsed < 0:
            raise ValueError("Control time cannot precede skill start")
        if elapsed >= self.reference.duration:
            self.running = False
            return None
        ref, velocity = self.reference.sample(elapsed)
        observation = self.history.observe(q, dq, ref, velocity, self.previous_action, elapsed, self.speed)
        with torch.no_grad():
            action = self.actor(torch.from_numpy(observation).unsqueeze(0)).numpy()[0]
        motor = self.reference.motor_target(elapsed, q, self.servo)
        if self.action_semantics == "absolute_wrist_position_velocity":
            lower, upper = self.limits["bounds_rad"][self.skill.wrist]
            motor[self.skill.wrist] = lower + (float(action[0]) + 1)*.5*(upper-lower)
            residual = np.zeros(1)
        else:
            residual = action * self.reference.residual_weight(elapsed)
        command = self.command.step(motor, residual, self.skill.wrist)
        if self.action_semantics == "absolute_wrist_position_velocity":
            desired_velocity = np.zeros(14)
            desired_velocity[self.skill.wrist] = float(action[1]) * self.limits["velocity_rad_s"][self.skill.wrist]
            acceleration = np.asarray(self.limits["acceleration_rad_s2"])
            self.output_velocity = np.clip(desired_velocity, self.output_velocity - acceleration*self.limits["period_s"],
                                            self.output_velocity + acceleration*self.limits["period_s"])
        else:
            self.output_velocity = self.command.previous_velocity.copy()
        if np.max(np.abs(command - q)) > self.limits["tracking_error_rad"]:
            raise RuntimeError("Encoder tracking error exceeds trained limit")
        self.previous_action, self.last_tick = action.copy(), now
        return command

    def cancel(self):
        self.running = False
