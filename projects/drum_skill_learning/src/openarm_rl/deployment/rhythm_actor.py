import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from ..control.servo import EncoderGravity, extrapolate_encoders
from ..music.reference import PerformanceReference
from ..music.score import ContactScore
from ..planning.strikes import scene_identity
from ..skills.registry import SKILLS
from ..music.reference import profile_hashes


# Tuned strokes end at the wrist limit, and a fast swing passes it by about .004 rad. Commands stay inside the limit.
LIMIT_OVERSHOOT_RAD = .01


class ScoreRuntime:
    """Encoder-checked 100 Hz playback of one compiled score; subclasses choose each joint target."""
    def __init__(self, scene_directory, profiles, midi_file, mapping=None, tempo_scale=1.):
        self.model = mujoco.MjModel.from_binary_path(str(Path(scene_directory) / "model.mjb"))
        metadata = json.loads((Path(scene_directory) / "scene.json").read_text())
        self.identity = scene_identity(metadata["robot"], metadata["scene"])
        self.score = ContactScore.read(midi_file, mapping, tempo_scale)
        self.reference = PerformanceReference(self.model, metadata, self.score, profiles)
        self.controller = self.reference.controller
        self.servo = EncoderGravity(self.model, self.controller.kp)
        surfaces = {surface["name"]: surface for surface in metadata["scene"]["surfaces"]}
        self.centers = np.array([surfaces[target]["center"] for target in SKILLS])
        c = self.controller
        self.limits = {"bounds_rad": c.bounds.tolist(), "velocity_rad_s": (.8*c.velocity).tolist(),
                       "acceleration_rad_s2": [120.]*14, "position_gains": c.kp.tolist(),
                       "velocity_gains": c.kd.tolist(), "feedback_timeout_s": .03, "period_s": .01,
                       "tracking_error_rad": .2}
        self.running = False
        self.trace = []

    def encode(self, q):
        bounds = self.controller.bounds
        return np.clip(2*(q-bounds[:, 0])/(bounds[:, 1]-bounds[:, 0])-1, -.999, .999).astype(np.float32)

    def validate(self, q, dq, sampled_at, now):
        if np.shape(q) != (14,) or np.shape(dq) != (14,) or not np.isfinite([q, dq]).all():
            raise ValueError("Both arms require fourteen finite encoder positions and velocities")
        if not np.isfinite([sampled_at, now]).all() or not 0 <= now-sampled_at <= self.limits["feedback_timeout_s"]+1e-9:
            raise RuntimeError("Rhythm encoder feedback is stale or has an invalid clock")
        if np.any(q < self.controller.bounds[:, 0]-LIMIT_OVERSHOOT_RAD) or np.any(q > self.controller.bounds[:, 1]+LIMIT_OVERSHOOT_RAD):
            raise RuntimeError("Measured rhythm joints exceed model limits")
        if np.any(np.abs(dq) > self.controller.velocity):
            raise RuntimeError("Measured rhythm velocity exceeds model limits")

    def begin(self, q, dq, sampled_at, now):
        q, dq = np.asarray(q), np.asarray(dq)
        self.validate(q, dq, sampled_at, now)
        if self.running:
            raise RuntimeError("A MIDI performance is already running")
        if np.max(np.abs(q-self.reference.initial)) > .03 or np.max(np.abs(dq)) > .08:
            raise ValueError("Move to the compiled ready pose and settle before MIDI playback")
        self.started, self.last_tick, self.history = now, None, []
        self.trace = []
        self.previous_target, self.previous_velocity = self.reference.initial.copy(), np.zeros(14)
        self.anchor_encoder = q.copy()
        self.last_action = self.encode(self.reference.initial)
        self.running = True

    def step(self, q, dq, sampled_at, now):
        if not self.running:
            raise RuntimeError("Begin MIDI playback before requesting commands")
        q, dq = np.asarray(q), np.asarray(dq)
        self.validate(q, dq, sampled_at, now)
        measured = q.copy()
        age = now-sampled_at
        q = extrapolate_encoders(q, dq, age)
        if self.last_tick is not None and not 0 < now-self.last_tick <= .05:
            raise RuntimeError("MIDI policy control deadline missed")
        elapsed = now-self.started
        if elapsed < 0:
            raise ValueError("Playback time cannot precede start")
        if elapsed >= self.reference.duration:
            self.running = False
            return None
        target, rate, motor, velocity, action = self.plan(q, dq, elapsed)
        c = self.controller
        self.trace.append({"time_s": elapsed, "feedback_age_s": age, "measured_rad": measured.tolist(),
                           "estimated_current_rad": q.tolist(),
                           "target_rad": target.tolist(), "action": action.tolist()})
        if np.max(np.abs(target-q)) + .5*120*age**2 > .2:
            joint = int(np.argmax(np.abs(target-q)))
            raise RuntimeError(f"MIDI tracking error exceeds the trained limit: joint={joint}, time_s={elapsed:.3f}, target={target[joint]:.4f}, measured={q[joint]:.4f}")
        if np.any(motor < c.bounds[:, 0]) or np.any(motor > c.bounds[:, 1]):
            raise RuntimeError("MIDI motor target exceeds model limits")
        self.previous_target, self.previous_velocity = target, rate
        self.last_action, self.last_tick = action.copy(), now
        return motor, velocity

    def cancel(self):
        self.running = False


class ReferenceRuntime(ScoreRuntime):
    """Plays the compiled reference with no learned policy, so every command is known before the robot moves."""
    @classmethod
    def from_rehearsal(cls, path):
        report = json.loads(Path(path).read_text())
        if report["schema"] != "openarm.midi_rehearsal.v1" or not report["passed"]:
            raise ValueError("MIDI rehearsal did not pass")
        files = report["inputs"]
        if hashlib.sha256(Path(files["midi"]).read_bytes()).hexdigest() != report["midi_sha256"]:
            raise ValueError("MIDI differs from its rehearsal")
        if hashlib.sha256(Path(files["mapping"]).read_bytes()).hexdigest() != report["mapping_sha256"]:
            raise ValueError("Percussion mapping differs from its rehearsal")
        if hashlib.sha256((Path(files["scene_directory"])/"model.mjb").read_bytes()).hexdigest() != report["model_sha256"]:
            raise ValueError("Robot model differs from its rehearsal")
        if profile_hashes(Path(files["profiles"])) != report["profiles_sha256"]:
            raise ValueError("Reference profiles differ from their rehearsal")
        runtime = cls(files["scene_directory"], files["profiles"], files["midi"], files["mapping"], files["tempo_scale"])
        if runtime.identity != report["scene_identity"]:
            raise ValueError("Scene identity differs from its rehearsal")
        return runtime

    def plan(self, q, dq, elapsed):
        c = self.controller
        ratio = c.kd/c.kp
        _, velocity = self.reference.sample(elapsed)
        base = self.reference.motor_target(elapsed, q, self.servo)-self.servo.position_target(np.zeros(14), q)+ratio*velocity
        rate = np.clip((base-self.previous_target)/(.01+ratio), -.8*c.velocity, .8*c.velocity)
        rate = np.clip(rate, self.previous_velocity-1.2, self.previous_velocity+1.2)
        target = np.clip(self.previous_target+rate*.01, c.bounds[:, 0], c.bounds[:, 1])
        # An idle arm rests on its joint stop, where the gravity offset alone would point past the limit.
        motor = np.clip(self.servo.position_target(target, q), c.bounds[:, 0], c.bounds[:, 1])
        return target, rate, motor, rate, self.encode(target)


class RhythmActorRuntime(ScoreRuntime):
    def __init__(self, actor_file, scene_directory, profiles, midi_file, mapping=None, tempo_scale=1.):
        actor_file = Path(actor_file)
        manifest = json.loads((actor_file.parent / "bundle.json").read_text())
        if manifest["schema"] not in ("openarm.rhythm_actor.v1", "openarm.score_actor.v3") or manifest["action_dim"] != 14 or manifest["control_hz"] != 100:
            raise ValueError("Unsupported MIDI actor bundle schema")
        self.score_only = manifest["schema"] == "openarm.score_actor.v3"
        if self.score_only and (manifest["actor_trajectory_inputs"] or manifest["motor_reference"]):
            raise ValueError("Score actor bundle must declare encoder-only motor control")
        if self.score_only and "reference_profiles_sha256" not in manifest:
            raise ValueError("Re-export the actor bundle with reference profile identities")
        if "reference_profiles_sha256" in manifest and profile_hashes(Path(profiles)) != manifest["reference_profiles_sha256"]:
            raise ValueError("MIDI actor reference profiles differ from training")
        if hashlib.sha256(actor_file.read_bytes()).hexdigest() != manifest["sha256"]["actor.pt"]:
            raise ValueError("MIDI actor checksum differs from its bundle")
        if hashlib.sha256((Path(scene_directory)/"model.mjb").read_bytes()).hexdigest() != manifest["model_sha256"]:
            raise ValueError("MIDI actor nominal model differs from its training model")
        self.actor = torch.jit.load(str(actor_file), map_location="cpu").eval()
        torch.set_num_threads(1)
        super().__init__(scene_directory, profiles, midi_file, mapping, tempo_scale)
        if self.identity != manifest["scene_identity"]:
            raise ValueError("MIDI actor belongs to a different registered scene")
        if self.score_only and any(goal.target not in manifest["trained_drums"] for goal in self.score.goals):
            raise ValueError("MIDI requests a drum absent from this actor's training corpus")
        actor_dim = 269 if self.score_only else 297
        if int(self.actor.mean.numel()) != actor_dim:
            raise ValueError("Rhythm actor observation schema differs from this runtime")
        with torch.no_grad():
            self.actor(torch.zeros((1, actor_dim)), torch.zeros((1, 14)))

    def plan(self, q, dq, elapsed):
        row = np.r_[q, dq, self.last_action].astype(np.float32)
        self.history = (self.history+[row])[-4:] if self.history else [row.copy() for _ in range(4)]
        if self.score_only:
            trajectory, nominal_q = np.empty(0), self.anchor_encoder
        else:
            reference, velocity = self.reference.sample(elapsed)
            trajectory, nominal_q = np.r_[reference, velocity], reference
        state = np.r_[np.concatenate(self.history), trajectory, self.centers.ravel(),
                      self.score.future(elapsed-self.reference.preroll)].astype(np.float32)
        nominal = np.arctanh(self.encode(nominal_q)).astype(np.float32)
        with torch.no_grad():
            action = self.actor(torch.from_numpy(state)[None], torch.from_numpy(nominal)[None]).numpy()[0]
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("MIDI actor returned invalid joint targets")
        c = self.controller
        desired = c.bounds[:, 0]+(action+1)*.5*(c.bounds[:, 1]-c.bounds[:, 0])
        rate = np.clip((desired-self.previous_target)/.01, -.8*c.velocity, .8*c.velocity)
        rate = np.clip(rate, self.previous_velocity-1.2, self.previous_velocity+1.2)
        target = np.clip(self.previous_target+rate*.01, c.bounds[:, 0], c.bounds[:, 1])
        if self.score_only:
            motor, velocity = self.servo.position_target(target, q), rate
        else:
            motor = self.reference.motor_target(elapsed, q, self.servo)+target-reference
        return target, rate, motor, velocity, action
