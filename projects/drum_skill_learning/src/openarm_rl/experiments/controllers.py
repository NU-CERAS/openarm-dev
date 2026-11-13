"""Reference, Jacobian IK, MuJoCo rollout MPC and a behavior-cloned policy.

All controllers see the same synchronous encoder positions/velocities. They use
the immutable nominal model, never the randomized evaluation plant. This is an
offline, reference-assisted benchmark, not a robot deployment controller.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np

from openarm_drummer.model import HoldController
from openarm_drummer.strike import STRIKE_CONFIG

from ..control.servo import EncoderGravity
from ..skills.registry import SKILLS
from .models import policy_features


@dataclass
class Command:
    motor: np.ndarray
    velocity: np.ndarray
    target: np.ndarray


class NominalPhysics:
    """Native MuJoCo forward simulation loaded separately from the plant.

    Only 14 encoder positions/velocities initialize a rollout. In particular,
    copying MjData from the evaluation environment would leak contact state and
    randomized parameters; this class does neither.
    """

    def __init__(self, scene_directory, metadata, period=.01):
        self.model = mujoco.MjModel.from_binary_path(str(Path(scene_directory)/"model.mjb"))
        self.data = mujoco.MjData(self.model)
        self.controller = HoldController(self.model, metadata["robot"])
        self.period = period
        self.substeps = round(period/self.model.opt.timestep)
        if not np.isclose(self.substeps*self.model.opt.timestep, period):
            raise ValueError("Control period must be a multiple of the MuJoCo timestep")
        self.servo = EncoderGravity(self.model, self.controller.kp)
        self.tip_sites = [self.model.site(f"{arm}_stick_tip").id for arm in ("left", "right")]
        self.tip_arms, self.head_targets = np.full(self.model.ngeom, -1), np.full(self.model.ngeom, -1)
        for arm_index, arm in enumerate(("left", "right")):
            for part in ("tip", "taper", "shaft"):
                self.tip_arms[self.model.geom(f"{arm}_stick_{part}_collision").id] = arm_index
        for index, target in enumerate(SKILLS):
            self.head_targets[self.model.geom(f"{target}_head_collision").id] = index
        self.safety = json.loads(STRIKE_CONFIG.read_text())
        self.forward_evaluations = 0

    def contact_safety(self):
        """Apply the plant's per-contact and summed arm/drum stop conditions."""
        forces, peak, unsafe = np.zeros(2*len(SKILLS)), 0., False
        wrench = np.zeros(6)
        for index, contact in enumerate(self.data.contact):
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            force = max(0., float(wrench[0]))
            arm = max(self.tip_arms[contact.geom1], self.tip_arms[contact.geom2])
            target = max(self.head_targets[contact.geom1], self.head_targets[contact.geom2])
            if arm >= 0 and target >= 0:
                forces[arm*len(SKILLS)+target] += force
                peak = max(peak, force)
                unsafe |= force > self.safety["force_stop_N"] or -contact.dist > self.safety["penetration_stop_m"]
            elif force > .1:
                unsafe = True
        peak = max(peak, float(np.max(forces)))
        unsafe |= peak > self.safety["force_stop_N"]
        return peak, bool(unsafe)

    def set_state(self, q, dq):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.controller.q] = q
        self.data.qvel[self.controller.v] = dq
        mujoco.mj_forward(self.model, self.data)

    def advance(self, command, residual=None):
        c, d, m = self.controller, self.data, self.model
        previous_q, previous_dq = d.qpos[c.q].copy(), d.qvel[c.v].copy()
        peak, unsafe = 0., False
        for _ in range(self.substeps):
            torque = c.kp*(command.motor-d.qpos[c.q])+c.kd*(command.velocity-d.qvel[c.v])
            d.ctrl[c.a] = np.clip(torque, -c.effort, c.effort)
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            force, unsafe_contact = self.contact_safety()
            peak = max(peak, force)
            unsafe |= unsafe_contact or np.max(np.abs(command.target-d.qpos[c.q])) > .2
        self.forward_evaluations += 1
        if residual is not None:
            correction = residual.correction(previous_q, previous_dq, command.motor, command.velocity)
            d.qpos[c.q] += correction[:14]
            d.qvel[c.v] += correction[14:]
            mujoco.mj_forward(m, d)
            corrected_peak, corrected_unsafe = self.contact_safety()
            peak = max(peak, corrected_peak)
            unsafe |= corrected_unsafe or np.max(np.abs(command.target-d.qpos[c.q])) > .2
        return d.qpos[c.q].copy(), d.qvel[c.v].copy(), peak, bool(unsafe)

    def predict(self, q, dq, command, residual=None):
        self.set_state(q, dq)
        return self.advance(command, residual)


class ReferenceController:
    description = "Compiled stroke reference with gravity compensation and bounded joint commands"
    reference_assistance = "Full reference trajectory at inference"

    def __init__(self, reference, physics):
        self.reference, self.physics = reference, physics
        self.c, self.period = physics.controller, physics.period
        self.previous_target = reference.initial.copy()
        self.previous_velocity = np.zeros(14)
        self.anchor = reference.initial.copy()

    def bounded(self, desired_velocity, previous_target, previous_velocity, q):
        c = self.c
        velocity = np.clip(desired_velocity, -.8*c.velocity, .8*c.velocity)
        velocity = np.clip(velocity, previous_velocity-120*self.period, previous_velocity+120*self.period)
        target = np.clip(previous_target+velocity*self.period, c.bounds[:, 0], c.bounds[:, 1])
        velocity = (target-previous_target)/self.period
        motor = np.clip(self.physics.servo.position_target(target, q), c.bounds[:, 0], c.bounds[:, 1])
        return Command(motor, velocity, target)

    def reference_command(self, q, time_s, previous_target, previous_velocity):
        ratio = self.c.kd/self.c.kp
        _, velocity = self.reference.sample(time_s)
        gravity = self.physics.servo.position_target(np.zeros(14), q)
        base = self.reference.motor_target(time_s, q, self.physics.servo)-gravity+ratio*velocity
        return self.bounded((base-previous_target)/(self.period+ratio), previous_target, previous_velocity, q)

    def choose(self, q, dq, time_s):
        return self.reference_command(q, time_s, self.previous_target, self.previous_velocity)

    def command(self, q, dq, time_s):
        command = self.choose(q, dq, time_s)
        self.previous_target, self.previous_velocity = command.target.copy(), command.velocity.copy()
        return command


class IKController(ReferenceController):
    description = "Damped Jacobian Cartesian stick-tip IK with orientation and posture regularization"
    reference_assistance = "Reference provides desired tip position/orientation and posture; IK solves joint target online"

    def __init__(self, reference, physics, iterations=5):
        super().__init__(reference, physics)
        self.kinematics = mujoco.MjData(physics.model)
        self.desired_data = mujoco.MjData(physics.model)
        self.iterations = iterations
        self.last_tip_error_m = None

    def solve(self, q, time_s):
        model, c = self.physics.model, self.c
        reference_q, reference_dq = self.reference.sample(time_s)
        # The wrist reference includes experimentally tuned contact compliance.
        # This IK baseline uses its geometry only, not its motor command.
        self.desired_data.qpos[c.q] = reference_q
        mujoco.mj_forward(model, self.desired_data)
        targets = self.desired_data.site_xpos[self.physics.tip_sites].copy()
        orientations = self.desired_data.site_xmat[self.physics.tip_sites].reshape(2, 3, 3).copy()
        solved = q.copy()
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        for _ in range(self.iterations):
            self.kinematics.qpos[c.q] = solved
            mujoco.mj_forward(model, self.kinematics)
            for index, site in enumerate(self.physics.tip_sites):
                arm = np.arange(index*7, (index+1)*7)
                mujoco.mj_jacSite(model, self.kinematics, jp, jr, site)
                position_error = targets[index]-self.kinematics.site_xpos[site]
                rotation = self.kinematics.site_xmat[site].reshape(3, 3)
                rotation_error = .5*sum((np.cross(rotation[:, j], orientations[index, :, j]) for j in range(3)))
                jacobian = np.vstack([jp[:, c.v[arm]], .08*jr[:, c.v[arm]]])
                error = np.r_[position_error, .08*rotation_error]
                regularization = 2e-5
                delta = np.linalg.solve(jacobian.T @ jacobian + regularization*np.eye(7),
                                        jacobian.T @ error + regularization*(reference_q[arm]-solved[arm]))
                solved[arm] += np.clip(delta, -.08, .08)
            solved = np.clip(solved, c.bounds[:, 0], c.bounds[:, 1])
        self.kinematics.qpos[c.q] = solved
        mujoco.mj_forward(model, self.kinematics)
        self.last_tip_error_m = float(np.max(np.linalg.norm(targets-self.kinematics.site_xpos[self.physics.tip_sites], axis=1)))
        return solved, reference_dq

    def choose(self, q, dq, time_s):
        solved, _ = self.solve(q, time_s)
        return self.bounded((solved-self.previous_target)/self.period,
                            self.previous_target, self.previous_velocity, q)


class MPCController(ReferenceController):
    description = "Finite-candidate receding-horizon control with native nominal MuJoCo rollouts"
    reference_assistance = "Full reference trajectory in rollout tracking cost and candidate proposal"

    def __init__(self, reference, physics, residual=None, horizon=4, candidates=5):
        super().__init__(reference, physics)
        if horizon < 1 or candidates not in (3, 5):
            raise ValueError("MPC needs positive horizon and 3 or 5 candidates")
        self.residual, self.horizon, self.candidates = residual, horizon, candidates
        self.active_joints = sorted({joint for stroke in reference.strokes for joint in stroke["reference"].active_joints})
        self.mask = np.zeros(14)
        self.mask[self.active_joints] = 1.
        self.last_cost, self.last_candidate = None, None
        self.last_unsafe_candidates = 0

    def choose(self, q, dq, time_s):
        # Biases are bounded command-velocity changes, not torques. Each candidate
        # simulates the same PD/effort-limited plant contract used by the episode.
        proposals = [(1., 0.), (.8, 0.), (1.2, 0.)]
        if self.candidates == 5:
            proposals += [(1., -.2), (1., .2)]
        best = None
        self.last_unsafe_candidates = 0
        for index, (scale, bias) in enumerate(proposals):
            self.physics.set_state(q, dq)
            predicted_q, predicted_dq = q.copy(), dq.copy()
            previous_target, previous_velocity = self.previous_target.copy(), self.previous_velocity.copy()
            cost, first = 0., None
            for step in range(self.horizon):
                scheduled = time_s+step*self.period
                nominal = self.reference_command(predicted_q, scheduled, previous_target, previous_velocity)
                velocity = nominal.velocity + self.mask*((scale-1.)*nominal.velocity+bias)
                command = self.bounded(velocity, previous_target, previous_velocity, predicted_q)
                if first is None:
                    first = command
                predicted_q, predicted_dq, peak, unsafe = self.physics.advance(command, self.residual)
                if unsafe:
                    self.last_unsafe_candidates += 1
                    cost = float("inf")
                    break
                desired_q, desired_dq = self.reference.sample(scheduled+self.period)
                # Reference tracking is a shaping objective; evaluation is only
                # independently detected physical hits, timing and safety.
                cost += float(30.*np.sum((predicted_q-desired_q)**2)
                              + .05*np.sum((predicted_dq-desired_dq)**2)
                              + .002*np.sum((command.velocity-previous_velocity)**2)
                              + .02*(peak/30.)**2 + 10.*max(0., peak/30.-1.)**2)
                if not np.isfinite(cost):
                    cost = float("inf")
                    break
                previous_target, previous_velocity = command.target, command.velocity
            if best is None or cost < best[0]:
                best = cost, index, first
        self.last_cost, self.last_candidate = best[:2]
        if not np.isfinite(self.last_cost):
            raise RuntimeError("Every MPC candidate was unsafe or produced a nonfinite prediction cost")
        return best[2]


class BCController(ReferenceController):
    description = "Supervised random-feature ridge behavior clone of the reference teacher; no RL training"
    reference_assistance = "Reference-generated training labels and initial ready pose; no reference trajectory at inference"

    def __init__(self, reference, physics, score, model):
        super().__init__(reference, physics)
        self.score, self.model = score, model

    def features(self, q, dq, time_s):
        return policy_features(q, dq, self.previous_velocity, self.previous_target, self.anchor,
                               self.score.future(time_s-self.reference.preroll))

    def choose(self, q, dq, time_s):
        return self.bounded(self.model.predict(self.features(q, dq, time_s)),
                            self.previous_target, self.previous_velocity, q)
