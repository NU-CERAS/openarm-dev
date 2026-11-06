import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from openarm_drummer.strike import HitDetector, STRIKE_CONFIG

from ..control.servo import EncoderGravity, extrapolate_encoders
from ..music.reference import PerformanceReference
from ..music.feedback import VelocityCalibration
from ..music.score import MATCH_HORIZON_S, ContactMatcher, ContactScore
from ..skills.registry import SKILLS


CAMERAS = {"side": {"lookat": (.35, 0, .8), "azimuth": 55, "elevation": -30, "distance": 2.3},
           "front": {"lookat": (.35, 0, .75), "azimuth": 180, "elevation": -50, "distance": 1.7}}


class RhythmicDrummingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    period = .01

    def __init__(self, scene_directory, profiles, midi_file, mapping=None, tempo_scale=1.,
                 domain_range=0., position_range=0., segment_seconds=None, render_mode=None,
                 score_only=False, velocity_calibration=None, force_penalty=None, source=None, camera="side"):
        if not 0 <= domain_range <= .2 or not 0 <= position_range <= .05:
            raise ValueError("Invalid rhythmic domain randomization ranges")
        if segment_seconds is not None and segment_seconds <= .5:
            raise ValueError("Training segments must exceed half a second")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Unsupported rhythm rendering mode")
        if camera not in CAMERAS:
            raise ValueError("Unsupported rhythm camera")
        self.camera = CAMERAS[camera]
        self.score_only = bool(score_only)
        self.force_penalty = (.05 if self.score_only else 0.) if force_penalty is None else float(force_penalty)
        if not np.isfinite(self.force_penalty) or not 0 <= self.force_penalty <= 1:
            raise ValueError("Contact-force penalty must lie between zero and one")
        directory = Path(scene_directory)
        # Environments of one corpus run one at a time, so a source environment lends its large compiled model.
        self.model = source.model if source else mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
        self.scene = json.loads((directory / "scene.json").read_text())
        if render_mode is not None:
            options = mujoco.MjvOption()
            kit_meshes = [index for index in range(self.model.ngeom)
                          if self.model.geom(index).name.startswith("kit_material_")]
            if not kit_meshes or any(self.model.geom_rgba[index, 3] <= 0 or
                                     not options.geomgroup[self.model.geom_group[index]] for index in kit_meshes):
                raise ValueError("MIDI video requires the visible original drum-kit mesh; prepare the scene without --layout")
        self.score = ContactScore.read(midi_file, mapping, tempo_scale)
        self.velocity_calibration = VelocityCalibration(velocity_calibration) if velocity_calibration else None
        if self.velocity_calibration and any(goal.target not in self.velocity_calibration.curves for goal in self.score.goals):
            raise ValueError("Every score drum requires measured velocity calibration")
        self.reference = PerformanceReference(self.model, self.scene, self.score, profiles)
        if not self.score_only and any(len(stroke["reference"].active_joints) != 1 for stroke in self.reference.strokes):
            raise ValueError("Full-arm MIDI references require --score-only joint command execution")
        self.controller = self.reference.controller
        self.servo = source.servo if source else EncoderGravity(self.model, self.controller.kp)
        self.data = mujoco.MjData(self.model)
        self.cfg = json.loads(STRIKE_CONFIG.read_text())
        self.domain_range, self.position_range, self.segment_seconds = domain_range, position_range, segment_seconds
        self.render_mode, self.renderer = render_mode, None
        origin = source or self
        if source is None:
            self.nominal_mass, self.nominal_inertia = self.model.body_mass.copy(), self.model.body_inertia.copy()
            self.nominal_friction, self.nominal_geom = self.model.geom_friction.copy(), self.model.geom_pos.copy()
            self.nominal_sites = self.model.site_pos.copy()
        self.nominal_mass, self.nominal_inertia = origin.nominal_mass, origin.nominal_inertia
        self.nominal_friction, self.nominal_geom = origin.nominal_friction, origin.nominal_geom
        self.nominal_sites = origin.nominal_sites
        prefixes = tuple(s["name"]+"_" for s in self.scene["scene"]["surfaces"])
        self.kit_geoms = [i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] == 0 and
                          self.model.geom(i).name.startswith(("kit_", "stand_", *prefixes))]
        self.kit_sites = [i for i in range(self.model.nsite) if self.model.site_bodyid[i] == 0 and self.model.site(i).name.startswith(prefixes)]
        self.surfaces = {s["name"]: s for s in self.scene["scene"]["surfaces"] if s["name"] in SKILLS}
        self.targets = list(SKILLS)
        self.target_index = {target: index for index, target in enumerate(self.targets)}
        self.centers = np.array([self.surfaces[t]["center"] for t in self.targets])
        self.normals = np.array([self.surfaces[t]["normal"] for t in self.targets])
        self.heads = {self.model.geom(f"{t}_head_collision").id: t for t in self.targets}
        # At playing angles the taper or shaft can meet a head or cymbal; only other geometry is an unsafe contact.
        self.tips = {self.model.geom(f"{arm}_stick_{part}_collision").id: arm for arm in ("left", "right")
                     for part in ("tip", "taper", "shaft")}
        self.tip_sites = {arm: self.model.site(f"{arm}_stick_tip").id for arm in ("left", "right")}
        self.actor_dim = 4*42 + (0 if self.score_only else 28) + 21 + 8*10
        self.critic_dim = self.actor_dim + 28 + 14 + 6 + 21 + 3 + 2 + 1
        self.observation_space = spaces.Dict({"actor": spaces.Box(-np.inf, np.inf, (self.actor_dim,), np.float32),
                                             "critic": spaces.Box(-np.inf, np.inf, (self.critic_dim,), np.float32),
                                             "nominal": spaces.Box(-10, 10, (14,), np.float32)})
        self.action_space = spaces.Box(-1, 1, (14,), np.float32)
        self.active = False

    def encode(self, positions):
        bounds = self.controller.bounds
        return np.clip(2*(positions-bounds[:, 0])/(bounds[:, 1]-bounds[:, 0])-1, -.999, .999).astype(np.float32)

    def settled_starts(self):
        c = self.controller
        candidates = {0., *[max(0., g.time_s+self.reference.preroll-.4) for g in self.score.goals],
                      *[max(0., stroke["start"]-.05) for stroke in self.reference.strokes]}
        settled = []
        for time_s in sorted(candidates):
            end = min(self.reference.duration, time_s+self.segment_seconds)
            if not any(time_s < goal.time_s+self.reference.preroll <= end-MATCH_HORIZON_S for goal in self.score.goals):
                continue
            positions, velocities = self.reference.sample(time_s)
            if np.max(np.abs(velocities)) >= .08:
                continue
            self.data.qpos[c.q] = positions
            mujoco.mj_forward(self.model, self.data)
            if not any(contact.dist < -1e-4 for contact in self.data.contact):
                settled.append(time_s)
        if not settled:
            raise ValueError("No settled collision-free music initialization contains a target")
        return settled

    def restore_nominal(self):
        self.model.body_mass[:] = self.nominal_mass
        self.model.body_inertia[:] = self.nominal_inertia
        self.model.geom_friction[:] = self.nominal_friction
        self.model.geom_pos[:] = self.nominal_geom
        self.model.site_pos[:] = self.nominal_sites

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("Use explicit score and domain configuration")
        self.parameters = self.np_random.uniform(1-self.domain_range, 1+self.domain_range, 3)
        self.shift = np.r_[self.np_random.uniform(-self.position_range, self.position_range, 2), 0.]
        self.restore_nominal()
        self.model.body_mass[:] *= self.parameters[0]
        self.model.body_inertia[:] *= self.parameters[0]
        self.model.geom_friction[:] *= self.parameters[1]
        self.model.geom_pos[self.kit_geoms] += self.shift
        self.model.site_pos[self.kit_sites] += self.shift
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setConst(self.model, self.data)
        c = self.controller
        self.elapsed = 0.
        if self.segment_seconds is not None:
            settled = self.settled_starts()
            self.elapsed = float(self.np_random.choice(settled))
        q, dq = self.reference.sample(self.elapsed)
        self.data.qpos[c.q], self.data.qvel[c.v] = q, dq
        self.data.qpos[c.q] += self.np_random.uniform(-.003, .003, 14)*self.domain_range/.1
        self.data.qpos[c.q] = np.clip(self.data.qpos[c.q], c.bounds[:, 0], c.bounds[:, 1])
        mujoco.mj_forward(self.model, self.data)
        if any(contact.dist < -1e-4 for contact in self.data.contact):
            raise ValueError("Rhythm initialization intersects collision geometry")
        self.started = self.elapsed
        self.end = self.reference.duration if self.segment_seconds is None else min(self.reference.duration, self.elapsed+self.segment_seconds)
        goals = [g for g in self.score.goals if self.started < g.time_s+self.reference.preroll <= self.end-MATCH_HORIZON_S]
        self.matcher = ContactMatcher(goals)
        self.cached_report = None
        self.detectors = {(arm, t): HitDetector(self.cfg) for arm in ("left", "right") for t in self.targets}
        self.last_action = self.encode(q)
        self.anchor_encoder = self.data.qpos[c.q].copy()
        self.history = []
        self.force = np.zeros(14)
        self.previous_tips = {arm: self.data.site_xpos[site].copy() for arm, site in self.tip_sites.items()}
        self.contact_speeds = {}
        self.contact_records = {}
        self.velocity_errors = []
        self.peak_force, self.max_penetration, self.max_contact_duration = 0., 0., 0.
        self.contact_cost_total = 0.
        self.unsafe, self.failure, self.steps = False, None, 0
        self.previous_target, self.previous_velocity = q.copy(), dq.copy()
        self.feedback_period = int(self.np_random.integers(1, 3)) if self.domain_range else 1
        self.feedback_delay = int(self.np_random.integers(0, 3)) if self.domain_range else 0
        self.sensor_samples = [(self.elapsed, self.data.qpos[c.q].copy(), self.data.qvel[c.v].copy()) for _ in range(4)]
        self.sensor_cache = self.sensor_samples[0]
        self.active, self.trace = True, []
        return self.observation(), self.info()

    def reference_action(self):
        if self.score_only:
            q, _ = self.encoders()
            gravity = self.servo.position_target(np.zeros(14), q)
            _, velocity = self.reference.sample(self.elapsed)
            ratio = self.controller.kd/self.controller.kp
            base = self.reference.motor_target(self.elapsed, q, self.servo)-gravity+ratio*velocity
            rate = np.clip((base-self.previous_target)/(self.period+ratio),
                           -.8*self.controller.velocity, .8*self.controller.velocity)
            rate = np.clip(rate, self.previous_velocity-120*self.period, self.previous_velocity+120*self.period)
            return self.encode(self.previous_target+rate*self.period)
        return self.encode(self.reference.sample(self.elapsed)[0])

    def encoders(self):
        sampled_at, q, dq = self.sensor_cache
        return extrapolate_encoders(q, dq, self.elapsed-sampled_at), dq

    def observation(self):
        c = self.controller
        q, dq = self.encoders()
        row = np.r_[q, dq, self.last_action].astype(np.float32)
        if not self.history:
            self.history = [row.copy() for _ in range(4)]
        if self.score_only:
            trajectory, nominal_q = np.empty(0), self.anchor_encoder
        else:
            ref, ref_velocity = self.reference.sample(self.elapsed)
            trajectory, nominal_q = np.r_[ref, ref_velocity], ref
        actor = np.r_[np.concatenate(self.history), trajectory, self.centers.ravel(),
                      self.score.future(self.elapsed-self.reference.preroll)].astype(np.float32)
        tips = self.data.site_xpos[list(self.tip_sites.values())].ravel()
        critic = np.r_[actor, self.data.qpos[c.q], self.data.qvel[c.v], self.force,
                       tips, (self.centers+self.shift).ravel(), self.parameters,
                       self.shift[:2], self.elapsed].astype(np.float32)
        nominal = np.arctanh(self.encode(nominal_q)).astype(np.float32)
        return {"actor": actor, "critic": critic, "nominal": nominal}

    def report(self):
        key = (len(self.matcher.matched), len(self.matcher.missed), len(self.matcher.actual))
        if self.cached_report is None or self.cached_report[0] != key:
            self.cached_report = key, self.matcher.report()
        return self.cached_report[1]

    def info(self):
        report = self.report()
        final, _ = self.reference.sample(self.end)
        recovered = (np.max(np.abs(self.data.qpos[self.controller.q]-final)) < .03
                     and np.max(np.abs(self.data.qvel[self.controller.v])) < .08)
        if self.score_only:
            recovered = (np.max(np.abs(self.data.qvel[self.controller.v])) < .08
                         and all(detector.armed for detector in self.detectors.values())
                         and np.max(self.force) <= self.cfg["release_force_N"])
        full_score = self.end == self.reference.duration
        return {**report, "unsafe_contact": self.unsafe, "failure": self.failure,
                "peak_force_N": self.peak_force, "max_penetration_m": self.max_penetration,
                "max_contact_duration_s": self.max_contact_duration, "recovered": bool(recovered),
                "contact_force_cost": self.contact_cost_total, "force_penalty": self.force_penalty,
                "score_time_s": self.elapsed-self.reference.preroll, "preroll_s": self.reference.preroll,
                "segment_start_s": self.started, "segment_end_s": self.end,
                "parameters": self.parameters.tolist(), "kit_shift_m": self.shift.tolist(),
                "actor_trajectory_inputs": not self.score_only,
                "motor_reference": not self.score_only,
                "velocity_mae": float(np.mean(self.velocity_errors)) if self.velocity_errors else None,
                "intensity_supervision": self.velocity_calibration.document["measurement_source"] if self.velocity_calibration else "disabled_no_measured_calibration",
                "is_success": bool(report["f1"] >= .95 and not self.unsafe and self.max_contact_duration <= .12
                                   and (not full_score or recovered))}

    def step(self, action):
        if not self.active:
            raise RuntimeError("Reset before executing a rhythm")
        action = np.asarray(action, np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Rhythm actions require fourteen finite normalized joint targets")
        c = self.controller
        bounds = c.bounds
        desired = bounds[:, 0] + (action+1)*.5*(bounds[:, 1]-bounds[:, 0])
        velocity = np.clip((desired-self.previous_target)/self.period, -.8*c.velocity, .8*c.velocity)
        velocity = np.clip(velocity, self.previous_velocity-120*self.period, self.previous_velocity+120*self.period)
        target = np.clip(self.previous_target+velocity*self.period, bounds[:, 0], bounds[:, 1])
        self.previous_target, self.previous_velocity = target.copy(), velocity.copy()
        previous_action = self.last_action.copy()
        self.last_action = action.copy()
        before_correct, before_extra = len(self.matcher.matched), len(self.matcher.actual)-len(self.matcher.matched)
        energy, intensity_cost, force_cost = 0., 0., 0.
        measured, _ = self.encoders()
        if self.score_only:
            applied = self.servo.position_target(target, measured)
            nominal_velocity = velocity
        else:
            nominal, nominal_velocity = self.reference.sample(self.elapsed)
            applied = self.reference.motor_target(self.elapsed, measured, self.servo) + target-nominal
        for _ in range(round(self.period/self.model.opt.timestep)):
            torque = c.kp*self.parameters[2]*(applied-self.data.qpos[c.q])+c.kd*np.sqrt(self.parameters[2])*(nominal_velocity-self.data.qvel[c.v])
            self.data.ctrl[c.a] = np.clip(torque, -c.effort, c.effort)
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.elapsed += self.model.opt.timestep
            self.force[:] = 0
            contacts = {}
            for index, contact in enumerate(self.data.contact):
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, index, wrench)
                force = max(0., float(wrench[0]))
                arm = self.tips.get(contact.geom1, self.tips.get(contact.geom2))
                drum = self.heads.get(contact.geom1, self.heads.get(contact.geom2))
                if arm is not None and drum is not None:
                    contacts.setdefault((arm, drum), []).append({
                        "geoms": [self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name],
                        "position_m": contact.pos.tolist(), "normal_force_N": force,
                        "penetration_m": max(0., -float(contact.dist))})
                    key = (0 if arm == "left" else 7)+self.targets.index(drum)
                    self.force[key] += force
                    self.peak_force = max(self.peak_force, force)
                    self.max_penetration = max(self.max_penetration, -float(contact.dist))
                    if force > self.cfg["force_stop_N"] or -float(contact.dist) > self.cfg["penetration_stop_m"]:
                        self.unsafe, self.failure = True, "Unsafe drum contact"
                elif force > .1:
                    names = [self.model.geom(g).name for g in (contact.geom1, contact.geom2)]
                    self.unsafe, self.failure = True, f"Unexpected physical contact: {names}, force={force:.3f}N"
            self.peak_force = max(self.peak_force, float(np.max(self.force)))
            force_cost += float(np.sum((self.force/self.cfg["force_stop_N"])**2))*self.model.opt.timestep/self.period
            if np.any(self.force > self.cfg["force_stop_N"]):
                self.unsafe, self.failure = True, "Unsafe summed drum contact force"
            tips = {arm: self.data.site_xpos[site].copy() for arm, site in self.tip_sites.items()}
            gaps = {arm: np.einsum("ij,ij->i", tip-self.centers-self.shift, self.normals) for arm, tip in tips.items()}
            speeds = {arm: np.maximum(0., self.normals@(self.previous_tips[arm]-tip))/self.model.opt.timestep
                      for arm, tip in tips.items()}
            for (arm, drum), detector in self.detectors.items():
                index = self.target_index[drum]
                tip, gap, speed = tips[arm], float(gaps[arm][index]), float(speeds[arm][index])
                detector_key = (arm, drum)
                if self.force[(0 if arm == "left" else 7)+index] > self.cfg["release_force_N"] and detector.episode is None:
                    self.contact_speeds[detector_key] = speed
                event = detector.update(self.elapsed-self.reference.preroll,
                                        self.force[(0 if arm == "left" else 7)+index], gap, self.model.opt.timestep)
                if event is not None:
                    self.matcher.hit(drum, event["onset_time_s"], arm)
                    self.matcher.actual[-1].update(downward_speed_m_s=self.contact_speeds.get(detector_key, speed),
                                                   peak_force_N=event["peak_force_N"],
                                                   normal_impulse_Ns=event["normal_impulse_Ns"],
                                                   tip_position_m=tip.tolist(),
                                                   drum_center_m=(self.centers[index]+self.shift).tolist(),
                                                   contacts=contacts[detector_key])
                    self.contact_records[detector_key] = self.matcher.actual[-1]
                    if self.velocity_calibration:
                        hit = self.matcher.actual[-1]
                        hit["midi_velocity"] = self.velocity_calibration.velocity(drum, hit["downward_speed_m_s"])
                        if hit["matched_event_id"] is not None:
                            expected = next(goal for goal in self.matcher.goals if goal.event_id == hit["matched_event_id"])
                            error = abs(hit["midi_velocity"]-expected.velocity)
                            self.velocity_errors.append(error)
                            curve = self.velocity_calibration.curves[drum]
                            excess = max(0., hit["downward_speed_m_s"]-curve[-1, 0])
                            intensity_cost += error/126+excess
                if detector.episode is not None:
                    self.max_contact_duration = max(self.max_contact_duration, detector.episode["contact_duration_s"])
                    record = self.contact_records.get(detector_key)
                    if record is not None and record["time_s"] == detector.episode["onset_time_s"]:
                        record.update({key: detector.episode[key] for key in
                                       ("peak_force_N", "normal_impulse_Ns", "contact_duration_s")})
            self.previous_tips = tips
            if np.max(np.abs(target-self.data.qpos[c.q])) > .2:
                self.unsafe, self.failure = True, "Joint tracking limit exceeded"
            energy += float(np.sum(self.data.ctrl[c.a]**2))*self.model.opt.timestep
            if self.unsafe:
                break
        missed = self.matcher.expire(self.elapsed-self.reference.preroll)
        correct, extra = len(self.matcher.matched), len(self.matcher.actual)-len(self.matcher.matched)
        self.contact_cost_total += force_cost
        reward = (correct-before_correct) - .5*(extra-before_extra) - 2*missed - .25*intensity_cost-self.force_penalty*force_cost
        score_time = self.elapsed-self.reference.preroll
        for goal in self.matcher.active(score_time):
            center = self.centers[self.target_index[goal.target]]+self.shift
            reward -= float(np.min(np.linalg.norm(self.data.site_xpos[list(self.tip_sites.values())]-center, axis=1)))
        reward -= .001*float(np.sum((action-previous_action)**2)) + 1e-6*energy
        reward -= .2*sum(detector.episode is not None and detector.episode["contact_duration_s"] > .12
                         for detector in self.detectors.values())
        if self.unsafe:
            remaining = {g.event_id for g in self.matcher.goals}-self.matcher.missed-set(self.matcher.matched)
            self.matcher.missed.update(remaining)
            reward -= 10 + 2*len(remaining)
        self.steps += 1
        self.sensor_samples.append((self.elapsed, self.data.qpos[c.q].copy(), self.data.qvel[c.v].copy()))
        self.sensor_samples = self.sensor_samples[-4:]
        if self.steps % self.feedback_period == 0:
            self.sensor_cache = self.sensor_samples[-1-self.feedback_delay]
        measured, speed = self.encoders()
        self.history.append(np.r_[measured, speed, self.last_action].astype(np.float32))
        self.history = self.history[-4:]
        self.active = self.elapsed < self.end and not self.unsafe
        self.trace.append({"performance_time_s": self.elapsed, "joint_target_rad": target.tolist(),
                           "motor_target_rad": applied.tolist(), "measured_joints_rad": self.data.qpos[c.q].tolist(),
                           "encoder_age_s": self.elapsed-self.sensor_cache[0],
                           "action": action.tolist()})
        return self.observation(), float(reward), not self.active, False, self.info()

    def render(self):
        if self.render_mode != "rgb_array":
            raise ValueError("Create the environment with rgb_array rendering")
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, width=1280, height=720)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.camera["lookat"]
        camera.azimuth, camera.elevation = self.camera["azimuth"], self.camera["elevation"]
        camera.distance = self.camera["distance"]
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
