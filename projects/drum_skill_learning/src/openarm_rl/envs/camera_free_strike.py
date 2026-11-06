from copy import deepcopy
import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from openarm_drummer.model import HoldController
from openarm_drummer.strike import HitDetector
from openarm_drummer.strike_planning import plan_wrist_strike

from ..control.reference import EncoderHistory, JointReference, ResidualCommand, SpeedConditionedReference
from ..control.servo import EncoderGravity
from ..planning.strikes import scene_identity
from ..skills.registry import SKILLS, StrikeSkill


REWARD_VERSION = "camera_free_center_strike_v4"


class CameraFreeStrikeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    period = .01

    def __init__(self, scene_directory, profile_file, position_range=.01, domain_range=.1,
                 history_frames=8, render_mode=None, target="snare", arm=None):
        if not 0 <= position_range <= .05 or not 0 <= domain_range <= .2:
            raise ValueError("Invalid position/domain randomization range")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Invalid render mode")
        directory = Path(scene_directory)
        self.model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
        metadata = json.loads((directory / "scene.json").read_text())
        self.robot, self.scene = metadata["robot"], metadata["scene"]
        self.profile = json.loads(Path(profile_file).read_text())
        self.action_semantics = self.profile.get("action_semantics", "wrist_residual")
        if self.action_semantics not in ("wrist_residual", "absolute_wrist_position_velocity"):
            raise ValueError("Unsupported strike action semantics")
        self.action_dim = 2 if self.action_semantics == "absolute_wrist_position_velocity" else 1
        if arm not in (None, "left", "right"):
            raise ValueError("Select a supported stick arm")
        self.skill = SKILLS[target] if arm is None else StrikeSkill(SKILLS[target].name, target, arm)
        if (self.profile["status"], self.profile["target"], self.profile["arm"]) != ("ready", target, self.skill.arm):
            raise ValueError("A matching validated center profile is required")
        if self.profile["scene_identity"] != scene_identity(self.robot, self.scene):
            raise ValueError("Profile scene identity differs")
        self.cfg = dict(self.profile["configuration"])
        self.controller = HoldController(self.model, self.robot)
        c = self.controller
        plan = plan_wrist_strike(self.model, self.robot, self.scene, self.cfg,
                                seed=np.array(self.profile["impact_seed"]), tip_direction=np.array(self.profile["tip_direction"]),
                                captured_ready=self.profile.get("captured_ready_joints"))
        if plan is None:
            raise ValueError("The fixed profile is no longer geometrically feasible")
        inactive = np.arange(7, 14) if self.skill.arm == "left" else np.arange(7)
        if "captured_ready_joints" not in self.profile:
            for key in ("ready", "approach", "bottom", "impact_pose", "staging"):
                plan[key][inactive] = 0
        probe = mujoco.MjData(self.model)
        intended = self.model.geom(f"{target}_head_collision").id
        tip_geom = self.model.geom(f"{self.skill.arm}_stick_tip_collision").id
        for row in plan["arc"]:
            probe.qpos[c.q] = plan["impact_pose"]
            probe.qpos[c.q[self.skill.wrist]] = row["joint7_rad"]
            mujoco.mj_forward(self.model, probe)
            for contact in probe.contact:
                virtual = row["height_m"] < 0 and intended in (contact.geom1, contact.geom2) and any(
                    self.model.geom(g).name.startswith(self.skill.arm + "_stick_") for g in (contact.geom1, contact.geom2))
                if contact.dist < -1e-5 and not virtual and {contact.geom1, contact.geom2} != {tip_geom, intended}:
                    names = [self.model.geom(g).name for g in (contact.geom1, contact.geom2)]
                    raise ValueError(f"Fixed parked-arm pose intersects the strike arc: {names}")
        pairs = [("approach_seconds", plan["ready"], plan["approach"]),
                 ("downstroke_seconds", plan["approach"], plan["bottom"]),
                 ("retract_seconds", plan["bottom"], plan["ready"])]
        self.acceleration = np.full(14, float(self.cfg.get("command_acceleration_rad_s2", 60.0)))
        if not np.isfinite(self.acceleration).all() or np.any(self.acceleration <= 0):
            raise ValueError("Command acceleration must be positive and finite")
        for key, start, end in pairs:
            delta = np.abs(end - start)
            self.cfg[key] = max(self.cfg[key], float(np.max(1.5 * delta / (.8 * c.velocity))),
                                float(np.max(np.sqrt(6 * delta / self.acceleration))))
        self.reference = JointReference.from_plan(plan, self.cfg, self.skill.wrist)
        self.reference_bank = SpeedConditionedReference(self.profile["reference_bank"]) if "reference_bank" in self.profile else None
        if self.reference_bank is not None:
            self.reference = self.reference_bank.at(.6)
        self.data = mujoco.MjData(self.model)
        self.surface = next(s for s in self.scene["surfaces"] if s["name"] == target)
        self.sid = self.model.site(f"{self.skill.arm}_stick_tip").id
        self.tip = self.model.geom(f"{self.skill.arm}_stick_tip_collision").id
        self.head = self.model.geom(f"{target}_head_collision").id
        self.nominal_geom = self.model.geom_pos.copy()
        self.nominal_sites = self.model.site_pos.copy()
        self.nominal_mass = self.model.body_mass.copy()
        self.nominal_inertia = self.model.body_inertia.copy()
        self.nominal_friction = self.model.geom_friction.copy()
        self.servo = EncoderGravity(self.model, c.kp)
        self.base_center = np.array(self.surface["center"])
        self.normal = np.array(self.surface["normal"])
        surface_names = tuple(s["name"] + "_" for s in self.scene["surfaces"])
        self.kit_geoms = [i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] == 0 and
                          self.model.geom(i).name.startswith(("kit_", "stand_", *surface_names))]
        self.kit_sites = [i for i in range(self.model.nsite) if self.model.site_bodyid[i] == 0 and
                          self.model.site(i).name.startswith(surface_names)]
        self.position_range, self.domain_range = position_range, domain_range
        self.history_frames, self.render_mode, self.renderer = history_frames, render_mode, None
        frame_dim = 58 + self.action_dim
        self.observation_space = spaces.Dict({"actor": spaces.Box(-np.inf, np.inf, (history_frames * frame_dim,), np.float32),
                                              "critic": spaces.Box(-np.inf, np.inf, (frame_dim + 16,), np.float32)})
        self.action_space = spaces.Box(-1, 1, (self.action_dim,), np.float32)
        self.active = False
        self.physics_observer = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("Use explicit environment configuration")
        self.shift = np.r_[self.np_random.uniform(-self.position_range, self.position_range, 2), 0.0]
        self.parameters = self.np_random.uniform(1 - self.domain_range, 1 + self.domain_range, 3)
        self.model.geom_pos[:] = self.nominal_geom
        self.model.site_pos[:] = self.nominal_sites
        self.model.geom_pos[self.kit_geoms] += self.shift
        self.model.site_pos[self.kit_sites] += self.shift
        self.center = self.base_center + self.shift
        self.model.body_mass[:] = self.nominal_mass * self.parameters[0]
        self.model.body_inertia[:] = self.nominal_inertia * self.parameters[0]
        self.model.geom_friction[:] = self.nominal_friction * self.parameters[1]
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setConst(self.model, self.data)
        c = self.controller
        self.data.qpos[c.q] = self.reference.positions[0]
        self.data.qpos[c.q] += self.np_random.uniform(-.005, .005, 14) * self.domain_range / .1
        inactive = np.arange(7, 14) if self.skill.arm == "left" else np.arange(7)
        self.data.qpos[c.q[inactive]] = self.reference.positions[0, inactive]
        self.data.qpos[c.q] = np.clip(self.data.qpos[c.q], c.bounds[:, 0] + .001, c.bounds[:, 1] - .001)
        self.data.qvel[c.v] = self.np_random.uniform(-.01, .01, 14) * self.domain_range / .1
        mujoco.mj_forward(self.model, self.data)
        if any(contact.dist < -1e-4 for contact in self.data.contact):
            raise ValueError("Randomized fixed ready pose intersects geometry")
        self.history = EncoderHistory(self.history_frames, self.action_dim)
        self.sensor_period_steps = int(self.np_random.integers(1, 3)) if self.domain_range else 1
        self.sensor_cache, self.observation_count = None, 0
        self.command = ResidualCommand(c.bounds, .8 * c.velocity, self.acceleration)
        self.detector = HitDetector(self.cfg)
        self.elapsed = self.force = self.quality = 0.0
        self.last_action = np.zeros(self.action_dim, np.float32)
        self.commanded_velocity = np.zeros(14)
        if self.action_semantics == "absolute_wrist_position_velocity":
            self.command.previous = np.clip(self.servo.position_target(self.data.qpos[c.q], self.data.qpos[c.q]), c.bounds[:, 0], c.bounds[:, 1])
            self.command.previous_velocity = np.zeros(14)
            self.last_action = self.encode_motor_command(self.command.previous[self.skill.wrist], 0.)
        self.peak_force = self.penetration = self.energy = 0.0
        self.unsafe = False
        self.failure = None
        self.target_speed = float(self.np_random.uniform(.35, .85))
        if self.reference_bank is not None:
            self.reference = self.reference_bank.at(self.target_speed)
        self.delay_steps = int(self.np_random.integers(0, 4)) if self.domain_range else 0
        self.queue = [(self.reference.positions[0].copy(), np.zeros(14)) for _ in range(self.delay_steps)]
        self.active = True
        self.trace = []
        return self.observation(), self.info()

    def observation(self):
        c = self.controller
        if self.sensor_cache is None or self.observation_count % self.sensor_period_steps == 0:
            q, dq = self.data.qpos[c.q].copy(), self.data.qvel[c.v].copy()
            noise = self.domain_range / .1
            q += self.np_random.normal(0, .0005 * noise, 14)
            dq += self.np_random.normal(0, .002 * noise, 14)
            self.sensor_cache = q, dq
        q, dq = self.sensor_cache
        self.observation_count += 1
        ref, velocity = self.reference.sample(self.elapsed)
        actor = self.history.observe(q, dq, ref, velocity, self.last_action, self.elapsed, self.target_speed)
        jacobian = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacobian, None, self.sid)
        critic = np.concatenate((actor[-self.history.frame_dim:], self.data.site_xpos[self.sid] - self.center,
                                  jacobian @ self.data.qvel, self.normal, self.parameters, self.shift, [self.force])).astype(np.float32)
        return {"actor": actor, "critic": critic}

    def info(self):
        hits = self.detector.events
        c = self.controller
        recovered = (self.elapsed >= self.reference.duration and np.max(np.abs(self.data.qpos[c.q] - self.reference.positions[-1])) < .03
                     and np.max(np.abs(self.data.qvel[c.v])) < .08)
        good = (len(hits) == 1 and hits[0]["radial_offset_m"] <= .01 and
                hits[0]["contact_duration_s"] <= .12 and
                abs(hits[0]["onset_time_s"] - self.reference.impact_time) <= .06 and
                abs(hits[0]["downward_speed_m_s"] - self.target_speed) <= .3)
        return {"is_success": bool(good and recovered and not self.unsafe), "recovered": bool(recovered),
                "hit_count": len(hits), "unsafe_contact": bool(self.unsafe), "failure": self.failure,
                "target": self.skill.target, "arm": self.skill.arm, "reward_version": REWARD_VERSION,
                "center_error_m": hits[0]["radial_offset_m"] if hits else None, "hit_quality": self.quality,
                "target_speed_m_s": self.target_speed, "target_impact_time_s": self.reference.impact_time,
                "peak_force_N": self.peak_force, "max_penetration_m": self.penetration,
                "kit_offset_m": self.shift.tolist(), "domain_scales": self.parameters.tolist(),
                "control_delay_steps": self.delay_steps, "hits": deepcopy(hits)}

    def step(self, action):
        if not self.active:
            raise RuntimeError("Reset before stepping")
        if not self.action_space.contains(np.asarray(action, dtype=np.float32)):
            raise ValueError("Invalid residual action")
        c = self.controller
        before, previous = len(self.detector.events), self.last_action
        self.last_action = np.asarray(action).copy()
        motor_reference = self.reference.motor_target(self.elapsed, self.data.qpos[c.q], self.servo)
        if self.action_semantics == "absolute_wrist_position_velocity":
            lower, upper = c.bounds[self.skill.wrist]
            motor_reference[self.skill.wrist] = lower + (float(action[0]) + 1) * .5 * (upper - lower)
            residual = np.zeros(1)
        else:
            residual = np.asarray(action) * self.reference.residual_weight(self.elapsed)
        target = self.command.step(motor_reference, residual, self.skill.wrist)
        if self.action_semantics == "absolute_wrist_position_velocity":
            desired_velocity = np.zeros(14)
            desired_velocity[self.skill.wrist] = float(action[1]) * .8 * c.velocity[self.skill.wrist]
            self.commanded_velocity = np.clip(desired_velocity, self.commanded_velocity - self.acceleration*self.period,
                                               self.commanded_velocity + self.acceleration*self.period)
        else:
            self.commanded_velocity = self.command.previous_velocity.copy()
        self.queue.append((target.copy(), self.commanded_velocity.copy()))
        applied, applied_velocity = self.queue.pop(0)
        self.energy = 0.0
        for _ in range(round(self.period / self.model.opt.timestep)):
            jacobian = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacobian, None, self.sid)
            speed = max(0., float(-(jacobian @ self.data.qvel) @ self.normal))
            kp = c.kp * self.parameters[2]
            torque = kp * (applied - self.data.qpos[c.q]) + c.kd * np.sqrt(self.parameters[2]) * (applied_velocity - self.data.qvel[c.v])
            self.data.ctrl[c.a] = np.clip(torque, -c.effort, c.effort)
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            if self.physics_observer is not None:
                self.physics_observer(self.data)
            self.elapsed = float(self.data.time)
            self.force = 0.0
            for index, contact in enumerate(self.data.contact):
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, index, wrench)
                if {contact.geom1, contact.geom2} == {self.tip, self.head}:
                    self.force += max(0., float(wrench[0]))
                    self.penetration = max(self.penetration, -float(contact.dist))
                elif wrench[0] > .05:
                    self.unsafe = True
            height = float((self.data.site_xpos[self.sid] - self.center) @ self.normal)
            hit = self.detector.update(self.elapsed, self.force, height, self.model.opt.timestep)
            if hit is not None:
                hit["radial_offset_m"] = float(np.linalg.norm(self.data.site_xpos[self.sid] - self.center - height * self.normal))
                hit["downward_speed_m_s"] = speed
            self.peak_force = max(self.peak_force, self.force)
            self.energy += float(np.sum(torque ** 2)) * self.model.opt.timestep
            if self.failure is not None:
                break
            if not np.isfinite(self.data.qpos).all() or self.data.warning.number.any():
                raise RuntimeError("Invalid native physics state")
        encoder_failure = (np.any(self.data.qpos[c.q] < c.bounds[:, 0] - .001) or np.any(self.data.qpos[c.q] > c.bounds[:, 1] + .001)
                           or np.max(np.abs(applied - self.data.qpos[c.q])) > .2 or np.any(np.abs(self.data.qvel[c.v]) > c.velocity))
        if encoder_failure:
            self.failure = "Encoder tracking or joint limit exceeded"
        self.unsafe |= self.peak_force > self.cfg["force_stop_N"] or self.penetration > self.cfg["penetration_stop_m"]
        terms = {"time": -.002, "effort": -.00001 * self.energy, "action_change": -.005 * float(np.square(self.last_action - previous).sum()),
                 "impact": 0., "extra_hit": -10. * max(0, len(self.detector.events) - max(1, before)),
                 "pressing": -.02 * sum(hit["contact_duration_s"] > .12 for hit in self.detector.events)}
        for hit in self.detector.events[before:]:
            self.quality = float(np.exp(-((hit["radial_offset_m"] / .01) ** 2) -
                                       ((hit["onset_time_s"] - self.reference.impact_time) / .06) ** 2 -
                                       ((hit["downward_speed_m_s"] - self.target_speed) / .3) ** 2))
            terms["impact"] += 5 * self.quality
        done = self.elapsed >= self.reference.duration or self.failure is not None
        info = self.info()
        terms["terminal"] = (20 * info["is_success"] - 10 * (info["hit_count"] == 0) - 10 * (not info["recovered"])
                             - 20 * self.unsafe - 10 * bool(self.failure)) if done else 0.
        self.trace.append({"time_s": self.elapsed, "joint_command_rad": applied.tolist(),
                           "joint_velocity_command_rad_s": applied_velocity.tolist(), "action": self.last_action.tolist()})
        self.active = not done
        return self.observation(), float(sum(terms.values())), done, False, {**info, "reward_terms": terms}

    def encode_motor_command(self, position, velocity):
        lower, upper = self.controller.bounds[self.skill.wrist]
        return np.array([np.clip(2*(position-lower)/(upper-lower) - 1, -1, 1),
                         np.clip(velocity / (.8*self.controller.velocity[self.skill.wrist]), -1, 1)], np.float32)

    def reference_action(self):
        if self.action_semantics == "wrist_residual":
            return np.zeros(1, np.float32)
        command = self.reference.motor_target(self.elapsed, self.data.qpos[self.controller.q], self.servo)
        velocity = self.reference.sample(self.elapsed)[1][self.skill.wrist]
        return self.encode_motor_command(command[self.skill.wrist], velocity)

    def render(self):
        if self.render_mode != "rgb_array":
            raise RuntimeError("Enable rgb_array rendering")
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 480, 640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.45, 0, .8]
        camera.distance, camera.azimuth, camera.elevation = 2.4, 145, -22
        options = mujoco.MjvOption()
        options.sitegroup[:] = 0
        self.renderer.update_scene(self.data, camera=camera, scene_option=options)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
