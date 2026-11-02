import copy
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from openarm_drummer.model import CONFIG, SOURCE
from openarm_drummer.scene import build_scene
from openarm_drummer.strike import STRIKE_CONFIG, StrikeSession


STATES = ("ready", "approach", "downstroke", "contact", "retract", "recovery", "complete")
REWARD_VERSION = "opendrummer_strike_v1"


def prepare_scene(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    model, robot, scene = build_scene()
    mujoco.mj_saveModel(model, str(directory / "model.mjb"), None)
    sources = [SOURCE, CONFIG, CONFIG.with_name("scene.json"), CONFIG.with_name("contact.json"), CONFIG.with_name("strike.json")]
    metadata = {"robot": robot, "scene": scene,
                "source_hashes": {str(p.relative_to(SOURCE.parents[3])): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (directory / "scene.json").write_text(json.dumps(metadata, indent=2) + "\n")


class OpenDrummerStrikeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    period = 0.01
    gamma = 0.99

    def __init__(self, scene_directory=None, render_mode=None, position_range=0.05, domain_range=0.1):
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or rgb_array")
        if not 0 <= position_range <= 0.05 or not 0 <= domain_range <= 0.2:
            raise ValueError("Position range must be in [0, 0.05] m and domain range in [0, 0.2]")
        if scene_directory is None:
            self.model, self.robot, self.scene = build_scene()
        else:
            directory = Path(scene_directory)
            self.model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
            metadata = json.loads((directory / "scene.json").read_text())
            self.robot, self.scene = metadata["robot"], metadata["scene"]
        self.base_scene = copy.deepcopy(self.scene)
        self.data = mujoco.MjData(self.model)
        self.render_mode, self.renderer = render_mode, None
        self.position_range, self.domain_range = position_range, domain_range
        self.action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (67,), dtype=np.float32)
        self.nominal_geom = self.model.geom_pos.copy()
        self.nominal_sites = self.model.site_pos.copy()
        self.nominal_friction = self.model.geom_friction.copy()
        self.nominal_mass = self.model.body_mass.copy()
        self.nominal_inertia = self.model.body_inertia.copy()
        self.kit_geoms = np.array([i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] == 0
                                  and self.model.geom(i).name not in ("floor_visual", "floor_collision", "robot_support_visual", "support_collision")])
        surface_names = [s["name"] for s in self.scene["surfaces"]]
        self.kit_sites = np.array([i for i in range(self.model.nsite) if any(self.model.site(i).name.startswith(n + "_") for n in surface_names)])
        self.reference_settings = {**json.loads(STRIKE_CONFIG.read_text()), "ready_height_m": 0.07, "approach_height_m": 0.04}
        self.impact_seed = StrikeSession(self.model, self.robot, self.scene, configuration=self.reference_settings).impact_pose.copy()
        self.active = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("Reset options are unsupported; use explicit environment ranges")
        self.scene = copy.deepcopy(self.base_scene)
        self.shift = np.r_[self.np_random.uniform(-self.position_range, self.position_range, 2), 0.0]
        self.model.geom_pos[:] = self.nominal_geom
        self.model.site_pos[:] = self.nominal_sites
        self.model.geom_pos[self.kit_geoms] += self.shift
        self.model.site_pos[self.kit_sites] += self.shift
        for surface in self.scene["surfaces"]:
            surface["center"] = (np.array(surface["center"]) + self.shift).tolist()
        self.parameters = self.np_random.uniform(1 - self.domain_range, 1 + self.domain_range, 3)
        self.model.body_mass[:] = self.nominal_mass * self.parameters[0]
        self.model.body_inertia[:] = self.nominal_inertia * self.parameters[0]
        self.model.geom_friction[:] = self.nominal_friction * self.parameters[1]
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setConst(self.model, self.data)
        self.reference = StrikeSession(self.model, self.robot, self.scene, impact_seed=self.impact_seed,
                                       configuration=self.reference_settings)
        self.reference.controller.kp = self.reference.controller.kp.astype(float) * self.parameters[2]
        self.reference.controller.kd = self.reference.controller.kd.astype(float) * np.sqrt(self.parameters[2])
        self.reference.reset(self.data)
        c = self.reference.controller
        self.data.qpos[c.q] += self.np_random.uniform(-0.01, 0.01, 14) * self.domain_range / 0.1
        self.data.qpos[c.q] = np.clip(self.data.qpos[c.q], c.bounds[:, 0] + 0.001, c.bounds[:, 1] - 0.001)
        self.data.qvel[c.v] = self.np_random.uniform(-0.02, 0.02, 14) * self.domain_range / 0.1
        mujoco.mj_forward(self.model, self.data)
        if any(contact.dist < -1e-5 for contact in self.data.contact):
            raise ValueError("Randomized initial pose intersects collision geometry")
        cfg = self.reference.cfg
        fraction = (self.reference.impact_pose[6] - self.reference.approach[6]) / (self.reference.bottom[6] - self.reference.approach[6])
        self.target_time = cfg["ready_seconds"] + cfg["approach_seconds"] + cfg["downstroke_seconds"] * float(fraction)
        self.target_time += self.np_random.uniform(-0.025, 0.025)
        self.target_speed = float(self.np_random.uniform(0.35, 0.85))
        self.last_action = 0.0
        self.offset = 0.0
        self.hit_quality = 0.0
        self.energy = 0.0
        self.active = True
        self.previous_potential = self.potential()
        return self.observation(), self.info()

    def tip_velocity(self):
        jacobian = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacobian, None, self.reference.sid)
        return jacobian @ self.data.qvel

    def observation(self):
        r, c = self.reference, self.reference.controller
        desired, _ = r.desired(self.data)
        phase = np.zeros(len(STATES))
        phase[STATES.index(r.state)] = 1
        return np.concatenate((self.data.qpos[c.q], self.data.qvel[c.v],
                               self.data.site_xpos[r.sid] - r.center, self.tip_velocity(), r.normal,
                               [self.target_time - self.data.time, self.target_speed], desired,
                               [self.last_action], phase, self.parameters, self.shift)).astype(np.float32)

    def potential(self):
        r = self.reference
        desired, _ = r.desired(self.data)
        distance = float(np.linalg.norm(self.data.qpos[r.controller.q] - desired))
        return 0.5 * float(np.exp(-(distance / 0.12) ** 2))

    def info(self):
        r = self.reference
        hits = r.detector.events
        recovered = r.finished and r.failure is None and r.completed == 1
        qualified = (len(hits) == 1 and abs(hits[0]["onset_time_s"] - self.target_time) <= 0.04
                     and abs(hits[0]["downward_speed_m_s"] - self.target_speed) <= 0.25)
        return {"is_success": bool(recovered and qualified), "recovered": bool(recovered),
                "hit_count": len(hits), "unsafe_contact": r.failure is not None and not r.failure.startswith("Missed strike"),
                "failure": r.failure, "phase": r.state, "hit_quality": self.hit_quality,
                "peak_force_N": r.metrics["peak_force_N"], "max_penetration_m": r.metrics["max_penetration_m"],
                "target_time_s": float(self.target_time), "target_speed_m_s": self.target_speed,
                "kit_offset_m": self.shift.tolist(), "domain_scales": self.parameters.tolist(),
                "hits": [dict(hit) for hit in hits], "reward_version": REWARD_VERSION}

    def step(self, action):
        if not self.active:
            raise RuntimeError("Reset the environment before stepping")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Action must contain one finite value in [-1, 1]")
        r = self.reference
        before = len(r.detector.events)
        previous = self.last_action
        self.last_action = float(action[0])
        requested = 0.08 * self.last_action
        start = self.offset
        self.offset += float(np.clip(requested - self.offset, -0.015, 0.015))
        substeps = round(self.period / self.model.opt.timestep)
        self.energy = 0.0
        for index in range(substeps):
            offset = start + (self.offset - start) * (index + 1) / substeps
            r.step(self.data, joint7_offset=offset)
            self.energy += float(np.sum(self.data.actuator_force ** 2)) * self.model.opt.timestep
            if r.finished:
                break
        terms = {"time": -0.002, "effort": -0.00001 * self.energy,
                 "action_change": -0.005 * (self.last_action - previous) ** 2, "impact": 0.0}
        for hit in r.detector.events[before:]:
            timing = (hit["onset_time_s"] - self.target_time) / 0.04
            speed = (hit["downward_speed_m_s"] - self.target_speed) / 0.25
            self.hit_quality = float(np.exp(-timing ** 2 - speed ** 2))
            terms["impact"] += 5 * self.hit_quality
        terminated = r.finished
        truncated = self.data.time >= 3.0 and not terminated
        info = self.info()
        terms["success"] = 20.0 * info["is_success"]
        terms["failure"] = -20.0 * info["unsafe_contact"] - 5.0 * (terminated and not info["recovered"])
        terms["extra_hit"] = -10.0 * max(0, len(r.detector.events) - max(1, before))
        terms["timeout"] = -5.0 * truncated
        potential = 0.0 if terminated or truncated else self.potential()
        terms["progress"] = self.gamma * potential - self.previous_potential
        self.previous_potential = potential
        self.active = not (terminated or truncated)
        return self.observation(), float(sum(terms.values())), terminated, truncated, {**info, "reward_terms": terms}

    def render(self):
        if self.render_mode != "rgb_array":
            raise RuntimeError("Use render_mode='rgb_array'")
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.45, 0, 0.85]
        camera.distance, camera.azimuth, camera.elevation = 2.4, 145, -22
        options = mujoco.MjvOption()
        options.sitegroup[:] = 0
        self.renderer.update_scene(self.data, camera=camera, scene_option=options)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
