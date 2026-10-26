import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .drumming import DrumStrikeEnv
from .execution import JointTrajectory, residual_strike_command
from .observations import residual_observation


class ResidualDrumStrikeEnv(gym.Wrapper):
    def __init__(self, render_mode=None, curriculum_stage: int = 2, residual_scale: float = 0.08, randomization: float = 0.1):
        if curriculum_stage not in (0, 1, 2) or not 0 <= randomization <= 0.2 or not 0 < residual_scale <= 0.15:
            raise ValueError("Invalid curriculum stage, residual scale, or randomization range")
        super().__init__(DrumStrikeEnv(render_mode=render_mode))
        self.curriculum_stage = curriculum_stage
        self.residual_scale = residual_scale
        self.randomization = randomization
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(51,), dtype=np.float32)
        self.sim = self.env.sim
        self.reference: JointTrajectory | None = None
        self.last_residual = np.zeros(8)
        self.nominal_mass = float(self.sim.model.body_mass[self.env.pickup.stick_body_id])
        self.nominal_inertia = self.sim.model.body_inertia[self.env.pickup.stick_body_id].copy()
        self.nominal_friction = self.sim.model.geom_friction[self.env.stick_id].copy()

    def _reference_tip(self, time_s):
        data = mujoco.MjData(self.sim.model)
        data.qpos[:] = self.sim.data.qpos
        data.qpos[self.sim.qpos_ids["right"]] = self.reference.sample(time_s)
        mujoco.mj_forward(self.sim.model, data)
        return data.site_xpos[self.sim.site_ids["right"]] + self.tip_offset

    def _observation(self, observation):
        t = min(self.env.elapsed + 0.02, 2.0)
        return residual_observation(observation, self.reference.sample(t), self._reference_tip(t),
                                    self.env.tip_position(), self.env.elapsed / 2.0, self.curriculum_stage)

    def reset(self, *, seed=None, options=None):
        rng = np.random.default_rng(seed)
        mass_scale, friction_scale = rng.uniform(1 - self.randomization, 1 + self.randomization, 2)
        body_id = self.env.pickup.stick_body_id
        self.sim.model.body_mass[body_id] = self.nominal_mass * mass_scale
        self.sim.model.body_inertia[body_id] = self.nominal_inertia * mass_scale
        self.sim.model.geom_friction[self.env.stick_id] = self.nominal_friction * friction_scale
        mujoco.mj_setConst(self.sim.model, self.sim.data)
        observation, info = self.env.reset(seed=seed, options=options)
        self.tip_offset = self.env.tip_position() - self.sim.tcp_position("right")
        start = self.sim.data.qpos[self.sim.qpos_ids["right"]].copy()
        contact_position = self.env.drum_center + [0, 0, 0.015] - self.tip_offset
        bottom_position = contact_position - [0, 0, 0.012]
        contact = self.sim._solve("right", contact_position, self.env.tool_orientation)
        bottom = self.sim._solve("right", bottom_position, self.env.tool_orientation)
        if contact is None or bottom is None:
            raise RuntimeError("The nominal strike reference is unreachable")
        ids = self.sim.qpos_ids["right"]
        data = mujoco.MjData(self.sim.model)
        data.qpos[:] = contact
        mujoco.mj_forward(self.sim.model, data)
        jacobian = np.zeros((3, self.sim.model.nv))
        rotation_jacobian = np.zeros_like(jacobian)
        mujoco.mj_jacSite(self.sim.model, data, jacobian, rotation_jacobian, self.sim.site_ids["right"])
        dofs = self.sim.model.jnt_dofadr[self.sim.joint_ids["right"]]
        contact_velocity = np.linalg.lstsq(
            np.vstack([jacobian[:, dofs], rotation_jacobian[:, dofs]]),
            [0, 0, -self.env.target_speed, 0, 0, 0], rcond=None,
        )[0]
        hit_time = self.env.target_hit_time
        self.reference = JointTrajectory(
            "right", np.array([0, hit_time, hit_time + 0.06, hit_time + 0.5, 2.0]),
            np.array([start, contact[ids], bottom[ids], start, start]),
            np.array([np.zeros(7), contact_velocity, np.zeros(7), np.zeros(7), np.zeros(7)]),
            "nominal_strike_reference",
        )
        self.last_residual = np.zeros(8)
        self.domain_parameters = {"stick_mass_kg": float(self.sim.model.body_mass[body_id]), "friction_scale": float(friction_scale)}
        return self._observation(observation), {**info, "curriculum_stage": self.curriculum_stage, **self.domain_parameters}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Residual action must contain eight finite values in [-1, 1]")
        reference = self.reference.sample(self.env.elapsed + 0.02)
        current = self.sim._desired[self.sim.actuator_ids["right"]]
        command = residual_strike_command(reference, current, action, self.residual_scale)
        low_level = np.zeros(8, dtype=np.float32)
        low_level[:7] = np.clip((command.position - current) / self.env.action_scale, -1, 1)
        low_level[7] = command.gripper / 0.022 - 1
        previous_hits = len(self.env.hit_events)
        observation, task_reward, terminated, truncated, info = self.env.step(low_level)
        tracking_error = float(np.linalg.norm(self._reference_tip(self.env.elapsed) - self.env.tip_position()))
        grasp_reward = float(info["grasped"])
        trajectory_reward = float(np.exp(-(tracking_error / 0.04)**2)) * grasp_reward
        energy_penalty = float(0.0001 * np.sum(self.sim.data.actuator_force**2))
        action_penalty = float(0.01 * np.sum((action - self.last_residual)**2))
        reward = grasp_reward + 2 * trajectory_reward - energy_penalty - action_penalty
        if self.curriculum_stage == 1:
            reward += 3.0 * (info["hit_count"] - previous_hits)
        elif self.curriculum_stage == 2:
            reward += task_reward
        if info["unsafe_contact"]:
            reward -= 5.0
        self.last_residual = action.copy()
        return self._observation(observation), reward, terminated, truncated, {
            **info, "curriculum_stage": self.curriculum_stage, "reference_error_m": tracking_error,
            "reward_terms": {"grasp": grasp_reward, "trajectory": trajectory_reward, "task": task_reward,
                             "energy_penalty": energy_penalty, "action_penalty": action_penalty},
            **self.domain_parameters,
        }
