import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from .camera import OverheadCamera
from .pickup import StickPickupEnv
from .observations import DrumObservation


class DrumStrikeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    action_scale = StickPickupEnv.action_scale
    tool_orientation = StickPickupEnv.tool_orientation

    def __init__(self, render_mode: str | None = None):
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or rgb_array")
        self.render_mode = render_mode
        self.pickup = StickPickupEnv()
        self.sim = self.pickup.sim
        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(39,), dtype=np.float32)
        self.camera: OverheadCamera | None = None
        self.drum_id = self.sim.model.geom("drumhead").id
        self.stick_id = self.sim.model.geom("drumstick").id
        self.tip_id = self.sim.model.site("stick_tip").id
        self.drum_center = self.sim.model.geom_pos[self.drum_id].copy()
        self.drum_top = self.drum_center[2] + 0.015
        self._episode_active = False

    def tip_position(self) -> NDArray[np.float64]:
        return self.sim.data.site_xpos[self.tip_id].copy()

    def _tip_velocity(self) -> NDArray[np.float64]:
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.sim.model, self.sim.data, mujoco.mjtObj.mjOBJ_SITE, self.tip_id, velocity, 0)
        return velocity[3:]

    def _observation(self) -> NDArray[np.float32]:
        qids = self.sim.qpos_ids["right"]
        vids = self.sim.model.jnt_dofadr[self.sim.joint_ids["right"]]
        return DrumObservation(
            self.sim.data.qpos[qids], self.sim.data.qvel[vids],
            float(self.sim.data.qpos[self.pickup.grip_qpos_id]), float(self.sim.data.qvel[self.pickup.grip_dof_id]),
            self.tip_position(), self._tip_velocity(), self.sim.data.xquat[self.pickup.stick_body_id],
            self.pickup.stick_position(), self.sim.tcp_position("right"),
            np.array([*self.drum_center[:2], self.drum_top]),
            self.target_hit_time - self.elapsed, self.target_speed, self.last_action,
        ).vector()

    def _info(self) -> dict:
        return {
            "is_success": self.success,
            "hit_count": len(self.hit_events),
            "last_hit": self.hit_events[-1] if self.hit_events else None,
            "target_hit_time_s": self.target_hit_time,
            "target_speed_m_s": self.target_speed,
            "elapsed_s": self.elapsed,
            "approach_error_m": self.approach_error,
            "approach_sim_time": self.approach_sim_time,
            "grasped": self.pickup.grasp_contacts() == 2,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options:
            raise ValueError("Strike reset does not accept additional options")
        self.pickup.reset(seed=seed)
        # 通过参考控制器生成已夹持 drumstick 的初始状态。
        for action in self.pickup.reference_pickup_actions():
            _, _, terminated, truncated, info = self.pickup.step(action)
            if terminated or truncated:
                break
        if not info["is_success"]:
            raise RuntimeError(f"Could not establish the strike starting grasp: {info}")
        self.drum_center = np.array([0.55, -0.13, 0.36]) + self.np_random.uniform(
            [-0.005, -0.005, -0.003], [0.005, 0.005, 0.003]
        )
        self.drum_top = self.drum_center[2] + 0.015
        self.sim.model.geom_pos[self.drum_id] = self.drum_center
        mujoco.mj_forward(self.sim.model, self.sim.data)
        tip_offset = self.tip_position() - self.sim.tcp_position("right")
        preparation = self.drum_center + [0, 0, 0.015 + 0.10] - tip_offset
        goal = self.sim._solve("right", preparation, self.tool_orientation)
        if goal is None:
            raise RuntimeError("Strike preparation pose is unreachable")
        ids = self.sim.actuator_ids["right"]
        start = self.sim._desired[ids].copy()
        goal = goal[self.sim.qpos_ids["right"]]
        for index in range(2000):
            s = min((index + 1) / 1500, 1.0)
            blend = 10 * s**3 - 15 * s**4 + 6 * s**5
            self.sim._desired[ids] = start + blend * (goal - start)
            self.sim.step()
            if self.pickup.unsafe_contacts():
                raise RuntimeError("Collision during strike preparation")
        if self.pickup.grasp_contacts() != 2:
            raise RuntimeError("The stick was lost during strike preparation")
        self.approach_error = float(np.linalg.norm(self.tip_position() - (self.drum_center + [0, 0, 0.115])))
        self.approach_sim_time = float(self.sim.data.time)
        if self.approach_error > 0.01:
            raise RuntimeError(f"Strike preparation error exceeds 10 mm: {self.approach_error}")
        self.elapsed = 0.0
        self.target_hit_time = float(self.np_random.uniform(0.7, 1.1))
        self.target_speed = float(self.np_random.uniform(0.4, 0.8))
        self.last_action = np.zeros(8)
        self.last_action[7] = -1
        self.hit_events: list[dict] = []
        self.success = False
        self._touching = False
        self._lost_time = 0.0
        self._episode_active = True
        return self._observation(), self._info()

    def step(self, action: NDArray[np.float32]):
        if not self._episode_active:
            raise RuntimeError("Call reset before stepping a finished episode")
        action = np.asarray(action, dtype=np.float32)
        self.pickup.set_action_targets(action)
        unsafe = False
        new_hit_reward = 0.0
        for _ in range(round(0.02 / self.sim.model.opt.timestep)):
            downward_speed = max(0.0, -float(self._tip_velocity()[2]))
            self.sim.step()
            self.elapsed += self.sim.model.opt.timestep
            unsafe |= self.pickup.unsafe_contacts()
            touching = False
            normal_force = 0.0
            for index, contact in enumerate(self.sim.data.contact):
                if {int(contact.geom1), int(contact.geom2)} == {self.drum_id, self.stick_id}:
                    touching = True
                    force = np.empty(6)
                    mujoco.mj_contactForce(self.sim.model, self.sim.data, index, force)
                    normal_force += max(0.0, float(force[0]))
            if touching and not self._touching and downward_speed > 0.02 and normal_force > 0:
                event = {"time_s": float(self.elapsed), "downward_speed_m_s": downward_speed, "normal_force_n": normal_force}
                self.hit_events.append(event)
                timing_error = abs(self.elapsed - self.target_hit_time)
                speed_error = abs(downward_speed - self.target_speed)
                new_hit_reward += 10 * np.exp(-(timing_error / 0.08)**2 - (speed_error / 0.3)**2)
            self._touching = touching
        tip = self.tip_position()
        reward = float(new_hit_reward - 0.1 * np.linalg.norm(tip[:2] - self.drum_center[:2]))
        reward -= float(0.002 * np.sum((action - self.last_action)**2))
        grasped = self.pickup.grasp_contacts() == 2
        self._lost_time = 0.0 if grasped else self._lost_time + 0.02
        correct_hits = [
            event for event in self.hit_events
            if abs(event["time_s"] - self.target_hit_time) < 0.08
            and abs(event["downward_speed_m_s"] - self.target_speed) < 0.3
        ]
        if correct_hits and len(self.hit_events) == 1 and grasped:
            reward += float(0.1 * np.clip((tip[2] - self.drum_top) / 0.08, 0, 1))
            self.success = bool(tip[2] > self.drum_top + 0.07 and np.linalg.norm(self._tip_velocity()) < 0.15)
        if len(self.hit_events) > 1:
            reward -= 0.5
        terminated = bool(self.success or unsafe or self._lost_time > 0.10)
        truncated = bool(self.elapsed >= 2.0 - 1e-9 and not terminated)
        self.last_action = action.copy()
        self._episode_active = not (terminated or truncated)
        info = {**self._info(), "unsafe_contact": bool(unsafe)}
        return self._observation(), reward - (5.0 if unsafe else 0.0), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            raise RuntimeError("Create the environment with render_mode='rgb_array'")
        if self.camera is None:
            self.camera = OverheadCamera(self.sim.model, plane_height=self.sim.table_height)
        return self.camera.capture(self.sim.data).rgb

    def close(self) -> None:
        if self.camera is not None:
            self.camera.close()
            self.camera = None
        self.pickup.close()
