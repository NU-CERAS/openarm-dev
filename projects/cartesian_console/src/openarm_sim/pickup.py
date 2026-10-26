from collections.abc import Iterator

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from .camera import OverheadCamera
from .simulation import ReachSimulation
from .pickup_reward import PickupReward, REWARD_VERSION


class StickPickupEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    action_scale = 0.025
    tool_orientation = np.array([[2**-0.5, 0, 2**-0.5], [0, -1, 0], [2**-0.5, 0, -(2**-0.5)]])

    def __init__(self, render_mode: str | None = None):
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or rgb_array")
        self.render_mode = render_mode
        self.sim = ReachSimulation(free_stick=True)
        self.camera: OverheadCamera | None = None
        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(57,), dtype=np.float32)
        self.reward_state = PickupReward(1)
        self.elapsed = 0.0
        self.stick_body_id = self.sim.model.body("stick").id
        self.stick_geom_id = self.sim.model.geom("drumstick").id
        self.stick_joint_id = self.sim.model.joint("stick_free").id
        self.grip_actuator_id = self.sim.model.actuator("right_finger1_ctrl").id
        grip_joint = self.sim.model.joint("openarm_right_finger_joint1").id
        self.grip_qpos_id = self.sim.model.jnt_qposadr[grip_joint]
        self.grip_dof_id = self.sim.model.jnt_dofadr[grip_joint]
        self.finger_geom_ids = {
            self.sim.model.geom("openarm_right_right_finger_collision").id,
            self.sim.model.geom("openarm_right_left_finger_collision").id,
        }
        self.active = False

    def stick_position(self) -> NDArray[np.float64]:
        return self.sim.data.xpos[self.stick_body_id].copy()

    def stick_velocity(self) -> NDArray[np.float64]:
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.sim.model, self.sim.data, mujoco.mjtObj.mjOBJ_BODY, self.stick_body_id, velocity, 0)
        return velocity

    def grasp_contacts(self) -> int:
        pads = set()
        for contact in self.sim.data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.stick_geom_id in pair:
                pads.update(pair & self.finger_geom_ids)
        return len(pads)

    def stick_supported(self) -> bool:
        for contact in self.sim.data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.stick_geom_id in pair and not (pair - {self.stick_geom_id}) <= self.finger_geom_ids:
                return True
        return False

    def _observation(self) -> NDArray[np.float32]:
        qids = self.sim.qpos_ids["right"]
        vids = self.sim.model.jnt_dofadr[self.sim.joint_ids["right"]]
        tcp_quat = np.empty(4)
        mujoco.mju_mat2Quat(tcp_quat, self.sim.data.site_xmat[self.sim.site_ids["right"]])
        return np.concatenate([
            self.sim.data.qpos[qids], self.sim.data.qvel[vids],
            [self.sim.data.qpos[self.grip_qpos_id], self.sim.data.qvel[self.grip_dof_id]],
            self.stick_position() - self.sim.tcp_position("right"),
            self.sim.data.xquat[self.stick_body_id], self.stick_velocity(), tcp_quat,
            [self.initial_height + 0.10 - self.stick_position()[2]], self.last_action,
            self.reward_state.features()[0], [self.grasp_contacts() / 2, float(self.stick_supported())],
            self.sim.model.geom_pos[self.sim.model.geom("drumhead").id] - self.sim.tcp_position("right"),
        ]).astype(np.float32)

    def info(self) -> dict:
        return {
            "is_success": self.success,
            "grasped": self.grasp_contacts() == 2,
            "gripper_contacts": self.grasp_contacts(),
            "lift_height_m": float(self.stick_position()[2] - self.initial_height),
            "stick_position": self.stick_position().tolist(),
            "elapsed_s": self.elapsed,
            "approach_error_m": self.approach_error,
            "max_lift_m": float(self.reward_state.best_height[0]),
            "milestone_2cm": bool(self.reward_state.awarded[0, 0]),
            "milestone_5cm": bool(self.reward_state.awarded[0, 1]),
            "reward_version": REWARD_VERSION,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options:
            raise ValueError("Pickup reset does not accept additional options")
        self.sim.reset()
        qadr = self.sim.model.jnt_qposadr[self.stick_joint_id]
        self.sim.data.qpos[qadr:qadr + 2] += self.np_random.uniform(-0.005, 0.005, 2)
        mujoco.mj_forward(self.sim.model, self.sim.data)
        for _ in range(100):
            self.sim.step()
        self.initial_height = float(self.stick_position()[2])
        target = self.stick_position()
        target[2] = self.sim.table_height + 0.12
        result = self.sim.start_reach("right", target, self.tool_orientation)
        if not result.accepted:
            raise RuntimeError(f"Cannot approach the stick: {result.reason}")
        for _ in range(12000):
            self.sim.step()
            if self.sim.status.state != "running":
                break
        if self.sim.status.state != "succeeded":
            raise RuntimeError(f"Stick approach failed: {self.sim.status}")
        self.approach_error = self.sim.status.error_m
        self.sim.status.state = "idle"
        self.elapsed = 0.0
        self.last_action = np.zeros(8)
        self.success = False
        distance = np.array([np.linalg.norm(self.stick_position() - self.sim.tcp_position("right"))])
        self.reward_state.reset(np.array([0]), distance)
        self.active = True
        return self._observation(), self.info()

    def step(self, action: NDArray[np.float32]):
        if not self.active:
            raise RuntimeError("Call reset before stepping a finished episode")
        action = np.asarray(action, dtype=np.float32)
        self.set_action_targets(action)
        unsafe = False
        for _ in range(round(0.02 / self.sim.model.opt.timestep)):
            self.sim.step()
            unsafe |= self.unsafe_contacts()
        self.elapsed += 0.02
        contacts = self.grasp_contacts()
        lift = float(self.stick_position()[2] - self.initial_height)
        distance = np.linalg.norm(self.stick_position() - self.sim.tcp_position("right"))
        reward, success, failed, timeout, terms = self.reward_state.step(
            np.array([distance]), np.array([contacts == 2]), np.array([lift]), np.array([self.stick_supported()]),
            np.array([np.linalg.norm(self.stick_velocity()[3:])]), np.array([unsafe]),
            np.array([self.stick_position()[2] < self.sim.table_height - 0.05]), action[None], self.last_action[None],
        )
        self.success = bool(success[0])
        terminated = bool(success[0] or failed[0] or timeout[0])
        truncated = False
        self.active = not (terminated or truncated)
        self.last_action = action.copy()
        info = {**self.info(), "unsafe_contact": bool(unsafe), "timeout": bool(timeout[0]),
                "reward_terms": {name: float(value[0]) for name, value in terms.items()}}
        return self._observation(), float(reward[0]), terminated, truncated, info

    def set_action_targets(self, action: NDArray[np.float32]) -> None:
        if not self.action_space.contains(action):
            raise ValueError("Action must contain eight finite values in [-1, 1]")
        ids = self.sim.actuator_ids["right"]
        limits = self.sim.model.actuator_ctrlrange[ids]
        self.sim._desired[ids] = np.clip(self.sim._desired[ids] + self.action_scale * action[:7], limits[:, 0], limits[:, 1])
        self.sim._desired[self.grip_actuator_id] = (float(action[7]) + 1) * 0.022

    def unsafe_contacts(self) -> bool:
        for contact in self.sim.data.contact:
            bodies = self.sim.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if contact.dist < -0.002 and self.stick_geom_id not in (contact.geom1, contact.geom2):
                if any(int(body) in self.sim.robot_bodies for body in bodies):
                    return True
        return False

    def reference_pickup_actions(self) -> Iterator[NDArray[np.float32]]:
        # 参考控制器通过物理接触完成抓取检查。
        for phase in ["lower", "close", "lift", "hold"]:
            start = self.sim._desired[self.sim.actuator_ids["right"]].copy()
            target = start.copy()
            if phase in ("lower", "lift"):
                position = self.stick_position()
                position[2] = self.sim.table_height + (0.012 if phase == "lower" else 0.15)
                goal = self.sim._solve("right", position, self.tool_orientation)
                if goal is None:
                    raise RuntimeError(f"Reference {phase} pose is unreachable")
                target = goal[self.sim.qpos_ids["right"]]
            steps = 75 if phase in ("lower", "lift") else 50 if phase == "close" else 30
            for index in range(steps):
                s = (index + 1) / steps
                blend = 10 * s**3 - 15 * s**4 + 6 * s**5
                desired = start + blend * (target - start)
                action = np.zeros(8, dtype=np.float32)
                action[:7] = np.clip((desired - self.sim._desired[self.sim.actuator_ids["right"]]) / self.action_scale, -1, 1)
                action[7] = 1.0 if phase == "lower" else 1 - 2 * blend if phase == "close" else -1.0
                yield action

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
        self.sim.close()
