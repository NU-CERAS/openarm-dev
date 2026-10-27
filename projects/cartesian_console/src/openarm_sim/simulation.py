from dataclasses import dataclass
from pathlib import Path

import mink
import mujoco
import numpy as np
from numpy.typing import NDArray


from .paths import BODY_REFERENCE, MODEL, ROBOT_CONFIGS, ROOT, RUNS
ARMS = ("right", "left")


def camera_quaternion(position: NDArray[np.float64], look_at: NDArray[np.float64]) -> NDArray[np.float64]:
    viewing_direction = look_at - position
    if np.linalg.norm(viewing_direction) < 1e-8:
        raise ValueError("Camera position and look-at point must differ")
    viewing_direction /= np.linalg.norm(viewing_direction)
    camera_right = np.cross(viewing_direction, [0, 0, 1])
    if np.linalg.norm(camera_right) < 1e-8:
        raise ValueError("Camera viewing direction must have a horizontal component")
    camera_right /= np.linalg.norm(camera_right)
    camera_up = np.cross(camera_right, viewing_direction)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.column_stack((camera_right, camera_up, -viewing_direction)).ravel())
    return quaternion


@dataclass(frozen=True)
class ReachAcceptance:
    accepted: bool
    reason: str


@dataclass
class ReachStatus:
    state: str = "idle"
    arm: str = "right"
    error_m: float = 0.0
    reason: str = ""
    target: NDArray[np.float64] | None = None


class ReachSimulation:
    table_height = 0.30
    control_period = 0.01

    def __init__(self, drumming: bool = False, free_stick: bool = False, planner: str = "mink", planning_assets: Path = RUNS / "planning/openarm.yml", obstacles: tuple[dict, ...] = (), drum_asset: Path | None = None, worktable: bool = True, tabletop_mount: bool = False):
        if planner not in ("mink", "ompl", "curobo"):
            raise ValueError("planner must be mink, ompl or curobo")
        self.planner_backend = planner
        self.planning_assets = planning_assets
        self.motion_planner = None
        self.trajectory = None
        if tabletop_mount and (drum_asset is not None or not worktable):
            raise ValueError("Tabletop mounting requires the worktable scene")
        self.tabletop_mount = tabletop_mount
        self.table_height = 0.0 if tabletop_mount else type(self).table_height
        free_stick = free_stick or drumming
        spec = mujoco.MjSpec.from_file(str(MODEL))
        spec.body(BODY_REFERENCE["body"]).add_site(name=BODY_REFERENCE["site"],
            pos=BODY_REFERENCE["position_m"], quat=BODY_REFERENCE["quaternion_wxyz"],
            size=[0.006, 0, 0], rgba=[0.3, 0.9, 1, 1], group=4)
        if worktable:
            spec.worldbody.add_geom(
                name="worktable", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0.30, 0, -0.04] if tabletop_mount else [0.46, 0, self.table_height - 0.025],
                size=[0.80, 0.65, 0.04] if tabletop_mount else [0.30, 0.60, 0.025],
                rgba=[0.24, 0.32, 0.37, 1], friction=[0.8, 0.01, 0.001],
            )
        if tabletop_mount:
            spec.geom("floor").pos = [0, 0, -0.75]
            for index, (x, y) in enumerate(((-0.43, -0.58), (-0.43, 0.58), (1.03, -0.58), (1.03, 0.58))):
                spec.worldbody.add_geom(name=f"table_leg_{index}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                        pos=[x, y, -0.415], size=[0.035, 0.035, 0.335], rgba=[0.18, 0.22, 0.25, 1])
        self.drum_asset = None
        if drum_asset is not None:
            from .drum_asset import load_asset

            self.drum_asset = load_asset(spec, drum_asset)
        for obstacle in obstacles:
            spec.worldbody.add_geom(name=obstacle["name"], type=mujoco.mjtGeom.mjGEOM_BOX,
                                    pos=obstacle["position"], size=obstacle["half_size"], rgba=[0.85, 0.25, 0.18, 1])
        # 官方 v2 Cell 头部相机中点，使用相对机械臂安装座的位置。
        # 来源：enactic/openarm_mujoco a3c8a57，v2/cell/cell.xml。
        official_mount = np.array([0.185, 0.0, 1.34])
        official_camera = np.array([0.223, 0.0, 1.45])
        local_mount = np.array([0.0, 0.0, spec.body("openarm_left_link0").pos[2]])
        camera_position = local_mount + official_camera - official_mount
        self.camera_look_at = camera_position + np.array([0.469472, 0.0, -0.882948])
        camera_quat = camera_quaternion(camera_position, self.camera_look_at)
        spec.worldbody.add_camera(name="overhead", pos=camera_position, quat=camera_quat, fovy=52, resolution=[1280, 720])
        spec.worldbody.add_geom(
            name="reach_target", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.009, 0, 0], pos=[0.43, -0.24, 0.44],
            contype=0, conaffinity=0, rgba=[1, 0.65, 0.1, 0.8], group=4,
        )
        for arm in ARMS:
            spec.body(f"openarm_{arm}_hand_tcp").add_site(
                name=f"{arm}_tcp", pos=[0, 0, 0],
                size=[0.006, 0, 0], rgba=[0.1, 1, 0.4, 1],
            )
        if drumming or free_stick:
            spec.worldbody.add_geom(
                name="drumhead", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[0.55, -0.13, 0.36] if free_stick else [0.43, -0.24, 0.36], size=[0.10, 0.015, 0],
                rgba=[0.8, 0.8, 0.75, 1], friction=[0.7, 0.01, 0.001],
                solref=[0.008, 0.5], solimp=[0.9, 0.95, 0.001, 0.5, 2],
            )
        if free_stick:
            stick_position = [0.40, -0.32, self.table_height + 0.008]
            stick = spec.worldbody.add_body(name="stick", pos=stick_position)
            stick.add_freejoint(name="stick_free")
            stick.add_geom(
                name="drumstick", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[-0.15, 0, 0, 0.15, 0, 0], size=[0.006, 0, 0],
                mass=0.04, rgba=[0.74, 0.49, 0.24, 1], friction=[0.9, 0.01, 0.001],
                condim=4, priority=1, solref=[0.005, 1],
            )
            stick.add_site(name="stick_tip", pos=[0.15, 0, 0], size=[0.005, 0, 0], rgba=[1, 0.3, 0.1, 1])
            spec.key("home").qpos = np.concatenate([spec.key("home").qpos, [*stick_position, 1, 0, 0, 0]])
        self.model = spec.compile()
        if free_stick:
            self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
            self.model.opt.impratio = 10
        self.model.vis.global_.offwidth = 1280
        self.model.vis.global_.offheight = 960
        self.model.vis.quality.offsamples = 4
        self.data = mujoco.MjData(self.model)
        self.joint_ids = {
            arm: np.array([self.model.joint(joint["name"]).id for joint in ROBOT_CONFIGS[arm]["joints"]])
            for arm in ARMS
        }
        self.qpos_ids = {arm: self.model.jnt_qposadr[ids] for arm, ids in self.joint_ids.items()}
        self.actuator_ids = {
            arm: np.array([self.model.actuator(f"{arm}_joint{i}_ctrl").id for i in range(1, 8)])
            for arm in ARMS
        }
        self.site_ids = {arm: self.model.site(f"{arm}_tcp").id for arm in ARMS}
        self.robot_bodies = {
            index for index in range(self.model.nbody)
            if self.model.body(index).name.startswith("openarm_")
        }
        self.actuated_dofs = self.model.jnt_dofadr[self.model.actuator_trnid[:, 0]]
        self.kp = self.model.actuator_gainprm[:, 0].copy()
        if np.any(self.kp <= 0):
            raise ValueError("Scene actuators must have positive position gains")
        self.status = ReachStatus()
        self._start_q = np.zeros(self.model.nu)
        self._goal_q = self._start_q.copy()
        self._motion_start = 0.0
        self._duration = 0.0
        self._stable_time = 0.0
        self.reset()
        if self.drum_asset is not None:
            self.set_camera(np.array([0.038, 0.0, 1.55]), np.array([0.70, -0.05, 0.65]), 60.0)

    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)
        self._desired = self.data.ctrl.copy()
        self.status = ReachStatus()
        self._stable_time = 0.0
        self.trajectory = None

    def reset_zero(self) -> None:
        self.reset()
        for arm in ARMS:
            self.data.qpos[self.qpos_ids[arm]] = 0
            self._desired[self.actuator_ids[arm]] = 0
        self.data.ctrl[:] = self._desired
        mujoco.mj_forward(self.model, self.data)

    def tcp_position(self, arm: str) -> NDArray[np.float64]:
        return self.data.site_xpos[self.site_ids[arm]].copy()

    def set_camera(self, position: NDArray[np.float64], look_at: NDArray[np.float64], fovy: float) -> None:
        position = np.asarray(position, dtype=float)
        look_at = np.asarray(look_at, dtype=float)
        if position.shape != (3,) or look_at.shape != (3,) or not np.isfinite([position, look_at]).all():
            raise ValueError("Camera position and look-at point must each contain three finite coordinates")
        if not np.isfinite(fovy) or not 5 <= fovy <= 120:
            raise ValueError("fovy must be between 5 and 120 degrees")
        quaternion = camera_quaternion(position, look_at)
        camera_id = self.model.camera("overhead").id
        self.model.cam_pos[camera_id] = position
        self.model.cam_quat[camera_id] = quaternion
        self.model.cam_fovy[camera_id] = fovy
        self.camera_look_at = look_at.copy()
        mujoco.mj_forward(self.model, self.data)

    def _collision(self, data: mujoco.MjData) -> bool:
        for contact in data.contact:
            if contact.dist >= -0.001:
                continue
            bodies = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if any(int(body) in self.robot_bodies for body in bodies):
                return True
        return False

    def _solve(self, arm: str, target: NDArray[np.float64], orientation: NDArray[np.float64] | None = None) -> NDArray[np.float64] | None:
        configuration = mink.Configuration(self.model, q=self.data.qpos)
        task = mink.FrameTask(
            frame_name=f"{arm}_tcp", frame_type="site", position_cost=1.0,
            orientation_cost=0.0 if orientation is None else 1.0, lm_damping=1e-3,
        )
        transform = configuration.get_transform_frame_to_world(f"{arm}_tcp", "site")
        rotation = transform.rotation() if orientation is None else mink.SO3.from_matrix(orientation)
        task.set_target(mink.SE3.from_rotation_and_translation(rotation, target))
        posture = mink.PostureTask(self.model, cost=1e-4)
        posture.set_target(self.data.qpos)
        selected = set(self.joint_ids[arm])
        velocities = {
            self.model.joint(i).name: 0.7 if i in selected else 0.0
            for i in range(self.model.njnt)
            if self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
        }
        limits = [mink.ConfigurationLimit(self.model), mink.VelocityLimit(self.model, velocities)]
        free_dofs = [
            int(self.model.jnt_dofadr[index]) + offset
            for index in range(self.model.njnt)
            if self.model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE
            for offset in range(6)
        ]
        constraints = [mink.DofFreezingTask(self.model, free_dofs)] if free_dofs else None
        for _ in range(350):
            velocity = mink.solve_ik(configuration, [task, posture], 0.02, "daqp", limits=limits, constraints=constraints)
            configuration.integrate_inplace(velocity, 0.02)
            position = configuration.data.site_xpos[self.site_ids[arm]]
            angular_error = np.linalg.norm(task.compute_error(configuration)[3:])
            if np.linalg.norm(position - target) < 0.002 and (orientation is None or angular_error < 0.03):
                return configuration.q.copy()
        return None

    def start_reach(self, arm: str, target: NDArray[np.float64], orientation: NDArray[np.float64] | None = None) -> ReachAcceptance:
        if arm not in ARMS:
            raise ValueError("arm must be right or left")
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Target must contain three finite coordinates")
        if self.status.state == "running":
            return ReachAcceptance(False, "Motion is running; wait for completion or cancel it")
        minimum_x = -0.10 if self.tabletop_mount else 0.14
        if not (minimum_x <= target[0] <= 0.70 and abs(target[1]) <= 0.52 and self.table_height + 0.04 <= target[2] <= 0.85):
            return ReachAcceptance(False, "Target is outside the demo workspace")
        if (arm == "right" and target[1] > -0.025) or (arm == "left" and target[1] < 0.025):
            return ReachAcceptance(False, "Select a target on the chosen arm's side")
        self.trajectory = None
        if self.planner_backend in ("ompl", "curobo"):
            if self.motion_planner is None:
                if self.planner_backend == "curobo":
                    from .planning.curobo import CuroboPlanner

                    self.motion_planner = CuroboPlanner(self, self.planning_assets)
                else:
                    from .planning.ompl import OmplPlanner

                    self.motion_planner = OmplPlanner(self)
            planning = self.motion_planner.plan(arm, target, orientation)
            if planning.trajectory is None:
                return ReachAcceptance(False, planning.reason)
            trajectory = planning.trajectory
            goal = self.data.qpos.copy()
            goal[self.qpos_ids[arm]] = trajectory.positions[-1]
            path = trajectory.positions
        else:
            goal = self._solve(arm, target, orientation)
            if goal is None:
                return ReachAcceptance(False, "IK could not reach the target within joint limits")
            trajectory = None
            path = np.array([self.data.qpos[self.qpos_ids[arm]] + fraction * (goal[self.qpos_ids[arm]] - self.data.qpos[self.qpos_ids[arm]]) for fraction in np.linspace(0, 1, 80)])
        collision_data = mujoco.MjData(self.model)
        for position in path:
            collision_data.qpos[:] = self.data.qpos
            collision_data.qpos[self.qpos_ids[arm]] = position
            mujoco.mj_forward(self.model, collision_data)
            if self._collision(collision_data):
                return ReachAcceptance(False, "The planned joint path contains a collision")
        self._start_q = self.data.qpos[self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]].copy()
        self._goal_q = self._start_q.copy()
        self._goal_q[self.actuator_ids[arm]] = goal[self.qpos_ids[arm]]
        delta = float(np.max(np.abs(self._goal_q - self._start_q)))
        self._duration = max(1.2, 1.875 * delta / 0.6, np.sqrt(5.8 * delta / 1.5))
        if trajectory is not None:
            self.trajectory = trajectory
            self._duration = trajectory.duration
        self._motion_start = float(self.data.time)
        self._stable_time = 0.0
        self.model.geom_pos[self.model.geom("reach_target").id] = target
        self.status = ReachStatus("running", arm, target=target.copy())
        return ReachAcceptance(True, "Target accepted")

    def cancel(self) -> None:
        if self.status.state == "running":
            self._desired = self.data.qpos[self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]].copy()
            self.status.state = "cancelled"
            self.status.reason = "Motion cancelled"

    def step(self) -> None:
        if self.status.state == "running":
            s = np.clip((self.data.time - self._motion_start) / self._duration, 0, 1)
            blend = 10 * s**3 - 15 * s**4 + 6 * s**5
            self._desired = self._start_q + blend * (self._goal_q - self._start_q)
            if self.trajectory is not None:
                self._desired[self.actuator_ids[self.status.arm]] = self.trajectory.sample(self.data.time - self._motion_start)
        # 通过 servo 目标位置加入重力前馈，并保留力矩限制。
        command = self._desired + self.data.qfrc_bias[self.actuated_dofs] / self.kp
        self.data.ctrl[:] = np.clip(command, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if self.data.warning.number.any() or not np.isfinite(self.data.qpos).all():
            raise RuntimeError("Invalid MuJoCo physics state")
        if self.status.state != "running":
            return
        self.status.error_m = float(np.linalg.norm(self.tcp_position(self.status.arm) - self.status.target))
        speed = np.max(np.abs(self.data.qvel[self.model.jnt_dofadr[self.joint_ids[self.status.arm]]]))
        if self.status.error_m < 0.01 and speed < 0.08:
            self._stable_time += self.model.opt.timestep
        else:
            self._stable_time = 0.0
        if self._stable_time >= 0.2:
            self.status.state = "succeeded"
        elif self.data.time - self._motion_start > self._duration + 4.0:
            self.status.state = "failed"
            self.status.reason = "Target was not reached before the execution timeout"
        elif self._collision(self.data):
            self.cancel()
            self.status.state = "failed"
            self.status.reason = "Collision detected during execution"

    def close(self) -> None:
        self.cancel()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
