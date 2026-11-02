import mujoco
import numpy as np

from openarm_drummer.model import OUTPUT, HoldController
from openarm_drummer.scene import build_scene

from .paths import RUNS
from .simulation import ARMS, ReachSimulation, ReachStatus, camera_quaternion


class OpenDrummerReachSimulation(ReachSimulation):
    robot_model = "opendrummer"

    def __init__(self, planner="ompl", drum_asset=None, worktable=False, tabletop_mount=False,
                 planning_assets=RUNS / "planning/opendrummer/openarm.yml"):
        if planner not in ("mink", "ompl", "curobo"):
            raise ValueError("Unknown motion planner")
        if drum_asset is not None:
            raise ValueError("OpenDrummer uses its own shared drum asset")
        model, self.robot_report, self.scene_report = build_scene()
        spec = mujoco.MjSpec.from_file(str(OUTPUT / "scene.xml"))
        base = 0
        points = []
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for index in range(model.ngeom):
            if model.geom_bodyid[index] != base or model.geom_type[index] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if model.geom(index).name.startswith("kit_"):
                continue
            mesh = model.geom_dataid[index]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start:start + count]
            points.extend(vertices @ data.geom_xmat[index].reshape(3, 3).T + data.geom_xpos[index])
        if not points:
            raise ValueError("Fixed torso mesh is required to define the torso-top reference")
        points = np.asarray(points)
        top = points.max(axis=0)
        center = (points.min(axis=0) + top) / 2
        center[2] = top[2]
        self.body_reference = {"body": "world", "site": "torso_top_reference"}
        spec.worldbody.add_site(name="torso_top_reference", pos=center, size=[.006, 0, 0], group=4)
        position, look_at = np.array([.04, 0, 1.55]), np.array([.65, 0, .7])
        self.camera_look_at = look_at
        spec.worldbody.add_camera(name="overhead", pos=position, quat=camera_quaternion(position, look_at), fovy=60)
        spec.worldbody.add_geom(name="reach_target", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                               size=[.009, 0, 0], contype=0, conaffinity=0, group=4, rgba=[1, .65, .1, .8])
        self.model = spec.compile()
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = 1280, 960
        self.model.vis.quality.offsamples = 4
        self.data = mujoco.MjData(self.model)
        self.controller = HoldController(self.model, self.robot_report)
        self.joint_ids = {arm: np.array([self.model.joint(f"{arm[0]}j{i}").id for i in range(1, 8)]) for arm in ARMS}
        self.qpos_ids = {arm: self.model.jnt_qposadr[ids] for arm, ids in self.joint_ids.items()}
        self.actuator_ids = {arm: np.array([self.model.actuator(f"{arm[0]}j{i}_motor").id for i in range(1, 8)]) for arm in ARMS}
        self.site_ids = {arm: self.model.site(f"{arm}_stick_tip").id for arm in ARMS}
        self.robot_bodies = set(range(1, self.model.nbody))
        self.planner_backend, self.planning_assets = planner, planning_assets
        self.motion_planner = self.trajectory = None
        self.drum_asset, self.tabletop_mount, self.table_height = None, False, 0.0
        self.workspace = (np.array([-.5, -.9, .1]), np.array([1.1, .9, 1.8]))
        self._start_q = np.zeros(self.model.nu)
        self._goal_q = self._start_q.copy()
        self._motion_start = self._duration = self._stable_time = 0.0
        self.reset_zero()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._desired = np.zeros(self.model.nu)
        self.status = ReachStatus()
        self._stable_time = 0.0
        self.trajectory = None

    def reset_zero(self):
        self.reset()

    def _apply_control(self):
        c = self.controller
        target = self._desired[c.a]
        torque = c.kp * (target - self.data.qpos[c.q]) - c.kd * self.data.qvel[c.v] + self.data.qfrc_bias[c.v]
        self.data.ctrl[c.a] = np.clip(torque, -c.effort, c.effort)
