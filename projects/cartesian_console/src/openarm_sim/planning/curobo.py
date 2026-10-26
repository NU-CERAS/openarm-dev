from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Scene
from curobo.types import GoalToolPose, JointState

from ..execution import JointTrajectory, joint_names
from ..paths import RUNS
from . import PlanningResult


class CuroboPlanner:
    def __init__(self, simulation, assets: Path = RUNS / "planning/openarm.yml"):
        if not torch.cuda.is_available():
            raise RuntimeError("cuRobo requires a compatible CUDA-enabled PyTorch environment")
        self.sim = simulation
        self.config = yaml.safe_load(assets.read_text())
        self.instances = {}
        self.last_report = {}

    def scene(self) -> dict:
        model, data = self.sim.model, self.sim.data
        cuboids = {}
        for index in range(model.ngeom):
            if int(model.geom_bodyid[index]) != 0 or not (model.geom_contype[index] or model.geom_conaffinity[index]):
                continue
            kind = model.geom_type[index]
            if kind == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                dimensions = model.geom_size[index] * 2
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                dimensions = np.array([2 * model.geom_size[index, 0]] * 2 + [2 * model.geom_size[index, 1]])
            else:
                raise ValueError(f"Unsupported world obstacle: {model.geom(index).name}")
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, data.geom_xmat[index])
            cuboids[model.geom(index).name] = {
                "dims": dimensions.tolist(), "pose": [*data.geom_xpos[index].tolist(), *quaternion.tolist()],
            }
        cuboids["ground"] = {"dims": [4, 4, 0.1], "pose": [0, 0, -0.05, 1, 0, 0, 0]}
        return {"cuboid": cuboids}

    def _planner(self, arm: str):
        selected = set(joint_names(arm))
        model, data = self.sim.model, self.sim.data
        locks = {
            model.joint(index).name: float(data.qpos[model.jnt_qposadr[index]])
            for index in range(model.njnt)
            if model.joint(index).name not in selected and model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE
        }
        signature = tuple(locks.items())
        if arm not in self.instances or self.instances[arm][0] != signature:
            config = deepcopy(self.config)
            kinematics = config.get("robot_cfg", config)["kinematics"]
            kinematics["tool_frames"] = [f"openarm_{arm}_hand_tcp"]
            kinematics["lock_joints"] = locks
            kinematics["cspace"]["default_joint_position"] = [
                float(data.qpos[model.jnt_qposadr[model.joint(name).id]])
                for name in kinematics["cspace"]["joint_names"]
            ]
            kinematics["cspace"]["max_acceleration"] = 1.5
            kinematics["cspace"]["max_jerk"] = 20.0
            planner = MotionPlanner(MotionPlannerCfg.create(
                robot=config, scene_model=self.scene(), interpolation_dt=0.01,
                position_tolerance=0.002, orientation_tolerance=0.03,
                optimizer_collision_activation_distance=0.015,
            ))
            if set(planner.joint_names) != selected:
                raise RuntimeError(f"Unexpected cuRobo joint mapping: {planner.joint_names}")
            self.instances[arm] = signature, planner
        planner = self.instances[arm][1]
        planner.update_world(Scene.create(self.scene()))
        return planner

    def plan(self, arm: str, target, orientation=None) -> PlanningResult:
        if self.sim.model.nq != 18:
            raise ValueError("Free-object transport needs an explicit attached-object collision model")
        if orientation is None:
            goal_q = self.sim._solve(arm, target)
            if goal_q is None:
                return PlanningResult(None, f"No feasible IK solution found for {arm} arm at {np.round(target, 4).tolist()} m")
            goal_data = mujoco.MjData(self.sim.model)
            goal_data.qpos[:] = goal_q
            mujoco.mj_forward(self.sim.model, goal_data)
            orientation = goal_data.site_xmat[self.sim.site_ids[arm]].reshape(3, 3)
        planner = self._planner(arm)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, np.asarray(orientation).ravel())
        start = JointState.from_position(torch.tensor([
            [self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint(name).id]] for name in planner.joint_names]
        ], dtype=torch.float32, device="cuda"), joint_names=planner.joint_names)
        goal = GoalToolPose(
            tool_frames=planner.tool_frames,
            position=torch.tensor(target, dtype=torch.float32, device="cuda").reshape(1, 1, 1, 1, 3),
            quaternion=torch.tensor(quaternion, dtype=torch.float32, device="cuda").reshape(1, 1, 1, 1, 4),
        )
        result = planner.plan_pose(goal, start)
        if result is None or not bool(result.success.all().item()):
            return PlanningResult(None, f"cuRobo found no collision-free trajectory for {arm} arm at {np.round(target, 4).tolist()} m")
        interpolated = result.get_interpolated_plan()
        indices = [interpolated.joint_names.index(name) for name in joint_names(arm)]
        width = len(interpolated.joint_names)
        positions = interpolated.position.reshape(-1, width).detach().cpu().numpy()[:, indices]
        velocities = interpolated.velocity.reshape(-1, width).detach().cpu().numpy()[:, indices]
        trajectory = JointTrajectory(arm, np.arange(len(positions)) * 0.01, positions, velocities, "curobo").retime()
        self.last_report = {
            "planner": "curobo", "waypoints": len(positions), "duration_s": trajectory.duration,
            "solve_time_s": float(result.solve_time), "obstacles": list(self.scene()["cuboid"]),
        }
        return PlanningResult(trajectory)
