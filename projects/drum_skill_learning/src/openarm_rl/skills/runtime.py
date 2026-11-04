import json
from pathlib import Path
import threading

import mujoco
import numpy as np

from openarm_drummer.model import HoldController
from openarm_drummer.strike import HitDetector
from openarm_drummer.transfers import TransferTrajectory, clearance_checker, plan_transfer

from ..deployment.drum_actor import DrumActor
from .registry import SKILLS


def prepare_transfers(model, controller, current, goal):
    state = np.array(current, dtype=float).copy()
    transfers = []
    for arm, selected in (("left", slice(0, 7)), ("right", slice(7, 14))):
        end = state.copy()
        end[selected] = goal[selected]
        if np.max(np.abs(end - state)) < .005:
            continue
        path = plan_transfer(model, controller, state, end, arm=arm)
        trajectory = TransferTrajectory(path, np.full(14, .5), np.full(14, 1.2),
                                        clearance_checker(model, controller, state, end, arm))
        transfers.append((arm, trajectory))
        state = end
    return transfers


class SimDrumTools:
    def __init__(self, manifest_file, allow_experimental=False):
        self.path = Path(manifest_file)
        self.document = json.loads(self.path.read_text())
        self.allow_experimental = allow_experimental
        if self.document["schema"] != "openarm.camera_free_tools.v1":
            raise ValueError("Unsupported tools manifest")
        directory = Path(self.document["scene_directory"])
        if not directory.is_absolute():
            directory = self.path.parent / directory
        self.model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
        self.metadata = json.loads((directory / "scene.json").read_text())
        self.controller = HoldController(self.model, self.metadata["robot"])
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.reservation = threading.Lock()
        self.cancelled = threading.Event()

    def schemas(self):
        return [skill.schema() for skill in SKILLS.values()]

    def checkpoint(self, target):
        row = self.document["targets"][target]
        if row["status"] != "simulation_qualified" and not (self.allow_experimental and row["status"] == "trained_unqualified"):
            raise ValueError(f"Skill is unavailable: {row['status']}")
        path = Path(row["checkpoint"])
        return path if path.is_absolute() else self.path.parent / path

    def call(self, name, target_speed_m_s=.6, timing_offset_s=0.0):
        if not np.isfinite(target_speed_m_s) or not .35 <= target_speed_m_s <= .85:
            raise ValueError("Requested speed is outside the trained range")
        matching = [skill for skill in SKILLS.values() if skill.name == name]
        if len(matching) != 1:
            raise ValueError("Unknown drum tool")
        if timing_offset_s != 0:
            raise ValueError("Camera-free tools use a fixed onset; timing offset must be zero")
        if not self.reservation.acquire(blocking=False):
            raise RuntimeError("Another robot skill is executing")
        try:
            return self.execute(matching[0], target_speed_m_s)
        finally:
            self.reservation.release()

    def execute(self, skill, speed):
        actor = DrumActor(self.checkpoint(skill.target))
        c = self.controller
        if actor.manifest["scene_identity"] != json.loads((actor.directory / "profile.json").read_text())["scene_identity"]:
            raise ValueError("Actor/profile identity differs")
        from ..planning.strikes import scene_identity

        if actor.manifest["scene_identity"] != scene_identity(self.metadata["robot"], self.metadata["scene"]):
            raise ValueError("Actor belongs to another annotated scene")
        self.cancelled.clear()
        start_time = float(self.data.time)
        for _, trajectory in prepare_transfers(self.model, c, self.data.qpos[c.q], actor.reference.positions[0]):
            start = float(self.data.time)
            while self.data.time - start <= trajectory.ends[-1] + .6:
                if self.cancelled.is_set():
                    raise InterruptedError("Skill preparation cancelled")
                desired, _ = trajectory.sample(min(self.data.time - start, trajectory.ends[-1]))
                torque = c.kp * (desired - self.data.qpos[c.q]) - c.kd * self.data.qvel[c.v] + self.data.qfrc_bias[c.v]
                self.data.ctrl[c.a] = np.clip(torque, -c.effort, c.effort)
                mujoco.mj_step(self.model, self.data)
        for _ in range(1000):
            desired = actor.reference.positions[0]
            self.data.ctrl[c.a] = np.clip(c.kp * (desired - self.data.qpos[c.q]) - c.kd * self.data.qvel[c.v] + self.data.qfrc_bias[c.v], -c.effort, c.effort)
            mujoco.mj_step(self.model, self.data)
            if np.max(np.abs(self.data.qvel[c.v])) < .05 and np.max(np.abs(self.data.qpos[c.q] - desired)) < .02:
                break
        mujoco.mj_forward(self.model, self.data)
        actor.begin(self.data.qpos[c.q], self.data.qvel[c.v], self.data.time, self.data.time, speed)
        cfg = json.loads((actor.directory / "profile.json").read_text())["configuration"]
        detector = HitDetector(cfg)
        tip = self.model.geom(f"{skill.arm}_stick_tip_collision").id
        head = self.model.geom(f"{skill.target}_head_collision").id
        sid = self.model.site(f"{skill.arm}_stick_tip").id
        surface = next(s for s in self.metadata["scene"]["surfaces"] if s["name"] == skill.target)
        center, normal = np.array(surface["center"]), np.array(surface["normal"])
        unsafe = False
        while actor.running:
            if self.cancelled.is_set():
                actor.cancel()
                raise InterruptedError("Skill cancelled")
            command = actor.step(self.data.qpos[c.q], self.data.qvel[c.v], self.data.time, self.data.time)
            if command is None:
                break
            for _ in range(round(.01 / self.model.opt.timestep)):
                jacobian = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.data, jacobian, None, sid)
                impact_speed = max(0., float(-(jacobian @ self.data.qvel) @ normal))
                self.data.ctrl[c.a] = np.clip(c.kp * (command - self.data.qpos[c.q]) + c.kd * (actor.output_velocity - self.data.qvel[c.v]), -c.effort, c.effort)
                mujoco.mj_step(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)
                force = 0.
                for index, contact in enumerate(self.data.contact):
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, index, wrench)
                    if {contact.geom1, contact.geom2} == {tip, head}:
                        force += max(0., float(wrench[0]))
                        unsafe |= -float(contact.dist) > cfg["penetration_stop_m"]
                    elif wrench[0] > .05:
                        unsafe = True
                height = float((self.data.site_xpos[sid] - center) @ normal)
                hit = detector.update(float(self.data.time), force, height, self.model.opt.timestep)
                if hit is not None:
                    hit["center_error_m"] = float(np.linalg.norm(self.data.site_xpos[sid] - center - height * normal))
                    hit["downward_speed_m_s"] = impact_speed
                unsafe |= force > cfg["force_stop_N"]
        recovered = np.max(np.abs(self.data.qpos[c.q] - actor.reference.positions[-1])) < .03 and np.max(np.abs(self.data.qvel[c.v])) < .08
        qualified = (len(detector.events) == 1 and detector.events[0]["center_error_m"] <= .01
                     and detector.events[0]["contact_duration_s"] <= .12
                     and abs(detector.events[0]["downward_speed_m_s"] - speed) <= .3
                     and abs(detector.events[0]["onset_time_s"] - actor.started - actor.reference.impact_time) <= .06)
        return {"tool": skill.name, "arm": skill.arm, "hit_count": len(detector.events), "recovered": bool(recovered),
                "is_success": bool(qualified and recovered and not unsafe), "hit_verification": "simulator_evaluation",
                "unsafe_contact": unsafe, "hits": detector.events, "encoder_verified": True,
                "start_time_s": start_time, "end_time_s": float(self.data.time), "final_joints_rad": self.data.qpos[c.q].tolist()}
