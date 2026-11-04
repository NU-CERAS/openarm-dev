from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from openarm_drummer.model import HoldController
from openarm_drummer.collisions import CONTACT_CONFIG
from openarm_drummer.scene import solve_approach
from openarm_drummer.strike import STRIKE_CONFIG, StrikeSession
from openarm_drummer.strike_planning import plan_wrist_strike
from openarm_sim.compute_guard import budgeted

from .scene import prepare_scene
from ..skills.registry import SKILLS


def scene_identity(robot, scene):
    from openarm_sim.annotation import identity

    return identity(robot, scene)


def reach_bound(model, surface, arm):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    points = [data.xanchor[model.joint(f"{arm[0]}j{i}").id].copy() for i in range(1, 8)]
    points.append(data.site_xpos[model.site(f"{arm}_stick_tip").id].copy())
    return {"distance_m": float(np.linalg.norm(np.array(surface["center"]) - points[0])),
            "upper_bound_m": float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())}


def plan_center(model, robot, scene, skill):
    surface = next(s for s in scene["surfaces"] if s["name"] == skill.target)
    if (surface["center"][1] > 0) != (skill.arm == "left"):
        raise ValueError("The fixed skill arm does not match the target side")
    bounds = reach_bound(model, surface, skill.arm)
    record = {**asdict(skill), "center_m": surface["center"], "reach_bound": bounds,
              "scene_identity": scene_identity(robot, scene),
              "status": "unavailable", "reason": "No collision-free center wrist stroke found in the configured IK search"}
    if bounds["distance_m"] > bounds["upper_bound_m"]:
        record["reason"] = "Center exceeds the arm chain's geometric reach upper bound"
        return record
    normal = np.array(surface["normal"])
    shoulder = np.array([0, .0925 if skill.arm == "left" else -.0925, 1.0018])
    horizontal = np.array(surface["center"]) - shoulder
    horizontal[2] = 0
    horizontal /= np.linalg.norm(horizontal)
    controller = HoldController(model, robot)
    annotation = scene.get("annotations", {}).get("targets", {}).get(skill.target, {})
    if "ready" in annotation:
        cfg = {**json.loads(STRIKE_CONFIG.read_text()), "arm": skill.arm, "target": skill.target,
               "staging_mode": "wrist_ready", "center_tolerance_m": .01}
        captured = np.array(annotation["ready"]["joints_rad"])
        plan = plan_wrist_strike(model, robot, scene, cfg, captured_ready=captured)
        if plan is None:
            return {**record, "reason": "Captured ready pose cannot make a clear center wrist stroke", "attempts": 1}
        routine = StrikeSession(model, robot, scene, configuration=cfg, geometric_plan=plan)
        data = mujoco.MjData(model)
        routine.reset(data)
        while not routine.finished:
            routine.step(data)
        report = routine.report()
        if report["status"] != "PASS":
            return {**record, "reason": report.get("failure", "Captured reference failed native validation"), "attempts": 1}
        return {**record, "status": "ready", "reason": None, "attempts": 1, "configuration": cfg,
                "captured_ready_joints": captured.tolist(), "ready_source": annotation["ready"]["source"],
                "ready_joints": captured.tolist(), "tip_direction": plan["tip_direction"],
                "impact_seed": plan["impact_pose"].tolist(), "reference_report": report}
    attempts = 0
    for lean in (.65, 1.2, 2.0, 3.0, 6.0):
        for yaw in (0, -25, 25):
            print(json.dumps({"planning": skill.target, "arm": skill.arm, "lean": lean, "yaw": yaw}), flush=True)
            angle = np.radians(yaw)
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            direction = lean * rotation @ horizontal - normal
            seed = solve_approach(model, robot, scene, skill.arm, skill.target, height=0,
                                  tip_direction=direction, attempts=6)
            if not seed["reachable"]:
                continue
            suggested = float(np.clip(seed["joint_positions"][skill.wrist], -1.4, 1.4))
            angles = [suggested, *([-.9, -.7, -1.1, 0, .7, -1.35] if skill.arm == "left" else [.9, .7, 1.1, 0, -.7, 1.35])]
            for wrist_angle in angles:
                for ready_height, approach_height in ((.07, .04), (.04, .02)):
                    cfg = {**json.loads(STRIKE_CONFIG.read_text()), "arm": skill.arm, "target": skill.target,
                           "impact_joint7_rad": wrist_angle, "ready_height_m": ready_height, "approach_height_m": approach_height,
                           "staging_mode": "wrist_ready", "center_tolerance_m": .01}
                    attempts += 1
                    plan = plan_wrist_strike(model, robot, scene, cfg, seed=np.array(seed["joint_positions"]), tip_direction=direction)
                    if plan is None:
                        continue
                    delta = abs(plan["bottom"][skill.wrist] - plan["approach"][skill.wrist])
                    cfg["downstroke_seconds"] = max(.06, .09 * approach_height / .04, 1.5 * delta / (0.8 * controller.velocity[skill.wrist]))
                    routine = StrikeSession(model, robot, scene, configuration=cfg, impact_seed=plan["impact_pose"], tip_direction=direction)
                    data = mujoco.MjData(model)
                    routine.reset(data)
                    while not routine.finished:
                        routine.step(data)
                    report = routine.report()
                    if report["status"] != "PASS":
                        continue
                    return {**record, "status": "ready", "reason": None, "attempts": attempts,
                            "configuration": cfg, "tip_direction": np.asarray(direction).tolist(),
                            "impact_seed": routine.impact_pose.tolist(), "ready_joints": routine.staging.tolist(),
                            "reference_report": report}
    return {**record, "attempts": attempts}


@budgeted("openarm_rl.planning.strikes")
def main():
    parser = argparse.ArgumentParser(description="Validate center strike geometry and physical reference motions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path)
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--targets", nargs="+", choices=list(SKILLS), default=list(SKILLS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prepare_scene(args.output / "scene", args.scene_config, args.layout)
    model = mujoco.MjModel.from_binary_path(str(args.output / "scene/model.mjb"))
    metadata = json.loads((args.output / "scene/scene.json").read_text())
    profiles = {}
    for target in args.targets:
        profile = plan_center(model, metadata["robot"], metadata["scene"], SKILLS[target])
        profiles[target] = profile
        if profile["status"] == "ready":
            (args.output / f"{target}.json").write_text(json.dumps(profile, indent=2) + "\n")
        (args.output / "preflight.json").write_text(json.dumps(profiles, indent=2) + "\n")
        print(json.dumps({"target": target, "status": profile["status"], "reason": profile["reason"]}), flush=True)


if __name__ == "__main__":
    main()
