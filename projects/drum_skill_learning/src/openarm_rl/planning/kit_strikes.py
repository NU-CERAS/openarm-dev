import argparse
from copy import deepcopy
from itertools import product
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import brentq

from openarm_drummer.model import HoldController
from openarm_drummer.strike_planning import plan_wrist_strike
from openarm_sim.compute_guard import budgeted, check_resources

from ..control.reference import JointReference
from ..skills.registry import SKILLS, StrikeSkill
from .strikes import plan_center, scene_identity


SHOULDER_HEIGHT = 1.0018
RIM_MARGIN = .03


def strike_point(surface: dict, arm: str, offset: float) -> np.ndarray:
    """Point `offset` metres from the surface center toward the arm's shoulder, in the surface plane."""
    center, normal = np.asarray(surface["center"]), np.asarray(surface["normal"])/np.linalg.norm(surface["normal"])
    inward = np.array([0., .0925 if arm == "left" else -.0925, SHOULDER_HEIGHT])-center
    inward -= (inward@normal)*normal
    return center+offset*inward/np.linalg.norm(inward)


def offset_scene(scene: dict, target: str, arm: str, offset: float) -> dict:
    shifted = deepcopy(scene)
    surface = next(row for row in shifted["surfaces"] if row["name"] == target)
    surface["center"] = strike_point(surface, arm, offset).tolist()
    surface["radius"] -= offset
    return shifted


def stroke_candidates(model, robot, scene, profile, wrist) -> list[dict]:
    documents = []
    for depth, downstroke, momentum in product((.003, .006), (.05, .07, .09), (0.,)):
        cfg = {**profile["configuration"], "command_depth_m": depth}
        plan = plan_wrist_strike(model, robot, scene, cfg, seed=np.asarray(profile["impact_seed"]),
                                 tip_direction=np.asarray(profile["tip_direction"]))
        if plan is None:
            continue
        inactive = np.arange(7, 14) if wrist == 6 else np.arange(7)
        for key in ("ready", "approach", "bottom", "impact_pose"):
            plan[key][inactive] = 0
        crossing = ((plan["impact_pose"][wrist]-plan["approach"][wrist])
                    / (plan["bottom"][wrist]-plan["approach"][wrist]))
        fraction = brentq(lambda x: x*x*(3-2*x)-crossing, 0, 1)
        times = np.cumsum([0, .10, .20, downstroke, .01, .28, .40])
        positions = np.array([plan["ready"], plan["ready"], plan["approach"], plan["bottom"],
                              plan["bottom"], plan["ready"], plan["ready"]])
        velocities = np.zeros_like(positions)
        velocities[2:4, wrist] = momentum*(plan["bottom"][wrist]-plan["ready"][wrist])/(.20+downstroke)
        documents.append(JointReference(times, positions, wrist, times[2]+fraction*downstroke, .3, .04, velocities).document())
    return documents


def tuned_stroke(ready, arm, impact, lift, depth, downstroke) -> dict:
    """Wrist stroke from a hand-tuned arm pose: joints 1-6 stay put and the wrist rises `lift` above impact."""
    wrist = 6 if arm == "left" else 13
    sign = float(np.sign(ready[6]-impact))
    pose = np.zeros(14)
    pose[slice(0, 7) if arm == "left" else slice(7, 14)] = ready
    top, approach, bottom = pose.copy(), pose.copy(), pose.copy()
    top[wrist], approach[wrist], bottom[wrist] = impact+sign*lift, impact+sign*.4*lift, impact-sign*depth
    crossing = (impact-approach[wrist])/(bottom[wrist]-approach[wrist])
    fraction = brentq(lambda x: x*x*(3-2*x)-crossing, 0, 1)
    times = np.cumsum([0, .10, .20, downstroke, .01, .28, .40])
    return JointReference(times, np.array([top, top, approach, bottom, bottom, top, top]), wrist,
                          times[2]+fraction*downstroke, .3, .04).document()


def first_contact_angle(model, controller, preset) -> float:
    """Wrist angle where a stick first meets the intended surface on the way to the tuned strike angle."""
    arm, drum = preset["arm"], preset["drum"]
    wrist = 6 if arm == "left" else 13
    head = model.geom(f"{drum}_head_collision").id
    pose = np.zeros(14)
    pose[slice(0, 7) if arm == "left" else slice(7, 14)] = preset["ready_rad"]
    data = mujoco.MjData(model)
    for angle in np.linspace(preset["ready_rad"][6], preset["strike_joint7_rad"], 600):
        pose[wrist] = angle
        data.qpos[controller.q] = pose
        mujoco.mj_forward(model, data)
        if any(contact.dist < 0 and head in (contact.geom1, contact.geom2) for contact in data.contact):
            return float(angle)
    raise ValueError(f"Tuned pose never reaches {drum}")


def tuned_profile(model, metadata, preset) -> dict:
    controller = HoldController(model, metadata["robot"])
    impact = first_contact_angle(model, controller, preset)
    full = abs(preset["ready_rad"][6]-impact)
    documents = [tuned_stroke(preset["ready_rad"], preset["arm"], impact, lift, depth, downstroke)
                 for lift, depth, downstroke in product(sorted({.3, .5, round(full, 3)}), (.015, .03), (.05, .07, .09))
                 if lift <= full+1e-6]
    return {"target": preset["drum"], "arm": preset["arm"], "status": "ready", "reason": None,
            "motion_type": "tuned_pose_wrist", "pose_source": "opendrummer_playground/configs/demo_poses.json",
            "scene_identity": scene_identity(metadata["robot"], metadata["scene"]), "impact_joint7_rad": impact,
            "reference_bank": {"schema": "openarm.speed_conditioned_strike.v1", "references": documents}}


def plan_assignment(model, metadata, target, arm, step=.04) -> dict:
    """Search strike points from the center outward; cymbals are normally played off center."""
    surface = next(row for row in metadata["scene"]["surfaces"] if row["name"] == target)
    identity = scene_identity(metadata["robot"], metadata["scene"])
    skill = StrikeSkill(f"strike_{target}", target, arm)
    attempts = []
    for offset in np.arange(0., surface["radius"]-RIM_MARGIN, step):
        check_resources()
        scene = offset_scene(metadata["scene"], target, arm, float(offset))
        profile = plan_center(model, metadata["robot"], scene, skill, allow_cross_body=True)
        attempts.append({"offset_m": float(offset), "status": profile["status"], "reason": profile["reason"]})
        print(json.dumps({"assignment": f"{target}.{arm}", **attempts[-1]}), flush=True)
        if profile["status"] != "ready":
            continue
        documents = stroke_candidates(model, metadata["robot"], scene, profile, skill.wrist)
        if not documents:
            continue
        profile.update(scene_identity=identity, strike_offset_m=float(offset), search=attempts,
                       strike_point_m=scene["surfaces"][[row["name"] for row in scene["surfaces"]].index(target)]["center"],
                       center_m=surface["center"],
                       reference_bank={"schema": "openarm.speed_conditioned_strike.v1", "references": documents})
        return profile
    return {"target": target, "arm": arm, "status": "unavailable", "search": attempts,
            "reason": "No collision-free wrist stroke at any searched strike point"}


@budgeted("openarm_rl.planning.kit_strikes")
def main():
    parser = argparse.ArgumentParser(description="Find strike points and wrist strokes for hand-to-drum assignments at original drum positions")
    parser.add_argument("--scene-directory", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--assignments", nargs="+", help="target:arm pairs, for example hi_hat:left")
    source.add_argument("--tuned-poses", type=Path, help="Hand-tuned ready poses, for example configs/demo_poses.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = [tuple(row.split(":")) for row in args.assignments or ()]
    if any(len(pair) != 2 or pair[0] not in SKILLS or pair[1] not in ("left", "right") for pair in pairs):
        raise ValueError("Assignments require target:left or target:right with a supported drum")
    args.output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_binary_path(str(args.scene_directory/"model.mjb"))
    metadata = json.loads((args.scene_directory/"scene.json").read_text())
    results = {}
    if args.tuned_poses:
        for preset in json.loads(args.tuned_poses.read_text())["presets"].values():
            profile = tuned_profile(model, metadata, preset)
            (args.output/f"{preset['drum']}.json").write_text(json.dumps(profile, indent=2))
            results[preset["drum"]] = {"impact_joint7_rad": profile["impact_joint7_rad"],
                                       "candidates": len(profile["reference_bank"]["references"])}
        (args.output/"reachability.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results), flush=True)
    for target, arm in pairs:
        profile = plan_assignment(model, metadata, target, arm)
        name = f"{target}.json" if SKILLS[target].arm == arm else f"{target}.{arm}.json"
        results[f"{target}.{arm}"] = {key: profile.get(key) for key in ("status", "reason", "strike_offset_m", "search")}
        if profile["status"] == "ready":
            (args.output/name).write_text(json.dumps(profile, indent=2))
        (args.output/"reachability.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({"assignment": f"{target}.{arm}", "status": profile["status"],
                          "strike_offset_m": profile.get("strike_offset_m")}), flush=True)


if __name__ == "__main__":
    main()
