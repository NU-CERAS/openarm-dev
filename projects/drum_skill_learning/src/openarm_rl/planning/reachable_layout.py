import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from openarm_drummer.model import HoldController
from openarm_drummer.strike_planning import plan_wrist_strike
from openarm_sim.annotation import DrumLayout
from openarm_sim.compute_guard import budgeted, check_resources

from .scene import prepare_scene
from .strikes import plan_center, scene_identity
from ..skills.registry import SKILLS


def find_layout(model, robot, scene, profiles, seed):
    controller = HoldController(model, robot)
    data = mujoco.MjData(model)
    layout = DrumLayout.from_scene(model, robot, scene)
    rng = np.random.default_rng(seed)
    changes = {}
    for target in ("hi_hat", "cymbal_left", "cymbal_right"):
        skill = SKILLS[target]
        surface = next(s for s in scene["surfaces"] if s["name"] == target)
        normal, original = np.array(surface["normal"]), np.array(surface["center"])
        selected = np.arange(7) if skill.arm == "left" else np.arange(7, 14)
        sid = model.site(f"{skill.arm}_stick_tip").id
        radius = float(model.geom_size[model.geom(f"{skill.arm}_stick_tip_collision").id, 0])
        bases = [np.array(profile["impact_seed"]) for profile in profiles.values() if profile["arm"] == skill.arm]
        candidates = []
        for index in range(2500):
            pose = bases[index % len(bases)].copy()
            pose[selected] += rng.normal(0, [.45, .4, .3, .35, .3, .25, .2])
            pose[selected] = np.clip(pose[selected], controller.bounds[selected, 0] + .06, controller.bounds[selected, 1] - .06)
            pose[np.setdiff1d(np.arange(14), selected)] = 0
            data.qpos[controller.q] = pose
            mujoco.mj_forward(model, data)
            center = data.site_xpos[sid].copy() - normal*radius
            direction = data.site_xmat[sid].reshape(3, 3)[:, 2].copy()
            if (.16 < center[0] < .68 and .2 < abs(center[1]) < .61 and .59 < center[2] < .98
                    and direction @ normal < -.15):
                candidates.append((float(np.linalg.norm(center-original)), pose, center, direction))
        candidates.sort(key=lambda row: row[0])
        for _, pose, center, direction in candidates:
            check_resources()
            layout.set_position(target, (center-np.array(layout.document["origin_m"]))*100)
            layout.apply(model, scene)
            cfg = {**next(iter(profiles.values()))["configuration"], "arm": skill.arm, "target": target,
                   "impact_joint7_rad": float(pose[skill.wrist]), "impact_height_m": radius,
                   "ready_height_m": .05, "approach_height_m": .025}
            plan = plan_wrist_strike(model, robot, scene, cfg, seed=pose, tip_direction=direction)
            if plan is None:
                continue
            valid = True
            for profile in profiles.values():
                existing = plan_wrist_strike(model, robot, scene, profile["configuration"], seed=np.array(profile["impact_seed"]),
                                              tip_direction=np.array(profile["tip_direction"]))
                if existing is None:
                    valid = False
                    break
            if not valid:
                continue
            profiles[target] = {"target": target, "arm": skill.arm, "configuration": cfg,
                                "impact_seed": plan["impact_pose"].tolist(), "tip_direction": direction.tolist()}
            changes[target] = {"source_center_m": original.tolist(), "training_center_m": center.tolist(),
                               "translation_m": (center-original).tolist()}
            print(json.dumps({"target": target, **changes[target]}), flush=True)
            break
        else:
            raise RuntimeError(f"No collision-clear layout candidate found for {target}")
    return layout, profiles, changes


@budgeted("openarm_rl.planning.reachable_layout")
def main():
    parser = argparse.ArgumentParser(description="Generate and verify a separate seven-center training layout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prepare_scene(args.output / "source_scene")
    model = mujoco.MjModel.from_binary_path(str(args.output / "source_scene/model.mjb"))
    metadata = json.loads((args.output / "source_scene/scene.json").read_text())
    robot, scene = metadata["robot"], metadata["scene"]
    profiles = {}
    for target in ("snare", "rack_tom_left", "rack_tom_right", "floor_tom"):
        profile = plan_center(model, robot, scene, SKILLS[target])
        if profile["status"] != "ready":
            raise RuntimeError(f"Base center could not be validated: {target}")
        profiles[target] = profile
    layout, profiles, changes = find_layout(model, robot, scene, profiles, args.seed)
    layout.save(args.output / "layout.json")
    (args.output / "layout_changes.json").write_text(json.dumps(changes, indent=2))
    prepare_scene(args.output / "scene", layout_file=args.output / "layout.json")
    model = mujoco.MjModel.from_binary_path(str(args.output / "scene/model.mjb"))
    metadata = json.loads((args.output / "scene/scene.json").read_text())
    controller = HoldController(model, metadata["robot"])
    output = args.output / "profiles"
    output.mkdir()
    final = {}
    for target, skill in SKILLS.items():
        profile = profiles[target]
        selected = np.arange(7) if skill.arm == "left" else np.arange(7, 14)
        pose = np.array(profile["impact_seed"])
        pose[selected] = np.clip(pose[selected], controller.bounds[selected, 0] + .05, controller.bounds[selected, 1] - .05)
        cfg = {**profile["configuration"], "ready_seconds": .10, "approach_seconds": .20, "downstroke_seconds": .12,
               "retract_seconds": .28, "recovery_seconds": .4, "impact_height_m": .004,
               "command_acceleration_rad_s2": 150. if target in ("rack_tom_right", "cymbal_right") else 120.}
        plan = plan_wrist_strike(model, metadata["robot"], metadata["scene"], cfg, seed=pose,
                                 tip_direction=np.array(profile["tip_direction"]))
        if plan is None:
            raise RuntimeError(f"Final center geometry could not be validated: {target}")
        profile = {**profile, "status": "ready", "validation_scope": "kinematic_arc", "configuration": cfg,
                   "action_semantics": "absolute_wrist_position_velocity", "impact_seed": plan["impact_pose"].tolist(),
                   "ready_joints": plan["ready"].tolist(), "scene_identity": scene_identity(metadata["robot"], metadata["scene"])}
        (output / f"{target}.json").write_text(json.dumps(profile, indent=2))
        final[target] = profile
    (args.output / "preflight.json").write_text(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
