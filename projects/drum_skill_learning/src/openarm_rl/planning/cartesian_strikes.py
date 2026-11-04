import argparse
from itertools import product
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from openarm_drummer.model import HoldController
from openarm_sim.compute_guard import budgeted, check_resources

from ..control.reference import JointReference
from ..envs.rhythmic import RhythmicDrummingEnv
from ..music.corpus import write_segment
from ..music.reference import ScoreFeasibilityError
from ..paths import PROJECT_ROOT
from ..skills.registry import SKILLS
from .strikes import scene_identity


def solve_pose(model, robot, scene, arm, target, *, height, tip_direction, seed=None, pose_validator=None, iterations=500):
    controller = HoldController(model, robot)
    selected = np.arange(7) if arm == "left" else np.arange(7, 14)
    surface = next(surface for surface in scene["surfaces"] if surface["name"] == target)
    position = np.asarray(surface["center"])+height*np.asarray(surface["normal"])
    direction = np.asarray(tip_direction)
    q = np.zeros(14)
    q[selected] = (controller.target if seed is None else seed)[selected]
    data = mujoco.MjData(model)
    sid = model.site(f"{arm}_stick_tip").id
    bounds = controller.bounds[selected]
    rng = np.random.default_rng(271828)
    best = None
    for attempt in range(8):
        initial = q[selected] if attempt == 0 else rng.uniform(bounds[:, 0]+.03, bounds[:, 1]-.03)
        def residual(joints):
            data.qpos[controller.q] = q
            data.qpos[controller.q[selected]] = joints
            mujoco.mj_forward(model, data)
            return np.r_[100.*(data.site_xpos[sid]-position),
                         .02*(data.site_xmat[sid].reshape(3, 3)[:, 2]-direction), .001*(joints-initial)]
        solution = least_squares(residual, np.clip(initial, bounds[:, 0]+1e-4, bounds[:, 1]-1e-4),
                                 bounds=(bounds[:, 0]+1e-4, bounds[:, 1]-1e-4), max_nfev=iterations,
                                 ftol=1e-10, xtol=1e-10, gtol=1e-10)
        result = q.copy()
        result[selected] = solution.x
        residual(solution.x)
        error = float(np.linalg.norm(data.site_xpos[sid]-position))
        pointing_down = data.site_xmat[sid].reshape(3, 3)[:, 2]@np.asarray(surface["normal"]) < -.05
        accepted = error < .0005 and pointing_down and (pose_validator is None or pose_validator(result))
        row = {"reachable": bool(accepted), "position_error_m": error, "joint_positions": result.tolist()}
        if accepted:
            return row
        if best is None or error < best["position_error_m"]:
            best = row
    return best


def prepare_target(scene_directory: Path, target: str, output: Path) -> dict:
    model = mujoco.MjModel.from_binary_path(str(scene_directory/"model.mjb"))
    metadata = json.loads((scene_directory/"scene.json").read_text())
    if "layout_sha256" in metadata["scene"]:
        raise ValueError("Cartesian strike preparation requires original drum positions")
    skill = SKILLS[target]
    controller = HoldController(model, metadata["robot"])
    surface = next(surface for surface in metadata["scene"]["surfaces"] if surface["name"] == target)
    center, normal = np.asarray(surface["center"]), np.asarray(surface["normal"])
    horizontal = center-np.asarray([0., .0925 if skill.arm == "left" else -.0925, 1.0018])
    horizontal[2] = 0
    horizontal /= np.linalg.norm(horizontal)
    probe = mujoco.MjData(model)
    tip_geom = model.geom(f"{skill.arm}_stick_tip_collision").id
    head_geom = model.geom(f"{target}_head_collision").id
    radius = model.geom_size[tip_geom, 0]
    directory = output/target
    directory.mkdir()
    profile_directory = directory/"profiles"
    profile_directory.mkdir()
    notes = json.loads((PROJECT_ROOT/"configs/music/percussion.json").read_text())["notes"]
    note = next(int(note) for note, name in notes.items() if name == target)
    midi = directory/"probe.mid"
    write_segment(midi, [{"time_s": .5, "note": note, "velocity": 96}], 0., 1.)
    records = []

    def clear(pose):
        probe.qpos[controller.q] = pose
        mujoco.mj_forward(model, probe)
        return not any(contact.dist < -1e-5 and {contact.geom1, contact.geom2} != {tip_geom, head_geom}
                       for contact in probe.contact)

    for lean, yaw in product((1., 2., 3., 5., 8.), (0., -20., 20.)):
        check_resources()
        angle = np.radians(yaw)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        direction = lean*(rotation@horizontal)-normal
        direction /= np.linalg.norm(direction)
        impact_height = radius*(1+float(direction@normal))
        initial = solve_pose(model, metadata["robot"], metadata["scene"], skill.arm, target,
            height=impact_height+.025, tip_direction=direction, iterations=400, pose_validator=clear)
        print(json.dumps({"cartesian_target": target, "lean": lean, "yaw": yaw,
            "ready_reachable": initial["reachable"], "position_error_m": initial["position_error_m"]}), flush=True)
        if not initial["reachable"]:
            records.append({"lean": lean, "yaw": yaw, "status": "ready_ik_unavailable", "ik": initial})
            continue
        ready = np.asarray(initial["joint_positions"])
        approach = solve_pose(model, metadata["robot"], metadata["scene"], skill.arm, target,
            height=impact_height+.012, tip_direction=direction, seed=ready, iterations=400, pose_validator=clear)
        if not approach["reachable"]:
            records.append({"lean": lean, "yaw": yaw, "status": "approach_ik_unavailable", "ik": approach})
            continue
        for depth in (.001, .002, .003):
            bottom = solve_pose(model, metadata["robot"], metadata["scene"], skill.arm, target,
                height=impact_height-depth, tip_direction=direction, seed=np.asarray(approach["joint_positions"]),
                iterations=400, pose_validator=clear)
            if not bottom["reachable"]:
                continue
            for downstroke in (.06, .09, .12, .18):
                times = np.cumsum([0., .15, downstroke, .01, .22, .12])
                positions = np.asarray([ready, approach["joint_positions"], bottom["joint_positions"],
                                        bottom["joint_positions"], ready, ready])
                reference = JointReference(times, positions, skill.wrist, .15+downstroke*.8,
                    active_joints=tuple(range(7)) if skill.arm == "left" else tuple(range(7, 14)))
                sampled = np.asarray([reference.sample(t)[1] for t in np.arange(0., reference.duration, .002)])
                if np.any(np.abs(sampled) > .8*controller.velocity):
                    continue
                profile = {"target": target, "arm": skill.arm, "status": "ready", "reason": None,
                    "motion_type": "cartesian_full_arm", "scene_identity": scene_identity(metadata["robot"], metadata["scene"]),
                    "center_m": center.tolist(), "tip_direction": direction.tolist(), "impact_seed": bottom["joint_positions"],
                    "configuration": {"arm": skill.arm, "target": target},
                    "reference_bank": {"schema": "openarm.speed_conditioned_strike.v1", "speeds_m_s": [.35, .85],
                        "references": [reference.document(), reference.document()]}}
                profile_file = profile_directory/f"{target}.json"
                profile_file.write_text(json.dumps(profile))
                row = {"lean": lean, "yaw": yaw, "depth_m": depth, "downstroke_s": downstroke}
                try:
                    env = RhythmicDrummingEnv(scene_directory, profile_directory, midi, score_only=True)
                except ScoreFeasibilityError as error:
                    records.append({**row, "status": "path_infeasible", "reason": str(error)})
                    continue
                with env:
                    env.reset(seed=314159)
                    while True:
                        _, _, done, truncated, info = env.step(env.reference_action())
                        if done or truncated:
                            break
                row.update(status="qualified" if info["is_success"] else "physics_unqualified", measurements=info)
                records.append(row)
                (directory/"search.json").write_text(json.dumps(records, indent=2))
                if info["is_success"] and info["peak_force_N"] < 22 and info["max_penetration_m"] < .002:
                    reference.impact_time += info["hits"][0]["time_s"]-.5
                    profile["reference_bank"]["references"] = [reference.document(), reference.document()]
                    profile["reference_bank"]["calibration_speed_m_s"] = [info["hits"][0]["downward_speed_m_s"]]*2
                    (output/"profiles"/f"{target}.json").write_text(json.dumps(profile, indent=2))
                    return {"target": target, "status": "ready", "duration_s": reference.duration,
                        "peak_force_N": info["peak_force_N"], "measured_speed_m_s": info["hits"][0]["downward_speed_m_s"]}
    (directory/"search.json").write_text(json.dumps(records, indent=2))
    return {"target": target, "status": "unavailable", "reason": "No qualified full-arm center stroke in the recorded search"}


@budgeted("openarm_rl.planning.cartesian_strikes")
def main():
    parser = argparse.ArgumentParser(description="Validate full-arm tip translations at original drum centers")
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=list(SKILLS), default=["hi_hat", "cymbal_left", "cymbal_right"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"profiles").mkdir()
    results = []
    for target in args.targets:
        results.append(prepare_target(args.scene_directory, target, args.output))
        (args.output/"reachability.json").write_text(json.dumps(results, indent=2))
    if any(row["status"] != "ready" for row in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
