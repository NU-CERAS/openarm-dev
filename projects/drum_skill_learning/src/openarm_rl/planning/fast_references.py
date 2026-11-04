import argparse
from itertools import product
import json
from pathlib import Path

import mujoco
import numpy as np

from openarm_drummer.strike_planning import plan_wrist_strike
from openarm_sim.compute_guard import budgeted, check_resources

from ..control.reference import JointReference
from ..envs.camera_free_strike import CameraFreeStrikeEnv


def calibrate(scene, profile_file, output):
    profile = json.loads(profile_file.read_text())
    target = profile["target"]
    measurements = []
    with CameraFreeStrikeEnv(scene, profile_file, target=target, arm=profile["arm"], position_range=0, domain_range=0) as env:
        for depth in (.001, .003, .006):
            cfg = {**profile["configuration"], "command_depth_m": depth}
            plan = plan_wrist_strike(env.model, env.robot, env.scene, cfg, seed=np.array(profile["impact_seed"]),
                                     tip_direction=np.array(profile["tip_direction"]))
            if plan is None:
                continue
            inactive = np.arange(7, 14) if env.skill.arm == "left" else np.arange(7)
            for key in ("ready", "approach", "bottom", "impact_pose"):
                plan[key][inactive] = 0
            crossing = (plan["impact_pose"][env.skill.wrist] - plan["approach"][env.skill.wrist]) / (plan["bottom"][env.skill.wrist] - plan["approach"][env.skill.wrist])
            from scipy.optimize import brentq

            fraction = brentq(lambda x: x*x*(3-2*x) - crossing, 0, 1)
            for downstroke, compliance, momentum in product((.05, .07, .09, .11, .13, .16, .20), (.15, .3, .45), (0., 1., 2.)):
                check_resources()
                times = np.cumsum([0, .10, .20, downstroke, .01, .28, .40])
                positions = np.array([plan["ready"], plan["ready"], plan["approach"], plan["bottom"],
                                      plan["bottom"], plan["ready"], plan["ready"]])
                velocities = np.zeros_like(positions)
                wrist = env.skill.wrist
                incoming = momentum * (plan["bottom"][wrist] - plan["ready"][wrist]) / (.20 + downstroke)
                velocities[2:4, wrist] = np.clip(incoming, -.8*env.controller.velocity[wrist], .8*env.controller.velocity[wrist])
                reference = JointReference(times, positions, wrist, times[2] + fraction*downstroke, compliance, .04, velocities)
                env.reference = reference
                env.reset(seed=43100)
                while True:
                    _, reward, done, truncated, info = env.step(env.reference_action())
                    if done or truncated:
                        break
                valid = (info["hit_count"] == 1 and info["recovered"] and not info["unsafe_contact"] and
                         info["center_error_m"] <= .01 and info["hits"][0]["contact_duration_s"] <= .12)
                row = {"valid": valid, "downstroke_s": downstroke, "compliance": compliance, "depth_m": depth, "momentum": momentum,
                       "reference": reference.document(), "measurements": info}
                if valid:
                    row["reference"]["impact_time_s"] = info["hits"][0]["onset_time_s"]
                measurements.append(row)
        candidates = [row for row in measurements if row["valid"]]
        if not candidates:
            output.mkdir(parents=True, exist_ok=True)
            (output / f"{target}_calibration.json").write_text(json.dumps(measurements, indent=2))
            raise RuntimeError(f"No physically valid fast reference for {target}")
        speeds = [.35, .45, .55, .65, .75, .85]
        selected = [min(candidates, key=lambda row: abs(row["measurements"]["hits"][0]["downward_speed_m_s"] - speed)) for speed in speeds]
        bank = {"schema": "openarm.speed_conditioned_strike.v1", "speeds_m_s": speeds,
                "references": [row["reference"] for row in selected], "calibration_seed": 43100,
                "calibration_speed_m_s": [row["measurements"]["hits"][0]["downward_speed_m_s"] for row in selected]}
        profile["reference_bank"] = bank
        output.mkdir(parents=True, exist_ok=True)
        (output / profile_file.name).write_text(json.dumps(profile, indent=2) + "\n")
        (output / f"{target}_calibration.json").write_text(json.dumps(measurements, indent=2))
        print(json.dumps({"target": target, "valid_candidates": len(candidates), "requested_speeds": speeds,
                          "measured_speeds": bank["calibration_speed_m_s"],
                          "durations_s": [row["reference"]["times_s"][-1] for row in selected]}), flush=True)


@budgeted("openarm_rl.planning.fast_references")
def main():
    parser = argparse.ArgumentParser(description="Calibrate short, speed-conditioned strike references in native physics")
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for profile in args.profiles:
        calibrate(args.scene_directory, profile, args.output)


if __name__ == "__main__":
    main()
