import argparse
import json
from pathlib import Path
import shutil

import mujoco

from openarm_sim.compute_guard import budgeted

from ..planning.fast_references import calibrate
from ..planning.strikes import plan_center
from ..skills.registry import SKILLS, StrikeSkill


@budgeted("openarm_rl.music.prepare")
def main():
    parser = argparse.ArgumentParser(description="Validate and calibrate additional hand-to-drum assignments")
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=list(SKILLS), default=["snare"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.profiles, args.output / "profiles")
    (args.output / "proposals").mkdir()
    model = mujoco.MjModel.from_binary_path(str(args.scene_directory / "model.mjb"))
    metadata = json.loads((args.scene_directory / "scene.json").read_text())
    outcomes = {}
    for target in args.targets:
        other = "right" if SKILLS[target].arm == "left" else "left"
        skill = StrikeSkill(f"strike_{target}", target, other)
        profile = plan_center(model, metadata["robot"], metadata["scene"], skill, allow_cross_body=True)
        key = f"{target}.{other}"
        outcomes[key] = profile
        (args.output / "reachability.json").write_text(json.dumps(outcomes, indent=2))
        if profile["status"] != "ready":
            print(json.dumps({"assignment": key, "status": "unavailable", "reason": profile["reason"]}), flush=True)
            continue
        profile["action_semantics"] = "absolute_wrist_position_velocity"
        profile["configuration"].update(ready_seconds=.10, approach_seconds=.20, retract_seconds=.28,
                                        recovery_seconds=.4, command_acceleration_rad_s2=120.)
        proposal = args.output / "proposals" / f"{key}.json"
        proposal.write_text(json.dumps(profile, indent=2))
        calibrate(args.scene_directory, proposal, args.output / "profiles")


if __name__ == "__main__":
    main()
