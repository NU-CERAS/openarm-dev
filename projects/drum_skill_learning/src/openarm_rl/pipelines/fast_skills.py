import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import h5py

from openarm_sim.compute_guard import budgeted

from ..skills.registry import SKILLS


@budgeted("openarm_rl.pipelines.fast_skills")
def main():
    parser = argparse.ArgumentParser(description="Serial live-demo expert, ACT and PPO training for fast drum-center tools")
    parser.add_argument("--scene-directory", type=Path)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=list(SKILLS), default=list(SKILLS))
    parser.add_argument("--act-directory", type=Path)
    parser.add_argument("--act-steps", type=int, default=2500)
    parser.add_argument("--demonstrations", type=int, default=96)
    parser.add_argument("--ppo-steps", type=int, default=262144)
    parser.add_argument("--ppo-rounds", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--position-range", type=float, default=.005)
    parser.add_argument("--domain-range", type=float, default=.1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--xvfb", type=Path)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if len(set(args.targets)) != len(args.targets) or not 1 <= args.num_envs <= 4 or min(args.ppo_rounds, args.ppo_steps, args.act_steps) <= 0:
        raise ValueError("Unique targets, positive counts and at most four environments are required")
    if (args.scene_directory is None) != (args.profiles is None):
        raise ValueError("Supply a scene and its profiles together")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "logs").mkdir()

    def run(command, log):
        with log.open("w") as stream:
            subprocess.run([sys.executable, "-m", *command], stdout=stream, stderr=subprocess.STDOUT, check=True)

    if args.scene_directory is None:
        preparation = args.output / "preparation"
        run(["openarm_rl.planning.reachable_layout", "--output", str(preparation)], args.output / "logs/layout.log")
        args.scene_directory = preparation / "scene"
        args.profiles = preparation / "calibrated_profiles"
        run(["openarm_rl.planning.fast_references", "--scene-directory", str(args.scene_directory), "--output", str(args.profiles),
             "--profiles", *[str(preparation / "profiles" / f"{target}.json") for target in SKILLS]], args.output / "logs/calibration.log")
    profiles = {target: args.profiles / f"{target}.json" for target in args.targets}
    for target, path in profiles.items():
        profile = json.loads(path.read_text())
        if profile["status"] != "ready" or profile["arm"] != SKILLS[target].arm or "reference_bank" not in profile:
            raise ValueError(f"A matching fast reference bank is required for {target}")
    shutil.copytree(args.scene_directory, args.output / "scene")
    (args.output / "profiles").mkdir()
    for target, path in profiles.items():
        shutil.copyfile(path, args.output / "profiles" / path.name)
    manifest = {"schema": "openarm.camera_free_tools.v1", "scene_directory": "scene", "minimum_success": .8,
                "pipeline": "live_demo_to_ACT_to_PPO", "targets": {
                    target: {"status": "pending", "arm": SKILLS[target].arm, "tool": SKILLS[target].name} for target in args.targets}}

    def save():
        (args.output / "tools.json").write_text(json.dumps(manifest, indent=2) + "\n")

    with ExitStack() as resources:
        if args.xvfb is not None:
            from ..recording.display import NativeDisplay

            display = NativeDisplay(args.xvfb, args.output / "display")
            resources.callback(display.close)
        save()
        for target in args.targets:
            row = manifest["targets"][target]
            destination = args.output / target
            destination.mkdir()
            profile = args.output / "profiles" / f"{target}.json"
            common = ["--scene-directory", str(args.output / "scene"), "--profile-file", str(profile), "--target", target,
                      "--position-range", str(args.position_range), "--domain-range", str(args.domain_range),
                      "--device", args.device, "--wandb-mode", args.wandb_mode]
            row["status"] = "ACT_training"
            save()
            act = destination / "ACT"
            supplied = None if args.act_directory is None else args.act_directory / target
            if supplied is not None and supplied.is_dir():
                with h5py.File(supplied / "train.h5") as data:
                    if data.attrs["profile_sha256"] != hashlib.sha256(profile.read_bytes()).hexdigest():
                        raise ValueError("Supplied ACT demonstrations belong to another reference profile")
                shutil.copytree(supplied, act)
            else:
                run(["openarm_rl.imitation.strike", *common, "--output", str(act), "--expert-source", "live_demo",
                     "--episodes", str(args.demonstrations), "--steps", str(args.act_steps)], args.output / "logs" / f"{target}-ACT.log")
            row["ACT"] = {k: v for k, v in json.loads((act / "heldout.json").read_text()).items() if k != "results"}
            previous = None
            for attempt in range(1, args.ppo_rounds + 1):
                row.update(status="PPO_training", round=attempt)
                save()
                output = destination / f"PPO_{attempt:02d}"
                command = ["openarm_rl.training.camera_free", *common, "--output", str(output), "--steps", str(args.ppo_steps),
                           "--num-envs", str(args.num_envs), "--checkpoint-steps", "65536", "--eval-episodes", "16",
                           "--heldout-episodes", "64"]
                if previous is None:
                    command += ["--initialize-act", str(act / "act.pt"), "--initialization-data", str(act / "train.h5")]
                else:
                    command += ["--resume-bundle", str(previous)]
                if args.video:
                    command.append("--video")
                run(command, args.output / "logs" / f"{target}-PPO-{attempt:02d}.log")
                comparison = json.loads((output / "comparison.json").read_text())
                previous = output / "best"
                row.update(checkpoint=str(previous.relative_to(args.output)), comparison=comparison)
                validation = comparison["validation_policy"]
                if validation["success_rate"] >= .9 and validation["unsafe_rate"] == 0 and validation["recovery_rate"] >= .95:
                    break
            report = row["comparison"]["policy"]
            qualified = (report["success_rate"] >= .8 and report["unsafe_rate"] == 0 and report["recovery_rate"] >= .95
                         and report["success_rate"] >= row["comparison"]["reference"]["success_rate"])
            row["status"] = "simulation_qualified" if qualified else "trained_unqualified"
            save()
            print(json.dumps({"target": target, "status": row["status"], "success_rate": report["success_rate"]}), flush=True)
        if any(row["status"] != "simulation_qualified" for row in manifest["targets"].values()):
            raise RuntimeError("Some trained policies did not meet the simulation qualification criteria; inspect tools.json")


if __name__ == "__main__":
    main()
