import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import shutil
import sys

from openarm_sim.compute_guard import budgeted

from ..skills.registry import SKILLS


@budgeted("openarm_rl.pipelines.camera_free")
def main():
    parser = argparse.ArgumentParser(description="Preflight and serially train independent camera-free drum skills")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--preflight-directory", type=Path)
    parser.add_argument("--targets", nargs="+", choices=list(SKILLS), default=list(SKILLS))
    parser.add_argument("--steps-per-policy", type=int, default=131072)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--heldout-episodes", type=int, default=64)
    parser.add_argument("--position-range", type=float, default=.01)
    parser.add_argument("--domain-range", type=float, default=.1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--xvfb", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--minimum-success", type=float, default=.8)
    args = parser.parse_args()
    if args.layout is not None and args.preflight_directory is not None:
        raise ValueError("A layout requires a fresh preflight")
    if len(set(args.targets)) != len(args.targets):
        raise ValueError("Targets must be unique")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as resources:
        if args.xvfb is not None:
            from ..recording.display import NativeDisplay

            display = NativeDisplay(args.xvfb, args.output / "display")
            resources.callback(display.close)
        preflight = args.preflight_directory.resolve() if args.preflight_directory else args.output / "preflight"
        if args.preflight_directory is None:
            command = [sys.executable, "-m", "openarm_rl.planning.strikes", "--output", str(preflight), "--targets", *args.targets]
            if args.layout:
                command += ["--layout", str(args.layout.resolve())]
            with (args.output / "preflight.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        profiles = json.loads((preflight / "preflight.json").read_text())
        shutil.copytree(preflight / "scene", args.output / "scene")
        (args.output / "profiles").mkdir()
        shutil.copyfile(preflight / "preflight.json", args.output / "profiles/preflight.json")
        for target in args.targets:
            if profiles[target]["status"] == "ready":
                shutil.copyfile(preflight / f"{target}.json", args.output / "profiles" / f"{target}.json")
        manifest = {"schema": "openarm.camera_free_tools.v1", "scene_directory": "scene",
                    "minimum_success": args.minimum_success,
                    "targets": {target: {"status": "pending" if profiles[target]["status"] == "ready" else "unavailable",
                                         "reason": profiles[target].get("reason"), "arm": SKILLS[target].arm,
                                         "tool": SKILLS[target].name} for target in args.targets}}
        def save():
            (args.output / "tools.json").write_text(json.dumps(manifest, indent=2) + "\n")
        save()
        for target in args.targets:
            row = manifest["targets"][target]
            if profiles[target]["status"] != "ready":
                row.update(status="unavailable", reason=profiles[target]["reason"])
                save()
                continue
            if args.preflight_only:
                row["status"] = "geometry_validated"
                save()
                continue
            row["status"] = "training"
            save()
            output = args.output / target
            command = [sys.executable, "-m", "openarm_rl.training.camera_free", "--scene-directory", str(args.output / "scene"),
                       "--profile-file", str(args.output / "profiles" / f"{target}.json"), "--target", target, "--output", str(output),
                       "--steps", str(args.steps_per_policy), "--num-envs", str(args.num_envs), "--device", args.device,
                       "--eval-episodes", str(args.eval_episodes), "--heldout-episodes", str(args.heldout_episodes),
                       "--position-range", str(args.position_range), "--domain-range", str(args.domain_range),
                       "--wandb-mode", args.wandb_mode]
            if args.video:
                command.append("--video")
            print(json.dumps({"target": target, "status": "training", "log": str(args.output / f"{target}.log")}), flush=True)
            with (args.output / f"{target}.log").open("w") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                row.update(status="failed", log=f"{target}.log", returncode=process.returncode)
                save()
                raise subprocess.CalledProcessError(process.returncode, command)
            comparison = json.loads((output / "comparison.json").read_text())
            policy, reference = comparison["policy"], comparison["reference"]
            qualified = (policy["success_rate"] >= args.minimum_success and policy["success_rate"] >= reference["success_rate"]
                         and policy["unsafe_rate"] == 0 and policy["recovery_rate"] >= .95)
            row.update(status="simulation_qualified" if qualified else "trained_unqualified", checkpoint=f"{target}/best", comparison=comparison)
            save()
        print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
