import argparse
import csv
import io
import json
import math
from pathlib import Path
import time
from uuid import uuid4

import wandb


def training_events(directory: Path) -> list[dict]:
    path = directory / "progress.csv"
    events = []
    if path.exists():
        text = path.read_text()
        complete = text[:text.rfind("\n") + 1]
        for row in csv.DictReader(io.StringIO(complete)):
            if row.get("time/total_timesteps") in (None, "") or None in row:
                continue
            values = {key: float(value) for key, value in row.items()
                      if value not in (None, "") and math.isfinite(float(value))}
            events.append({"id": f"train:{row['time/total_timesteps']}",
                           "steps": int(values["time/total_timesteps"]), "metrics": values})
    for path in sorted(directory.glob("*/evaluation.json")):
        report = json.loads(path.read_text())
        results = list(report["results"].values())
        metrics = {"eval/success_rate": report["success_rate"],
                                   "eval/mean_return": report["mean_return"],
                                   "eval/episodes": report["episodes"],
                                   "eval/unsafe_rate": sum(r["unsafe_contact"] for r in results) / len(results)}
        if all("max_lift_m" in result for result in results):
            for name, key in (("mean_max_lift_m", "max_lift_m"), ("reached_2cm_rate", "milestone_2cm"),
                              ("reached_5cm_rate", "milestone_5cm"), ("timeout_rate", "timeout")):
                metrics[f"eval/{name}"] = sum(float(result[key]) for result in results) / len(results)
        events.append({"id": f"evaluation:{path.parent.name}", "steps": report["steps"],
                       "metrics": metrics,
                       "directory": str(path.parent)})
    return sorted(events, key=lambda event: (event["steps"], event["id"]))


def main():
    parser = argparse.ArgumentParser(description="Synchronize OpenArm training metrics, videos, and checkpoints to W&B")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--project", default="openarm-teachers")
    parser.add_argument("--entity")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("Polling interval must be positive")
    directory = args.directory.resolve()
    config = json.loads((directory / "run_config.json").read_text())
    state_path = directory / "wandb_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"run_id": uuid4().hex[:8], "logged": []}
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    config.update({"algorithm": config.get("algorithm", "PPO"), "physics": "MuJoCo Warp", "oracle_observations": True,
                   "learning_rate": config.get("learning_rate", 0.0003), "gamma": 0.99, "gae_lambda": 0.95,
                   "clip_range": 0.2, "ent_coef": config.get("ent_coef", 0.0), "vf_coef": 0.5,
                   "max_grad_norm": 0.5, "network": [128, 128], "control_hz": 50})
    with wandb.init(project=args.project, entity=args.entity, id=state["run_id"], resume="allow",
                    name=directory.name, config=config, mode="online", force=True,
                    dir=str(directory), settings=wandb.Settings(init_timeout=30)) as run:
        run.define_metric("env_steps")
        run.define_metric("*", step_metric="env_steps")
        run.define_metric("eval/success_rate", summary="max", step_metric="env_steps")
        (directory / "wandb_url.txt").write_text(run.url + "\n")
        print(f"W&B run: {run.url}", flush=True)
        logged = set(state["logged"])
        while True:
            for event in training_events(directory):
                if event["id"] in logged:
                    continue
                metrics = {"env_steps": event["steps"], **event["metrics"]}
                if "directory" in event:
                    checkpoint = Path(event["directory"])
                    metrics["eval/execution"] = wandb.Video(str(checkpoint / "execution.mp4"), format="mp4")
                    model_path = checkpoint / "policy.zip"
                    if model_path.exists():
                        artifact = wandb.Artifact(f"pickup-teacher-{run.id}", type="model", metadata={"env_steps": event["steps"]})
                        artifact.add_file(str(model_path))
                        artifact.add_file(str(checkpoint / "evaluation.json"))
                        if config.get("normalize", False):
                            artifact.add_file(str(checkpoint / "vecnormalize.pkl"))
                        if config["algorithm"] == "ACT+residual_PPO":
                            artifact.add_file(str(checkpoint / "act.pt"))
                        run.log_artifact(artifact, aliases=[f"step-{event['steps']}", "latest"])
                run.log(metrics)
                logged.add(event["id"])
                state["logged"] = sorted(logged)
                state_path.write_text(json.dumps(state, indent=2) + "\n")
            final_report = directory / "final" / "evaluation.json"
            if args.once or (final_report.exists() and "evaluation:final" in logged):
                break
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
