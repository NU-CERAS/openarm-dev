import argparse
import json
from pathlib import Path

from openarm_sim.compute_guard import budgeted, check_resources

from ..envs.multiscore import MultiScoreDrummingEnv
from .perform import rollout


@budgeted("openarm_rl.music.audit")
def main():
    parser = argparse.ArgumentParser(description="Audit every accepted corpus segment in native contact physics")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("Audit episodes must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for split in ("train", "validation", "test"):
        with MultiScoreDrummingEnv(args.corpus, split, scene_directory=args.scene_directory,
                                  profiles=args.profiles, domain_range=.05, position_range=.002) as env:
            for clip in env.clips:
                for episode in range(args.episodes):
                    check_resources()
                    report = rollout(env, args.output/clip["id"]/f"episode_{episode:03d}",
                                     seed=900000+episode, reset_options={"clip_id": clip["id"]})
                    results.append(report)
                    print(json.dumps({"clip": clip["id"], "seed": report["seed"], "f1": report["f1"],
                                      "success": report["is_success"], "failure": report["failure"]}), flush=True)
    (args.output/"evaluation.json").write_text(json.dumps({"results": results,
        "episodes": len(results), "successful": sum(row["is_success"] for row in results)}, indent=2))
    if not all(row["is_success"] and row["completed"] for row in results):
        raise RuntimeError("Native corpus audit has failed episodes; inspect evaluation.json")


if __name__ == "__main__":
    main()
