import argparse
import json
from pathlib import Path

import numpy as np
import torch

from openarm_sim.compute_guard import budgeted

from ..envs.multiscore import make_rhythm_env
from .perform import rollout


@budgeted("openarm_rl.music.benchmark")
def main():
    parser = argparse.ArgumentParser(description="Compare an exported learned MIDI actor and its native reference on matched seeds")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="final")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--episodes-per-clip", type=int)
    parser.add_argument("--seed", type=int, default=910000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("Episode count must be positive")
    if args.episodes_per_clip is not None and args.episodes_per_clip <= 0:
        raise ValueError("Episodes per clip must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.run / "run_config.json").read_text())
    options = config["environment"]
    manifest = json.loads((args.run / args.checkpoint / "bundle.json").read_text())
    actor = torch.jit.load(str(args.run / args.checkpoint / "actor.pt"), map_location="cpu").eval()
    torch.set_num_threads(1)
    def command(observation):
        with torch.no_grad():
            return actor(torch.from_numpy(observation["actor"])[None], torch.from_numpy(observation["nominal"])[None]).numpy()[0]
    result = {"checkpoint": args.checkpoint, "steps": manifest["steps"], "optimization_updates": manifest["optimization_updates"]}
    if "corpus_manifest" in options:
        patterns = [("training_segments", options), ("validation_segments", config["validation_environment"]),
                    ("test_segments", config["test_environment"])]
    else:
        patterns = [("training_pattern", options), ("unseen_pattern", {**options, "midi_file": str(args.run / "heldout.mid")})]
    for name, environment in patterns:
        reports = {}
        for controller, policy in (("reference", None), ("actor", command)):
            rows = []
            with make_rhythm_env(**environment) as env:
                episodes = len(env.clips)*args.episodes_per_clip if "corpus_manifest" in options and args.episodes_per_clip else args.episodes
                for episode in range(episodes):
                    reset_options = {"clip_id": env.clips[episode % len(env.clips)]["id"]} if "corpus_manifest" in options else None
                    rows.append(rollout(env, args.output / name / controller / f"episode_{episode:03d}",
                                        args.seed+episode, policy, reset_options=reset_options))
            reports[controller] = {"episodes": episodes,
                **{key: float(np.mean([row[key] for row in rows])) for key in ("f1", "precision", "recall", "unsafe_contact", "is_success", "recovered")},
                "timing_p95_max_s": max((row["timing_p95_s"] for row in rows if row["timing_p95_s"] is not None), default=None),
                "results": rows}
        result[name] = reports
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2))
    names = [name for name, _ in patterns]
    print(json.dumps({**{k: v for k, v in result.items() if k not in names},
                      **{name: {controller: {k: v for k, v in report.items() if k != "results"} for controller, report in result[name].items()}
                         for name in names}}), flush=True)


if __name__ == "__main__":
    main()
