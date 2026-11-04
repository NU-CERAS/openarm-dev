import argparse
import json
from pathlib import Path

from openarm_sim.compute_guard import budgeted

from ..models.act import ActionChunkTransformer
from ..training.rhythmic import evaluate


@budgeted("openarm_rl.music.evaluate_act")
def main():
    parser = argparse.ArgumentParser(description="Evaluate a collected-data ACT controller on complete corpus clips")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes <= 0 or args.output.exists():
        raise ValueError("Require positive episodes and a new output directory")
    keys = {"train": "environment", "validation": "validation_environment", "test": "test_environment"}
    options = json.loads((args.run/"run_config.json").read_text())[keys[args.split]]
    actor = ActionChunkTransformer.load(args.run/"act.pt", "cpu")
    report = evaluate(options, args.output, act=actor, episodes=args.episodes, seed=910000)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}), flush=True)


if __name__ == "__main__":
    main()
