import argparse
import json
from pathlib import Path
import pickle

from stable_baselines3 import PPO

from openarm_sim.compute_guard import budgeted

from ..training.rhythmic import save_bundle


@budgeted("openarm_rl.music.export")
def main():
    parser = argparse.ArgumentParser(description="Export a saved rhythmic PPO checkpoint with model identities and command limits")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    options = json.loads((args.run / "run_config.json").read_text())["environment"]
    checkpoint = args.run / args.checkpoint
    with (checkpoint / "vecnormalize.pkl").open("rb") as source:
        normalizer = pickle.load(source)
    normalizer.training = False
    model = PPO.load(checkpoint / "policy.zip", device="cpu")
    save_bundle(model, normalizer, args.output, options)
    print(str(args.output), flush=True)


if __name__ == "__main__":
    main()
