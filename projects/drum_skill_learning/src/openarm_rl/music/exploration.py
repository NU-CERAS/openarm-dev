import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import torch
from stable_baselines3 import PPO

from openarm_sim.compute_guard import budgeted

from ..envs.multiscore import make_rhythm_env
from ..models.act_ppo import ACTRhythmPolicy
from .perform import rollout


@budgeted("openarm_rl.music.exploration")
def main():
    parser = argparse.ArgumentParser(description="Measure deterministic and exploratory robot execution on identical native reset states")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=930000)
    parser.add_argument("--sde", action="store_true")
    parser.add_argument("--log-std-init", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("Episode count must be positive")
    if args.log_std_init is not None and not -20 <= args.log_std_init <= 0:
        raise ValueError("Initial log standard deviation must lie in [-20, 0]")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    options = json.loads((args.run/"run_config.json").read_text())["validation_environment"]
    policy = PPO.load(args.run/args.checkpoint/"policy.zip", device="cpu")
    if args.sde:
        if not isinstance(policy.policy, ACTRhythmPolicy):
            raise ValueError("SDE comparison requires a score-only ACT checkpoint")
        parameters = policy.policy._get_constructor_parameters()
        parameters.update(use_sde=True, squash_output=True, log_std_init=-9. if args.log_std_init is None else args.log_std_init)
        replacement = ACTRhythmPolicy(**parameters)
        weights = policy.policy.state_dict()
        weights["log_std"] = replacement.log_std.detach().clone()
        replacement.load_state_dict(weights)
        replacement.set_training_mode(False)
        policy.policy, policy.use_sde = replacement, True
    elif args.log_std_init is not None:
        with torch.no_grad():
            policy.policy.log_std.fill_(args.log_std_init)
    with (args.run/args.checkpoint/"vecnormalize.pkl").open("rb") as source:
        normalizer = pickle.load(source)
    normalizer.training = False
    reports = {}
    for name, deterministic in (("deterministic", True), ("exploratory", False)):
        rows = []
        with make_rhythm_env(**options) as env:
            for episode in range(args.episodes):
                torch.manual_seed(args.seed+episode)
                if policy.use_sde:
                    policy.policy.reset_noise(1)
                def command(observation):
                    return policy.predict(normalizer.normalize_obs(observation), deterministic=deterministic)[0]
                report = rollout(env, args.output/name/f"episode_{episode:03d}", args.seed+episode, command,
                    reset_options={"clip_id": env.clips[episode % len(env.clips)]["id"]})
                report["final_max_joint_speed_rad_s"] = float(np.max(np.abs(env.data.qvel[env.controller.v])))
                report["mean_action_change"] = float(np.mean(np.abs(np.diff(np.asarray([step["action"] for step in env.trace]), axis=0))))
                rows.append(report)
                print(json.dumps({"mode": name, "episode": episode, "success": report["is_success"],
                    "recovered": report["recovered"], "final_speed": report["final_max_joint_speed_rad_s"]}), flush=True)
        reports[name] = {"episodes": len(rows), **{key: float(np.mean([row[key] for row in rows]))
            for key in ("f1", "is_success", "unsafe_contact", "recovered", "peak_force_N",
                        "final_max_joint_speed_rad_s", "mean_action_change")}, "results": rows}
    result = {"checkpoint": args.checkpoint, "seed": args.seed, "gSDE": policy.use_sde,
              "noise_resampling": "episode" if policy.use_sde else "control_step",
              "log_std": policy.policy.log_std.detach().cpu().tolist(), "reports": reports}
    (args.output/"comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"gSDE": policy.use_sde, **{name: {key: value for key, value in report.items() if key != "results"}
        for name, report in reports.items()}}), flush=True)


if __name__ == "__main__":
    main()
