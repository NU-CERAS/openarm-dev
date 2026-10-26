import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

from ..act_model import ActionChunkTransformer
from ..act_residual import ACTResidualVecEnv, residual_configuration
from .act_train import ACTExecutor
from ..warp_pickup import WarpPickupVecEnv, build_reset_bank
from ..compute_guard import budgeted, check_resources


def evaluate_pickup(policy, env, *, residual=False):
    if isinstance(env, VecNormalize) and policy.get_env() is not None:
        sync_envs_normalization(policy.get_env(), env)
    raw = env
    while hasattr(raw, "venv"):
        raw = raw.venv
    if not raw.sequential_resets:
        raise ValueError("Evaluation requires unique sequential reset states")
    observation = env.reset()
    results = {}
    residual_sum = 0.0
    residual_count = 0
    for _ in range(301):
        check_resources()
        actions, _ = policy.predict(observation, deterministic=True)
        active = np.array([index not in results for index in range(env.num_envs)])
        if residual:
            residual_sum += float(np.abs(actions[active]).sum())
            residual_count += int(active.sum()) * actions.shape[1]
        observation, _, done, infos = env.step(actions)
        for index in np.flatnonzero(done & active):
            results[int(index)] = {key: value for key, value in infos[index].items()
                                   if key not in ("terminal_observation", "terminal_qpos", "terminal_command")}
        if len(results) == env.num_envs:
            break
    if len(results) != env.num_envs:
        raise RuntimeError("Evaluation did not finish every initial episode")
    ordered = [results[index] for index in range(env.num_envs)]
    indices = [item["reset_index"] for item in ordered]
    if len(set(indices)) != len(indices):
        raise RuntimeError("Evaluation repeated an initial state")
    failures = Counter()
    for item in ordered:
        if item["is_success"]:
            continue
        reason = ("unsafe_contact" if item["unsafe_contact"] else
                  "no_two_finger_contact" if item["grasp_fraction"] == 0 else
                  "no_lift" if not item["milestone_2cm"] else
                  "timeout_after_lift" if item["timeout"] else "lost_grasp")
        failures[reason] += 1
    interval = binomtest(sum(r["is_success"] for r in ordered), len(ordered)).proportion_ci()
    return {"episodes": len(ordered), "success_rate": float(np.mean([r["is_success"] for r in ordered])),
            "success_ci95": [interval.low, interval.high],
            "mean_return": float(np.mean([r["episode"]["r"] for r in ordered])),
            "mean_max_lift_m": float(np.mean([r["max_lift_m"] for r in ordered])),
            "failure_counts": dict(failures),
            "mean_absolute_residual": residual_sum / max(residual_count, 1), "results": ordered}


def paired_comparison(baseline, candidate):
    original = {item["reset_index"]: item for item in baseline["results"]}
    current = {item["reset_index"]: item for item in candidate["results"]}
    if original.keys() != current.keys():
        raise ValueError("Paired evaluation requires identical reset indices")
    rescued = [key for key in original if not original[key]["is_success"] and current[key]["is_success"]]
    regressed = [key for key in original if original[key]["is_success"] and not current[key]["is_success"]]
    discordant = len(rescued) + len(regressed)
    return {"rescued": rescued, "regressed": regressed,
            "success_difference": candidate["success_rate"] - baseline["success_rate"],
            "paired_binomial_p": float(binomtest(len(rescued), discordant).pvalue) if discordant else 1.0}


def compare_checkpoints(teacher, bank, output, residual=None, worlds=None):
    output.mkdir(parents=True, exist_ok=True)
    with np.load(bank) as data:
        count = len(data["qpos"]) if worlds is None else worlds
    model = ActionChunkTransformer.load(teacher)
    env = WarpPickupVecEnv(count, bank, sequential_resets=True)
    try:
        baseline = evaluate_pickup(ACTExecutor(model, env, query_interval=1), env)
    finally:
        env.close()
    reports = {"act": baseline, "provenance": {"bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
                                                 "teacher_sha256": hashlib.sha256(teacher.read_bytes()).hexdigest()}}
    if residual is not None:
        if hashlib.sha256((residual / "act.pt").read_bytes()).hexdigest() != reports["provenance"]["teacher_sha256"]:
            raise ValueError("Residual checkpoint uses a different ACT teacher")
        wrapped = ACTResidualVecEnv(WarpPickupVecEnv(count, bank, sequential_resets=True), teacher,
                                    **residual_configuration(residual))
        env = VecNormalize.load(str(residual / "vecnormalize.pkl"), wrapped)
        env.training, env.norm_reward = False, False
        try:
            policy = PPO.load(residual / "policy.zip", device="cuda")
            reports["residual"] = evaluate_pickup(policy, env, residual=True)
            reports["paired"] = paired_comparison(baseline, reports["residual"])
        finally:
            env.close()
    (output / "comparison.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "results"}
                      for key, value in reports.items()}), flush=True)
    return reports


@budgeted("openarm_sim.training.pickup_evaluation")
def main():
    parser = argparse.ArgumentParser(description="Paired, unique-state pickup checkpoint evaluation")
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--residual", type=Path)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--create-bank", type=int)
    parser.add_argument("--seed", type=int, default=410000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.create_bank is not None:
        if args.bank.exists():
            raise ValueError("Refusing to replace an existing evaluation bank")
        build_reset_bank(args.bank, args.create_bank, args.seed)
    compare_checkpoints(args.teacher, args.bank, args.output, args.residual)


if __name__ == "__main__":
    main()
