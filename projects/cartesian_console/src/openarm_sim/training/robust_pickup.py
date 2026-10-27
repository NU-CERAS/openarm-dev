import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from ..act_residual import ACTResidualVecEnv, SquashedResidualPolicy
from .pickup_evaluation import evaluate_pickup, paired_comparison
from ..warp_pickup import WarpPickupVecEnv
from .warp_train import evaluate_and_record, save_teacher
from ..compute_guard import budgeted, check_resources


class RobustnessCheckpoint(BaseCallback):
    def __init__(self, output, evaluation, interval, teacher, baseline, run):
        super().__init__()
        self.output, self.evaluation, self.interval = output, evaluation, interval
        self.teacher, self.baseline, self.run = teacher, baseline, run
        self.next_save = interval
        self.best = (baseline["success_rate"], baseline["mean_return"])

    def evaluate(self, name):
        report = evaluate_pickup(self.model, self.evaluation, residual=True)
        report["paired"] = paired_comparison(self.baseline, report)
        directory = self.output / name
        save_teacher(self.model, directory)
        shutil.copyfile(self.teacher, directory / "act.pt")
        shutil.copyfile(self.output / "run_config.json", directory / "execution_config.json")
        (directory / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        score = (report["success_rate"], report["mean_return"])
        if score > self.best:
            self.best = score
            best = self.output / "best"
            best.mkdir(exist_ok=True)
            for filename in ("policy.zip", "vecnormalize.pkl", "act.pt", "evaluation.json", "execution_config.json"):
                shutil.copyfile(directory / filename, best / filename)
        if self.run:
            self.run.log({"env_steps": self.num_timesteps,
                          "validation/success_rate": report["success_rate"],
                          "validation/mean_return": report["mean_return"],
                          "validation/rescued": len(report["paired"]["rescued"]),
                          "validation/regressed": len(report["paired"]["regressed"]),
                          "validation/mean_absolute_residual": report["mean_absolute_residual"],
                          **{f"validation/failures/{key}": value for key, value in report["failure_counts"].items()}})
        print(json.dumps({"env_steps": self.num_timesteps, "validation": report["success_rate"],
                          "paired": report["paired"]}), flush=True)
        return report

    def _on_rollout_start(self):
        check_resources()
        if self.run:
            metrics = {key: float(value) for key, value in self.model.logger.name_to_value.items()
                       if isinstance(value, (int, float, np.number))}
            self.run.log({"env_steps": self.num_timesteps, **metrics})
        if self.num_timesteps >= self.next_save:
            self.evaluate(f"checkpoint_{self.num_timesteps:09d}")
            self.next_save += self.interval

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.logger.record_mean("pickup/success_rate", float(info["is_success"]))
                self.logger.record_mean("pickup/max_lift_m", info["max_lift_m"])
                self.logger.record_mean("pickup/grasp_fraction", info["grasp_fraction"])
        return True


@budgeted("openarm_sim.training.robust_pickup")
def main():
    parser = argparse.ArgumentParser(description="Validation-selected ACT residual PPO with paired pickup assessment")
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--train-bank", type=Path, required=True)
    parser.add_argument("--validation-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1048576)
    parser.add_argument("--save-every", type=int, default=131072)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--log-std", type=float, default=-1.5)
    parser.add_argument("--joint-scale", type=float, default=0.06)
    parser.add_argument("--gripper-scale", type=float, default=0.004)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--action-smoothing", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new experiment output directory")
    if min(args.steps, args.save_every, args.rollout, args.num_envs, args.epochs) <= 0:
        raise ValueError("Training sizes must be positive")
    hashes = {name + "_sha256": hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
              (("teacher", args.teacher), ("train_bank", args.train_bank), ("validation_bank", args.validation_bank))}
    if hashes["train_bank_sha256"] == hashes["validation_bank_sha256"]:
        raise ValueError("Training and validation reset banks must differ")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    teacher = args.output / "act.pt"
    shutil.copyfile(args.teacher, teacher)
    settings = {**vars(args), **hashes, "algorithm": "frozen_ACT_residual_PPO",
                "joint_residual_rad": args.joint_scale, "gripper_residual_m": args.gripper_scale,
                "entropy": "current_policy_reparameterized", "gae_lambda": 0.98, "target_kl": 0.01,
                "gamma": 0.99, "query_interval": 1}
    (args.output / "run_config.json").write_text(json.dumps(settings, default=str, indent=2) + "\n")
    with np.load(args.validation_bank) as data:
        evaluation_count = len(data["qpos"])
    raw = ACTResidualVecEnv(WarpPickupVecEnv(args.num_envs, args.train_bank, seed=args.seed), teacher,
                            query_interval=1, joint_scale=args.joint_scale, gripper_scale=args.gripper_scale,
                            action_smoothing=args.action_smoothing)
    test = ACTResidualVecEnv(WarpPickupVecEnv(evaluation_count, args.validation_bank, sequential_resets=True), teacher,
                             query_interval=1, joint_scale=args.joint_scale, gripper_scale=args.gripper_scale,
                             action_smoothing=args.action_smoothing)
    env = VecNormalize(raw, norm_obs=True, norm_reward=True)
    evaluation = VecNormalize(test, training=False, norm_obs=True, norm_reward=False)
    run = wandb.init(project="openarm-teachers", name=args.output.name, dir=str(args.output),
                     config=json.loads(json.dumps(settings, default=str)), mode="online", force=True) if args.wandb else None
    if run:
        run.define_metric("env_steps")
        run.define_metric("*", step_metric="env_steps")
        (args.output / "wandb_url.txt").write_text(run.url + "\n")
    try:
        policy = PPO(SquashedResidualPolicy, env, device="cuda", n_steps=args.rollout,
                     batch_size=min(2048, args.num_envs * args.rollout), n_epochs=args.epochs,
                     learning_rate=args.learning_rate, ent_coef=args.ent_coef, gae_lambda=0.98,
                     target_kl=0.01, seed=args.seed,
                     policy_kwargs={"net_arch": [128, 128], "log_std_init": args.log_std}, verbose=1)
        policy.set_logger(configure(str(args.output), ["stdout", "csv"]))
        baseline = evaluate_pickup(policy, evaluation, residual=True)
        (args.output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
        callback = RobustnessCheckpoint(args.output, evaluation, args.save_every, teacher, baseline, run)
        policy.learn(args.steps, callback=callback)
        callback.evaluate("final")
        selected = args.output / "best"
        if selected.exists():
            selected_env = VecNormalize.load(str(selected / "vecnormalize.pkl"), test)
            selected_env.training, selected_env.norm_reward = False, False
            policy = PPO.load(selected / "policy.zip", env=selected_env, device="cuda")
            evaluate_and_record(policy, selected_env, selected / "video", policy.num_timesteps)
        if hashlib.sha256(teacher.read_bytes()).hexdigest() != hashes["teacher_sha256"]:
            raise RuntimeError("Frozen teacher changed during training")
    finally:
        env.close()
        evaluation.close()
        if run:
            run.finish()


if __name__ == "__main__":
    main()
