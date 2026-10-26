import argparse
import json
from contextlib import closing
from functools import partial
from pathlib import Path

import numpy as np
import yaml
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

from .drumming import DrumStrikeEnv
from .pickup import StickPickupEnv
from .residual_drumming import ResidualDrumStrikeEnv
from .paths import RUNS


def environment_factory(task: str, curriculum_stage=2, randomization=0.1, residual_scale=0.08):
    if task == "residual_strike":
        return partial(ResidualDrumStrikeEnv, curriculum_stage=curriculum_stage, randomization=randomization, residual_scale=residual_scale)
    return {"pickup": StickPickupEnv, "strike": DrumStrikeEnv}[task]


def evaluate(policy: PPO, episodes: int, seed: int = 1000, task: str = "residual_strike", **environment_options) -> dict:
    results = []
    environment = environment_factory(task, **environment_options)
    with environment() as env:
        for index in range(episodes):
            observation, _ = env.reset(seed=seed + index)
            episode_return = 0.0
            while True:
                action, _ = policy.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                if terminated or truncated:
                    results.append({**info, "episode_return": episode_return})
                    break
    return {
        "episodes": episodes,
        "success_rate": float(np.mean([result["is_success"] for result in results])),
        "mean_return": float(np.mean([result["episode_return"] for result in results])),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate an OpenArm pickup or strike policy")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", choices=["pickup", "strike", "residual_strike"], default="residual_strike")
    parser.add_argument("--curriculum-stage", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--randomization", type=float, default=0.1)
    parser.add_argument("--residual-scale", type=float, default=0.08)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=50_000)
    parser.add_argument("--train", action="store_true", help="Explicitly enable learning")
    parser.add_argument("--check-env", action="store_true", help="Validate the environment without learning (default)")
    parser.add_argument("--resume", type=Path, help="Continue learning from an existing checkpoint with --train")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--rollout", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output", type=Path, default=RUNS / "rl/drum")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate an existing checkpoint")
    preliminary, _ = parser.parse_known_args()
    if preliminary.config is not None:
        configuration = yaml.safe_load(preliminary.config.read_text())
        allowed = {action.dest for action in parser._actions} - {"help", "config", "train", "checkpoint", "resume"}
        if not isinstance(configuration, dict) or set(configuration) - allowed:
            raise ValueError("Training configuration contains unknown fields")
        parser.set_defaults(**configuration)
    args = parser.parse_args()
    if (args.train and args.checkpoint is not None) or (args.check_env and (args.train or args.checkpoint is not None)):
        raise ValueError("Choose environment validation, training, or checkpoint evaluation")
    if args.resume is not None and not args.train:
        raise ValueError("--resume requires --train")
    if min(args.steps, args.rollout, args.batch_size, args.epochs, args.eval_episodes, args.num_envs, args.save_every) <= 0:
        raise ValueError("Training and evaluation parameters must be positive integers")
    if (args.rollout * args.num_envs) % args.batch_size or args.batch_size < 2:
        raise ValueError("Total rollout must be divisible by batch-size, with batch-size at least 2")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    options = {"curriculum_stage": args.curriculum_stage, "randomization": args.randomization, "residual_scale": args.residual_scale}
    environment = environment_factory(args.task, **options)
    if not args.train and args.checkpoint is None:
        with environment() as env:
            check_env(env.unwrapped if args.task != "residual_strike" else env, skip_render_check=True)
            observation, info = env.reset(seed=args.seed)
            report = {"task": args.task, "observation_shape": list(observation.shape),
                      "action_shape": list(env.action_space.shape), "learning_started": False, "reset": info, **options}
        (args.output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
        return
    if args.checkpoint is not None:
        policy = PPO.load(args.checkpoint, device=args.device)
    else:
        with closing(make_vec_env(
            environment, n_envs=args.num_envs, seed=args.seed,
            monitor_dir=str(args.output / "monitor"),
            vec_env_cls=SubprocVecEnv if args.num_envs > 1 else None,
            vec_env_kwargs={"start_method": "spawn"} if args.num_envs > 1 else None,
        )) as env, closing(make_vec_env(environment, seed=1000)) as eval_env:
            if args.resume is not None:
                policy = PPO.load(args.resume, env=env, device=args.device)
            else:
                policy = PPO(
                    "MlpPolicy", env, n_steps=args.rollout, batch_size=args.batch_size,
                    n_epochs=args.epochs, learning_rate=3e-4, seed=args.seed,
                    device=args.device, verbose=1, policy_kwargs={"net_arch": [128, 128]},
                )
            policy.set_logger(configure(str(args.output), ["stdout", "csv"]))
            frequency = max(1, args.save_every // args.num_envs)
            callbacks = [
                CheckpointCallback(save_freq=frequency, save_path=str(args.output / "checkpoints")),
                EvalCallback(eval_env, eval_freq=frequency, n_eval_episodes=args.eval_episodes,
                             best_model_save_path=str(args.output / "best"), log_path=str(args.output / "eval")),
            ]
            policy.learn(total_timesteps=args.steps, callback=callbacks)
            policy.save(args.output / "policy")
    report = evaluate(policy, args.eval_episodes, task=args.task, **options)
    (args.output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
