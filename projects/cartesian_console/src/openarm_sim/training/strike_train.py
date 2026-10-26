import argparse
from contextlib import closing
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, sync_envs_normalization

from ..act_residual import SquashedResidualPolicy
from ..residual_drumming import ResidualDrumStrikeEnv
from .warp_train import save_teacher
from ..compute_guard import budgeted, check_resources


def evaluate_strike(policy, env, output, episodes=16, video=False, seed=700000):
    output.mkdir(parents=True, exist_ok=True)
    sync_envs_normalization(policy.get_env(), env)
    results = []
    frames = []
    task = env.venv.envs[0].unwrapped
    with mujoco.Renderer(task.sim.model, height=360, width=640) as renderer:
        for episode in range(episodes):
            env.seed(seed + episode)
            observation = env.reset()
            episode_return = 0.0
            for step in range(101):
                action, _ = policy.predict(observation, deterministic=True)
                observation, reward, done, infos = env.step(action)
                episode_return += float(reward[0])
                if video and episode == 0 and not done[0] and step % 2 == 0:
                    renderer.update_scene(task.sim.data, camera="overhead")
                    frames.append(renderer.render().copy())
                if done[0]:
                    info = {key: value for key, value in infos[0].items() if key != "terminal_observation"}
                    results.append({**info, "episode_return": episode_return, "seed": seed + episode})
                    break
            else:
                raise RuntimeError("Strike evaluation did not complete its episode")
    report = {"episodes": len(results), "success_rate": float(np.mean([r["is_success"] for r in results])),
              "contact_rate": float(np.mean([r["hit_count"] > 0 for r in results])),
              "mean_return": float(np.mean([r["episode_return"] for r in results])), "results": results}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    if video:
        if not frames:
            raise RuntimeError("Strike evaluation produced no execution frames")
        with imageio.get_writer(output / "execution.mp4", fps=25, codec="libx264", macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(frame)
    return report


class StrikeCheckpoint(BaseCallback):
    def __init__(self, output, evaluation, interval, run, episodes=16):
        super().__init__()
        self.output, self.evaluation, self.interval, self.run = output, evaluation, interval, run
        self.next_save = interval
        self.best = (-1, -np.inf)
        self.episodes = episodes

    def evaluate(self, name, video=False):
        directory = self.output / name
        save_teacher(self.model, directory)
        report = evaluate_strike(self.model, self.evaluation, directory, episodes=self.episodes, video=video)
        score = (report["success_rate"], report["mean_return"])
        if score > self.best:
            self.best = score
            save_teacher(self.model, self.output / "best")
        self.run.log({"env_steps": self.num_timesteps, "eval/success_rate": report["success_rate"],
                      "eval/contact_rate": report["contact_rate"], "eval/mean_return": report["mean_return"]})
        if video:
            self.run.log({"env_steps": self.num_timesteps,
                          "eval/execution": wandb.Video(str(directory / "execution.mp4"), format="mp4")})
        print(json.dumps({key: value for key, value in report.items() if key != "results"}), flush=True)

    def _on_rollout_start(self):
        check_resources()
        save_teacher(self.model, self.output / "latest")
        metrics = {key: float(value) for key, value in self.logger.name_to_value.items()
                   if isinstance(value, (int, float, np.number))}
        self.run.log({"env_steps": self.num_timesteps, **metrics})
        if self.num_timesteps >= self.next_save:
            self.evaluate(f"checkpoint_{self.num_timesteps:09d}")
            self.next_save += self.interval

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.logger.record_mean("strike/success_rate", float(info["is_success"]))
                self.logger.record_mean("strike/hit_count", info["hit_count"])
                self.logger.record_mean("strike/unsafe_rate", float(info["unsafe_contact"]))
        return True


@budgeted("openarm_sim.training.strike_train")
def main():
    parser = argparse.ArgumentParser(description="Train an oracle nominal-reference strike teacher")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=524288)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    if args.output.exists() and args.resume_checkpoint is None:
        raise ValueError("Use a new strike output directory")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=args.resume_checkpoint is not None)
    settings = {**vars(args), "algorithm": "nominal_strike_residual_PPO", "reward_version": "strike_terminal_potential_v1",
                "oracle": True, "device": "cuda", "randomization": 0.1, "rollout": 256,
                "learning_rate": 1e-4, "ent_coef": 0.001, "target_kl": 0.01}
    (args.output / "run_config.json").write_text(json.dumps(settings, default=str, indent=2) + "\n")
    run = wandb.init(project="openarm-teachers", name=args.output.name, dir=str(args.output),
                     config=json.loads(json.dumps(settings, default=str)), mode="online", force=True)
    run.define_metric("env_steps")
    run.define_metric("*", step_metric="env_steps")
    (args.output / "wandb_url.txt").write_text(run.url + "\n")
    try:
        with closing(make_vec_env(ResidualDrumStrikeEnv, n_envs=args.num_envs, seed=args.seed,
                                  vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "spawn"})) as raw, \
                closing(make_vec_env(ResidualDrumStrikeEnv, n_envs=1)) as raw_eval:
            env = (VecNormalize.load(str(args.resume_checkpoint / "vecnormalize.pkl"), raw)
                   if args.resume_checkpoint is not None else VecNormalize(raw, norm_obs=True, norm_reward=True))
            env.training, env.norm_reward = True, True
            evaluation = VecNormalize(raw_eval, training=False, norm_obs=True, norm_reward=False)
            if args.resume_checkpoint is not None:
                policy = PPO.load(args.resume_checkpoint / "policy.zip", env=env, device="cuda")
            else:
                policy = PPO(SquashedResidualPolicy, env, device="cuda", seed=args.seed, n_steps=256,
                             batch_size=256, n_epochs=5, learning_rate=1e-4, ent_coef=0.001, target_kl=0.01,
                             policy_kwargs={"net_arch": [128, 128], "log_std_init": -3.0}, verbose=1)
                with torch.no_grad():
                    policy.policy.action_net.bias[7] = -3.0
            policy.set_logger(configure(str(args.output), ["stdout", "csv"]))
            callback = StrikeCheckpoint(args.output, evaluation, 65536, run, args.eval_episodes)
            callback.init_callback(policy)
            callback.next_save = (policy.num_timesteps // callback.interval + 1) * callback.interval
            callback.evaluate("baseline", video=True)
            if not args.check_only:
                policy.learn(args.steps, callback=callback, reset_num_timesteps=args.resume_checkpoint is None)
                callback.evaluate("final", video=True)
                best_env = VecNormalize.load(str(args.output / "best/vecnormalize.pkl"), raw_eval)
                best_env.training, best_env.norm_reward = False, False
                best_policy = PPO.load(args.output / "best/policy.zip", env=best_env, device="cuda")
                heldout = evaluate_strike(best_policy, best_env, args.output / "heldout", episodes=64, video=True, seed=800000)
                run.log({"env_steps": policy.num_timesteps, "heldout/success_rate": heldout["success_rate"],
                         "heldout/contact_rate": heldout["contact_rate"]})
    finally:
        run.finish()


if __name__ == "__main__":
    main()
