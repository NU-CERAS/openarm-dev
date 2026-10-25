import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

from .warp_pickup import WarpPickupVecEnv, build_reset_bank
from .pickup_reward import REWARD_VERSION


def evaluate_and_record(policy: PPO, env: WarpPickupVecEnv, directory: Path, steps: int) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(env, VecNormalize):
        sync_envs_normalization(policy.get_env(), env)
    render_env = env.venv if isinstance(env, VecNormalize) else env
    while hasattr(render_env, "venv"):
        render_env = render_env.venv
    env.seed(1000)
    observation = env.reset()
    results = {}
    trace = []
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.40, -0.25, 0.40]
    camera.distance = 1.15
    camera.azimuth = 135
    camera.elevation = -30
    with mujoco.Renderer(render_env.model, height=480, width=640) as renderer, imageio.get_writer(
        directory / "execution.mp4", fps=25, codec="libx264", macro_block_size=16,
        output_params=["-movflags", "+faststart"],
    ) as writer:
        for tick in range(301):
            actions, _ = policy.predict(observation, deterministic=True)
            observation, rewards, dones, infos = env.step(actions)
            if 0 not in results and (tick % 2 == 0 or dones[0]):
                data = render_env.snapshot(0)
                if dones[0]:
                    data.qpos[:] = infos[0]["terminal_qpos"]
                    mujoco.mj_forward(render_env.model, data)
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 45), fill="black")
                draw.text((10, 8), f"Pickup teacher | {steps:,} transitions | deterministic evaluation", fill="white")
                draw.text((10, 26), f"MuJoCo Warp | t={(tick + 1) * 0.02:.2f}s | XY randomization +/- 5 cm", fill="white")
                writer.append_data(np.asarray(frame))
                trace.append({"time_s": (tick + 1) * 0.02, "qpos": data.qpos.tolist(), "action": actions[0].tolist()})
            for i in np.flatnonzero(dones):
                if int(i) not in results:
                    results[int(i)] = {key: value for key, value in infos[i].items()
                                       if key not in ("terminal_observation", "terminal_qpos", "terminal_command")}
            if len(results) == env.num_envs:
                break
    if len(results) != env.num_envs:
        raise RuntimeError("Evaluation did not finish all pickup episodes")
    report = {"steps": steps, "episodes": env.num_envs,
              "success_rate": float(np.mean([r["is_success"] for r in results.values()])),
              "mean_return": float(np.mean([r["episode"]["r"] for r in results.values()])),
              "results": results}
    (directory / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    (directory / "execution.json").write_text(json.dumps(trace) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}), flush=True)
    return report


def save_teacher(policy: PPO, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    policy.save(directory / "policy")
    environment = policy.get_env()
    if isinstance(environment, VecNormalize):
        environment.save(str(directory / "vecnormalize.pkl"))


class TeacherCheckpoint(BaseCallback):
    def __init__(self, output: Path, evaluation_env: WarpPickupVecEnv, interval: int):
        super().__init__()
        self.output = output
        self.evaluation_env = evaluation_env
        self.interval = interval
        self.next_save = interval

    def _on_step(self) -> bool:
        self.logger.record_mean("pickup/action_clip_fraction", float(np.mean(np.abs(self.locals["actions"]) > 1)))
        for info in self.locals["infos"]:
            if "episode" not in info:
                continue
            for metric, key in (("max_lift_m", "max_lift_m"), ("reached_2cm_rate", "milestone_2cm"),
                                ("reached_5cm_rate", "milestone_5cm"), ("grasp_fraction", "grasp_fraction"),
                                ("timeout_rate", "timeout")):
                self.logger.record_mean(f"pickup/{metric}", float(info[key]))
            for name, value in info["reward_terms"].items():
                self.logger.record_mean(f"reward/{name}", value)
        if self.num_timesteps >= self.next_save:
            directory = self.output / f"checkpoint_{self.num_timesteps:09d}"
            directory.mkdir(parents=True, exist_ok=True)
            save_teacher(self.model, directory)
            evaluate_and_record(self.model, self.evaluation_env, directory, self.num_timesteps)
            self.next_save += self.interval
        return True


def main():
    parser = argparse.ArgumentParser(description="Train an oracle pickup teacher with GPU MuJoCo physics")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50_000_000)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5_000_000)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--bank-size", type=int, default=1024)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--position-range", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(".work/rl/pickup-teacher-milestones50m"))
    args = parser.parse_args()
    if min(args.num_envs, args.steps, args.rollout, args.batch_size, args.epochs, args.save_every,
           args.bank_size, args.eval_episodes) <= 0 or not 0 < args.position_range <= 0.05:
        raise ValueError("Invalid GPU training parameters")
    if args.num_envs * args.rollout % args.batch_size:
        raise ValueError("Total rollout must be divisible by minibatch size")
    if not np.isfinite(args.ent_coef) or args.ent_coef < 0:
        raise ValueError("Entropy coefficient must be finite and nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU teacher training requires CUDA")
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = {**vars(args), "reward_version": REWARD_VERSION, "observation_size": 57,
                     "gamma": 0.99, "timeout_is_task_failure": True}
    (args.output / "run_config.json").write_text(json.dumps(configuration, default=str, indent=2) + "\n")
    torch.set_num_threads(1)
    train_bank = args.output / "train_resets.npz"
    eval_bank = args.output / "eval_resets.npz"
    build_reset_bank(train_bank, args.bank_size, args.seed, args.position_range)
    build_reset_bank(eval_bank, max(32, args.eval_episodes), 1000, args.position_range)
    env = WarpPickupVecEnv(args.num_envs, train_bank, args.seed)
    eval_env = WarpPickupVecEnv(args.eval_episodes, eval_bank, 1000)
    if args.normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10, clip_reward=10, gamma=0.99)
        eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10, gamma=0.99)
    try:
        policy = PPO("MlpPolicy", env, device="cuda", learning_rate=3e-4, n_steps=args.rollout,
                     ent_coef=args.ent_coef,
                     batch_size=args.batch_size, n_epochs=args.epochs, seed=args.seed,
                     policy_kwargs={"net_arch": [128, 128]}, verbose=1)
        policy.set_logger(configure(str(args.output), ["stdout", "csv"]))
        free_memory, total_memory = torch.cuda.mem_get_info()
        print(f"GPU physics initialized: {args.num_envs} worlds; "
              f"device memory in use {(total_memory - free_memory) / 2**20:.1f} MiB", flush=True)
        callback = TeacherCheckpoint(args.output, eval_env, args.save_every)
        policy.learn(total_timesteps=args.steps, callback=callback)
        save_teacher(policy, args.output)
        save_teacher(policy, args.output / "final")
        evaluate_and_record(policy, eval_env, args.output / "final", policy.num_timesteps)
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
