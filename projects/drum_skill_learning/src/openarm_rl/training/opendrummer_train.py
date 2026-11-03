import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import time

import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from openarm_sim.compute_guard import budgeted, check_resources
from ..envs.opendrummer import OpenDrummerStrikeEnv, REWARD_VERSION, prepare_scene
from ..models.policies import SquashedResidualPolicy, load_residual_policy
from .vector_env import TrainingSubprocVecEnv
from ..planning.strikes import plan_center
from ..skills.registry import SKILLS


def evaluate(policy, normalizer, options, output, episodes, seed, video=False):
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with ExitStack() as resources:
        env = resources.enter_context(OpenDrummerStrikeEnv(**options, render_mode="rgb_array" if video else None))
        writer = resources.enter_context(imageio.get_writer(output / "execution.mp4", fps=50, codec="libx264")) if video else None
        for index in range(episodes):
            observation, _ = env.reset(seed=seed + index)
            total = 0.0
            trace = []
            for step in range(301):
                if policy is None:
                    action = np.zeros(1, np.float32)
                else:
                    action, _ = policy.predict(normalizer.normalize_obs(observation), deterministic=True)
                observation, reward, done, truncated, info = env.step(action)
                total += reward
                if index == 0:
                    r = env.reference
                    trace.append({"time_s": float(env.data.time), "action": action.tolist(),
                                  "joint_positions_rad": env.data.qpos[r.controller.q].tolist(),
                                  "joint_command_rad": r.rows[-1]["commanded_joint_positions_rad"],
                                  "phase": info["phase"], "reward_terms": info["reward_terms"]})
                    if writer is not None and step % 2 == 0:
                        writer.append_data(env.render())
                if done or truncated:
                    results.append({**info, "episode_return": float(total), "seed": seed + index})
                    break
            else:
                raise RuntimeError("Strike episode did not terminate within 301 control steps")
            if index == 0:
                (output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    report = {"episodes": episodes, "seed": seed, "reward_version": REWARD_VERSION,
              "target": options["target"], "arm": options["arm"],
              "success_rate": float(np.mean([r["is_success"] for r in results])),
              "recovery_rate": float(np.mean([r["recovered"] for r in results])),
              "contact_rate": float(np.mean([r["hit_count"] > 0 for r in results])),
              "unsafe_rate": float(np.mean([r["unsafe_contact"] for r in results])),
              "mean_hit_quality": float(np.mean([r["hit_quality"] for r in results])),
              "mean_return": float(np.mean([r["episode_return"] for r in results])), "results": results}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def save(policy, directory, options):
    directory.mkdir(parents=True, exist_ok=True)
    policy.save(directory / "policy.zip")
    policy.get_vec_normalize_env().save(str(directory / "vecnormalize.pkl"))
    profile_path = Path(options["profile_file"])
    shutil.copyfile(profile_path, directory / "profile.json")
    profile = json.loads(profile_path.read_text())
    skill = {"schema": "openarm.strike_checkpoint.v1", "target": options["target"], "arm": options["arm"],
             "tool": SKILLS[options["target"]].name, "reward_version": REWARD_VERSION,
             "scene_identity": profile["scene_identity"], "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
             "observation_dim": 67, "action_dim": 1, "control_hz": 100, "steps": policy.num_timesteps}
    (directory / "skill.json").write_text(json.dumps(skill, indent=2) + "\n")


class StrikeEvaluation(BaseCallback):
    def __init__(self, output, options, interval, episodes, run):
        super().__init__()
        self.output, self.options = output, options
        self.interval, self.episodes, self.run = interval, episodes, run
        self.next_save = interval
        self.best = (-np.inf, -np.inf)

    def assess(self, name, video=False):
        directory = self.output / name
        save(self.model, directory, self.options)
        report = evaluate(self.model, self.model.get_vec_normalize_env(), self.options, directory,
                          self.episodes, 700000, video)
        score = (report["success_rate"], report["mean_return"])
        if score > self.best:
            self.best = score
            save(self.model, self.output / "best", self.options)
        summary = {k: v for k, v in report.items() if k != "results"}
        print(json.dumps({"steps": self.num_timesteps, "evaluation": name, **summary}), flush=True)
        if self.run is not None:
            self.run.log({"env_steps": self.num_timesteps, **{f"eval/{k}": v for k, v in summary.items() if isinstance(v, (int, float))}})
            if video:
                import wandb

                self.run.log({"env_steps": self.num_timesteps,
                              "eval/execution": wandb.Video(str(directory / "execution.mp4"), format="mp4")})
        return report

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                for key in ("is_success", "hit_count", "unsafe_contact", "hit_quality"):
                    self.logger.record_mean(f"strike/{key}", float(info[key]))
        if self.num_timesteps >= self.next_save:
            self.assess(f"checkpoint_{self.num_timesteps:09d}")
            self.next_save += self.interval
        return True

    def _on_rollout_start(self):
        check_resources()
        if self.run is not None:
            values = {k: float(v) for k, v in self.logger.name_to_value.items() if isinstance(v, (int, float, np.number))}
            self.run.log({"env_steps": self.num_timesteps, **values})


@budgeted("openarm_rl.training.opendrummer_train")
def main():
    parser = argparse.ArgumentParser(description="Train an oracle J7 residual PPO teacher on the OpenDrummer drum scene")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=list(SKILLS), default="snare")
    parser.add_argument("--arm", choices=["left", "right"])
    parser.add_argument("--profile-file", type=Path)
    parser.add_argument("--steps", type=int, default=262144)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--checkpoint-steps", type=int, default=32768)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--heldout-episodes", type=int, default=64)
    parser.add_argument("--position-range", type=float, default=0.05)
    parser.add_argument("--domain-range", type=float, default=0.1)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    skill = SKILLS[args.target]
    if args.arm is not None and args.arm != skill.arm:
        raise ValueError("The arm is fixed by the selected drum side")
    args.arm = skill.arm
    if args.profile_file is not None:
        args.profile_file = args.profile_file.resolve()
        if not args.profile_file.is_file():
            raise FileNotFoundError(args.profile_file)
    if args.steps <= 0 or not 1 <= args.num_envs <= 8 or min(args.eval_episodes, args.heldout_episodes, args.checkpoint_steps) <= 0:
        raise ValueError("Positive step/episode counts and 1–8 parallel environments are required")
    if not 0 <= args.position_range <= 0.05 or not 0 <= args.domain_range <= 0.2:
        raise ValueError("Position range must be in [0, 0.05] m and domain range in [0, 0.2]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = args.resume_checkpoint.resolve()
        for name in ("policy.zip", "vecnormalize.pkl"):
            if not (args.resume_checkpoint / name).is_file():
                raise FileNotFoundError(args.resume_checkpoint / name)
    torch.set_num_threads(1)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    prepare_scene(args.output / "scene")
    if args.profile_file is None:
        import mujoco

        model = mujoco.MjModel.from_binary_path(str(args.output / "scene/model.mjb"))
        metadata = json.loads((args.output / "scene/scene.json").read_text())
        profile = plan_center(model, metadata["robot"], metadata["scene"], skill)
        args.profile_file = args.output / "center-profile.json"
        args.profile_file.write_text(json.dumps(profile, indent=2) + "\n")
        if profile["status"] != "ready":
            raise ValueError(f"No center strike profile: {profile['reason']}")
    profile_hash = hashlib.sha256(args.profile_file.read_bytes()).hexdigest()
    if args.resume_checkpoint is not None:
        previous = json.loads((args.resume_checkpoint / "skill.json").read_text())
        if (previous["target"], previous["arm"], previous["reward_version"], previous["profile_sha256"]) != (
                args.target, args.arm, REWARD_VERSION, profile_hash):
            raise ValueError("Resume checkpoint target, arm, reward or reference profile differs")
    options = {"scene_directory": str(args.output / "scene"), "position_range": args.position_range,
               "domain_range": args.domain_range, "target": args.target, "arm": args.arm,
               "profile_file": str(args.profile_file)}
    settings = {**vars(args), "output": str(args.output), "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
                "profile_file": str(args.profile_file), "profile_sha256": profile_hash, "environment": options,
                "algorithm": "OpenDrummer_J7_residual_PPO", "reward_version": REWARD_VERSION,
                "oracle_observations": True, "action": "J7 reference offset, 0.08 rad maximum, slew 0.015 rad/control step",
                "physics": "native MuJoCo CPU", "control_hz": 100, "rollout": 256,
                "batch_size": 256, "n_epochs": 5, "learning_rate": 1e-4, "ent_coef": 0.001,
                "gamma": 0.99, "target_kl": 0.01, "net_arch": [128, 128], "normalize": True}
    (args.output / "run_config.json").write_text(json.dumps(settings, indent=2) + "\n")
    started = time.monotonic()
    with ExitStack() as resources:
        run = None
        if args.wandb_mode != "disabled":
            import wandb

            run = resources.enter_context(wandb.init(project="openarm-teachers", name=args.output.name,
                                                    dir=str(args.output), config=settings, mode=args.wandb_mode,
                                                    settings=wandb.Settings(init_timeout=30)))
            run.define_metric("env_steps")
            run.define_metric("*", step_metric="env_steps")
            if run.url:
                (args.output / "wandb_url.txt").write_text(run.url + "\n")
        baseline = evaluate(None, None, options, args.output / "reference", args.eval_episodes, 700000, args.video)
        if run is not None:
            run.log({"env_steps": 0, "reference/success_rate": baseline["success_rate"],
                     "reference/mean_return": baseline["mean_return"]})
            if args.video:
                run.log({"env_steps": 0, "reference/execution": wandb.Video(str(args.output / "reference/execution.mp4"), format="mp4")})
        raw = make_vec_env(OpenDrummerStrikeEnv, n_envs=args.num_envs, seed=args.seed, env_kwargs=options,
                           vec_env_cls=TrainingSubprocVecEnv, vec_env_kwargs={"start_method": "spawn"})
        resources.callback(raw.close)
        if args.resume_checkpoint is None:
            env = VecNormalize(raw, norm_obs=True, norm_reward=True, gamma=0.99)
            policy = PPO(SquashedResidualPolicy, env, device=args.device, seed=args.seed, n_steps=256,
                         batch_size=256, n_epochs=5, learning_rate=1e-4, ent_coef=0.001,
                         gamma=0.99, target_kl=0.01, policy_kwargs={"net_arch": [128, 128], "log_std_init": -2.0}, verbose=1)
        else:
            env = VecNormalize.load(str(args.resume_checkpoint / "vecnormalize.pkl"), raw)
            env.training, env.norm_reward = True, True
            policy = load_residual_policy(args.resume_checkpoint / "policy.zip", env=env, device=args.device)
            policy.set_random_seed(args.seed)
        policy.set_logger(configure(str(args.output), ["stdout", "csv"]))
        callback = StrikeEvaluation(args.output, options, args.checkpoint_steps, args.eval_episodes, run)
        callback.init_callback(policy)
        callback.num_timesteps = policy.num_timesteps
        callback.next_save = (policy.num_timesteps // args.checkpoint_steps + 1) * args.checkpoint_steps
        callback.assess("initial")
        policy.learn(args.steps, callback=callback, reset_num_timesteps=args.resume_checkpoint is None)
        trained_steps = policy.num_timesteps
        final = callback.assess("final", args.video)
        policy = load_residual_policy(args.output / "best/policy.zip", env=env, device=args.device)
        normalizer = VecNormalize.load(str(args.output / "best/vecnormalize.pkl"), raw)
        normalizer.training, normalizer.norm_reward = False, False
        heldout_reference = evaluate(None, None, options, args.output / "heldout_reference", args.heldout_episodes, 800000)
        heldout = evaluate(policy, normalizer, options, args.output / "heldout", args.heldout_episodes, 800000, args.video)
        comparison = {"steps": trained_steps, "selected_checkpoint_steps": policy.num_timesteps,
                      "elapsed_seconds": time.monotonic() - started,
                      "validation_reference_success": baseline["success_rate"], "validation_final_success": final["success_rate"],
                      "heldout_reference_success": heldout_reference["success_rate"], "heldout_policy_success": heldout["success_rate"],
                      "heldout_reference_return": heldout_reference["mean_return"], "heldout_policy_return": heldout["mean_return"],
                      "improved_heldout_success": heldout["success_rate"] > heldout_reference["success_rate"]}
        (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        if run is not None:
            run.log({"env_steps": trained_steps, **{f"heldout/{k}": v for k, v in comparison.items() if isinstance(v, (int, float))}})
            if args.video:
                run.log({"env_steps": trained_steps, "heldout/execution": wandb.Video(str(args.output / "heldout/execution.mp4"), format="mp4")})
        print(json.dumps(comparison), flush=True)


if __name__ == "__main__":
    main()
