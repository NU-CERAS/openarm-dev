import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
import torch

from openarm_sim.compute_guard import budgeted, check_resources

from ..envs.camera_free_strike import CameraFreeStrikeEnv, REWARD_VERSION
from ..evaluation.camera_free import evaluate
from ..models.asymmetric import AsymmetricStrikePolicy, DeployableActor
from ..skills.registry import SKILLS
from .vector_env import TrainingSubprocVecEnv


def save_bundle(policy, output, options):
    output.mkdir(parents=True, exist_ok=True)
    policy.save(output / "policy.zip")
    normalizer = policy.get_vec_normalize_env()
    normalizer.save(str(output / "vecnormalize.pkl"))
    with CameraFreeStrikeEnv(**options) as env:
        reference = env.reference.document()
        limits = {"bounds_rad": env.controller.bounds.tolist(), "velocity_rad_s": (.8 * env.controller.velocity).tolist(),
                  "acceleration_rad_s2": env.acceleration.tolist(), "period_s": .01,
                  "tracking_error_rad": .2, "feedback_timeout_s": .1, "scale_rad": .08, "slew_rad": .015}
        limits["position_gains"] = env.controller.kp.tolist()
        limits["velocity_gains"] = env.controller.kd.tolist()
        action_semantics = env.action_semantics
        action_dim = env.action_dim
        mujoco.mj_saveModel(env.servo.model, str(output / "gravity_model.mjb"), None)
    actor = DeployableActor(policy.policy, normalizer).eval()
    example = torch.zeros((1, actor.mean.numel()))
    torch.jit.trace(actor, example).save(str(output / "actor.pt"))
    (output / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    (output / "limits.json").write_text(json.dumps(limits, indent=2) + "\n")
    shutil.copyfile(options["profile_file"], output / "profile.json")
    stats = normalizer.obs_rms["actor"]
    (output / "normalization.json").write_text(json.dumps({"mean": stats.mean.tolist(), "variance": stats.var.tolist(),
                                                          "epsilon": normalizer.epsilon, "clip": normalizer.clip_obs}) + "\n")
    files = ["actor.pt", "reference.json", "limits.json", "profile.json", "normalization.json", "policy.zip", "vecnormalize.pkl", "gravity_model.mjb"]
    manifest = {"schema": "openarm.camera_free_skill.v1", "target": options["target"],
                "arm": SKILLS[options["target"]].arm, "tool": SKILLS[options["target"]].name,
                "reward_version": REWARD_VERSION, "history_frames": options.get("history_frames", 8),
                "action_semantics": action_semantics,
                "actor_dim": int(actor.mean.numel()), "action_dim": action_dim, "control_hz": 100,
                "command_semantics": "position_velocity_PD_with_nominal_encoder_gravity", "mujoco_version": mujoco.__version__,
                "steps": int(policy.num_timesteps), "optimization_updates": int(policy._n_updates),
                "scene_identity": json.loads(Path(options["profile_file"]).read_text())["scene_identity"],
                "sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in files}}
    (output / "skill.json").write_text(json.dumps(manifest, indent=2) + "\n")


class EvaluationCallback(BaseCallback):
    def __init__(self, output, options, interval, episodes, run):
        super().__init__()
        self.output, self.options, self.interval, self.episodes, self.run = output, options, interval, episodes, run
        self.next_save, self.best = interval, (-np.inf, -np.inf)

    def assess(self, name, video=False):
        output = self.output / name
        save_bundle(self.model, output, self.options)
        report = evaluate(self.options, output, self.episodes, 700000, self.model, self.model.get_vec_normalize_env(), video)
        score = (report["success_rate"] - report["unsafe_rate"], report["mean_return"])
        if self.model._n_updates > 0 and score > self.best:
            self.best = score
            save_bundle(self.model, self.output / "best", self.options)
        summary = {key: value for key, value in report.items() if key != "results"}
        print(json.dumps({"evaluation": name, "steps": self.num_timesteps, **summary}), flush=True)
        if self.run is not None:
            self.run.log({"env_steps": self.num_timesteps, **{f"eval/{k}": v for k, v in summary.items() if isinstance(v, (int, float))}})
        return report

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                for key in ("is_success", "hit_count", "unsafe_contact"):
                    self.logger.record_mean(f"strike/{key}", float(info[key]))
        if self.num_timesteps >= self.next_save:
            self.assess(f"checkpoint_{self.num_timesteps:09d}")
            self.next_save += self.interval
        return True

    def _on_rollout_start(self):
        check_resources()
        if self.run is not None:
            self.run.log({"env_steps": self.num_timesteps, **{k: float(v) for k, v in self.logger.name_to_value.items() if isinstance(v, (int, float, np.number))}})


@budgeted("openarm_rl.training.camera_free")
def main():
    parser = argparse.ArgumentParser(description="Train a camera-free strike actor with an oracle critic")
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--profile-file", type=Path, required=True)
    parser.add_argument("--target", choices=list(SKILLS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=131072)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--checkpoint-steps", type=int, default=32768)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--heldout-episodes", type=int, default=64)
    parser.add_argument("--position-range", type=float, default=.01)
    parser.add_argument("--domain-range", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--initialize-act", type=Path)
    parser.add_argument("--initialization-data", type=Path)
    parser.add_argument("--bc-steps", type=int, default=2000)
    parser.add_argument("--resume-bundle", type=Path)
    args = parser.parse_args()
    if min(args.steps, args.checkpoint_steps, args.eval_episodes, args.heldout_episodes) <= 0 or not 1 <= args.num_envs <= 4:
        raise ValueError("Positive counts and at most four environments are required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if bool(args.initialize_act) != bool(args.initialization_data) or args.bc_steps <= 0:
        raise ValueError("ACT initialization requires its demonstrations and positive BC steps")
    if args.resume_bundle is not None and args.initialize_act is not None:
        raise ValueError("Choose ACT initialization or a PPO resume bundle")
    options = {"scene_directory": str(args.scene_directory.resolve()), "profile_file": str(args.profile_file.resolve()),
               "target": args.target, "position_range": args.position_range, "domain_range": args.domain_range, "history_frames": 8}
    with CameraFreeStrikeEnv(**options) as configured:
        control_components = configured.action_dim
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "environment": options,
                "algorithm": "ACT_initialized_asymmetric_PPO" if args.initialize_act else "asymmetric_PPO",
                "actor_oracle_inputs": False, "runtime_contact_inputs": False,
                "reference_replanned_on_reset": False, "reward_version": REWARD_VERSION,
                "learning_rate": .00003 if args.initialize_act else .0001,
                "entropy_coefficient": .0001 if args.initialize_act else .001,
                "actor_architecture": [128, 128], "critic_architecture": [128, 128]}
    (args.output / "run_config.json").write_text(json.dumps(settings, indent=2) + "\n")
    with ExitStack() as resources:
        run = None
        if args.wandb_mode != "disabled":
            import wandb

            run = resources.enter_context(wandb.init(project="openarm-drum-skills", name=args.output.name, dir=str(args.output),
                                                    config=settings, mode=args.wandb_mode))
            run.define_metric("env_steps")
            run.define_metric("*", step_metric="env_steps")
            if run.url:
                (args.output / "wandb_url.txt").write_text(run.url + "\n")
        reference = evaluate(options, args.output / "reference", args.eval_episodes, 700000)
        raw = make_vec_env(CameraFreeStrikeEnv, n_envs=args.num_envs, seed=args.seed, env_kwargs=options,
                           vec_env_cls=TrainingSubprocVecEnv, vec_env_kwargs={"start_method": "spawn"})
        resources.callback(raw.close)
        if args.resume_bundle is not None:
            normalizer = VecNormalize.load(str(args.resume_bundle / "vecnormalize.pkl"), raw)
            normalizer.training, normalizer.norm_reward = True, True
            policy = PPO.load(args.resume_bundle / "policy.zip", env=normalizer, device=args.device)
        else:
            normalizer = VecNormalize(raw, norm_obs=True, norm_reward=True)
            policy = PPO(AsymmetricStrikePolicy, normalizer, n_steps=256, batch_size=256, n_epochs=5,
                         learning_rate=.00003 if args.initialize_act else .0001,
                         ent_coef=.0001 if args.initialize_act else .001, gamma=.99, target_kl=.01,
                         device=args.device, seed=args.seed,
                         policy_kwargs={"log_std_init": -3.5 if args.initialize_act else -2.0}, verbose=1)
            if args.initialize_act is not None and control_components == 2:
                with torch.no_grad():
                    policy.policy.log_std.copy_(torch.tensor([-5., -4.], device=policy.device))
        if args.initialize_act is not None:
            from ..imitation.strike import initialize_ppo

            report = initialize_ppo(policy, normalizer, args.initialize_act, args.initialization_data, args.bc_steps)
            (args.output / "initialization.json").write_text(json.dumps(report, indent=2))
            initialization = evaluate(options, args.output / "initialized_actor", args.eval_episodes, 700000, policy, normalizer)
            print(json.dumps({"initialized_ACT_actor_success": initialization["success_rate"]}), flush=True)
        actual = {"learning_rate": float(policy.lr_schedule(1.)), "entropy_coefficient": policy.ent_coef,
                  "initial_log_std": policy.policy.log_std.detach().cpu().tolist(), "action_dim": control_components}
        settings.update(actual)
        (args.output / "run_config.json").write_text(json.dumps(settings, indent=2) + "\n")
        if run is not None:
            run.config.update(actual, allow_val_change=True)
        policy.set_logger(configure(str(args.output), ["stdout", "csv"]))
        callback = EvaluationCallback(args.output, options, args.checkpoint_steps, args.eval_episodes, run)
        callback.next_save = policy.num_timesteps + args.checkpoint_steps
        policy.learn(args.steps, callback=callback, reset_num_timesteps=args.resume_bundle is None)
        callback.assess("final")
        best = PPO.load(args.output / "best/policy.zip", device=args.device)
        stats = VecNormalize.load(str(args.output / "best/vecnormalize.pkl"), raw)
        stats.training, stats.norm_reward = False, False
        validation = evaluate(options, args.output / "validation_best", 64, 700000, best, stats)
        baseline = evaluate(options, args.output / "heldout_reference", args.heldout_episodes, 800000)
        result = evaluate(options, args.output / "heldout", args.heldout_episodes, 800000, best, stats, args.video)
        exported = torch.jit.load(str(args.output / "best/actor.pt"))
        exported_result = evaluate(options, args.output / "exported_check", min(4, args.heldout_episodes), 800000,
                                   best, stats, exported=exported)
        for actual, expected in zip(exported_result["results"], result["results"]):
            if any(actual[key] != expected[key] for key in ("is_success", "hit_count", "unsafe_contact")):
                raise RuntimeError("Exported actor outcome differs from training policy")
        comparison = {"reference": {k: v for k, v in baseline.items() if k != "results"},
                      "policy": {k: v for k, v in result.items() if k != "results"},
                      "improved_success": result["success_rate"] > baseline["success_rate"],
                      "validation_reference_success": reference["success_rate"], "export_verified": True,
                      "validation_policy": {k: v for k, v in validation.items() if k != "results"}}
        (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        if run is not None:
            run.log({"env_steps": policy.num_timesteps, "heldout/success_rate": result["success_rate"],
                     "heldout/reference_success_rate": baseline["success_rate"], "heldout/unsafe_rate": result["unsafe_rate"]})
            if args.video:
                run.log({"env_steps": policy.num_timesteps, "heldout/video": wandb.Video(str(args.output / "heldout/execution.mp4"), format="mp4")})
        print(json.dumps(comparison), flush=True)


if __name__ == "__main__":
    main()
