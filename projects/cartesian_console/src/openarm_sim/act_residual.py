import argparse
import hashlib
import json
from pathlib import Path
import shutil

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.logger import configure
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize

from .act_data import command_action, command_vector, limited_command
from .act_model import ACTController, ActionChunkTransformer
from .paths import RUNS
from .warp_pickup import WarpPickupVecEnv
from .warp_train import TeacherCheckpoint, evaluate_and_record, save_teacher


class SquashedResidualPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_dist = SquashedDiagGaussianDistribution(self.action_space.shape[0])
        self._squash_output = True
        torch.nn.init.zeros_(self.action_net.weight)
        torch.nn.init.zeros_(self.action_net.bias)


class ACTResidualVecEnv(VecEnvWrapper):
    def __init__(self, env: WarpPickupVecEnv, teacher: Path, query_interval=5, joint_scale=0.03, gripper_scale=0.002):
        if not 0 < joint_scale <= 0.08 or not 0 <= gripper_scale <= 0.005:
            raise ValueError("Invalid ACT residual limits")
        self.teacher = ActionChunkTransformer.load(teacher)
        if self.teacher.config.vision or self.teacher.config.state_dim != 57:
            raise ValueError("Residual teacher must be an oracle-state ACT")
        self.teacher.requires_grad_(False)
        self.controller = ACTController(self.teacher, env.num_envs, query_interval)
        self.joint_scale, self.gripper_scale = joint_scale, gripper_scale
        self.last_residual = torch.zeros((env.num_envs, 8), device="cuda")
        self.nominal = command_vector(env).clone()
        self.executed = self.nominal.clone()
        self.raw_observation = None
        size = 57 + 8 + 8 * self.teacher.config.chunk + 1 + 8
        super().__init__(env, observation_space=gym.spaces.Box(-np.inf, np.inf, (size,), dtype=np.float32))

    def _observation(self):
        normalized_chunk = (self.controller.chunk - self.teacher.action_mean) / self.teacher.action_std
        nominal = (self.nominal - self.teacher.action_mean) / self.teacher.action_std
        return torch.cat((torch.as_tensor(self.raw_observation, device="cuda"), nominal,
                          normalized_chunk.flatten(1), self.controller.phase[:, None] / self.controller.query_interval,
                          self.last_residual), dim=1).cpu().numpy()

    def reset(self):
        self.raw_observation = self.venv.reset()
        self.controller.reset(slice(None))
        self.last_residual.zero_()
        self.nominal = self.controller.command(self.raw_observation)
        return self._observation()

    def compose(self, residual):
        residual = torch.as_tensor(residual, device="cuda", dtype=torch.float32)
        if residual.shape != (self.num_envs, 8) or not bool(torch.isfinite(residual).all()) or bool((residual.abs() > 1).any()):
            raise ValueError("Residual actions must be finite and bounded")
        command = self.nominal.clone()
        command[:, :7] += self.joint_scale * residual[:, :7]
        command[:, 7] = torch.clamp(command[:, 7] + self.gripper_scale * residual[:, 7], 0, 0.044)
        current = command_vector(self.venv)
        command[:, :7] = torch.clamp(command[:, :7], self.venv.limits[self.venv.arm_aids, 0],
                                    self.venv.limits[self.venv.arm_aids, 1])
        command[:, :7] = current[:, :7] + torch.clamp(command[:, :7] - current[:, :7], -0.025, 0.025)
        return command

    def step_async(self, actions):
        self.executed = self.compose(actions)
        self.last_residual.copy_(torch.as_tensor(actions, device="cuda"))
        self.venv.step_async(command_action(self.venv, self.executed).cpu().numpy())

    def step_command(self, command):
        self.executed = limited_command(self.venv, command)
        residual = self.executed - self.nominal
        residual[:, :7] /= self.joint_scale
        if self.gripper_scale > 0:
            residual[:, 7] /= self.gripper_scale
        else:
            residual[:, 7] = 0
        self.last_residual.copy_(residual.clamp(-1, 1))
        self.venv.step_async(command_action(self.venv, self.executed).cpu().numpy())
        return self.step_wait()

    def step_wait(self):
        self.raw_observation, reward, done, infos = self.venv.step_wait()
        terminal = self._observation()
        commands = self.executed.cpu().numpy()
        for index in np.flatnonzero(done):
            terminal[index, :57] = infos[index]["terminal_observation"]
            infos[index]["terminal_observation"] = terminal[index]
            infos[index]["terminal_command"] = commands[index].copy()
        self.controller.reset(torch.as_tensor(np.flatnonzero(done), device="cuda"))
        self.last_residual[torch.as_tensor(done, device="cuda")] = 0
        self.nominal = self.controller.command(self.raw_observation)
        return self._observation(), reward, done, infos


class ResidualCheckpoint(TeacherCheckpoint):
    def __init__(self, output, evaluation_env, interval, teacher):
        super().__init__(output, evaluation_env, interval)
        self.teacher_path = teacher

    def _on_step(self):
        if self.num_timesteps >= self.next_save:
            directory = self.output / f"checkpoint_{self.num_timesteps:09d}"
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.teacher_path, directory / "act.pt")
        return super()._on_step()


def main():
    parser = argparse.ArgumentParser(description="Refine a frozen oracle ACT with bounded residual PPO")
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=RUNS / "act/data")
    parser.add_argument("--output", type=Path, default=RUNS / "act/residual")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--save-every", type=int, default=100_000)
    parser.add_argument("--query-interval", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    teacher_path = args.output / "act.pt"
    shutil.copyfile(args.teacher, teacher_path)
    settings = {**vars(args), "algorithm": "ACT+residual_PPO", "normalize": True, "ent_coef": 0.001,
                "learning_rate": 0.0001, "rollout": 64, "batch_size": min(1024, args.num_envs * 64),
                "log_std_init": -3.0, "network": [128, 128],
                "teacher_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
                "joint_residual_rad": 0.03, "gripper_residual_m": 0.002}
    (args.output / "run_config.json").write_text(json.dumps(settings, default=str, indent=2) + "\n")
    raw = ACTResidualVecEnv(WarpPickupVecEnv(args.num_envs, args.data / "train_resets.npz"), teacher_path, args.query_interval)
    evaluation = ACTResidualVecEnv(WarpPickupVecEnv(32, args.data / "eval_resets.npz"), teacher_path, args.query_interval)
    env = VecNormalize(raw, norm_obs=True, norm_reward=True)
    eval_env = VecNormalize(evaluation, training=False, norm_obs=True, norm_reward=False)
    try:
        policy = PPO(SquashedResidualPolicy, env, device="cuda", n_steps=64, batch_size=min(1024, args.num_envs * 64),
                     ent_coef=0.001, learning_rate=0.0001, policy_kwargs={"net_arch": [128, 128], "log_std_init": -3.0}, verbose=1)
        policy.set_logger(configure(str(args.output), ["stdout", "csv"]))
        policy.learn(total_timesteps=args.steps, callback=ResidualCheckpoint(args.output, eval_env, args.save_every, teacher_path))
        save_teacher(policy, args.output / "final")
        shutil.copyfile(teacher_path, args.output / "final/act.pt")
        evaluate_and_record(policy, eval_env, args.output / "final", policy.num_timesteps)
        if hashlib.sha256(teacher_path.read_bytes()).hexdigest() != settings["teacher_sha256"]:
            raise RuntimeError("Frozen ACT checkpoint changed during PPO training")
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
