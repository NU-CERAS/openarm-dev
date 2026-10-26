import argparse
import hashlib
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
import torch

from .warp_pickup import WarpPickupVecEnv, build_reset_bank
from .paths import RUNS


SCHEMA = "openarm.act_episodes.v1"


def command_vector(env: WarpPickupVecEnv) -> torch.Tensor:
    return torch.cat((env.desired[:, env.arm_aids], env.desired[:, env.cpu.grip_actuator_id, None]), dim=1)


def command_action(env: WarpPickupVecEnv, command: torch.Tensor) -> torch.Tensor:
    current = command_vector(env)
    result = torch.zeros_like(command)
    result[:, :7] = torch.clamp((command[:, :7] - current[:, :7]) / 0.025, -1, 1)
    result[:, 7] = torch.clamp(command[:, 7] / 0.022 - 1, -1, 1)
    return result


def limited_command(env: WarpPickupVecEnv, command: torch.Tensor) -> torch.Tensor:
    action = command_action(env, command)
    current = command_vector(env)
    result = current.clone()
    result[:, :7] = torch.clamp(current[:, :7] + 0.025 * action[:, :7],
                               env.limits[env.arm_aids, 0], env.limits[env.arm_aids, 1])
    result[:, 7] = (action[:, 7] + 1) * 0.022
    return result


def proprioception(observation: np.ndarray) -> np.ndarray:
    return np.concatenate((observation[..., :16], observation[..., 34:42]), axis=-1)


def collect_reference(bank_path: Path, output: Path, episodes: int, worlds: int, seed: int) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    env = WarpPickupVecEnv(worlds, bank_path, seed)
    report = {"schema": SCHEMA, "requested": episodes, "successful": 0, "attempted": 0, "failures": []}
    with h5py.File(output, "w") as file:
        file.attrs.update(schema=SCHEMA, source="reference_controller", bank_seed=seed,
                          bank_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest(),
                          command_units="rad,rad,rad,rad,rad,rad,rad,m", control_hz=50)
        group = file.create_group("episodes")
        while report["attempted"] < episodes:
            count = min(worlds, episodes - report["attempted"])
            env.seed(seed + report["attempted"])
            observation = env.reset()
            active = np.arange(worlds) < count
            records = [{"observation": [], "command": [], "qpos": [], "qvel": []} for _ in range(worlds)]
            initial = env.qpos.cpu().numpy().copy()
            drums = env.geom_positions[:, env.drum_id].cpu().numpy().copy()
            goal = command_vector(env).clone()
            for phase, duration in (("lower", 75), ("close", 50), ("lift", 75), ("hold", 30)):
                start = command_vector(env).clone()
                goal.copy_(start)
                if phase in ("lower", "lift"):
                    positions = env.qpos.cpu().numpy()
                    for index in np.flatnonzero(active):
                        env.model.geom_pos[env.drum_id] = drums[index]
                        env.sim.data.qpos[:] = positions[index]
                        mujoco.mj_forward(env.model, env.sim.data)
                        target = env.cpu.stick_position()
                        target[2] = env.sim.table_height + (0.012 if phase == "lower" else 0.15)
                        solution = env.sim._solve("right", target, env.cpu.tool_orientation)
                        if solution is None:
                            active[index] = False
                            report["failures"].append({"attempt": report["attempted"] + int(index), "reason": f"{phase}_ik"})
                            continue
                        goal[index, :7] = torch.as_tensor(solution[env.sim.qpos_ids["right"]], device="cuda", dtype=torch.float32)
                for step in range(duration):
                    if not active.any():
                        break
                    fraction = (step + 1) / duration
                    blend = 10 * fraction**3 - 15 * fraction**4 + 6 * fraction**5
                    command = start + blend * (goal - start)
                    command[:, 7] = 0.044 if phase == "lower" else 0.044 * (1 - blend) if phase == "close" else 0
                    action = command_action(env, command)
                    applied = command_vector(env).clone()
                    applied[:, :7] += 0.025 * action[:, :7]
                    applied[:, 7] = (action[:, 7] + 1) * 0.022
                    commands = applied.cpu().numpy()
                    qpos, qvel = env.qpos.cpu().numpy(), env.qvel.cpu().numpy()
                    for index in np.flatnonzero(active):
                        records[index]["observation"].append(observation[index].copy())
                        records[index]["command"].append(commands[index].copy())
                        records[index]["qpos"].append(qpos[index].copy())
                        records[index]["qvel"].append(qvel[index].copy())
                    observation, _, done, infos = env.step(action.cpu().numpy())
                    for index in np.flatnonzero(done & active):
                        info = infos[index]
                        if info["is_success"]:
                            episode = group.create_group(f"{report['successful']:06d}")
                            for name, values in records[index].items():
                                episode.create_dataset(name, data=np.array(values, dtype=np.float32), compression="gzip")
                            episode.create_dataset("initial_qpos", data=initial[index])
                            episode.create_dataset("drum_position", data=drums[index])
                            episode.attrs.update(success=True, attempt=report["attempted"] + int(index),
                                                 max_lift_m=info["max_lift_m"], steps=len(records[index]["command"]))
                            report["successful"] += 1
                        else:
                            report["failures"].append({"attempt": report["attempted"] + int(index), "reason": info})
                        active[index] = False
            for index in np.flatnonzero(active):
                report["failures"].append({"attempt": report["attempted"] + int(index), "reason": "reference_incomplete"})
            report["attempted"] += count
            file.flush()
            print(f"Reference collection: {report['attempted']}/{episodes}, successes={report['successful']}", flush=True)
        file.attrs["successful"] = report["successful"]
    env.close()
    if report["successful"] == 0:
        raise RuntimeError("Reference controller produced no successful demonstration")
    output.with_suffix(".json").write_text(json.dumps(report, default=lambda value: value.tolist(), indent=2) + "\n")
    return report


class EpisodeDataset:
    def __init__(self, path: Path, chunk: int, student: bool = False):
        self.path, self.chunk, self.student = path, chunk, student
        self.episodes = []
        with h5py.File(path) as file:
            if file.attrs["schema"] != SCHEMA:
                raise ValueError("Unsupported ACT dataset")
            for name, episode in file["episodes"].items():
                if not episode.attrs["success"] and file.attrs.get("supervision", "executed") != "teacher_query":
                    raise ValueError("Imitation dataset contains an unsuccessful episode")
                observation = episode["observation"][:]
                command = episode["command"][:]
                if observation.shape != (len(command), 57) or command.shape[1:] != (8,):
                    raise ValueError("Invalid ACT observation or command dimensions")
                if not np.isfinite(observation).all() or not np.isfinite(command).all():
                    raise ValueError("Non-finite ACT demonstration")
                if np.any(command[:, 7] < 0) or np.any(command[:, 7] > 0.044001):
                    raise ValueError("Invalid gripper command units")
                state = proprioception(observation) if student else observation
                self.episodes.append((name, state, command))
        if not self.episodes:
            raise ValueError("ACT dataset is empty")
        self.ends = np.cumsum([len(item[1]) for item in self.episodes])
        self.state = np.concatenate([item[1] for item in self.episodes])
        self.command = np.concatenate([item[2] for item in self.episodes])

    def __len__(self):
        return int(self.ends[-1])

    def batch(self, indices: np.ndarray, device="cuda") -> tuple:
        episode_ids = np.searchsorted(self.ends, indices, side="right")
        positions = indices[:, None] + np.arange(self.chunk)
        masks = positions >= self.ends[episode_ids, None]
        positions = np.minimum(positions, self.ends[episode_ids, None] - 1)
        images = []
        if self.student:
            with h5py.File(self.path) as file:
                for index, episode_id in zip(indices, episode_ids):
                    offset = int(index - (self.ends[episode_id - 1] if episode_id else 0))
                    images.append(file[f"episodes/{self.episodes[episode_id][0]}/image"][offset])
        return (torch.as_tensor(self.state[indices], dtype=torch.float32, device=device),
                torch.as_tensor(self.command[positions], dtype=torch.float32, device=device),
                torch.as_tensor(masks, dtype=torch.bool, device=device),
                torch.as_tensor(np.array(images), device=device).permute(0, 3, 1, 2).float() / 255 if self.student else None)


def main():
    parser = argparse.ArgumentParser(description="Collect physical OpenArm reference-controller demonstrations")
    parser.add_argument("--output", type=Path, default=RUNS / "act/data")
    parser.add_argument("--train-episodes", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--worlds", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    for split, count, seed in (("train", args.train_episodes, 0), ("eval", args.eval_episodes, 100000)):
        bank = args.output / f"{split}_resets.npz"
        build_reset_bank(bank, count, seed)
        collect_reference(bank, args.output / f"{split}.h5", count, min(args.worlds, count), seed)


if __name__ == "__main__":
    main()
