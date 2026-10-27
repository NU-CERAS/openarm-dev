import hashlib
from pathlib import Path
import uuid

import h5py
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from .act_data import SCHEMA, command_action, limited_command
from .act_model import ACTController, ActionChunkTransformer
from .act_residual import ACTResidualVecEnv, residual_configuration
from .collection_io import collection_start, save_progress
from .compute_guard import check_resources
from .warp_pickup import WarpPickupVecEnv


def collect_batched_teacher(teacher: Path, residual: Path | None, bank: Path, output: Path,
                            episodes: int, seed: int, width: int, height: int, resume: bool, worlds: int):
    completed, successes = collection_start(teacher, residual, bank, output, episodes, seed, None, width, height, resume)
    if completed == episodes:
        return {"attempts": completed, "successes": successes}
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = WarpPickupVecEnv(min(worlds, episodes - completed), bank, seed)
    model = ActionChunkTransformer.load(teacher)
    controller = ACTController(model, raw.num_envs, query_interval=1)
    if residual is not None:
        if hashlib.sha256((residual / "act.pt").read_bytes()).hexdigest() != hashlib.sha256(teacher.read_bytes()).hexdigest():
            raise ValueError("Residual and ACT checkpoints differ")
        combined = ACTResidualVecEnv(raw, teacher, **residual_configuration(residual))
        environment = VecNormalize.load(str(residual / "vecnormalize.pkl"), combined)
        environment.training, environment.norm_reward = False, False
        policy = PPO.load(residual / "policy.zip", env=environment, device="cuda")
    else:
        environment = raw
    results = []
    try:
        with h5py.File(output, "a" if resume else "w-") as file, \
                mujoco.Renderer(raw.model, height=height, width=width) as renderer:
            file.attrs.update(schema=SCHEMA, source="ACT+PPO" if residual else "ACT",
                              teacher_sha256=hashlib.sha256(teacher.read_bytes()).hexdigest(),
                              bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest(), bank_seed=seed,
                              control_hz=50, command_units="rad,rad,rad,rad,rad,rad,rad,m",
                              camera="overhead", image_resolution=[width, height], supervision="executed",
                              collection_worlds=raw.num_envs)
            if residual is not None:
                file.attrs["residual_sha256"] = hashlib.sha256((residual / "policy.zip").read_bytes()).hexdigest()
            group = file.require_group("episodes")
            pending = file.require_group("pending")
            for offset in range(completed, episodes, raw.num_envs):
                check_resources()
                count = min(raw.num_envs, episodes - offset)
                environment.seed(seed + offset)
                observation = environment.reset()
                controller.reset(slice(None))
                initial = raw.qpos.cpu().numpy().copy()
                drums = raw.geom_positions[:, raw.drum_id].cpu().numpy().copy()
                active = np.arange(raw.num_envs) < count
                records = [{key: [] for key in ("observation", "command", "executed_command", "qpos", "qvel")}
                           for _ in range(count)]
                outcomes = {}
                for _ in range(300):
                    raw_observation = raw._observation().cpu().numpy()
                    if residual is None:
                        command = controller.command(raw_observation)
                        action = command_action(raw, command).cpu().numpy()
                        executed = limited_command(raw, command)
                    else:
                        action, _ = policy.predict(observation, deterministic=True)
                        executed = combined.compose(action)
                    commands = executed.cpu().numpy()
                    positions, velocities = raw.qpos.cpu().numpy(), raw.qvel.cpu().numpy()
                    for index in np.flatnonzero(active):
                        for key, value in (("observation", raw_observation), ("command", commands),
                                           ("executed_command", commands), ("qpos", positions), ("qvel", velocities)):
                            records[index][key].append(value[index].copy())
                    observation, _, done, infos = environment.step(action)
                    for index in np.flatnonzero(done & active):
                        outcomes[int(index)] = {key: value for key, value in infos[index].items()
                                                if key not in ("terminal_observation", "terminal_qpos", "terminal_command")}
                        active[index] = False
                    if not active.any():
                        break
                if len(outcomes) != count:
                    raise RuntimeError("Batched teacher collection did not finish every active episode")
                for index in range(count):
                    check_resources()
                    result = outcomes[index]
                    results.append(result)
                    attempt = offset + index
                    if result["is_success"]:
                        name = f"{attempt:06d}"
                        if name in pending:
                            file.require_group("interrupted")
                            file.move(pending[name].name, f"interrupted/{name}_{uuid.uuid4().hex[:8]}")
                        episode = pending.create_group(name)
                        for key, values in records[index].items():
                            episode.create_dataset(key, data=np.asarray(values), compression="gzip")
                        episode.create_dataset("initial_qpos", data=initial[index])
                        episode.create_dataset("drum_position", data=drums[index])
                        steps = len(records[index]["command"])
                        images = episode.create_dataset("image", (steps, height, width, 3), dtype=np.uint8,
                                                        compression="gzip", chunks=(1, height, width, 3))
                        raw.model.geom_pos[raw.drum_id] = drums[index]
                        for step in range(steps):
                            raw.sim.data.qpos[:] = records[index]["qpos"][step]
                            raw.sim.data.qvel[:] = records[index]["qvel"][step]
                            raw.sim.data.time = step * 0.02
                            mujoco.mj_forward(raw.model, raw.sim.data)
                            renderer.update_scene(raw.sim.data, camera="overhead")
                            images[step] = renderer.render()
                        episode.attrs.update(success=True, attempt=attempt, steps=steps,
                                             reset_index=result["reset_index"], sampling_seed=seed + offset)
                        file.flush()
                        file.move(episode.name, f"episodes/{name}")
                        successes += 1
                    save_progress(file, output, attempt + 1, successes, completed, results)
                    print(f"Teacher collection {attempt + 1}/{episodes}, successes={successes}", flush=True)
    finally:
        environment.close()
    return {"attempts": episodes, "successes": successes}
