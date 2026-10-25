import argparse
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from .act_data import SCHEMA, command_action, limited_command, proprioception
from .act_model import ACTController, ActionChunkTransformer
from .act_residual import ACTResidualVecEnv
from .warp_pickup import WarpPickupVecEnv


def render_reference_images(source: Path, output: Path, limit: int, width=192, height=108):
    from .pickup import StickPickupEnv

    output.parent.mkdir(parents=True, exist_ok=True)
    with StickPickupEnv() as env, h5py.File(source) as original, h5py.File(output, "w") as target:
        for name, value in original.attrs.items():
            target.attrs[name] = value
        target.attrs["camera"] = "overhead"
        target.attrs["image_resolution"] = [width, height]
        episodes = target.create_group("episodes")
        with mujoco.Renderer(env.sim.model, height=height, width=width) as renderer:
            for index, (name, item) in enumerate(original["episodes"].items()):
                if index >= limit:
                    break
                original.copy(item, episodes, name=name)
                episode = episodes[name]
                images = episode.create_dataset("image", (len(item["qpos"]), height, width, 3), dtype=np.uint8,
                                                compression="gzip", chunks=(1, height, width, 3))
                env.sim.model.geom_pos[env.sim.model.geom("drumhead").id] = item["drum_position"][:]
                for step in range(len(images)):
                    env.sim.data.qpos[:] = item["qpos"][step]
                    env.sim.data.qvel[:] = item["qvel"][step]
                    mujoco.mj_forward(env.sim.model, env.sim.data)
                    renderer.update_scene(env.sim.data, camera="overhead")
                    images[step] = renderer.render()
                target.flush()
                print(f"Rendered reference episode {index + 1}/{limit}", flush=True)
    bank_source = source.with_name(source.stem + "_resets.npz")
    bank_target = output.with_name(output.stem + "_resets.npz")
    if bank_source.resolve() != bank_target.resolve():
        shutil.copyfile(bank_source, bank_target)


def collect_teacher(teacher: Path, residual: Path | None, bank: Path, output: Path, episodes: int, seed=200000, student: Path | None = None):
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = WarpPickupVecEnv(1, bank, seed)
    model = ActionChunkTransformer.load(teacher)
    model.requires_grad_(False)
    controller = ACTController(model, 1, query_interval=1)
    if student is not None:
        student_model = ActionChunkTransformer.load(student)
        if not student_model.config.vision or student_model.config.state_dim != 24:
            raise ValueError("Student-state aggregation requires a camera/proprioception ACT")
        student_controller = ACTController(student_model, 1, query_interval=1)
    if residual is not None:
        if hashlib.sha256((residual / "act.pt").read_bytes()).hexdigest() != hashlib.sha256(teacher.read_bytes()).hexdigest():
            raise ValueError("PPO checkpoint and ACT teacher do not match")
        combined = ACTResidualVecEnv(raw, teacher, query_interval=1)
        environment = VecNormalize.load(str(residual / "vecnormalize.pkl"), combined)
        environment.training, environment.norm_reward = False, False
        policy = PPO.load(residual / "policy.zip", env=environment, device="cuda")
    else:
        environment = raw
    successes = 0
    reports = []
    with h5py.File(output, "w") as file, mujoco.Renderer(raw.model, height=108, width=192) as renderer:
        file.attrs.update(schema=SCHEMA, source="ACT+PPO" if residual is not None else "ACT",
                          teacher_sha256=hashlib.sha256(teacher.read_bytes()).hexdigest(),
                          bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest(),
                          bank_seed=seed, control_hz=50, command_units="rad,rad,rad,rad,rad,rad,rad,m",
                          camera="overhead", image_resolution=[192, 108])
        file.attrs["supervision"] = "teacher_query" if student is not None else "executed"
        if residual is not None:
            file.attrs["residual_sha256"] = hashlib.sha256((residual / "policy.zip").read_bytes()).hexdigest()
        if student is not None:
            file.attrs["student_sha256"] = hashlib.sha256(student.read_bytes()).hexdigest()
        group = file.create_group("episodes")
        for episode_index in range(episodes):
            environment.seed(seed + episode_index)
            observation = environment.reset()
            controller.reset(slice(None))
            if student is not None:
                student_controller.reset(slice(None))
            initial = raw.qpos[0].cpu().numpy().copy()
            drum = raw.geom_positions[0, raw.drum_id].cpu().numpy().copy()
            records = {name: [] for name in ("observation", "command", "executed_command", "qpos", "qvel", "image")}
            for step in range(300):
                raw_observation = raw._observation().cpu().numpy()
                data = raw.snapshot(0)
                renderer.update_scene(data, camera="overhead")
                image = renderer.render().copy()
                if residual is None:
                    command = controller.command(raw_observation)
                    action = command_action(raw, command).cpu().numpy()
                    executed = limited_command(raw, command)
                else:
                    action, _ = policy.predict(observation, deterministic=True)
                    executed = combined.compose(action)
                records["observation"].append(raw_observation[0])
                records["command"].append(executed[0].cpu().numpy().copy())
                records["qpos"].append(data.qpos.copy())
                records["qvel"].append(data.qvel.copy())
                records["image"].append(image)
                if student is not None:
                    images = torch.as_tensor(image[None], device="cuda").permute(0, 3, 1, 2).float() / 255
                    student_command = student_controller.command(proprioception(raw_observation), images)
                    student_command = limited_command(raw, student_command)
                    if residual is None:
                        student_action = command_action(raw, student_command)
                        observation, _, done, info = raw.step(student_action.cpu().numpy())
                    else:
                        observation, _, done, info = combined.step_command(student_command)
                        observation = environment.normalize_obs(observation)
                    records["executed_command"].append(student_command[0].cpu().numpy().copy())
                else:
                    records["executed_command"].append(executed[0].cpu().numpy().copy())
                    observation, _, done, info = environment.step(action)
                if done[0]:
                    break
            result = {key: value for key, value in info[0].items() if key not in ("terminal_observation", "terminal_qpos", "terminal_command")}
            reports.append(result)
            if result["is_success"] or student is not None:
                episode = group.create_group(f"{episode_index:06d}")
                episode.attrs.update(success=result["is_success"], attempt=episode_index, steps=len(records["command"]))
                for name, values in records.items():
                    episode.create_dataset(name, data=np.array(values), compression="gzip")
                episode.create_dataset("initial_qpos", data=initial)
                episode.create_dataset("drum_position", data=drum)
                successes += int(result["is_success"])
                file.flush()
            print(f"Teacher collection {episode_index + 1}/{episodes}, successes={successes}", flush=True)
    environment.close()
    report = {"attempts": episodes, "successes": successes, "results": reports}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if successes == 0 and student is None:
        raise RuntimeError("Teacher produced no successful trajectory for student distillation")
    return report


def evaluate_student(model, bank: Path, output: Path, episodes=16):
    import imageio.v2 as imageio

    if not model.config.vision or model.config.state_dim != 24:
        raise ValueError("Student must use camera and 24-dimensional robot-only state")
    output.mkdir(parents=True, exist_ok=True)
    raw = WarpPickupVecEnv(1, bank, seed=300000)
    controller = ACTController(model, 1, query_interval=1)
    results = []
    with mujoco.Renderer(raw.model, height=108, width=192) as renderer, imageio.get_writer(
        output / "execution.mp4", fps=25, codec="libx264", macro_block_size=1, output_params=["-movflags", "+faststart"],
    ) as writer:
        for episode in range(episodes):
            raw.seed(300000 + episode)
            observation = raw.reset()
            controller.reset(slice(None))
            for step in range(300):
                renderer.update_scene(raw.snapshot(0), camera="overhead")
                image = renderer.render().copy()
                if episode == 0 and step % 2 == 0:
                    writer.append_data(image)
                images = torch.as_tensor(image[None], device="cuda").permute(0, 3, 1, 2).float() / 255
                command = controller.command(proprioception(observation), images)
                observation, _, done, infos = raw.step(command_action(raw, command).cpu().numpy())
                if done[0]:
                    results.append({key: value for key, value in infos[0].items() if key not in ("terminal_observation", "terminal_qpos")})
                    break
    raw.close()
    report = {"episodes": episodes, "success_rate": float(np.mean([item["is_success"] for item in results])),
              "mean_return": float(np.mean([item["episode"]["r"] for item in results])), "results": results}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Collect camera-labelled ACT/residual-teacher rollouts or render reference data")
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--residual", type=Path)
    parser.add_argument("--student", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=200000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.source is not None:
        render_reference_images(args.source, args.output, args.episodes)
    else:
        if args.teacher is None or args.bank is None:
            raise ValueError("Teacher collection requires --teacher and --bank")
        collect_teacher(args.teacher, args.residual, args.bank, args.output, args.episodes, args.seed, args.student)


if __name__ == "__main__":
    main()
