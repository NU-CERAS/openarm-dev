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

from .act_data import SCHEMA, command_action, limited_command, proprioception, visual_state
from .act_model import ACTController, ActionChunkTransformer
from .act_residual import ACTResidualVecEnv, residual_configuration
from .warp_pickup import WarpPickupVecEnv
from .compute_guard import budgeted, check_resources
from .collection_io import collection_start, save_progress


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
        target.attrs["successful"] = len(episodes)
        target.attrs["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    bank_source = source.with_name(source.stem + "_resets.npz")
    bank_target = output.with_name(output.stem + "_resets.npz")
    if bank_source.resolve() != bank_target.resolve():
        shutil.copyfile(bank_source, bank_target)


def merge_datasets(sources: list[Path], output: Path):
    if not sources or any(source.resolve() == output.resolve() for source in sources):
        raise ValueError("Dataset aggregation requires distinct input and output paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    bank_hash = None
    with h5py.File(output, "w") as target:
        episodes = target.create_group("episodes")
        for source in sources:
            with source.open("rb") as data:
                source_hash = hashlib.file_digest(data, "sha256").hexdigest()
            with h5py.File(source) as original:
                if original.attrs["schema"] != SCHEMA:
                    raise ValueError("Unsupported teacher-query dataset")
                current_hash = original.attrs["bank_sha256"]
                if bank_hash is not None and current_hash != bank_hash:
                    raise ValueError("Do not merge datasets from different reset splits")
                if bank_hash is not None and not np.array_equal(original.attrs.get("image_resolution", ()), target.attrs.get("image_resolution", ())):
                    raise ValueError("Dataset image resolutions differ")
                if bank_hash is None:
                    bank_hash = current_hash
                    for name, value in original.attrs.items():
                        target.attrs[name] = value
                for item in original["episodes"].values():
                    name = f"{len(episodes):06d}"
                    original.copy(item, episodes, name=name)
                    episodes[name].attrs["source_sha256"] = source_hash
                    episodes[name].attrs["supervision"] = original.attrs.get("supervision", "executed")
        target.attrs["supervision"] = "teacher_query"
        target.attrs["successful"] = sum(int(item.attrs["success"]) for item in episodes.values())
        target.attrs["source"] = "aggregated_teacher_supervision"


def collect_teacher(teacher: Path, residual: Path | None, bank: Path, output: Path, episodes: int, seed=200000, student: Path | None = None,
                    width=320, height=180, resume=False):
    output.parent.mkdir(parents=True, exist_ok=True)
    completed, previous_successes = collection_start(teacher, residual, bank, output, episodes, seed,
                                                     student, width, height, resume)
    if completed >= episodes:
        if completed != episodes:
            raise ValueError("Saved collection already exceeds the requested attempt count")
        return {"attempts": completed, "successes": previous_successes}
    raw = WarpPickupVecEnv(1, bank, seed)
    model = ActionChunkTransformer.load(teacher)
    model.requires_grad_(False)
    controller = ACTController(model, 1, query_interval=1)
    if student is not None:
        student_model = ActionChunkTransformer.load(student)
        if not student_model.config.vision or student_model.config.state_dim not in (9, 24):
            raise ValueError("Student-state aggregation requires a camera/proprioception ACT")
        if (width, height) != (student_model.config.image_width, student_model.config.image_height):
            raise ValueError("Teacher-query image resolution differs from the student configuration")
        student_controller = ACTController(student_model, 1, query_interval=1)
    if residual is not None:
        if hashlib.sha256((residual / "act.pt").read_bytes()).hexdigest() != hashlib.sha256(teacher.read_bytes()).hexdigest():
            raise ValueError("PPO checkpoint and ACT teacher do not match")
        combined = ACTResidualVecEnv(raw, teacher, **residual_configuration(residual))
        environment = VecNormalize.load(str(residual / "vecnormalize.pkl"), combined)
        environment.training, environment.norm_reward = False, False
        policy = PPO.load(residual / "policy.zip", env=environment, device="cuda")
    else:
        environment = raw
    successes = previous_successes
    reports = []
    with h5py.File(output, "a" if resume else "w-") as file, mujoco.Renderer(raw.model, height=height, width=width) as renderer:
        file.attrs.update(schema=SCHEMA, source="ACT+PPO" if residual is not None else "ACT",
                          teacher_sha256=hashlib.sha256(teacher.read_bytes()).hexdigest(),
                          bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest(),
                          bank_seed=seed, control_hz=50, command_units="rad,rad,rad,rad,rad,rad,rad,m",
                          camera="overhead", image_resolution=[width, height])
        file.attrs["supervision"] = "teacher_query" if student is not None else "executed"
        if residual is not None:
            file.attrs["residual_sha256"] = hashlib.sha256((residual / "policy.zip").read_bytes()).hexdigest()
        if student is not None:
            file.attrs["student_sha256"] = hashlib.sha256(student.read_bytes()).hexdigest()
        group = file["episodes"] if "episodes" in file else file.create_group("episodes")
        for episode_index in range(completed, episodes):
            check_resources()
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
                data = raw.render_snapshot(0)
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
                    student_command = student_controller.command(visual_state(raw_observation, step * 0.02, student_model.config.state_dim), images)
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
            save_progress(file, output, episode_index + 1, successes, completed, reports)
    environment.close()
    report = {"attempts": episodes, "successes": successes, "previous_completed_attempts": completed, "results": reports}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if successes == 0 and student is None:
        raise RuntimeError("Teacher produced no successful trajectory for student distillation")
    return report


def evaluate_student(model, bank: Path, output: Path, episodes=16, markers="visible", worlds=1):
    import imageio.v2 as imageio

    if not model.config.vision or model.config.state_dim not in (9, 24) or model.config.physical_oracle:
        raise ValueError("Student must use camera and robot-only state")
    if markers not in ("visible", "no_tcp", "hidden") or not 1 <= worlds <= episodes:
        raise ValueError("Invalid marker visibility or evaluation world count")
    output.mkdir(parents=True, exist_ok=True)
    with np.load(bank) as source:
        if not 0 < episodes <= len(source["qpos"]):
            raise ValueError("Student evaluation needs distinct reset states")
    raw = WarpPickupVecEnv(worlds, bank, seed=300000, sequential_resets=True)
    controller = ACTController(model, worlds, query_interval=1)
    options = mujoco.MjvOption()
    if markers == "no_tcp":
        for name in ("right_tcp", "left_tcp"):
            raw.model.site_rgba[raw.model.site(name).id, 3] = 0
    elif markers == "hidden":
        options.sitegroup[:] = 0
    results = []
    with mujoco.Renderer(raw.model, height=model.config.image_height, width=model.config.image_width) as renderer, imageio.get_writer(
        output / "execution.mp4", fps=25, codec="libx264", macro_block_size=1, output_params=["-movflags", "+faststart"],
    ) as writer:
        for episode in range(0, episodes, worlds):
            offset = min(episode, episodes - worlds)
            raw.set_options({"bank_offset": offset})
            observation = raw.reset()
            controller.reset(slice(None))
            completed = np.zeros(worlds, dtype=bool)
            elapsed = np.zeros(worlds)
            batch_results = {}
            for step in range(300):
                check_resources()
                frames = []
                for world in range(worlds):
                    renderer.update_scene(raw.render_snapshot(world), camera="overhead", scene_option=options)
                    frames.append(renderer.render().copy())
                if episode == 0 and step % 2 == 0:
                    if not completed[0]:
                        writer.append_data(frames[0])
                images = torch.as_tensor(np.stack(frames), device="cuda").permute(0, 3, 1, 2).float() / 255
                command = controller.command(visual_state(observation, elapsed, model.config.state_dim), images)
                observation, _, done, infos = raw.step(command_action(raw, command).cpu().numpy())
                elapsed += 0.02
                for world in np.flatnonzero(done & ~completed):
                    result = {key: value for key, value in infos[world].items() if key not in ("terminal_observation", "terminal_qpos")}
                    if result["reset_index"] != offset + world:
                        raise RuntimeError("Evaluation reset index differs from the requested state")
                    batch_results[offset + world] = result
                    completed[world] = True
                elapsed[done] = 0
                controller.reset(np.flatnonzero(done))
                if completed.all():
                    break
            results.extend(batch_results[index] for index in range(episode, min(episode + worlds, episodes)))
            print(f"Student evaluation {markers}: {len(results)}/{episodes}", flush=True)
    raw.close()
    if len(results) != episodes:
        raise RuntimeError("Student evaluation did not finish every requested episode")
    report = {"episodes": episodes, "markers": markers, "worlds": worlds,
              "success_rate": float(np.mean([item["is_success"] for item in results])),
              "mean_return": float(np.mean([item["episode"]["r"] for item in results])), "results": results}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


@budgeted("openarm_sim.act_distill")
def main():
    parser = argparse.ArgumentParser(description="Collect camera-labelled ACT/residual-teacher rollouts or render reference data")
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--residual", type=Path)
    parser.add_argument("--student", type=Path)
    parser.add_argument("--evaluate-student", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worlds", type=int)
    parser.add_argument("--markers", choices=["visible", "no_tcp", "hidden"], default="visible")
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--merge", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=200000)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=180)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.evaluate_student is not None:
        if args.bank is None:
            raise ValueError("Student evaluation requires a reset bank")
        evaluate_student(ActionChunkTransformer.load(args.evaluate_student), args.bank, args.output, args.episodes,
                         args.markers, args.worlds or 1)
    elif args.merge is not None:
        merge_datasets(args.merge, args.output)
    elif args.source is not None:
        render_reference_images(args.source, args.output, args.episodes)
    else:
        if args.teacher is None or args.bank is None:
            raise ValueError("Teacher collection requires --teacher and --bank")
        worlds = args.worlds if args.worlds is not None else (1 if args.student else 32)
        if worlds < 1 or (args.student is not None and worlds != 1):
            raise ValueError("Student-query collection uses one world; teacher world count must be positive")
        if worlds == 1:
            collect_teacher(args.teacher, args.residual, args.bank, args.output, args.episodes, args.seed, args.student,
                            args.image_width, args.image_height, args.resume)
        else:
            from .teacher_collection import collect_batched_teacher

            collect_batched_teacher(args.teacher, args.residual, args.bank, args.output, args.episodes, args.seed,
                                    args.image_width, args.image_height, args.resume, worlds)


if __name__ == "__main__":
    main()
