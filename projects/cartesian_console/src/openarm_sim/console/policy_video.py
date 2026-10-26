import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ..act_data import command_action, visual_state
from ..act_model import ACTController, ActionChunkTransformer
from ..compute_guard import budgeted, check_resources
from ..residual_drumming import ResidualDrumStrikeEnv
from ..warp_pickup import WarpPickupVecEnv


class PolicyVideo:
    fps = 25

    def __init__(self, path: Path, model: mujoco.MjModel, task: str = "pickup"):
        self.task = task
        self.writer = imageio.get_writer(path, fps=self.fps, codec="libx264", quality=8,
                                        macro_block_size=1, output_params=["-movflags", "+faststart"])
        self.renderer = mujoco.Renderer(model, height=720, width=1280)
        self.options = mujoco.MjvOption()
        self.options.sitegroup[:] = 0
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.28, 0, 0.48]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.5, 145, -22
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        self.title_font = ImageFont.truetype(font, 40)
        self.font = ImageFont.truetype(font, 25)
        self.small = ImageFont.truetype(font, 21)
        self.frames = 0

    def capture(self, data: mujoco.MjData, rgb: np.ndarray, episode: int, total: int,
                elapsed: float, metrics: list[str], repeat: int = 1) -> None:
        self.renderer.update_scene(data, camera=self.camera, scene_option=self.options)
        overview = self.renderer.render().copy()
        if rgb.std() < 2 or overview.std() < 2:
            raise RuntimeError("Policy recording contains an empty rendered image")
        canvas = Image.new("RGB", (1920, 1080), "#0b1220")
        canvas.paste(Image.fromarray(overview), (32, 150))
        canvas.paste(Image.fromarray(rgb).resize((544, 306), Image.Resampling.NEAREST), (1344, 150))
        draw = ImageDraw.Draw(canvas)
        pickup = self.task == "pickup"
        heading = "VISION-CONDITIONED PICKUP" if pickup else "ORACLE STRIKE"
        actor = "Visual ACT student" if pickup else "Oracle residual PPO"
        image_label = f"POLICY RGB INPUT  {rgb.shape[1]} x {rgb.shape[0]}" if pickup else "OBSERVER CAMERA"
        inputs = ("RGB + robot joint positions", "Gripper position + clock") if pickup else (
            "51-dimensional oracle state", "Reference + PPO residuals")
        draw.text((32, 22), f"OPENARM  /  {heading}", font=self.small, fill="#56dbd0")
        draw.text((32, 57), f"{actor}  |  Trial {episode + 1}/{total}", font=self.title_font, fill="#f3f6fc")
        draw.text((1344, 110), image_label, font=self.small, fill="#b5c3d6")
        for y, line in zip((490, 532, 570, 608), ("POLICY INPUTS", *inputs, f"Control: 50 Hz | t = {elapsed:.2f} s")):
            draw.text((1344, y), line, font=self.font, fill="#e2eaf5")
        draw.text((1344, 680), "TASK EVALUATION", font=self.small, fill="#56dbd0")
        for index, line in enumerate(metrics):
            draw.text((1344, 721 + index * 34), line, font=self.font, fill="#f3f6fc")
        draw.rounded_rectangle((32, 902, 1888, 1008), radius=14, fill="#142238")
        details = ("RGB-conditioned ACT student distilled from an ACT + PPO teacher",
                   "Physical grasp and lift in MuJoCo Warp; hidden TCP and target markers") if pickup else (
            "Trained PPO residuals correct a nominal joint trajectory; oracle object/contact features",
            "Strike trials start from a physically established grasp and prepared arm position")
        draw.text((55, 918), details[0], font=self.font, fill="#e2eaf5")
        draw.text((55, 953), details[1], font=self.font, fill="#e2eaf5")
        footer = "Consecutive held-out resets  |  Task metrics are evaluator-only" if pickup else "Consecutive evaluation seeds  |  RGB is an observer view"
        draw.text((32, 1030), f"{footer}  |  Simulation execution", font=self.small, fill="#98adc5")
        frame = np.asarray(canvas)
        for _ in range(repeat):
            self.writer.append_data(frame)
            self.frames += 1

    def close(self) -> None:
        self.writer.close()
        self.renderer.close()


def record_pickup_policy(checkpoint: Path, bank: Path, output: Path, episodes: int = 3) -> dict:
    if episodes < 1:
        raise ValueError("The recording needs at least one consecutive held-out reset")
    model = ActionChunkTransformer.load(checkpoint)
    if not model.config.vision or model.config.physical_oracle or model.config.state_dim != 9:
        raise ValueError("Pickup recording requires RGB, eight robot positions and a controller clock")
    with np.load(bank) as resets:
        if episodes > len(resets["qpos"]):
            raise ValueError("The reset bank has fewer states than the requested episodes")
    output.mkdir(parents=True, exist_ok=False)
    raw = WarpPickupVecEnv(1, bank, seed=300000, sequential_resets=True)
    controller = ACTController(model, 1, query_interval=1)
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    results, trace = [], []
    video = None
    try:
        video = PolicyVideo(output / "pickup_policy.mp4", raw.model)
        with mujoco.Renderer(raw.model, height=model.config.image_height, width=model.config.image_width) as renderer:
            for episode in range(episodes):
                raw.set_options({"bank_offset": episode})
                observation = raw.reset()
                controller.reset(slice(None))
                drum_position = raw.geom_positions[0, raw.drum_id].cpu().numpy().copy()
                for step in range(300):
                    check_resources()
                    data = raw.render_snapshot(0)
                    renderer.update_scene(data, camera="overhead", scene_option=options)
                    rgb = renderer.render().copy()
                    state = visual_state(observation, step * 0.02, model.config.state_dim)
                    images = torch.as_tensor(rgb[None], device="cuda").permute(0, 3, 1, 2).float() / 255
                    command = controller.command(state, images)
                    action = command_action(raw, command).cpu().numpy()
                    if step % 2 == 0:
                        video.capture(data, rgb, episode, episodes, step * 0.02, ["Policy: executing"],
                                      repeat=25 if step == 0 else 1)
                    trace.append({"episode": episode, "reset_index": episode, "step": step,
                                  "state": state[0].tolist(), "command": command[0].cpu().tolist(),
                                  "action": action[0].tolist(), "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
                    observation, _, done, infos = raw.step(action)
                    if done[0]:
                        info = infos[0]
                        if info["reset_index"] != episode:
                            raise RuntimeError("The recorded reset differs from its declared held-out index")
                        result = {key: value for key, value in info.items()
                                  if key not in ("terminal_observation", "terminal_qpos")}
                        results.append(result)
                        data.qpos[:] = info["terminal_qpos"]
                        raw.model.geom_pos[raw.drum_id] = drum_position
                        mujoco.mj_forward(raw.model, data)
                        renderer.update_scene(data, camera="overhead", scene_option=options)
                        terminal_rgb = renderer.render().copy()
                        video.capture(data, terminal_rgb, episode, episodes, (step + 1) * 0.02,
                                      [f"Success: {info['is_success']}", f"Max lift: {info['max_lift_m'] * 100:.1f} cm",
                                       f"Unsafe contact: {info['unsafe_contact']}"], repeat=50)
                        print(json.dumps({"episode": episode, **result}), flush=True)
                        break
                else:
                    raise RuntimeError("The visual pickup policy did not finish its episode")
    finally:
        if video is not None:
            video.close()
        raw.close()
    report = {"task": "pickup", "actor": "visual_ACT_distilled_from_ACT_PPO_teacher", "config": asdict(model.config),
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "bank_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(), "reset_indices": list(range(episodes)),
              "markers": "hidden", "success_rate": float(np.mean([r["is_success"] for r in results])),
              "results": results, "frames": video.frames, "fps": video.fps,
              "training_started": False, "hardware_executed": False}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "policy_trace.jsonl").write_text("".join(json.dumps(item) + "\n" for item in trace))
    return report


def record_strike_policy(checkpoint: Path, output: Path, episodes: int = 3, seed: int = 810000) -> dict:
    if episodes < 1:
        raise ValueError("The recording needs at least one consecutive evaluation seed")
    policy = PPO.load(checkpoint / "policy.zip", device="cuda")
    if policy.num_timesteps <= 0 or policy.observation_space.shape != (51,):
        raise ValueError("Strike recording requires a trained 51-dimensional oracle PPO checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    normalization_env = DummyVecEnv([ResidualDrumStrikeEnv])
    normalizer = VecNormalize.load(str(checkpoint / "vecnormalize.pkl"), normalization_env)
    normalizer.training, normalizer.norm_reward = False, False
    task = normalization_env.envs[0]
    video = None
    results, trace = [], []
    try:
        video = PolicyVideo(output / "strike_policy.mp4", task.sim.model, task="strike")
        options = mujoco.MjvOption()
        options.sitegroup[:] = 0
        with mujoco.Renderer(task.sim.model, height=306, width=544) as renderer:
            for episode in range(episodes):
                observation, _ = task.reset(seed=seed + episode)
                for step in range(101):
                    check_resources()
                    action, _ = policy.predict(normalizer.normalize_obs(observation[None]), deterministic=True)
                    renderer.update_scene(task.sim.data, camera="overhead", scene_option=options)
                    rgb = renderer.render().copy()
                    if step % 2 == 0:
                        video.capture(task.sim.data, rgb, episode, episodes, task.env.elapsed,
                                      [f"PPO steps: {policy.num_timesteps}", f"Impacts: {len(task.env.hit_events)}"],
                                      repeat=25 if step == 0 else 1)
                    trace.append({"episode": episode, "seed": seed + episode, "step": step,
                                  "oracle_observation": observation.tolist(), "action": action[0].tolist()})
                    observation, _, terminated, truncated, info = task.step(action[0])
                    if terminated or truncated:
                        results.append({"seed": seed + episode, **info})
                        renderer.update_scene(task.sim.data, camera="overhead", scene_option=options)
                        rgb = renderer.render().copy()
                        video.capture(task.sim.data, rgb, episode, episodes, task.env.elapsed,
                                      [f"Success: {info['is_success']}", f"Impacts: {info['hit_count']}",
                                       f"Grasp retained: {info['grasped']}", f"Unsafe: {info['unsafe_contact']}"], repeat=50)
                        break
                else:
                    raise RuntimeError("The oracle strike policy did not finish its episode")
    finally:
        if video is not None:
            video.close()
        normalizer.close()
    report = {"task": "strike", "actor": "oracle_nominal_reference_residual_PPO", "num_timesteps": policy.num_timesteps,
              "checkpoint_sha256": hashlib.sha256((checkpoint / "policy.zip").read_bytes()).hexdigest(),
              "normalization_sha256": hashlib.sha256((checkpoint / "vecnormalize.pkl").read_bytes()).hexdigest(),
              "seeds": list(range(seed, seed + episodes)), "results": results,
              "success_rate": float(np.mean([r["is_success"] for r in results])), "oracle": True,
              "frames": video.frames, "fps": video.fps, "training_started": False, "hardware_executed": False}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "policy_trace.jsonl").write_text("".join(json.dumps(item) + "\n" for item in trace))
    return report


@budgeted("openarm_sim.console.policy_video")
def main() -> None:
    parser = argparse.ArgumentParser(description="Record visual pickup ACT and trained oracle strike PPO inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--console-video", type=Path)
    parser.add_argument("--console-start", type=float, default=0.0)
    parser.add_argument("--console-duration", type=float)
    parser.add_argument("--strike-checkpoint", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.console_start) or args.console_start < 0:
        raise ValueError("Console start time must be finite and nonnegative")
    if args.console_duration is not None and (not np.isfinite(args.console_duration) or args.console_duration <= 0):
        raise ValueError("Console duration must be finite and positive")
    paths = [args.checkpoint, args.bank]
    if args.console_video is not None:
        paths.append(args.console_video)
    if args.strike_checkpoint is not None:
        paths += [args.strike_checkpoint / name for name in ("policy.zip", "vecnormalize.pkl")]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    torch.set_num_threads(1)
    report = record_pickup_policy(args.checkpoint, args.bank, args.output, args.episodes)
    clips = [args.output / "pickup_policy.mp4"]
    reports = {"pickup": report}
    if args.strike_checkpoint is not None:
        reports["strike"] = record_strike_policy(args.strike_checkpoint, args.output / "strike", args.episodes)
        clips.append(args.output / "strike/strike_policy.mp4")
    if args.console_video is not None:
        clips.insert(0, args.console_video)
        reports["console"] = {"source_sha256": hashlib.sha256(args.console_video.read_bytes()).hexdigest(),
                              "start_s": args.console_start, "duration_s": args.console_duration}
    if len(clips) > 1:
        filters = []
        for index in range(len(clips)):
            filters.append(f"[{index}:v]fps=25,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[v{index}]")
        filters.append("".join(f"[v{index}]" for index in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=0[v]")
        with (args.output / "encoding.log").open("w") as log:
            inputs = []
            for index, clip in enumerate(clips):
                if index == 0 and args.console_video is not None:
                    inputs += ["-ss", str(args.console_start)]
                    if args.console_duration is not None:
                        inputs += ["-t", str(args.console_duration)]
                inputs += ["-i", str(clip)]
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", *inputs, "-filter_complex", ";".join(filters),
                            "-map", "[v]", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
                            "-movflags", "+faststart", str(args.output / "openarm_visual_demo.mp4")],
                           check=True, stdout=subprocess.DEVNULL, stderr=log)
    else:
        shutil.copyfile(clips[0], args.output / "openarm_visual_demo.mp4")
    (args.output / "demo_report.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    main()
