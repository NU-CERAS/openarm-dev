import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .camera import OverheadCamera
from .pickup import StickPickupEnv
from .residual_drumming import ResidualDrumStrikeEnv
from .simulation import ReachSimulation
from .paths import RUNS


class DemoRecorder:
    fps = 30

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(path), fps=self.fps, codec="libx264", quality=8,
                                        pixelformat="yuv420p", macro_block_size=1,
                                        ffmpeg_params=["-movflags", "+faststart"])
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        self.title_font = ImageFont.truetype(font_path, 42)
        self.font = ImageFont.truetype(font_path, 25)
        self.small_font = ImageFont.truetype(font_path, 21)
        self.trace = []
        self.segments = []
        self.credit = None

    def attach(self, sim, title, method):
        self.sim = sim
        self.title = title
        self.method = method
        self.overview = mujoco.Renderer(sim.model, height=720, width=1280)
        self.head = OverheadCamera(sim.model, width=544, height=306)
        self.view = mujoco.MjvCamera()
        self.view.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.view.lookat[:] = [0.28, 0, 0.48]
        self.view.distance = 1.5
        self.view.azimuth = 145
        self.view.elevation = -22
        self.next_capture = float(sim.data.time)
        self.segments.append({"title": title, "method": method, "start_frame": len(self.trace)})
        print(f"Recording: {title}", flush=True)

    def detach(self):
        self.overview.close()
        self.head.close()

    def capture(self, details, metrics):
        if self.sim.data.time + 1e-9 < self.next_capture:
            return
        self.next_capture += 1 / self.fps
        self.overview.update_scene(self.sim.data, camera=self.view)
        overview = self.overview.render().copy()
        head = self.head.capture(self.sim.data).rgb
        if overview.std() < 2 or head.std() < 2:
            raise RuntimeError("Rendered frame contains insufficient image variation")
        canvas = Image.new("RGB", (1920, 1080), "#0b1220")
        canvas.paste(Image.fromarray(overview), (32, 150))
        canvas.paste(Image.fromarray(head), (1344, 150))
        draw = ImageDraw.Draw(canvas)
        draw.text((32, 22), "OPENARM  /  SIMULATION SKILLS", font=self.small_font, fill="#56dbd0")
        draw.text((32, 57), self.title, font=self.title_font, fill="#f3f6fc")
        head_label = "ROBOT HEAD CAMERA" if self.sim.drum_asset is not None else "OFFICIAL HEAD-CAMERA MIDPOINT"
        draw.text((1344, 110), head_label, font=self.small_font, fill="#b5c3d6")
        draw.text((1344, 490), "CONTROL", font=self.small_font, fill="#56dbd0")
        for index, line in enumerate(self.method):
            draw.text((1344, 530 + index * 35), line, font=self.font, fill="#f3f6fc")
        draw.text((1344, 680), "MEASURED STATE", font=self.small_font, fill="#56dbd0")
        for index, line in enumerate(metrics):
            draw.text((1344, 721 + index * 34), line, font=self.font, fill="#f3f6fc")
        draw.rounded_rectangle((32, 902, 1888, 1008), radius=14, fill="#142238")
        for index, line in enumerate(details):
            draw.text((55, 918 + index * 35), line, font=self.font, fill="#e2eaf5")
        footer = self.credit or "Actual MuJoCo execution  |  No RL learning in this recording  |  No hardware execution"
        if self.small_font.getlength(footer) > 1856:
            raise ValueError("Video attribution text exceeds the available width")
        draw.text((32, 1030), footer, font=self.small_font, fill="#98adc5")
        record = {
            "frame": len(self.trace), "video_time_s": len(self.trace) / self.fps,
            "simulation_time_s": float(self.sim.data.time), "segment": self.title,
            "right_tcp_m": self.sim.tcp_position("right").tolist(),
            "left_tcp_m": self.sim.tcp_position("left").tolist(),
            "qpos": self.sim.data.qpos.tolist(), "metrics": metrics,
        }
        self.trace.append(record)
        self.writer.append_data(np.asarray(canvas))

    def hold(self, seconds, details, metrics):
        end = self.sim.data.time + seconds
        while self.sim.data.time < end:
            self.sim.step()
            self.capture(details, metrics)

    def close(self):
        self.writer.close()
        report = {"fps": self.fps, "frames": len(self.trace), "duration_s": len(self.trace) / self.fps,
                  "resolution": [1920, 1080], "segments": self.segments,
                  "learning_started": False, "hardware_executed": False}
        self.path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        self.path.with_suffix(".jsonl").write_text("".join(json.dumps(frame) + "\n" for frame in self.trace))


def record_reach(recorder):
    with ReachSimulation(planner="curobo") as sim:
        recorder.attach(sim, "01  Camera-guided reach", ["cuRobo motion planning", "Collision-aware trajectory", "Physical position servos"])
        recorder.hold(2.0, ["Select a tabletop pixel in the calibrated head-camera image.", "Known table height converts the pixel ray into a 3D reach target."], ["Planner: cuRobo", "Arms: 7 DoF each"])
        for arm, point in [("right", [0.43, -0.24, 0.30]), ("left", [0.45, 0.24, 0.30])]:
            frame = recorder.head.capture(sim.data)
            pixel = recorder.head.project(np.array(point), frame)
            target = recorder.head.unproject(*pixel, frame) + [0, 0, 0.14]
            result = sim.start_reach(arm, target)
            if not result.accepted:
                raise RuntimeError(result.reason)
            trajectory = sim.trajectory
            trajectory_path = recorder.path.parent / f"{arm}_reach_trajectory.json"
            trajectory_path.write_text(json.dumps(trajectory.to_dict(0.044), indent=2) + "\n")
            deadline = sim.data.time + trajectory.duration + 5
            while sim.status.state == "running" and sim.data.time < deadline:
                sim.step()
                recorder.capture([f"{arm.capitalize()} arm | pixel {pixel} | target [m]: {np.round(target, 3).tolist()}",
                                  "GPU planning + timed trajectory execution with actual TCP feedback."],
                                 [f"Arm: {arm}", f"TCP error: {sim.status.error_m * 1000:.1f} mm", f"State: {sim.status.state}"])
            if sim.status.state != "succeeded":
                raise RuntimeError(f"Recorded reach failed: {sim.status}")
            recorder.hold(2.0, ["Target reached and held.", "The displayed error is measured from the simulated TCP."],
                          [f"Arm: {arm}", f"Final error: {sim.status.error_m * 1000:.2f} mm", "State: reached"])
        recorder.detach()


def record_pickup(recorder):
    with StickPickupEnv() as env:
        env.reset(seed=0)
        recorder.attach(env.sim, "03  Physical drumstick pickup", ["IK reference controller", "Close parallel gripper", "Lift through friction"])
        recorder.hold(1.5, ["The arm has approached the stick using IK.", "Pickup uses the simulator object pose; image-based object detection is not connected."], ["Stick: free body", "Finger contacts: 0 / 2"])
        for index, action in enumerate(env.reference_pickup_actions()):
            _, _, terminated, truncated, info = env.step(action)
            phase = "Lower" if index < 75 else "Close" if index < 125 else "Lift" if index < 200 else "Hold"
            recorder.capture([f"Reference phase: {phase}", "The stick moves through finger contacts and friction; it remains a free body."],
                             [f"Contacts: {info['gripper_contacts']} / 2", f"Lift: {info['lift_height_m'] * 1000:.1f} mm", f"Success: {info['is_success']}"])
            if terminated or truncated:
                break
        if not info["is_success"]:
            raise RuntimeError(f"Recorded pickup failed: {info}")
        recorder.hold(2.0, ["Pickup verified: two-finger contact, lift above 100 mm, and stable hold.", "This sequence is controlled by an explicit reference controller."],
                      ["Contacts: 2 / 2", f"Lift: {info['lift_height_m'] * 1000:.1f} mm", "Pickup: succeeded"])
        recorder.detach()


def record_strike(recorder):
    with ResidualDrumStrikeEnv(randomization=0.0) as env:
        env.reset(seed=0)
        recorder.attach(env.sim, "04  Contact-rich striking environment", ["Nominal strike reference", "Zero arm RL residual", "Physical impact detection"])
        recorder.segments[-1]["reset_seed"] = 0
        recorder.hold(1.5, ["Reset physically picks up the stick and moves it above the drum.", "The RL interface exposes arm residuals and a gripper command at 50 Hz."], ["Grasp: established", "RL actions: 8", "Observation values: 51"])
        for _ in range(101):
            action = np.zeros(8, dtype=np.float32)
            action[7] = -1
            _, _, terminated, truncated, info = env.step(action)
            hit = info["last_hit"]
            recorder.capture(["Executing the nominal reference with zero learned arm correction.", "Contact forces and impact speed come from the MuJoCo contact solver."],
                             [f"Impact events: {info['hit_count']}", f"Last force: {hit['normal_force_n']:.1f} N" if hit else "Last force: --",
                              f"Grasp retained: {info['grasped']}", f"Task success: {info['is_success']}"])
            if terminated or truncated:
                break
        if info["hit_count"] < 1 or not info["grasped"]:
            raise RuntimeError(f"Recorded strike did not retain the stick after impact: {info}")
        result_text = "This reference rollout met the configured impact and recovery criteria." if info["is_success"] else "Contact demonstrated. Timing/speed task success remains a training objective."
        recorder.hold(3.0, ["RL preparation: reference tracking, grasp retention, impact timing, and speed rewards.", result_text],
                      [f"Impact events: {info['hit_count']}", "Grasp retained: True", f"Task success: {info['is_success']}"])
        recorder.detach()


def record_obstacle(recorder):
    obstacle = {"name": "reach_obstacle", "position": [0.4, -0.24, 0.4], "half_size": [0.012, 0.012, 0.08]}
    with ReachSimulation(planner="curobo", obstacles=(obstacle,)) as sim:
        target = np.array([0.52, -0.35, 0.44])
        goal = sim._solve("right", target)
        data = mujoco.MjData(sim.model)
        blocked_samples = 0
        for fraction in np.linspace(0, 1, 40):
            data.qpos[:] = sim.data.qpos + fraction * (goal - sim.data.qpos)
            mujoco.mj_forward(sim.model, data)
            blocked_samples += sim._collision(data)
        if blocked_samples == 0:
            raise RuntimeError("The obstacle demonstration requires a blocked direct joint path")
        recorder.attach(sim, "02  Obstacle-aware motion planning", ["cuRobo trajectory optimizer", "Static obstacle geometry", "Physical trajectory tracking"])
        recorder.hold(2.0, ["A red obstacle intersects the direct joint-interpolation path.", "cuRobo searches for a collision-free trajectory to the same target."],
                      ["Obstacle: present", f"Blocked samples: {blocked_samples}/40", "Planner: cuRobo"])
        result = sim.start_reach("right", target)
        if not result.accepted:
            raise RuntimeError(result.reason)
        deadline = sim.data.time + sim.trajectory.duration + 5
        while sim.status.state == "running" and sim.data.time < deadline:
            sim.step()
            recorder.capture(["Following the cuRobo path around the obstacle.", "Every recorded robot state comes from MuJoCo dynamics."],
                             ["Obstacle: present", f"TCP error: {sim.status.error_m * 1000:.1f} mm", f"State: {sim.status.state}"])
        if sim.status.state != "succeeded":
            raise RuntimeError(f"Recorded obstacle reach failed: {sim.status}")
        recorder.hold(2.0, ["Collision-free reach completed with the obstacle in place.", "The final TCP error is measured from the physical simulation state."],
                      ["State: reached", f"Final error: {sim.status.error_m * 1000:.2f} mm", "Planner: cuRobo"])
        recorder.detach()


def record_kit(recorder, manifest: Path):
    with ReachSimulation(planner="curobo", drum_asset=manifest, worktable=False) as sim:
        recorder.credit = "SINNIK | CC BY 4.0 | sketchfab.com/3d-models/electronic-drum-set-51b95e62da844b95b6ca871c23b6e858 | creativecommons.org/licenses/by/4.0/"
        recorder.attach(sim, "Electronic drum set in MuJoCo", ["Original model geometry", "45 imported visual meshes", "6 pad contact surfaces"])
        recorder.segments[-1]["asset_manifest"] = str(manifest)
        recorder.segments[-1]["asset_author"] = "SINNIK"
        recorder.segments[-1]["asset_license"] = "CC-BY-4.0"
        recorder.view.lookat[:] = [0.15, -0.05, 0.60]
        recorder.view.distance = 3.0
        recorder.view.elevation = -22
        for _ in range(240):
            for _ in range(33):
                sim.step()
            recorder.view.azimuth = 135 + sim.data.time * 5
            recorder.capture(["Electronic Drum Set by SINNIK, licensed under CC BY 4.0.",
                              "Geometry and material parameters converted for MuJoCo. No RL learning or hardware execution."],
                             ["Model scale: 1.0", "Placement: configurable", "Camera: overview + head"])
        pad = next(pad for pad in sim.drum_asset["pads"] if pad["source_node"] == "Object_28")
        target = np.array(pad["center"]) + np.array(pad["normal"]) * 0.14
        result = sim.start_reach("left", target)
        if not result.accepted:
            raise RuntimeError(result.reason)
        deadline = sim.data.time + sim.trajectory.duration + 5
        while sim.status.state == "running" and sim.data.time < deadline:
            sim.step()
            recorder.capture(["cuRobo reach to 0.14 m above a fitted pad surface.",
                              "Collision geometry represents the six pads. Rack and stool are visual geometry."],
                             ["Arm: left", f"TCP error: {sim.status.error_m * 1000:.1f} mm", f"State: {sim.status.state}"])
        if sim.status.state != "succeeded":
            raise RuntimeError(f"Electronic kit reach failed: {sim.status}")
        recorder.hold(3.0, ["The arm reached the selected pad preparation point.",
                           "Physical dimensions, placement, and contact response require measurement on the actual kit."],
                      ["State: reached", f"Final error: {sim.status.error_m * 1000:.2f} mm", "Controller: cuRobo + servos"])
        recorder.detach()


def main():
    parser = argparse.ArgumentParser(description="Record actual OpenArm simulation execution as an English-captioned MP4")
    parser.add_argument("--output", type=Path, default=RUNS / "demo/openarm_demo.mp4")
    assets = parser.add_mutually_exclusive_group()
    assets.add_argument("--drum-asset", type=Path, help="Record only the imported kit")
    assets.add_argument("--include-kit", type=Path, help="Append the imported kit to the skill demonstration")
    args = parser.parse_args()
    recorder = DemoRecorder(args.output)
    try:
        if args.drum_asset is not None:
            record_kit(recorder, args.drum_asset)
        else:
            record_reach(recorder)
            record_obstacle(recorder)
            record_pickup(recorder)
            record_strike(recorder)
            if args.include_kit is not None:
                record_kit(recorder, args.include_kit)
    finally:
        recorder.close()
    print(json.dumps({"video": str(args.output), "frames": len(recorder.trace), "duration_s": len(recorder.trace) / recorder.fps}))


if __name__ == "__main__":
    main()
