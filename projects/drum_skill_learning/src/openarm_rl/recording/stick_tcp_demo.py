import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageGrab
import torch

from openarm_sim.annotation import DrumLayout
from openarm_sim.compute_guard import budgeted, check_resources
from openarm_sim.console.app import SyncConsole
from openarm_sim.console.launch import stop_backend
from openarm_sim.console.session import Preview, RobotWorker, SyncSession
from openarm_sim.execution import JointTrajectory
from ..envs.camera_free_strike import CameraFreeStrikeEnv
from ..skills.runtime import prepare_transfers
from .compilation import title_frame
from .display import NativeDisplay
from .showcase import GuiRecording


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


class StickRecording(GuiRecording):
    def __init__(self, console, output, display):
        super().__init__(console, output, display)
        self.detail = "MuJoCo execution backend | ROS 2 joint feedback | no physical hardware"
        self.positions = []
        self.stage_events = []

    def mark(self, text):
        self.chapter.set(text)
        self.events.append({"chapter": text, "frame": self.frames})
        self.stage_events.append({"title": text, "start_s": self.frames / self.fps})
        print(json.dumps({"stage": text, "video_time_s": self.frames / self.fps}), flush=True)
        self.pump(1)

    def pump(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            check_resources()
            self.console.root.update()
            if self.console.error is not None:
                raise self.console.error
            if self.console.frame is None:
                time.sleep(.005)
                continue
            elapsed = time.monotonic() - self.started
            if self.frames <= elapsed * self.fps:
                session = self.console.session
                arm = self.console.arm.get()
                measured = session.twin.tcp_position(arm)
                preview = session.preview.tcp_position(arm)
                camera = self.console.cameras[self.console.frame_index]
                canvas = self.console.canvas
                canvas.delete("demo_tip")
                pixel = camera.project(preview, self.console.frame)
                canvas.create_oval(pixel[0] - 15, pixel[1] - 15, pixel[0] + 15, pixel[1] + 15,
                                   outline="#ffdb63", width=4, tags="demo_tip")
                canvas.create_text(pixel[0] + 23, pixel[1] - 22, text="DRUMSTICK TIP = TCP",
                                   anchor="w", fill="#ffdb63", font=("DejaVu Sans", 16, "bold"), tags="demo_tip")
                if not self.positions or np.linalg.norm(self.positions[-1] - measured) > .002:
                    self.positions.append(measured.copy())
                if len(self.positions) > 1:
                    pixels = [camera.project(point, self.console.frame) for point in self.positions]
                    canvas.create_line(*np.array(pixels).ravel().tolist(), fill="#ffdb63", width=3,
                                       dash=(6, 4), tags="demo_tip")
                canvas.update_idletasks()
                root = self.console.root
                frame = ImageGrab.grab((root.winfo_rootx(), root.winfo_rooty(),
                                        root.winfo_rootx() + 1640, root.winfo_rooty() + 1020), xdisplay=self.display)
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 1640, 76), fill="#101b2c")
                draw.text((24, 18), self.chapter.get(), font=ImageFont.truetype(FONT, 30), fill="white")
                draw.rectangle((0, 928, 1640, 1020), fill="#101b2c")
                coordinates = session.reference.from_world(measured)
                draw.text((24, 938), f"Measured tip | X {coordinates[0]:+.1f} cm   Y {coordinates[1]:+.1f} cm   Z {coordinates[2]:+.1f} cm",
                          font=ImageFont.truetype(FONT, 25), fill="#ffdb63")
                draw.text((24, 979), self.detail, font=ImageFont.truetype(FONT, 20), fill="#d5e2f1")
                while self.frames <= elapsed * self.fps:
                    self.writer.append_data(np.asarray(frame))
                    self.frames += 1
                self.trace.append({"elapsed_s": elapsed, "chapter": self.chapter.get(),
                                   "measured_tip_m": measured.tolist(), "preview_tip_m": preview.tolist(),
                                   "worker_state": session.worker.state})
            time.sleep(.005)


def execute_preview(console, recording):
    session = console.session
    plan = session.plan
    if plan is None:
        raise RuntimeError("No TCP preview to execute")
    console.execute_button.invoke()
    recording.wait(lambda: session.worker.state == "completed" and session.worker.completed_plan is plan, timeout=60)
    session.sync()
    np.testing.assert_allclose(session.twin.data.qpos[session.twin.qpos_ids[plan.trajectory.arm]],
                               plan.trajectory.positions[-1], atol=.03)
    recording.pump(1)
    return dict(session.worker.report)


def record_controls(args, display):
    output = args.output
    with ExitStack() as resources:
        namespace = f"/stick_tcp_demo_{uuid.uuid4().hex[:8]}"
        log = resources.enter_context((output / "backend.log").open("w"))
        backend = subprocess.Popen([sys.executable, "-m", "openarm_sim.console.ros_backend", "--namespace", namespace,
                                    "--model", "opendrummer"], stdout=log, stderr=subprocess.STDOUT)
        resources.callback(stop_backend, backend)
        worker = RobotWorker("ros", ros_namespace=namespace, robot_model="opendrummer")
        resources.callback(worker.close)
        if not worker.ready.wait(15):
            raise TimeoutError("OpenDrummer ROS simulation did not start")
        session = SyncSession(worker, "mink", robot_model="opendrummer")
        console = SyncConsole(session, output / "positions.json")
        resources.callback(console.close)
        console.arm.set("left")
        session.preview.set_camera(np.array([1.4, 1.15, 1.35]), np.array([.32, .05, .82]), 44)
        recording = StickRecording(console, output, display)
        resources.callback(recording.close)
        recording.wait(lambda: console.frame is not None)
        reference = json.loads((args.manifest.parent / "snare/best/reference.json").read_text())
        current = np.concatenate([session.sync().positions[arm] for arm in ("left", "right")])
        goal = np.array(reference["positions_rad"][0])
        reports = []
        recording.mark("01 | Drumstick tip is the TCP")
        recording.detail = "Preparing a collision-checked working pose through the ROS trajectory interface"
        for arm, transfer in prepare_transfers(session.twin.model, session.twin.controller, current, goal):
            times = np.linspace(0, transfer.ends[-1], max(2, int(transfer.ends[-1] / .02) + 1))
            selected = slice(0, 7) if arm == "left" else slice(7, 14)
            rows = [transfer.sample(t) for t in times]
            trajectory = JointTrajectory(arm, times, np.array([q[selected] for q, _ in rows]),
                                         np.array([v[selected] for _, v in rows]), "collision_checked_transfer")
            probe = mujoco.MjData(session.twin.model)
            probe.qpos[:] = session.twin.data.qpos
            probe.qpos[session.twin.qpos_ids[arm]] = trajectory.positions[-1]
            mujoco.mj_forward(session.twin.model, probe)
            session.plan = Preview(trajectory, probe.site_xpos[session.twin.site_ids[arm]].copy(), session.sync())
            preparation = session.plan
            session.execute()
            recording.wait(lambda: worker.state == "completed" and worker.completed_plan is preparation, timeout=60)
            session.sync()
        recording.positions.clear()
        recording.pump(1)
        for axis, distance, label in ((2, 8, "Lift the stick tip by 8 cm"), (0, 6, "Move the stick tip forward by 6 cm")):
            recording.mark(f"02 | {label}: drag to PREVIEW")
            before = session.sync().positions["left"].copy()
            initial_tip = session.twin.tcp_position("left").copy()
            recording.detail = "PREVIEW ONLY | measured robot holds position | yellow circle marks the stick tip"
            recording.drag(axis, distance)
            plan = session.plan
            np.testing.assert_allclose(session.sync().positions["left"], before, atol=.002)
            expected = initial_tip.copy()
            expected[axis] += distance / 100
            np.testing.assert_allclose(plan.target, expected, atol=.004)
            recording.pump(2)
            recording.mark("03 | EXECUTE ACTION: move, read encoders, synchronize")
            recording.detail = "Executing through ROS 2 | dashed yellow line shows measured stick-tip motion"
            reports.append(execute_preview(console, recording))
            np.testing.assert_allclose(session.twin.tcp_position("left"), expected, atol=.008)
        recording.mark("04 | Register a coordinate point using the measured stick tip")
        recording.detail = "Human-observed point registration | torso-top reference | centimeters"
        console.layout_path = output / "annotated_layout.json"
        console.open_annotation()
        panel = console.annotation
        panel.window.geometry("430x580+1190+190")
        recording.pump(2)
        panel.use_tip()
        measured = session.twin.tcp_position("left").copy()
        np.testing.assert_allclose(panel.layout.center("snare"), measured, atol=1e-10)
        recording.detail = "Point A = measured stick tip | coordinates copied into the snare-center annotation"
        recording.pump(4)
        panel.save()
        saved = DrumLayout.load(console.layout_path, session.twin)
        np.testing.assert_allclose(saved.center("snare"), measured, atol=1e-10)
        recording.mark("05 | Saved: torso-relative drum coordinates for the next session")
        recording.detail = "Demonstration point captured from simulated encoders | physical registration requires operator observation"
        recording.pump(3)
        return {"executions": reports, "registered_point_world_m": measured.tolist(),
                "registered_point_cm": session.reference.from_world(measured).tolist(),
                "layout_verified": True, "control_chapters": recording.stage_events,
                "preparation": "collision-checked motion from zero to working pose"}


def record_policy_details(args):
    results = {}
    for target in ("snare", "floor_tom"):
        directory = args.manifest.parent / target / "best"
        actor = torch.jit.load(str(directory / "actor.pt"), map_location="cpu").eval()
        with ExitStack() as resources:
            env = resources.enter_context(CameraFreeStrikeEnv(
                args.manifest.parent / "scene", directory / "profile.json", target=target,
                position_range=.005, domain_range=.1))
            observation, _ = env.reset(seed=800000)
            env.model.vis.global_.offwidth = max(1280, env.model.vis.global_.offwidth)
            env.model.vis.global_.offheight = max(720, env.model.vis.global_.offheight)
            renderer = mujoco.Renderer(env.model, 720, 1280)
            resources.callback(renderer.close)
            writer = resources.enter_context(imageio.get_writer(args.output / f"{target}_detail.mp4", fps=50, codec="libx264"))
            camera = mujoco.MjvCamera()
            camera.lookat[:] = env.data.geom_xpos[env.model.geom(f"{target}_head_collision").id] + [0, 0, .12]
            camera.distance, camera.azimuth, camera.elevation = 1.05, 135, -28
            options = mujoco.MjvOption()
            options.sitegroup[:] = 0
            for step in range(1000):
                check_resources()
                with torch.no_grad():
                    action = actor(torch.tensor(observation["actor"]).unsqueeze(0)).numpy()[0]
                observation, _, done, truncated, info = env.step(action)
                if step % 2 == 0:
                    renderer.update_scene(env.data, camera=camera, scene_option=options)
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 1280, 100), fill="#101b2c")
                    draw.text((24, 12), f"{target} | exported camera-free actor | 0.5x slow motion",
                              font=ImageFont.truetype(FONT, 25), fill="white")
                    draw.text((24, 57), f"Simulation t={env.data.time:.2f}s | detected hits: {info['hit_count']}",
                              font=ImageFont.truetype(FONT, 23), fill="#ffdb63")
                    writer.append_data(np.asarray(frame))
                if done or truncated:
                    results[target] = {"seed": 800000, "is_success": info["is_success"],
                                       "hit_count": info["hit_count"], "recovered": info["recovered"],
                                       "unsafe_contact": info["unsafe_contact"]}
                    break
            else:
                raise RuntimeError("Policy detail recording exceeded its episode bound")
        print(json.dumps({"policy_detail": target, **results[target]}), flush=True)
    return results


def compose(args, report):
    destination = args.output / "drumstick_tcp_calibration_demo.mp4"
    frames = 0
    with imageio.get_writer(destination, fps=24, codec="libx264", quality=8, macro_block_size=1,
                            output_params=["-movflags", "+faststart"]) as writer:
        def card(title, lines, seconds):
            nonlocal frames
            image = title_frame(title, lines)
            for _ in range(seconds * 24):
                writer.append_data(image)
                frames += 1

        def clip(path, label=None, speed=1):
            nonlocal frames
            with imageio.get_reader(path) as reader:
                fps = reader.get_meta_data()["fps"]
                emitted = 0
                for index, raw in enumerate(reader):
                    image = Image.fromarray(raw).convert("RGB")
                    image.thumbnail((1640, 1020))
                    canvas = Image.new("RGB", (1640, 1020), "#101b2c")
                    canvas.paste(image, ((1640 - image.width) // 2, (1020 - image.height) // 2))
                    if label:
                        draw = ImageDraw.Draw(canvas)
                        draw.rectangle((0, 0, 1640, 90), fill="#101b2c")
                        draw.text((30, 25), label, font=ImageFont.truetype(FONT, 30), fill="white")
                    desired = round((index + 1) * 24 / fps / speed)
                    while emitted < desired:
                        writer.append_data(np.asarray(canvas))
                        emitted += 1
                        frames += 1

        card("Move the drumstick tip. Register the drum coordinates.", [
            "1. Drag the stick-tip TCP and preview the motion",
            "2. Execute and synchronize from measured joint feedback",
            "3. Save a torso-relative coordinate point using the stick tip",
            "Actual console operations | OpenDrummer model | MuJoCo + ROS 2"], 4)
        clip(args.output / "console.mp4")
        manifest = json.loads(args.manifest.read_text())
        metrics = []
        for name in ("snare", "rack_tom_left", "rack_tom_right", "floor_tom"):
            row = manifest["targets"][name]
            comparison = row["comparison"]
            metrics.append(f"{name}: {comparison['reference']['success_rate']:.0%} reference -> {comparison['policy']['success_rate']:.0%} RL")
        card("RL progress | four independent camera-free strike policies", [
            *metrics, "131,072 steps / policy | 64 evaluation episodes / target",
            "Experimental policies; success threshold is 80%. Physical transfer is unverified."], 6)
        for name in ("snare", "floor_tom"):
            result = report["policy_details"][name]
            clip(args.output / f"{name}_detail.mp4", speed=.5)
            card(f"{name} | recorded episode outcome", [
                f"Detected hits: {result['hit_count']} | Recovered: {result['recovered']} | Unsafe contact: {result['unsafe_contact']}",
                f"Full task success (center, timing, speed and recovery): {result['is_success']}",
                "Fixed evaluation seed 800000 | results across 64 episodes appear above"], 2)
        card("What the demo verifies", [
            "Drumstick-tip TCP preview, explicit execution and measured-state synchronization",
            "Coordinate point registration and saved-layout reload",
            "Four trained camera-free actors; two evaluation episodes shown",
            "ROS policy execution: repeated calls validated with native integration tests.",
            "Physical hardware execution and physical point registration remain unverified."], 5)
    with imageio.get_reader(destination) as reader:
        if reader.count_frames() != frames:
            raise RuntimeError("Video frame verification failed")
    report.update(video=str(destination), duration_s=frames / 24, verified_frames=frames, hardware_executed=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


@budgeted("openarm_rl.recording.stick_tcp_demo")
def main():
    parser = argparse.ArgumentParser(description="Record stick-tip TCP control, coordinate registration and RL progress")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--xvfb", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as resources:
        display = NativeDisplay(args.xvfb, args.output / "display")
        resources.callback(display.close)
        report = record_controls(args, display.display)
        report["policy_details"] = record_policy_details(args)
    compose(args, report)


if __name__ == "__main__":
    main()
