import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from openarm_sim.annotation import DrumLayout
from openarm_sim.compute_guard import budgeted
from openarm_sim.console.app import SyncConsole
from openarm_sim.console.launch import stop_backend
from openarm_sim.console.session import RobotWorker, SyncSession
from .display import NativeDisplay
from .showcase import GuiRecording


def record_console(output, display, model):
    with ExitStack() as resources:
        namespace = f"/compilation_{uuid.uuid4().hex[:10]}"
        log = resources.enter_context((output / f"{model}_backend.log").open("w"))
        backend = subprocess.Popen(
            [sys.executable, "-m", "openarm_sim.console.ros_backend", "--namespace", namespace, "--model", model],
            stdout=log, stderr=subprocess.STDOUT)
        resources.callback(stop_backend, backend)
        worker = RobotWorker("ros", ros_namespace=namespace, robot_model=model)
        resources.callback(worker.close)
        if not worker.ready.wait(15):
            raise TimeoutError("ROS backend did not become ready")
        session = SyncSession(worker, "mink", robot_model=model)
        console = SyncConsole(session, output / "positions.json")
        resources.callback(console.close)
        recording = GuiRecording(console, output, display)
        resources.callback(recording.close)
        recording.pump(2)
        reports = []
        if model == "v1.0":
            recording.mark("TCP control | torso-top coordinates in centimeters | ROS simulation")
            for axis, distance in ((2, 6), (0, 3)):
                before = session.sync().positions["right"].copy()
                recording.mark("Drag TCP | preview from the latest measured joint state")
                recording.drag(axis, distance)
                plan = session.plan
                recording.pump(1)
                np.testing.assert_allclose(session.sync().positions["right"], before, atol=.002)
                recording.mark("Execute action | independent MuJoCo backend | automatic feedback sync")
                console.execute_button.invoke()
                recording.wait(lambda: worker.state == "completed" and worker.completed_plan is plan)
                session.sync()
                np.testing.assert_allclose(session.twin.data.qpos[session.twin.qpos_ids["right"]],
                                           plan.trajectory.positions[-1], atol=.03)
                reports.append(dict(worker.report))
                recording.pump(2)
        else:
            console.layout_path = output / "layout.json"
            recording.mark("OpenDrummer | drumstick tip TCP | no camera input")
            console.open_annotation()
            panel = console.annotation
            panel.window.geometry("410x580+1220+160")
            before = session.sync().positions
            recording.mark("Annotate drum center | X forward, Y left, Z up | centimeters")
            original = np.array(panel.layout.document["targets"]["snare"]["position_cm"])
            for field, value in zip(panel.values, original + [2, 0, 2]):
                field.set(f"{value:.3f}")
            panel.apply_fields()
            recording.pump(3)
            panel.drag_enabled.set(True)
            panel.select()
            recording.mark("Drag drum center | edit simulation geometry | robot remains stationary")
            recording.wait(lambda: console.frame is not None and 2 in console.gizmo_handles)
            tip = console.gizmo_handles[2][1].astype(int)
            console.canvas.event_generate("<ButtonPress-1>", x=int(tip[0]), y=int(tip[1]))
            drag = console.drag
            if drag is None or drag.index != 2:
                raise RuntimeError("Drum center drag did not select Z")
            for index in range(1, 13):
                pixel = drag.camera.project(drag.origin + drag.axis * (drag.start_parameter + .02 * index / 12), drag.frame)
                console.canvas.event_generate("<B1-Motion>", x=pixel[0], y=pixel[1])
                recording.pump(.08)
            console.canvas.event_generate("<ButtonRelease-1>", x=pixel[0], y=pixel[1])
            recording.pump(2)
            recording.mark("Capture measured joint pose and save layout | simulated encoder feedback")
            panel.capture()
            panel.save()
            loaded = DrumLayout.load(console.layout_path, session.twin)
            if loaded.document["targets"]["snare"]["ready"]["source"] != "simulation_feedback":
                raise RuntimeError("Saved pose lacks simulation feedback provenance")
            for arm in before:
                np.testing.assert_allclose(session.sync().positions[arm], before[arm], atol=.002)
            recording.pump(3)
            reports.append({"layout_saved": True, "robot_motion_commanded": False,
                            "captured_pose": "current pose; strike feasibility is checked separately"})
        return reports


def title_frame(title, lines):
    image = Image.new("RGB", (1640, 1020), "#101b2c")
    draw = ImageDraw.Draw(image)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    draw.text((90, 130), title, font=ImageFont.truetype(font, 46), fill="#ffffff")
    for index, line in enumerate(lines):
        draw.text((90, 250 + index * 68), line, font=ImageFont.truetype(font, 27), fill="#c9d8ed")
    return np.asarray(image)


@budgeted("openarm_rl.recording.compilation")
def main():
    parser = argparse.ArgumentParser(description="Record console operations and compile actual policy evaluation clips")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--xvfb", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    with ExitStack() as resources:
        display = NativeDisplay(args.xvfb, args.output / "display")
        resources.callback(display.close)
        for model in ("v1.0", "opendrummer"):
            directory = args.output / model
            directory.mkdir()
            reports[model] = record_console(directory, display.display, model)
    manifest = json.loads(args.manifest.read_text())
    chapters = []
    destination = args.output / "openarm_changes_demo.mp4"
    with imageio.get_writer(destination, fps=24, codec="libx264", quality=8, macro_block_size=1,
                            output_params=["-movflags", "+faststart"]) as writer:
        frame_count = 0

        def card(title, lines, seconds=5):
            nonlocal frame_count
            chapters.append({"title": title, "start_s": frame_count / 24})
            frame = title_frame(title, lines)
            for _ in range(seconds * 24):
                writer.append_data(frame)
                frame_count += 1

        def clip(path):
            nonlocal frame_count
            with imageio.get_reader(path) as reader:
                fps = reader.get_meta_data()["fps"]
                emitted = 0
                for index, frame in enumerate(reader):
                    desired = round((index + 1) * 24 / fps)
                    resized = Image.fromarray(frame).convert("RGB")
                    resized.thumbnail((1640, 1020))
                    canvas = Image.new("RGB", (1640, 1020), "black")
                    canvas.paste(resized, ((1640 - resized.width) // 2, (1020 - resized.height) // 2))
                    while emitted < desired:
                        writer.append_data(np.asarray(canvas))
                        emitted += 1
                        frame_count += 1

        card("OpenARM | console, annotation and camera-free drum skills", [
            "Actual GUI operations and native MuJoCo policy evaluation recordings",
            "Code organization: cartesian_console / drum_skill_learning / opendrummer_playground",
            "Simulation demonstration. Physical hardware has not been validated."])
        clip(args.output / "v1.0/console.mp4")
        clip(args.output / "opendrummer/console.mp4")
        card("Camera-free residual PPO | one fixed-arm tool per drum", [
            "Actor: encoder history, reference trajectory, previous action, requested speed",
            "Critic and reward: privileged simulation state and contact measurements",
            "Reference motion + learned wrist residual -> bounded joint position commands",
            "Four serial GPU training runs; 131,072 steps per target; 64 evaluation episodes."])
        for target in ("snare", "rack_tom_left", "rack_tom_right", "floor_tom"):
            row = manifest["targets"][target]
            comparison = row["comparison"]
            card(f"{row['tool']} | {row['arm']} arm", [
                f"Reference success: {comparison['reference']['success_rate']:.1%}",
                f"Learned policy success: {comparison['policy']['success_rate']:.1%} (64 episodes)",
                "Experimental policy; qualification threshold: 80%",
                "Next: recorded held-out evaluation episode, shown at normal speed."], seconds=4)
            clip(args.manifest.parent / target / "heldout/execution.mp4")
        card("Current verification status", [
            "Four trained policies, exported actors and evaluation videos are available.",
            "Three targets lack a collision-free center stroke in the configured search.",
            "ROS TCP preview / execution is demonstrated in the opening console segment.",
            "ROS continuous policy streaming is covered by native integration tests.",
            "Physical drum placement, hardware execution and transfer remain unverified."], seconds=7)
    reports.update(video=str(destination), chapters=chapters, hardware_executed=False,
                   duration_s=frame_count / 24, source_manifest=str(args.manifest.resolve()))
    (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
    with imageio.get_reader(destination) as reader:
        if reader.count_frames() != frame_count:
            raise RuntimeError("Encoded compilation frame count differs")
    print(json.dumps({"video": str(destination), "duration_s": frame_count / 24, "verified_frames": frame_count}))


if __name__ == "__main__":
    main()
