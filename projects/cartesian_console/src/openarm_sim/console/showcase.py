import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import time
from tkinter import StringVar, ttk

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import ImageGrab

from ..compute_guard import budgeted, check_resources
from ..paths import RUNS
from ..record_demo import DemoRecorder, record_obstacle, record_pickup, record_strike
from .app import SyncConsole
from .session import RobotWorker, SyncSession


class PrivateDisplay:
    def __init__(self, executable, output, number):
        self.display = f"127.0.0.1:{number}.0"
        self.authority = output / "display.auth"
        self.authority.touch(mode=0o600, exist_ok=False)
        subprocess.run(["xauth", "-f", str(self.authority), "add", self.display,
                        "MIT-MAGIC-COOKIE-1", secrets.token_hex(16)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.log = (output / "display.log").open("w")
        self.process = subprocess.Popen([str(executable), f":{number}", "-screen", "0", "1720x1080x24",
                                         "-nolock", "-nolisten", "unix", "-listen", "tcp", "-auth", str(self.authority)],
                                        stdout=self.log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(f"Private X server failed; inspect {output / 'display.log'}")
            with socket.socket() as connection:
                if connection.connect_ex(("127.0.0.1", 6000 + number)) == 0:
                    break
            if time.monotonic() - started > 10:
                self.close()
                raise TimeoutError("Private X server did not become ready")
            time.sleep(0.1)
        os.environ["DISPLAY"] = self.display
        os.environ["XAUTHORITY"] = str(self.authority)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=5)
        self.log.close()


class GuiRecording:
    fps = 12

    def __init__(self, console, output, display):
        self.console, self.output, self.display = console, output, display
        self.chapter = StringVar(value="01  Torso-top reference and centimeter coordinates")
        heading = ttk.Label(console.root, textvariable=self.chapter, style="Step.TLabel", padding=(18, 8))
        heading.pack(fill="x", before=console.root.winfo_children()[0])
        console.root.geometry("1640x1020+0+0")
        console.root.update()
        console.root.focus_force()
        self.writer = imageio.get_writer(output / "console.mp4", fps=self.fps, codec="libx264", quality=7,
                                         macro_block_size=1, output_params=["-movflags", "+faststart"])
        self.started = time.monotonic()
        self.frames = 0
        self.trace = []
        self.events = []

    def pump(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            check_resources()
            self.console.root.update()
            if self.console.error is not None:
                raise self.console.error
            elapsed = time.monotonic() - self.started
            if self.frames <= elapsed * self.fps:
                root = self.console.root
                x, y = root.winfo_rootx(), root.winfo_rooty()
                frame = np.asarray(ImageGrab.grab((x, y, x + 1640, y + 1020), xdisplay=self.display))
                while self.frames <= elapsed * self.fps:
                    self.writer.append_data(frame)
                    self.frames += 1
                self.trace.append({"elapsed_s": elapsed, "chapter": self.chapter.get(),
                                   "state": self.console.session.worker.state,
                                   "target_cm": [v.get() for v in self.console.coordinates]})
            time.sleep(0.005)

    def wait(self, predicate, timeout=30):
        started = time.monotonic()
        while not predicate():
            if time.monotonic() - started > timeout:
                raise TimeoutError("Showcase step did not finish")
            self.pump(0.08)

    def mark(self, text):
        self.chapter.set(text)
        self.events.append({"chapter": text, "frame": self.frames})
        self.pump(1.5)

    def drag(self, axis, centimeters, cancel=False):
        self.wait(lambda: self.console.frame is not None and axis in self.console.gizmo_handles and self.console.resize_pending is None)
        tip = self.console.gizmo_handles[axis][1].astype(int)
        self.console.canvas.event_generate("<ButtonPress-1>", x=int(tip[0]), y=int(tip[1]))
        drag = self.console.drag
        if drag is None or drag.index != axis:
            raise RuntimeError("The recorded mouse gesture did not select the requested TCP axis")
        for step in range(1, 13):
            point = drag.origin + drag.axis * (0.10 + centimeters / 100 * step / 12)
            pixel = drag.camera.project(point, drag.frame)
            self.console.canvas.event_generate("<B1-Motion>", x=pixel[0], y=pixel[1])
            self.pump(0.08)
        if cancel:
            self.console.root.event_generate("<Escape>")
        self.console.canvas.event_generate("<ButtonRelease-1>", x=pixel[0], y=pixel[1])
        if not cancel and self.console.session.plan is None:
            raise RuntimeError(self.console.session.rejection_reason)
        self.pump(1.5)

    def wait_for_motion(self):
        session = self.console.session
        plan = session.plan
        target = session.plan.target.copy()
        measured = session.sync().positions[plan.trajectory.arm].copy()
        self.pump(plan.trajectory.duration + 0.2)
        np.testing.assert_allclose(session.sync().positions[plan.trajectory.arm], measured, atol=0.002)
        self.chapter.set("Execute action - move, read feedback and synchronize")
        self.console.execute_button.invoke()
        self.wait(lambda: session.worker.state == "completed" and session.worker.completed_plan is plan)
        session.sync()
        if np.linalg.norm(session.twin.tcp_position(session.plan.trajectory.arm) - target) > 0.015:
            raise RuntimeError("Recorded execution did not reach its target")
        self.pump(1)
        return dict(session.worker.report)

    def close(self):
        self.writer.close()
        (self.output / "console_trace.json").write_text(json.dumps({"fps": self.fps, "frames": self.frames,
                                                                   "chapters": self.events, "trace": self.trace}, indent=2) + "\n")


def record_gui(output, planner, display):
    worker = RobotWorker()
    session = None
    console = None
    recording = None
    try:
        if not worker.ready.wait(10):
            raise TimeoutError("Demonstration robot did not become ready")
        session = SyncSession(worker, planner)
        console = SyncConsole(session, output / "saved_positions.json")
        recording = GuiRecording(console, output, display)
        recording.pump(2)
        recording.mark("02  Drag Z - preview a lift from joint zero")
        recording.drag(2, 6)
        recording.wait_for_motion()
        console.name.set("right_lift")
        console.save_button.invoke()
        recording.mark("03  Drag X - preview from measured feedback")
        recording.drag(0, 4)
        recording.wait_for_motion()
        recording.mark("04  Drag Y - preview from the current robot pose")
        recording.drag(1, -2)
        right = recording.wait_for_motion()
        initial = session.sync().positions["right"].copy()
        recording.mark("05  Escape cancels a drag and restores its target")
        previous = [v.get() for v in console.coordinates]
        recording.drag(0, 5, cancel=True)
        if [v.get() for v in console.coordinates] != previous:
            raise RuntimeError("Escape did not restore the original target")
        np.testing.assert_allclose(session.sync().positions["right"], initial, atol=0.002)
        recording.mark("06  Switch arms and drag the left TCP")
        console.arm.set("left")
        recording.pump(1)
        recording.drag(2, 6)
        recording.wait_for_motion()
        recording.drag(0, 4)
        left = recording.wait_for_motion()
        console.name.set("left_demo")
        console.save_button.invoke()
        recording.mark("07  Preview a saved position from current feedback")
        console.name.set("right_lift")
        console.load_button.invoke()
        if session.plan is None:
            raise RuntimeError("Loading a saved position did not create its coordinate motion")
        right = recording.wait_for_motion()
        recording.pump(3)
        return {"right_execution": right, "left_execution": left,
                "reference_site": session.reference.site, "reference_origin_m": session.reference.origin_m.tolist()}
    finally:
        if recording is not None:
            recording.close()
        if console is not None and not console.closed:
            console.close()
        elif session is not None:
            session.close()
        else:
            worker.close()


@budgeted("openarm_sim.console.showcase")
def main():
    parser = argparse.ArgumentParser(description="Record actual GUI interaction and native physics in a private display")
    parser.add_argument("--output", type=Path, default=RUNS / "showcase")
    parser.add_argument("--xvfb", type=Path, default=shutil.which("Xvfb"))
    parser.add_argument("--display-number", type=int, default=96)
    parser.add_argument("--planner", choices=["ompl", "curobo"], default="ompl")
    args = parser.parse_args()
    if args.xvfb is None:
        raise ValueError("Install Xvfb or supply its executable with --xvfb")
    args.output.mkdir(parents=True, exist_ok=False)
    display = PrivateDisplay(args.xvfb, args.output, args.display_number)
    try:
        report = record_gui(args.output, args.planner, display.display)
        recorder = DemoRecorder(args.output / "physics.mp4")
        try:
            record_obstacle(recorder, args.planner)
            record_pickup(recorder)
            record_strike(recorder)
        finally:
            recorder.close()
        with (args.output / "encoding.log").open("w") as encoding_log:
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(args.output / "console.mp4"),
                            "-i", str(args.output / "physics.mp4"), "-filter_complex",
                            "[0:v]fps=20,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[a];[1:v]fps=20,setsar=1[b];[a][b]concat=n=2:v=1:a=0[v]",
                            "-map", "[v]", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-movflags", "+faststart",
                            str(args.output / "openarm_showcase.mp4")], check=True, stdout=subprocess.DEVNULL, stderr=encoding_log)
        report.update(planner=args.planner, backend="independent_mujoco_physics", hardware_executed=False, training_started=False,
                      video="openarm_showcase.mp4", physics_segments=recorder.segments)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        display.close()


if __name__ == "__main__":
    main()
