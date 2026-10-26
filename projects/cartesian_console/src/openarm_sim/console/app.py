import argparse
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from openarm_tools.recording.storage import write_json

from ..camera import OverheadCamera
from ..compute_guard import budgeted, check_resources
from ..paths import RUNS
from ..execution import joint_names
from .session import RobotWorker, SyncSession, hardware_configuration


class SyncConsole:
    def __init__(self, session: SyncSession, targets: Path, width=640, height=360):
        self.session = session
        self.targets = targets
        self.session.sync()
        self.root = tk.Tk()
        self.root.title(f"OpenArm | Sim-to-real console | {session.worker.backend.upper()}")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.report_callback_exception = self.callback_error
        self.error = None
        self.closed = False
        self.preview_started = None
        self.last_report = None
        self.session_directory = RUNS / "sync/sessions" / str(time.time_ns())
        self.session_directory.mkdir(parents=True)
        self.feedback_log = (self.session_directory / "feedback.jsonl").open("x")
        self.arm = tk.StringVar(value="right")
        initial_target = (0.43, -0.34, 0.56) if session.preview.drum_asset is not None else (0.43, -0.24, 0.44)
        self.coordinates = [tk.StringVar(value=str(value)) for value in initial_target]
        self.increment = tk.StringVar(value="0.01")
        self.clearance = tk.StringVar(value="0.14")
        self.name = tk.StringVar(value="target_1")
        self.message = tk.StringVar(value="Connected. Select a sim target and create a preview.")
        self.feedback_text = tk.StringVar()
        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill="x")
        ttk.Label(controls, text="Arm").pack(side="left")
        selector = ttk.Combobox(controls, textvariable=self.arm, values=["right", "left"], state="readonly", width=7)
        selector.pack(side="left", padx=4)
        for axis, variable in zip("XYZ", self.coordinates):
            ttk.Label(controls, text=f"{axis} [m]").pack(side="left")
            ttk.Entry(controls, textvariable=variable, width=9).pack(side="left", padx=3)
        ttk.Button(controls, text="Use current TCP", command=self.use_tcp).pack(side="left", padx=5)
        self.preview_button = ttk.Button(controls, text="Plan / Preview", command=self.plan)
        self.preview_button.pack(side="left", padx=5)
        self.execute_button = ttk.Button(controls, text="Execute on robot", command=self.execute, state="disabled")
        self.execute_button.pack(side="left", padx=5)
        ttk.Button(controls, text="Cancel motion", command=self.cancel).pack(side="left", padx=5)
        adjustments = ttk.Frame(self.root, padding=(8, 0, 8, 6))
        adjustments.pack(fill="x")
        ttk.Label(adjustments, text="Adjustment [m]").pack(side="left")
        ttk.Entry(adjustments, textvariable=self.increment, width=7).pack(side="left", padx=4)
        for index, axis in enumerate("XYZ"):
            for direction in (-1, 1):
                ttk.Button(adjustments, text=f"{axis}{'+' if direction > 0 else '-'}", width=4,
                           command=lambda i=index, d=direction: self.adjust(i, d)).pack(side="left", padx=2)
        ttk.Label(adjustments, text="Click clearance [m]").pack(side="left", padx=6)
        ttk.Entry(adjustments, textvariable=self.clearance, width=7).pack(side="left")
        ttk.Button(adjustments, text="Overview", command=lambda: self.view(True)).pack(side="left", padx=8)
        ttk.Button(adjustments, text="Robot view", command=lambda: self.view(False)).pack(side="left")
        saved = ttk.Frame(self.root, padding=(8, 0, 8, 6))
        saved.pack(fill="x")
        ttk.Label(saved, text="Saved target").pack(side="left")
        self.names = ttk.Combobox(saved, textvariable=self.name, width=28)
        self.names.pack(side="left", padx=5)
        ttk.Button(saved, text="Save target + observed joints", command=self.save).pack(side="left", padx=5)
        ttk.Button(saved, text="Load target", command=self.load).pack(side="left", padx=5)
        ttk.Label(saved, text="Loading requires a new preview before execution.").pack(side="left", padx=8)
        views = ttk.Frame(self.root)
        views.pack()
        self.canvases = []
        self.image_items = []
        self.photos = [None, None]
        self.cameras = []
        for index, (title, sim) in enumerate((("Live robot joint feedback", session.twin),
                                            ("Simulation trajectory preview - click to select target", session.preview))):
            panel = ttk.LabelFrame(views, text=title)
            panel.grid(row=0, column=index)
            canvas = tk.Canvas(panel, width=width, height=height, highlightthickness=0)
            canvas.pack()
            self.canvases.append(canvas)
            self.image_items.append(canvas.create_image(0, 0, anchor="nw"))
            self.cameras.append(OverheadCamera(sim.model, width, height, sim.table_height))
        self.canvases[1].bind("<Button-1>", self.click)
        ttk.Label(self.root, textvariable=self.message, padding=(8, 5), wraplength=1260).pack(fill="x")
        ttk.Label(self.root, textvariable=self.feedback_text, padding=(8, 5)).pack(fill="x")
        text = "DEMO: a separate MuJoCo robot executes commands and supplies feedback. No camera input."
        if session.worker.backend == "can":
            text = "CAN: seven arm joints only; grippers are unchanged. Motors release after motion/cancel. Support both arms."
        ttk.Label(self.root, text=text, padding=(8, 5)).pack(fill="x")
        self.camera_defaults = [(sim.model.cam_pos[camera.camera_id].copy(), sim.camera_look_at.copy(),
                                 float(sim.model.cam_fovy[camera.camera_id]))
                                for sim, camera in zip((session.twin, session.preview), self.cameras)]
        if session.twin.tabletop_mount:
            self.view(True)
        for variable in [self.arm, *self.coordinates]:
            variable.trace_add("write", self.invalidate)
        self.refresh_names()
        self.frame = None
        self.tick()

    def invalidate(self, *_):
        self.session.plan = None
        self.preview_started = None
        self.execute_button.configure(state="disabled")

    def use_tcp(self):
        self.session.sync()
        for variable, value in zip(self.coordinates, self.session.twin.tcp_position(self.arm.get())):
            variable.set(f"{value:.5f}")

    def adjust(self, axis, direction):
        step = float(self.increment.get())
        if not np.isfinite(step) or not 0 < step <= 0.1:
            raise ValueError("Adjustment must be in (0, 0.1] meters")
        self.coordinates[axis].set(f"{float(self.coordinates[axis].get()) + direction * step:.5f}")
        self.message.set("Target adjusted. Create a new preview.")

    def click(self, event):
        if self.frame is None:
            raise RuntimeError("Preview image is not available")
        clearance = float(self.clearance.get())
        if not np.isfinite(clearance) or not 0 <= clearance <= 0.3:
            raise ValueError("Click clearance must be in [0, 0.3] meters")
        target = self.cameras[1].unproject(event.x, event.y, self.frame) + [0, 0, clearance]
        for variable, value in zip(self.coordinates, target):
            variable.set(f"{value:.5f}")
        self.message.set("Sim target selected. Create a preview before execution.")

    def plan(self):
        result = self.session.prepare(self.arm.get(), [float(variable.get()) for variable in self.coordinates])
        if result is None:
            self.preview_started = None
            self.execute_button.configure(state="disabled")
            self.message.set(self.session.rejection_reason)
            return
        write_json(self.session_directory / f"plan_{time.time_ns()}.json", {
            "schema": "openarm.sync_plan.v1", "arm": result.trajectory.arm,
            "joint_names": joint_names(result.trajectory.arm), "target_m": result.target.tolist(),
            "times_s": result.trajectory.times.tolist(), "positions_rad": result.trajectory.positions.tolist(),
            "velocities_rad_s": result.trajectory.velocities.tolist(), "source": result.trajectory.source,
            "gripper_commanded": False,
        })
        self.preview_started = time.monotonic()
        self.execute_button.configure(state="normal")
        self.message.set(f"Preview ready | {result.trajectory.source} | {result.trajectory.duration:.2f} s | Robot has not moved.")

    def execute(self):
        self.session.execute()
        self.execute_button.configure(state="disabled")
        self.message.set("Executing the previewed trajectory. Live view follows measured joint feedback.")

    def cancel(self):
        self.session.worker.cancel()
        self.invalidate()
        self.message.set("Cancellation requested.")

    def view(self, overview):
        for index, sim in enumerate((self.session.twin, self.session.preview)):
            look_height = 0.25 if sim.tabletop_mount else 0.48
            values = (np.array([1.55, 1.15, 1.25]), np.array([0.25, 0, look_height]), 48) if overview else self.camera_defaults[index]
            sim.set_camera(*values)

    def refresh_names(self):
        names = []
        if self.targets.exists():
            payload = json.loads(self.targets.read_text())
            if payload["schema"] != "openarm.sync_targets.v1":
                raise ValueError("Unsupported target file")
            names = list(payload["targets"])
        self.names.configure(values=names)

    def save(self):
        if self.session.worker.state == "executing":
            raise RuntimeError("Wait for execution to finish before saving observed joints")
        self.session.save(self.targets, self.name.get())
        self.refresh_names()
        self.message.set(f"Saved {self.name.get()} to {self.targets}")

    def load(self):
        arm, target = self.session.load(self.targets, self.name.get())
        self.arm.set(arm)
        for variable, value in zip(self.coordinates, target):
            variable.set(str(value))
        self.message.set("Saved target loaded. Create a new preview from current feedback.")

    def tick(self):
        if self.closed:
            return
        check_resources()
        feedback = self.session.sync()
        if self.preview_started is not None and self.session.plan is not None:
            self.session.show_preview(min(time.monotonic() - self.preview_started, self.session.plan.trajectory.duration))
        else:
            from .session import apply_feedback

            apply_feedback(self.session.preview, feedback)
        for index, (sim, camera) in enumerate(zip((self.session.twin, self.session.preview), self.cameras)):
            frame = camera.capture(sim.data)
            if index == 1:
                self.frame = frame
            rendered = Image.fromarray(frame.rgb)
            if self.session.plan is not None:
                u, v = camera.project(self.session.plan.target, frame)
                drawing = ImageDraw.Draw(rendered)
                drawing.ellipse((u - 6, v - 6, u + 6, v + 6), outline="#ffba40", width=2)
                drawing.text((u + 8, v - 8), "Target", fill="#ffba40")
            self.photos[index] = ImageTk.PhotoImage(rendered)
            self.canvases[index].itemconfigure(self.image_items[index], image=self.photos[index])
        arm = self.arm.get()
        tcp = self.session.twin.tcp_position(arm)
        self.feedback_text.set(f"Transport: {self.session.worker.state} | Feedback age: {(time.monotonic() - feedback.received_at) * 1000:.0f} ms | "
                               f"TCP in sim [m]: {np.round(tcp, 4).tolist()} | Joints [rad]: {np.round(feedback.positions[arm], 3).tolist()}")
        self.feedback_log.write(json.dumps({"timestamp_unix": time.time(), "state": self.session.worker.state,
                                           "positions_rad": {name: value.tolist() for name, value in feedback.positions.items()}}) + "\n")
        self.feedback_log.flush()
        if self.session.worker.report is not None and self.session.worker.report is not self.last_report:
            self.last_report = self.session.worker.report
            write_json(self.session_directory / f"execution_{time.time_ns()}.json", self.last_report)
            self.message.set(f"Execution completed | {json.dumps(self.session.worker.report)}")
        self.root.after(80, self.tick)

    def callback_error(self, kind, error, traceback):
        self.error = error
        self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.session.close()
        finally:
            self.feedback_log.close()
            for camera in self.cameras:
                camera.close()
            self.root.destroy()

    def run(self):
        self.root.mainloop()
        if self.error is not None:
            raise self.error


@budgeted("openarm_sim.console.app")
def main():
    parser = argparse.ArgumentParser(description="Feedback-driven OpenArm twin and explicit trajectory execution")
    parser.add_argument("--backend", choices=["demo", "can"], default="demo")
    parser.add_argument("--hardware-config", type=Path)
    parser.add_argument("--supported", action="store_true", help="Both physical arms are supported for motor release")
    parser.add_argument("--planner", choices=["ompl", "mink", "curobo"], default="ompl")
    parser.add_argument("--drum-asset", type=Path)
    parser.add_argument("--targets", type=Path, default=RUNS / "sync/targets.json")
    parser.add_argument("--speed", type=float, default=0.2)
    parser.add_argument("--record", type=Path, help="Run and record a physical MuJoCo demo without opening Tk")
    parser.add_argument("--smoke-test", action="store_true", help="Exercise the real demo GUI and close automatically")
    args = parser.parse_args()
    if args.backend == "can":
        if args.record or args.smoke_test:
            raise ValueError("Automatic demonstration is available only with the MuJoCo backend")
        if args.hardware_config is None or not args.supported:
            raise ValueError("CAN requires --hardware-config and --supported")
        hardware_configuration(args.hardware_config)
    if args.record:
        from .video import record

        record(args.record, args.planner, args.drum_asset)
        return
    worker = RobotWorker(args.backend, args.hardware_config, args.drum_asset)
    session = None
    try:
        if not worker.ready.wait(10):
            raise TimeoutError("Robot connection did not become ready")
        worker.read()
        session = SyncSession(worker, args.planner, args.drum_asset, args.speed)
        console = SyncConsole(session, args.targets)
        print(json.dumps({"sync_console": "ready", "backend": args.backend, "planner": args.planner,
                          "session": str(console.session_directory)}), flush=True)
        if args.smoke_test:
            started = time.monotonic()

            def exercise():
                if time.monotonic() - started > 90:
                    raise TimeoutError("GUI smoke test exceeded 90 seconds")
                if worker.state == "completed":
                    console.save()
                    console.load()
                    console.close()
                else:
                    console.root.after(100, exercise)

            console.arm.set("left")
            console.plan()
            if session.plan is not None or not console.root.winfo_exists():
                raise RuntimeError("Opposite-side target must be rejected while keeping the interface available")
            console.arm.set("right")
            console.plan()
            console.execute()
            console.root.after(100, exercise)
        console.run()
        if args.smoke_test:
            print(json.dumps({"gui_smoke_test": "passed", "execution": worker.report,
                              "targets": str(args.targets), "session": str(console.session_directory)}), flush=True)
    finally:
        if session is not None:
            session.close()
        else:
            worker.close()


if __name__ == "__main__":
    main()
