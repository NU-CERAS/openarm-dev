import argparse
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import mujoco
from PIL import Image, ImageTk

from openarm_tools.recording.storage import write_json

from ..camera import OverheadCamera
from ..compute_guard import budgeted, check_resources
from ..execution import joint_names
from ..paths import RUNS
from .session import RobotWorker, SyncSession, apply_feedback, hardware_configuration
from .gizmo import AxisDrag, hit_axis


class SyncConsole:
    def __init__(self, session: SyncSession, targets: Path, width=None, height=None):
        self.session, self.targets = session, targets
        initial_feedback = self.session.sync()
        self.root = tk.Tk()
        self.root.title("OpenArm | Tabletop reach")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.report_callback_exception = self.callback_error
        self.width = width or min(1280, max(640, self.root.winfo_screenwidth() - 390))
        self.height = height or min(760, max(400, self.root.winfo_screenheight() - 240))
        self.root.minsize(980, 620)
        self.root.geometry(f"{self.width + 340}x{self.height + 190}")
        self.error = None
        self.closed = False
        self.preview_started = None
        self.last_report = None
        self.last_state = None
        self.frame = None
        self.frame_index = 0
        self.resize_pending = None
        self.drag = None
        self.drag_target = None
        self.drag_changed = False
        self.gizmo_handles = {}
        self.gizmo_origin = None
        self.saved_names = set()
        self.session_directory = RUNS / "sync/sessions" / str(time.time_ns())
        self.session_directory.mkdir(parents=True)
        write_json(self.session_directory / "startup.json", {
            "schema": "openarm.console_startup.v1", "backend": session.worker.backend,
            "reference_site": session.reference.site,
            "joint_positions_rad": {arm: values.tolist() for arm, values in initial_feedback.positions.items()},
            "tcp_relative_cm": {arm: session.reference.from_world(session.twin.tcp_position(arm)).tolist()
                                for arm in ("right", "left")},
            "motion_commanded": False,
        })
        self.feedback_log = (self.session_directory / "feedback.jsonl").open("x")
        self.arm = tk.StringVar(value="right")
        initial_target_cm = session.reference.from_world(session.twin.tcp_position(self.arm.get()))
        self.coordinates = [tk.StringVar(value=f"{value:.2f}") for value in initial_target_cm]
        self.increment = tk.StringVar(value="1")
        self.name = tk.StringVar(value="target_1")
        self.display_mode = tk.StringVar(value="Live")
        self.message = tk.StringVar(value="Drag a TCP axis, then release to preview. Choose Move robot when ready.")
        self.view_label = tk.StringVar(value="LIVE ROBOT")
        self.connection = tk.StringVar()
        self.feedback_text = tk.StringVar()
        self.detail_text = tk.StringVar()
        self._style()
        self._layout()
        self.cameras = [OverheadCamera(sim.model, self.width, self.height, sim.table_height)
                        for sim in (session.twin, session.preview)]
        self.camera_defaults = [(sim.model.cam_pos[camera.camera_id].copy(), sim.camera_look_at.copy(),
                                 float(sim.model.cam_fovy[camera.camera_id]))
                                for sim, camera in zip((session.twin, session.preview), self.cameras)]
        self.view(True)
        self.arm.trace_add("write", self.select_arm)
        for variable in self.coordinates:
            variable.trace_add("write", self.invalidate)
        self.name.trace_add("write", self.update_load_button)
        self.refresh_names()
        self.canvas.bind("<Configure>", self.schedule_resize)
        self.root.bind("<Escape>", lambda event: self.cancel_drag())
        self.tick()

    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        self.root.option_add("*Font", "{DejaVu Sans} 11")
        style.configure(".", font=("DejaVu Sans", 11))
        style.configure("TFrame", background="#f3f5f8")
        style.configure("TLabel", background="#f3f5f8", foreground="#172b4d")
        style.configure("TButton", padding=(10, 8), font=("DejaVu Sans", 11))
        style.configure("Title.TLabel", font=("DejaVu Sans", 20, "bold"))
        style.configure("Step.TLabel", font=("DejaVu Sans", 12, "bold"))
        style.configure("Muted.TLabel", foreground="#46566e", font=("DejaVu Sans", 10))
        style.configure("Primary.TButton", foreground="white", background="#245bd7", font=("DejaVu Sans", 12, "bold"))
        style.map("Primary.TButton", background=[("disabled", "#d5dce8"), ("active", "#1746ae")],
                  foreground=[("disabled", "#59677b")])
        style.configure("Stop.TButton", foreground="#a61b2b", font=("DejaVu Sans", 12, "bold"))
        style.configure("Status.TLabel", padding=(12, 10), background="#e5edf9", font=("DejaVu Sans", 11))

    def _layout(self):
        header = ttk.Frame(self.root, padding=(18, 12))
        header.pack(fill="x")
        ttk.Label(header, text="OpenArm", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="Tabletop reach", padding=(16, 0)).pack(side="left")
        badge = "SIMULATION DEMO" if self.session.worker.backend == "demo" else "PHYSICAL ROBOT"
        ttk.Label(header, text=badge, style="Step.TLabel").pack(side="right")
        body = ttk.Frame(self.root, padding=(12, 0, 12, 6))
        body.pack(fill="both", expand=True)
        side = ttk.Frame(body, width=306, padding=(8, 4, 16, 4))
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        self.edit_controls = []
        ttk.Label(side, text="1  Select a target", style="Step.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(side, text="Origin: torso top center", style="Muted.TLabel").pack(anchor="w", pady=(0, 6))
        arms = ttk.Frame(side)
        arms.pack(fill="x")
        for arm, label in (("right", "Right arm"), ("left", "Left arm")):
            button = ttk.Radiobutton(arms, text=label, variable=self.arm, value=arm)
            button.pack(side="left", padx=(0, 12))
            self.edit_controls.append(button)
        fields = ttk.Frame(side)
        fields.pack(fill="x", pady=8)
        for row, (label, variable) in enumerate(zip(("X (cm)", "Y (cm)", "Z (cm)"), self.coordinates)):
            ttk.Label(fields, text=label).grid(row=row, column=0, sticky="w", pady=4)
            entry = ttk.Entry(fields, textvariable=variable, width=11, font=("DejaVu Sans", 12))
            entry.grid(row=row, column=1, sticky="e", padx=(10, 0), pady=4)
            self.edit_controls.append(entry)
        ttk.Label(side, text="Drag a TCP arrow along X, Y or Z.\nCoordinates are from the torso top.\nTable clicks select X/Y and keep Z.", style="Muted.TLabel", wraplength=275).pack(anchor="w")
        ttk.Button(side, text="Use current TCP", command=self.use_tcp).pack(fill="x", pady=(8, 12))
        ttk.Separator(side).pack(fill="x", pady=12)
        ttk.Label(side, text="2  Check the preview", style="Step.TLabel").pack(anchor="w")
        self.preview_button = ttk.Button(side, text="Preview reach", command=self.plan)
        self.preview_button.pack(fill="x", pady=8)
        self.edit_controls.append(self.preview_button)
        ttk.Label(side, text="Preview never moves the robot.", style="Muted.TLabel").pack(anchor="w")
        ttk.Label(side, text="3  Move when ready", style="Step.TLabel").pack(anchor="w", pady=(14, 0))
        self.execute_button = ttk.Button(side, text="Move robot", command=self.execute, style="Primary.TButton", state="disabled")
        self.execute_button.pack(fill="x", pady=8)
        ttk.Button(side, text="Stop", command=self.cancel, style="Stop.TButton").pack(fill="x")
        notebook = ttk.Notebook(side)
        notebook.pack(fill="both", expand=True, pady=(16, 0))
        saved = ttk.Frame(notebook, padding=8)
        adjustments = ttk.Frame(notebook, padding=8)
        details = ttk.Frame(notebook, padding=8)
        notebook.add(saved, text="Saved positions")
        notebook.add(adjustments, text="Adjust")
        notebook.add(details, text="Feedback")
        ttk.Label(adjustments, text="Step (cm)").grid(row=0, column=0, sticky="w")
        ttk.Combobox(adjustments, textvariable=self.increment, values=["0.1", "0.5", "1", "2", "5"], state="readonly", width=7).grid(row=0, column=1, columnspan=2, sticky="e")
        for row, axis in enumerate("XYZ", start=1):
            ttk.Label(adjustments, text=axis).grid(row=row, column=0, sticky="w", pady=2)
            for column, direction in enumerate((-1, 1), start=1):
                button = ttk.Button(adjustments, text="-" if direction < 0 else "+", width=7,
                                    command=lambda i=row - 1, d=direction: self.adjust(i, d))
                button.grid(row=row, column=column, padx=3, pady=2)
                self.edit_controls.append(button)
        self.names = ttk.Combobox(saved, textvariable=self.name, width=19)
        self.names.pack(fill="x", pady=(0, 8))
        self.save_button = ttk.Button(saved, text="Save this position", command=self.save, state="disabled")
        self.save_button.pack(fill="x", pady=3)
        self.load_button = ttk.Button(saved, text="Load position", command=self.load, state="disabled")
        self.load_button.pack(fill="x", pady=3)
        ttk.Label(details, textvariable=self.feedback_text, wraplength=240, justify="left").pack(anchor="w")
        ttk.Label(details, textvariable=self.detail_text, wraplength=240, justify="left", style="Muted.TLabel").pack(anchor="w", pady=12)
        scene = ttk.Frame(body)
        scene.pack(side="left", fill="both", expand=True)
        toolbar = ttk.Frame(scene, padding=(6, 0, 6, 8))
        toolbar.pack(fill="x")
        ttk.Label(toolbar, textvariable=self.view_label, style="Step.TLabel").pack(side="left")
        for mode in ("Live", "Preview"):
            ttk.Radiobutton(toolbar, text=mode, variable=self.display_mode, value=mode).pack(side="left", padx=12)
        ttk.Button(toolbar, text="Overview", command=lambda: self.view(True)).pack(side="right", padx=4)
        ttk.Button(toolbar, text="Head view", command=lambda: self.view(False)).pack(side="right", padx=4)
        self.canvas = tk.Canvas(scene, width=self.width, height=self.height, highlightthickness=0, background="#233043")
        self.canvas.pack(fill="both", expand=True)
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")
        self.photo = None
        self.canvas.bind("<ButtonPress-1>", self.pointer_down)
        self.canvas.bind("<B1-Motion>", self.pointer_move)
        self.canvas.bind("<ButtonRelease-1>", self.pointer_up)
        self.canvas.bind("<Motion>", self.pointer_hover)
        ttk.Label(scene, textvariable=self.message, style="Status.TLabel", wraplength=self.width - 20).pack(fill="x", pady=(8, 0))
        footer = ttk.Frame(self.root, padding=(18, 4, 18, 10))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.connection, style="Muted.TLabel").pack(side="left")
        if self.session.worker.backend == "can":
            ttk.Label(footer, text="Arm joints only. Motors release after motion/stop; support both arms.", style="Muted.TLabel").pack(side="right")

    def action(self, event, **details):
        self.feedback_log.write(json.dumps({"event": event, "timestamp_unix": time.time(), **details}) + "\n")
        self.feedback_log.flush()

    def invalidate(self, *_):
        self.session.plan = None
        self.preview_started = None
        self.execute_button.configure(state="disabled")
        self.save_button.configure(state="disabled")

    def use_tcp(self):
        if self.session.worker.state == "executing" or self.drag is not None:
            return
        self.session.sync()
        position_cm = self.session.reference.from_world(self.session.twin.tcp_position(self.arm.get()))
        for variable, value in zip(self.coordinates, position_cm):
            variable.set(f"{value:.2f}")

    def select_arm(self, *_):
        self.cancel_drag()
        self.invalidate()
        self.use_tcp()

    def adjust(self, axis, direction):
        target = self.target_values()
        if target is None:
            return
        step = float(self.increment.get())
        if not np.isfinite(step) or not 0 < step <= 10:
            raise ValueError("Adjustment must be in (0, 10] centimeters")
        self.coordinates[axis].set(f"{target[axis] + direction * step:.2f}")
        self.message.set("Target adjusted. Choose Preview reach.")

    def target_values(self):
        values = [variable.get().strip() for variable in self.coordinates]
        if not all(self.root.tk.getboolean(self.root.tk.call("string", "is", "double", "-strict", value)) for value in values):
            self.invalidate()
            self.message.set("Enter a number in each X, Y and Z field.")
            return None
        target = np.array([float(value) for value in values])
        if not np.isfinite(target).all():
            self.invalidate()
            self.message.set("X, Y and Z must be finite numbers.")
            return None
        return target

    def click(self, event):
        if self.session.worker.state == "executing":
            self.message.set("The robot is moving. Stop it before selecting another target.")
            return
        if self.frame is None:
            self.message.set("Wait for the simulation view to finish resizing.")
            return
        if not 0 <= event.x < self.width or not 0 <= event.y < self.height:
            self.message.set("Click inside the simulation view.")
            return
        camera = self.cameras[self.frame_index]
        sim = (self.session.twin, self.session.preview)[self.frame_index]
        if sim.tabletop_mount:
            surface = camera.intersect_plane(event.x, event.y, self.frame, sim.table_height)
            table = sim.model.geom("worktable").id
            if surface is None or np.any(np.abs(surface[:2] - sim.model.geom_pos[table, :2]) > sim.model.geom_size[table, :2]):
                self.invalidate()
                self.message.set("Choose a point on the tabletop.")
                return
        else:
            surface = camera.unproject(event.x, event.y, self.frame)
        relative = self.session.reference.from_world(surface)
        for variable, value in zip(self.coordinates[:2], relative[:2]):
            variable.set(f"{value:.2f}")
        target = self.target_values()
        if target is None:
            return
        self.action("target_selected", arm=self.arm.get(), target_relative_cm=target.tolist(), reference_site=self.session.reference.site)
        self.plan()

    def plan(self):
        if self.drag is not None:
            self.message.set("Release the mouse to preview this target.")
            return
        target = self.target_values()
        if target is None:
            return
        self.message.set("Planning a collision-free reach...")
        self.root.update_idletasks()
        result = self.session.prepare_relative(self.arm.get(), target)
        if result is None:
            self.preview_started = None
            self.execute_button.configure(state="disabled")
            self.save_button.configure(state="disabled")
            reason = self.session.rejection_reason
            self.detail_text.set(reason)
            if "chosen arm" in reason:
                self.message.set("Choose the other arm, or select a point on this arm's side of the table.")
            elif "IK" in reason or "workspace" in reason:
                self.message.set("Cannot reach this position. Try a closer point or increase Z.")
            else:
                self.message.set(f"Preview unavailable: {reason}")
            self.action("plan_rejected", arm=self.arm.get(), target_relative_cm=target.tolist(), reason=reason)
            return
        write_json(self.session_directory / f"plan_{time.time_ns()}.json", {
            "schema": "openarm.sync_plan.v1", "arm": result.trajectory.arm,
            "joint_names": joint_names(result.trajectory.arm), "target_m": result.target.tolist(),
            "reference_site": self.session.reference.site, "target_relative_cm": target.tolist(),
            "times_s": result.trajectory.times.tolist(), "positions_rad": result.trajectory.positions.tolist(),
            "velocities_rad_s": result.trajectory.velocities.tolist(), "source": result.trajectory.source,
            "gripper_commanded": False,
        })
        self.preview_started = time.monotonic()
        self.display_mode.set("Preview")
        self.execute_button.configure(state="normal")
        self.save_button.configure(state="disabled")
        self.detail_text.set(f"{result.trajectory.source.upper()} | {result.trajectory.duration:.1f} seconds")
        self.message.set("Preview ready. Check the motion, then choose Move robot.")

    def execute(self):
        if self.drag is not None:
            self.message.set("Release the mouse and check the preview before moving.")
            return
        self.session.execute()
        self.display_mode.set("Live")
        self.execute_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.message.set("Moving to the target. Live view follows robot feedback.")

    def cancel(self):
        self.cancel_drag()
        moving = self.session.worker.state == "executing"
        self.session.worker.cancel()
        self.invalidate()
        self.display_mode.set("Live")
        self.message.set("Stopping motion..." if moving else "Ready. Click the tabletop to preview a reach.")

    def view(self, overview):
        self.cancel_drag()
        for index, sim in enumerate((self.session.twin, self.session.preview)):
            look_height = 0.25 if sim.tabletop_mount else 0.48
            values = (np.array([1.55, 1.15, 1.25]), np.array([0.25, 0, look_height]), 48) if overview else self.camera_defaults[index]
            sim.set_camera(*values)

    def schedule_resize(self, event):
        if self.resize_pending is not None:
            self.root.after_cancel(self.resize_pending)
        width = min(1920, max(400, event.width))
        height = min(1080, max(320, event.height))
        self.resize_pending = self.root.after(150, lambda: self.resize_view(width, height))

    def resize_view(self, width, height):
        self.resize_pending = None
        if self.closed or (width, height) == (self.width, self.height):
            return
        self.cancel_drag()
        for camera in self.cameras:
            camera.close()
        for sim in (self.session.twin, self.session.preview):
            sim.model.vis.global_.offwidth = max(1920, width)
            sim.model.vis.global_.offheight = max(1080, height)
        self.width, self.height = width, height
        self.cameras = [OverheadCamera(sim.model, width, height, sim.table_height)
                        for sim in (self.session.twin, self.session.preview)]
        self.frame = None

    def pointer_hover(self, event):
        if self.drag is None:
            selected = hit_axis((event.x, event.y), self.gizmo_handles)
            self.canvas.configure(cursor="hand2" if selected is not None else "crosshair")

    def pointer_down(self, event):
        if self.session.worker.state == "executing" or self.frame is None:
            self.click(event)
            return
        axis = hit_axis((event.x, event.y), self.gizmo_handles)
        if axis is None:
            if self.gizmo_origin is not None and np.linalg.norm(np.array([event.x, event.y]) - self.gizmo_origin) < 13:
                self.message.set("Drag one of the colored X, Y or Z arrows.")
                return
            self.click(event)
            return
        camera = self.cameras[self.frame_index]
        drag = AxisDrag.begin(camera, self.frame, self.gizmo_world, self.session.reference.rotation[:, axis], axis,
                              (event.x, event.y))
        if drag is None:
            self.message.set("Choose Overview to drag this axis from a clearer angle.")
            return
        self.drag_values = [value.get() for value in self.coordinates]
        self.drag_plan = self.session.plan
        self.drag_preview_started = self.preview_started
        self.drag_pose = self.session.preview.data.qpos.copy()
        self.drag_start_pixel = np.array([event.x, event.y])
        self.drag = drag
        self.drag_target = drag.origin.copy()
        self.drag_changed = False
        self.preview_started = None
        self.display_mode.set("Live" if self.frame_index == 0 else "Preview")
        self.execute_button.configure(state="disabled")
        self.canvas.configure(cursor="fleur")
        self.message.set(f"Dragging {'XYZ'[axis]}. Release to preview; Esc cancels.")
        self.action("drag_started", axis="XYZ"[axis], origin_cm=self.session.reference.from_world(drag.origin).tolist())

    def pointer_move(self, event):
        if self.drag is None:
            return
        if not 0 <= event.x < self.width or not 0 <= event.y < self.height:
            return
        if not self.drag_changed and np.linalg.norm(np.array([event.x, event.y]) - self.drag_start_pixel) < 3:
            return
        target = self.drag.target((event.x, event.y))
        if target is None:
            self.message.set("Keep the drag inside the simulation view.")
            return
        self.drag_target = target
        self.drag_changed = True
        for variable, value in zip(self.coordinates, self.session.reference.from_world(target)):
            variable.set(f"{value:.2f}")

    def pointer_up(self, event):
        if self.drag is None:
            return
        self.pointer_move(event)
        changed = self.drag_changed
        axis = self.drag.index
        self.drag = None
        self.drag_target = None
        self.canvas.configure(cursor="crosshair")
        if changed:
            self.action("drag_finished", axis="XYZ"[axis], target_relative_cm=[float(v.get()) for v in self.coordinates])
            self.plan()
        else:
            self.preview_started = self.drag_preview_started
            self.execute_button.configure(state="normal" if self.session.plan is not None else "disabled")
            self.message.set("Drag an axis arrow to change the TCP target.")

    def cancel_drag(self):
        if self.drag is None:
            return
        axis = self.drag.index
        self.drag = None
        self.drag_target = None
        for variable, value in zip(self.coordinates, self.drag_values):
            variable.set(value)
        self.session.plan = self.drag_plan
        self.preview_started = self.drag_preview_started
        self.execute_button.configure(state="normal" if self.session.plan is not None else "disabled")
        completed = self.session.plan is not None and self.session.worker.completed_plan is self.session.plan
        self.save_button.configure(state="normal" if completed else "disabled")
        self.canvas.configure(cursor="crosshair")
        self.message.set("Drag cancelled. The robot position is unchanged.")
        self.action("drag_cancelled", axis="XYZ"[axis])

    def update_load_button(self, *_):
        self.load_button.configure(state="normal" if self.name.get() in self.saved_names else "disabled")

    def refresh_names(self):
        names = []
        if self.targets.exists():
            payload = json.loads(self.targets.read_text())
            if payload["schema"] != "openarm.sync_targets.v1":
                raise ValueError("Unsupported target file")
            names = list(payload["targets"])
        self.saved_names = set(names)
        self.names.configure(values=names)
        self.update_load_button()

    def save(self):
        if not self.name.get().strip():
            self.message.set("Enter a name for this position.")
            return
        self.session.save(self.targets, self.name.get())
        self.refresh_names()
        self.message.set(f"Saved position: {self.name.get()}")

    def load(self):
        if self.name.get() not in self.saved_names:
            self.message.set("Save a position before loading it.")
            return
        if self.session.worker.state == "executing":
            self.message.set("Stop the robot before loading a position.")
            return
        arm, target = self.session.load(self.targets, self.name.get())
        self.arm.set(arm)
        for variable, value in zip(self.coordinates, self.session.reference.from_world(target)):
            variable.set(f"{value:.2f}")
        self.plan()

    def visible_pixel(self, camera, point):
        if (self.frame.rotation.T @ (point - self.frame.position))[2] >= -1e-5:
            return None
        pixel = camera.project(point, self.frame, subpixel=True)
        return pixel if 0 <= pixel[0] < self.width and 0 <= pixel[1] < self.height else None

    def draw_gizmo(self, camera, sim):
        self.canvas.delete("gizmo")
        self.gizmo_handles = {}
        self.gizmo_origin = None
        if self.session.worker.state == "executing":
            return
        if self.drag is not None:
            anchor = self.drag_target
        elif self.frame_index == 1 and self.session.plan is not None:
            anchor = self.session.plan.target
        else:
            anchor = sim.tcp_position(self.arm.get())
        origin = self.visible_pixel(camera, anchor)
        if origin is None:
            return
        self.gizmo_world = anchor.copy()
        self.gizmo_origin = np.array(origin, dtype=float)
        self.canvas.create_oval(origin[0] - 7, origin[1] - 7, origin[0] + 7, origin[1] + 7,
                                fill="#ffffff", outline="#162337", width=2, tags="gizmo")
        for index, (axis, color) in enumerate((("X", "#ff6b76"), ("Y", "#61e49b"), ("Z", "#75b5ff"))):
            end = self.visible_pixel(camera, anchor + self.session.reference.rotation[:, index] * 0.10)
            if end is None or np.linalg.norm(np.array(end) - origin) < 18:
                continue
            self.gizmo_handles[index] = (self.gizmo_origin.copy(), np.array(end, dtype=float))
            width = 6 if self.drag is not None and self.drag.index == index else 4
            self.canvas.create_line(*origin, *end, fill=color, width=width, arrow="last", arrowshape=(12, 15, 6), tags="gizmo")
            self.canvas.create_oval(end[0] - 5, end[1] - 5, end[0] + 5, end[1] + 5,
                                    fill=color, outline="#142134", width=1, tags="gizmo")
            self.canvas.create_text(end[0], end[1] - 15, text=axis, fill=color,
                                    font=("DejaVu Sans", 13, "bold"), tags="gizmo")
        if self.session.plan is None:
            self.canvas.create_text(origin[0] + 14, origin[1] + 18, anchor="nw", text="Drag TCP axes",
                                    fill="white", font=("DejaVu Sans", 11, "bold"), tags="gizmo")

    def _overlay(self, camera, sim):
        self.canvas.delete("overlay")
        reference = self.session.reference
        origin = self.visible_pixel(camera, reference.origin_m)
        if origin is not None:
            self.canvas.create_oval(origin[0] - 6, origin[1] - 6, origin[0] + 6, origin[1] + 6,
                                    fill="#55dcec", outline="#102b3d", width=2, tags="overlay")
            self.canvas.create_text(origin[0] + 10, origin[1] - 14, anchor="sw", text="Torso top (0, 0, 0)",
                                    fill="#80f4ff", font=("DejaVu Sans", 11, "bold"), tags="overlay")
            for index, (axis, color) in enumerate((("+X", "#ff8b91"), ("+Y", "#81f0af"), ("+Z", "#9ac6ff"))):
                end = self.visible_pixel(camera, reference.origin_m + reference.rotation[:, index] * 0.12)
                if end is not None:
                    self.canvas.create_line(*origin, *end, fill=color, width=3, arrow="last", tags="overlay")
                    self.canvas.create_text(end[0], end[1] - 10, text=axis, fill=color, font=("DejaVu Sans", 11, "bold"), tags="overlay")
        if sim.tabletop_mount:
            for arm, y in (("Right (-Y)", -0.28), ("Left (+Y)", 0.28)):
                point = self.visible_pixel(camera, np.array([0.30, y, sim.table_height]))
                if point is not None:
                    self.canvas.create_text(*point, text=arm, fill="#ffffff", font=("DejaVu Sans", 12, "bold"), tags="overlay")
        if self.session.plan is not None:
            target = self.session.plan.target
            coordinates = reference.from_world(target)
            point = self.visible_pixel(camera, target)
            if point is None:
                return
            u, v = point
            self.canvas.create_oval(u - 8, v - 8, u + 8, v + 8, outline="#ffb340", width=3, tags="overlay")
            self.canvas.create_text(u + 14, v - 12, anchor="sw", text=f"TCP ({coordinates[0]:.1f}, {coordinates[1]:.1f}, {coordinates[2]:.1f}) cm",
                                    fill="#ffcb75", font=("DejaVu Sans", 12, "bold"), tags="overlay")

    def tick(self):
        if self.closed:
            return
        check_resources()
        feedback = self.session.sync()
        if self.drag is not None:
            self.session.preview.data.qpos[:] = self.drag_pose
            mujoco.mj_forward(self.session.preview.model, self.session.preview.data)
        elif self.preview_started is not None and self.session.plan is not None:
            self.session.show_preview(min(time.monotonic() - self.preview_started, self.session.plan.trajectory.duration))
        else:
            apply_feedback(self.session.preview, feedback)
        self.frame_index = 0 if self.display_mode.get() == "Live" else 1
        sim = (self.session.twin, self.session.preview)[self.frame_index]
        camera = self.cameras[self.frame_index]
        self.frame = camera.capture(sim.data)
        self.photo = ImageTk.PhotoImage(Image.fromarray(self.frame.rgb))
        self.canvas.itemconfigure(self.image_item, image=self.photo)
        self._overlay(camera, sim)
        self.draw_gizmo(camera, sim)
        self.view_label.set("LIVE ROBOT" if self.frame_index == 0 else "MOTION PREVIEW")
        moving = self.session.worker.state == "executing"
        for widget in self.edit_controls:
            widget.state(["disabled"] if moving or self.drag is not None else ["!disabled"])
        if moving or self.drag is not None:
            self.load_button.configure(state="disabled")
        else:
            self.update_load_button()
        arm = self.arm.get()
        tcp = self.session.twin.tcp_position(arm)
        self.connection.set(f"{self.session.worker.backend.upper()} connected | {self.session.preview.planner_backend.upper()} | "
                            f"{self.width} x {self.height} native | Feedback {(time.monotonic() - feedback.received_at) * 1000:.0f} ms")
        self.feedback_text.set(f"TCP from torso top (cm)\n{np.round(self.session.reference.from_world(tcp), 2).tolist()}\n\nJoints (rad)\n{np.round(feedback.positions[arm], 3).tolist()}")
        self.action("feedback", state=self.session.worker.state,
                    positions_rad={name: value.tolist() for name, value in feedback.positions.items()})
        if self.session.worker.report is not None and self.session.worker.report is not self.last_report:
            self.last_report = self.session.worker.report
            write_json(self.session_directory / f"execution_{time.time_ns()}.json", self.last_report)
            self.save_button.configure(state="normal")
            self.message.set("Reached target. Save this position or click another target.")
        if self.session.worker.state == "cancelled" and self.last_state != "cancelled":
            self.message.set("Motion stopped. Click a target to preview another reach.")
        self.last_state = self.session.worker.state
        self.root.after(60, self.tick)

    def callback_error(self, kind, error, traceback):
        self.error = error
        self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.cancel_drag()
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
    parser.add_argument("--record", type=Path, help="Record an independent MuJoCo demo without opening Tk")
    parser.add_argument("--smoke-test", action="store_true", help="Exercise the actual GUI click and execution flow")
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
                          "render_resolution": [console.width, console.height], "session": str(console.session_directory)}), flush=True)
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

            def begin():
                if time.monotonic() - started > 90:
                    raise TimeoutError("GUI view did not become ready")
                if console.frame is None:
                    console.root.after(100, begin)
                    return
                if not console.saved_names:
                    console.load()
                    if not console.load_button.instate(["disabled"]):
                        raise RuntimeError("Loading must remain disabled until a position is saved")
                if session.preview.tabletop_mount:
                    for variable, value in zip(console.coordinates, [40, -25, -30]):
                        variable.set(str(value))
                    pixel = console.cameras[console.frame_index].project(np.array([0.43, -0.24, 0.0]), console.frame)
                    console.canvas.event_generate("<Button-1>", x=pixel[0], y=pixel[1])
                    if abs(float(console.coordinates[2].get()) + 30.0) > 1e-6 or session.plan is None:
                        raise RuntimeError("Tabletop click must preserve torso-relative Z and produce a preview")
                else:
                    console.plan()
                if console.frame.rgb.shape[:2] != (console.height, console.width):
                    raise RuntimeError("The displayed frame must use the native viewport resolution")
                console.execute()
                console.root.after(100, exercise)

            console.root.after(300, begin)
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
