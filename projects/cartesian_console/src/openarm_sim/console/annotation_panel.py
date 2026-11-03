from pathlib import Path
import tkinter as tk
from tkinter import ttk

import mujoco
import numpy as np

from ..annotation import DrumLayout


class AnnotationPanel:
    def __init__(self, console, path):
        self.console, self.path = console, Path(path)
        sim = console.session.twin
        self.layout = DrumLayout.load(self.path, sim) if self.path.is_file() else DrumLayout.from_simulation(sim)
        self.window = tk.Toplevel(console.root)
        self.window.title("OpenDrummer | Drum layout annotation")
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.report_callback_exception = console.callback_error
        self.target = tk.StringVar(value="snare")
        self.drag_enabled = tk.BooleanVar(value=False)
        self.values = [tk.StringVar() for _ in range(3)]
        self.status = tk.StringVar(value="Coordinates from torso top, in centimeters. Layout edits do not move the robot.")
        body = ttk.Frame(self.window, padding=16)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Drum center annotation", style="Step.TLabel").pack(anchor="w")
        selection = ttk.Combobox(body, textvariable=self.target, values=list(self.layout.document["targets"]), state="readonly")
        selection.pack(fill="x", pady=8)
        selection.bind("<<ComboboxSelected>>", lambda _: self.select())
        fields = ttk.Frame(body)
        fields.pack(fill="x")
        for index, axis in enumerate("XYZ"):
            ttk.Label(fields, text=f"{axis} (cm)").grid(row=index, column=0, padx=5, pady=5)
            ttk.Entry(fields, textvariable=self.values[index], width=12).grid(row=index, column=1)
        ttk.Button(body, text="Apply center coordinates", command=self.apply_fields).pack(fill="x", pady=6)
        ttk.Checkbutton(body, text="Drag selected drum in preview", variable=self.drag_enabled, command=self.select).pack(anchor="w", pady=6)
        ttk.Button(body, text="Place center at measured stick tip", command=self.use_tip).pack(fill="x", pady=4)
        ttk.Button(body, text="Capture measured ready joints", command=self.capture).pack(fill="x", pady=4)
        ttk.Button(body, text="Save layout and ready poses", command=self.save).pack(fill="x", pady=4)
        ttk.Label(body, textvariable=self.status, wraplength=350).pack(fill="x", pady=8)
        ttk.Label(body, text="Editable drum proxies represent the annotated heads.\nStand geometry retains its source placement.", wraplength=350).pack(fill="x")
        self.select()
        self.apply()

    def check_idle(self):
        if self.console.session.worker.state == "executing":
            raise RuntimeError("Wait for robot execution before editing the layout")

    def select(self):
        self.check_idle()
        self.console.cancel_drag()
        self.console.invalidate()
        item = self.layout.document["targets"][self.target.get()]
        if self.console.arm.get() != item["arm"]:
            self.console.arm.set(item["arm"])
        for variable, value in zip(self.values, item["position_cm"]):
            variable.set(f"{value:.3f}")
        ready = item.get("ready")
        self.status.set(f"{self.target.get()} | {item['arm']} arm | ready pose: {ready['source'] if ready else 'not recorded'}")

    def apply(self):
        self.check_idle()
        self.console.invalidate()
        for sim in (self.console.session.twin, self.console.session.preview):
            self.layout.apply(sim.model, sim.scene_report)
            mujoco.mj_forward(sim.model, sim.data)
            sim.motion_planner = None
        self.console.frame = None

    def apply_fields(self):
        self.layout.set_position(self.target.get(), [float(v.get()) for v in self.values])
        self.apply()
        self.select()

    def drag_to(self, world):
        self.layout.set_position(self.target.get(), (np.array(world) - self.layout.document["origin_m"]) * 100)
        self.apply()
        self.select()

    def use_tip(self):
        self.console.session.sync()
        arm = self.layout.document["targets"][self.target.get()]["arm"]
        self.drag_to(self.console.session.twin.tcp_position(arm))

    def capture(self):
        self.check_idle()
        feedback = self.console.session.sync()
        joints = np.concatenate([feedback.positions[arm] for arm in ("left", "right")])
        source = "physical_feedback" if getattr(self.console.session.worker, "physical", False) else "simulation_feedback"
        self.layout.capture_ready(self.target.get(), joints, source)
        self.select()

    def save(self):
        self.check_idle()
        self.layout.save(self.path)
        self.status.set(f"Saved: {self.path}")

    def close(self):
        self.console.cancel_drag()
        self.console.annotation = None
        self.window.destroy()
