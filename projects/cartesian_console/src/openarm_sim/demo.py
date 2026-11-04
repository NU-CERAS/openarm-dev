import argparse
from pathlib import Path
import time
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .camera import OverheadCamera
from .simulation import ReachSimulation


class ReachDemo:
    target_clearance = 0.14

    def __init__(self, width=1280, height=720, task="reach", planner="ompl", drum_asset=None):
        if task not in ("reach", "kit") or (task == "kit") != (drum_asset is not None):
            raise ValueError("Choose reach, or kit with --drum-asset manifest.json")
        self.is_kit = task == "kit"
        self.sim = ReachSimulation(planner=planner, drum_asset=drum_asset, worktable=not self.is_kit)
        self.camera = OverheadCamera(self.sim.model, width, height, self.sim.table_height)
        self.root = tk.Tk()
        self.root.title("OpenArm | Camera-guided reach")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.report_callback_exception = self.callback_error
        self.arm = tk.StringVar(value="right")
        self.message = tk.StringVar(value="Select an arm and click a surface. TCP clearance: 0.14 m.")
        controls = tk.Frame(self.root)
        controls.pack(fill="x")
        for arm in ("right", "left"):
            tk.Radiobutton(controls, text=f"{arm.title()} arm", variable=self.arm, value=arm).pack(side="left")
        tk.Button(controls, text="Cancel motion", command=self.sim.cancel).pack(side="left")
        tk.Button(controls, text="Reset simulation", command=self.sim.reset).pack(side="left")
        cid = self.sim.model.camera("overhead").id
        self.defaults = [*self.sim.model.cam_pos[cid], *self.sim.camera_look_at, self.sim.model.cam_fovy[cid]]
        self.variables = [tk.DoubleVar(value=float(value)) for value in self.defaults]
        panel = tk.LabelFrame(self.root, text="Camera position and look-at [m]")
        panel.pack(fill="x")
        for label, variable in zip(("x", "y", "z", "look_x", "look_y", "look_z", "fovy"), self.variables):
            tk.Label(panel, text=label).pack(side="left")
            tk.Entry(panel, textvariable=variable, width=7).pack(side="left")
        tk.Button(panel, text="Apply camera", command=self.apply_camera).pack(side="left")
        tk.Button(panel, text="Default camera", command=self.restore_camera).pack(side="left")
        if self.is_kit:
            tk.Button(panel, text="Kit overview", command=self.kit_camera).pack(side="left")
            tk.Label(self.root, text="Electronic Drum Set by SINNIK | CC BY 4.0").pack(fill="x")
        self.canvas = tk.Canvas(self.root, width=width, height=height, highlightthickness=0)
        self.canvas.pack()
        self.item = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.bind("<Button-1>", self.click)
        tk.Label(self.root, textvariable=self.message).pack(fill="x")
        self.feedback = tk.StringVar()
        tk.Label(self.root, textvariable=self.feedback).pack(fill="x")
        self.closed, self.error = False, None
        self.selected = None
        self.frame = self.camera.capture(self.sim.data)

    def apply_camera(self):
        values = [v.get() for v in self.variables]
        self.sim.set_camera(np.array(values[:3]), np.array(values[3:6]), values[6])

    def restore_camera(self):
        for variable, value in zip(self.variables, self.defaults):
            variable.set(value)
        self.apply_camera()

    def kit_camera(self):
        for variable, value in zip(self.variables, [1.75, 1.5, 1.45, 0.30, 0, 0.55, 52]):
            variable.set(value)
        self.apply_camera()

    def click(self, event):
        if self.is_kit:
            from .drum_asset import pick_pad

            selected = pick_pad(event.x, event.y, self.frame, self.camera.width, self.camera.height, self.sim.drum_asset["pads"])
            if selected is None:
                self.message.set("Select a visible drum-pad surface.")
                return
            pad, surface = selected
            target = surface + np.array(pad["normal"]) * self.target_clearance
        else:
            target = self.camera.unproject(event.x, event.y, self.frame) + [0, 0, self.target_clearance]
        result = self.sim.start_reach(self.arm.get(), target)
        self.selected = (event.x, event.y)
        self.message.set(f"Target [m]: {np.round(target, 4).tolist()}" if result.accepted else result.reason)

    def tick(self):
        for _ in range(round(1 / 30 / self.sim.model.opt.timestep)):
            self.sim.step()
        self.frame = self.camera.capture(self.sim.data)
        image = Image.fromarray(self.frame.rgb)
        draw = ImageDraw.Draw(image)
        if self.selected is not None:
            u, v = self.selected
            draw.ellipse((u - 7, v - 7, u + 7, v + 7), outline="#ffb62e", width=3)
        for arm, color in (("right", "#45e7ff"), ("left", "#a6ff60")):
            u, v = self.camera.project(self.sim.tcp_position(arm), self.frame)
            draw.line((u - 6, v, u + 6, v), fill=color, width=2)
            draw.line((u, v - 6, u, v + 6), fill=color, width=2)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.itemconfigure(self.item, image=self.photo)
        status = self.sim.status
        self.feedback.set(f"State: {status.state} | TCP [m]: {np.round(self.sim.tcp_position(self.arm.get()), 4).tolist()} | Target error: {status.error_m * 1000:.1f} mm")
        if self.smoke_started is not None:
            if status.state == "succeeded":
                self.close()
                return
            if status.state == "failed" or time.monotonic() - self.smoke_started > 25:
                raise RuntimeError(f"Reach check failed: {status}")
        self.root.after(20, self.tick)

    def callback_error(self, kind, error, traceback):
        self.error = error
        self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.camera.close()
        self.sim.close()
        self.root.destroy()

    def run(self, smoke=False):
        self.smoke_started = time.monotonic() if smoke else None
        if smoke:
            if self.is_kit:
                self.arm.set("left")
                pad = next(p for p in self.sim.drum_asset["pads"] if p["source_node"] == "Object_28")
                point = np.array(pad["center"])
            else:
                point = np.array([0.43, -0.24, self.sim.table_height])
            pixel = self.camera.project(point, self.frame)
            self.root.after(300, lambda: self.canvas.event_generate("<Button-1>", x=pixel[0], y=pixel[1]))
        self.root.after(1, self.tick)
        try:
            self.root.mainloop()
        finally:
            self.close()
        if self.error is not None:
            raise self.error


def main():
    parser = argparse.ArgumentParser(description="Reach a surface selected in the calibrated camera image")
    parser.add_argument("--scene", choices=["reach", "kit"], default="reach")
    parser.add_argument("--planner", choices=["ompl", "curobo", "mink"], default="ompl")
    parser.add_argument("--drum-asset", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    ReachDemo(task=args.scene, planner=args.planner, drum_asset=args.drum_asset).run(args.smoke_test)


if __name__ == "__main__":
    main()
