import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from ..camera import OverheadCamera
from ..simulation import ReachSimulation


class MujocoPlantWindow:
    def __init__(self, sim: ReachSimulation):
        self.sim = sim
        self.root = tk.Tk()
        self.root.title("OpenArm | MuJoCo test robot")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        ttk.Label(self.root, text="CONTROLLED MUJOCO ROBOT", font=("DejaVu Sans", 16, "bold"),
                  padding=12).pack(fill="x")
        ttk.Label(self.root, text="Independent physics session | Receives joint commands after Execute action",
                  padding=(12, 0, 12, 8)).pack(fill="x")
        self.image = ttk.Label(self.root)
        self.image.pack()
        self.joints = tk.StringVar()
        ttk.Label(self.root, textvariable=self.joints, padding=12).pack(fill="x")
        sim.set_camera(np.array([1.55, 1.15, 1.25]), np.array([0.25, 0, 0.25]), 48)
        self.camera = OverheadCamera(sim.model, 960, 600, sim.table_height)
        self.photo = None
        self.closed = False
        self.next_frame = 0.0
        self.sync()

    def sync(self):
        if self.closed:
            return
        now = time.monotonic()
        if now >= self.next_frame:
            frame = self.camera.capture(self.sim.data)
            self.photo = ImageTk.PhotoImage(Image.fromarray(frame.rgb), master=self.root)
            self.image.configure(image=self.photo)
            self.joints.set(" | ".join(f"{arm.title()} joints (rad): {np.round(self.sim.data.qpos[ids], 3).tolist()}"
                                       for arm, ids in self.sim.qpos_ids.items()))
            self.next_frame = now + 1 / 15
        self.root.update()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.camera.close()
        self.root.destroy()
