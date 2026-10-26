import argparse
import json
import time
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .camera import OverheadCamera
from .pickup import StickPickupEnv
from .residual_drumming import ResidualDrumStrikeEnv
from .simulation import ReachSimulation


class ReachDemo:
    target_clearance = 0.14

    def __init__(self, width: int = 1280, height: int = 720, drumming: bool = False, task: str = "reach", planner: str = "curobo", drum_asset: Path | None = None):
        if task == "kit" and drum_asset is None:
            raise ValueError("The kit scene requires --drum-asset manifest.json")
        if drum_asset is not None and task != "kit":
            raise ValueError("--drum-asset requires the kit scene")
        if drumming or task in ("drum", "pickup"):
            self.drum_env = StickPickupEnv()
        else:
            self.drum_env = ResidualDrumStrikeEnv(randomization=0.0) if task == "strike" else None
        if self.drum_env is not None:
            self.drum_env.reset(seed=0)
            self.sim = self.drum_env.sim
        else:
            self.sim = ReachSimulation(planner=planner, drum_asset=drum_asset, worktable=task != "kit")
        camera_id = self.sim.model.camera("overhead").id
        self.default_camera = [*self.sim.model.cam_pos[camera_id], *self.sim.camera_look_at, self.sim.model.cam_fovy[camera_id]]
        self.is_kit = task == "kit"
        self.camera = OverheadCamera(self.sim.model, width, height, self.sim.table_height)
        self.root = tk.Tk()
        self.root.title("OpenArm | Electronic drum set" if self.is_kit else "OpenArm | Drumstick pickup" if isinstance(self.drum_env, StickPickupEnv) else "OpenArm | Strike environment" if self.drum_env is not None else "OpenArm | Camera-guided reach")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.report_callback_exception = self._callback_error
        self.arm = tk.StringVar(value="right")
        self.message = tk.StringVar(value="Click the tabletop to select a target. TCP clearance: 0.14 m.")
        if self.is_kit:
            self.message.set("Select an arm and click a drum pad to reach 0.14 m above its fitted surface.")
        self.feedback = tk.StringVar()
        controls = tk.Frame(self.root, padx=10, pady=8)
        controls.pack(fill="x")
        tk.Radiobutton(controls, text="Right arm", variable=self.arm, value="right").pack(side="left")
        tk.Radiobutton(controls, text="Left arm", variable=self.arm, value="left").pack(side="left")
        tk.Button(controls, text="Cancel motion", command=self.cancel).pack(side="left", padx=8)
        tk.Button(controls, text="Reset simulation", command=self.reset).pack(side="left")
        self.reference_actions = None
        self.reference_label = "pickup" if isinstance(self.drum_env, StickPickupEnv) else "strike"
        if isinstance(self.drum_env, StickPickupEnv):
            self.reference_button = tk.Button(controls, text="Reference pickup (controller)", command=self.start_reference_pickup)
            self.reference_button.pack(side="left", padx=8)
        elif isinstance(self.drum_env, ResidualDrumStrikeEnv):
            self.reference_button = tk.Button(controls, text="Reference strike (zero arm residual)", command=self.start_reference_strike)
            self.reference_button.pack(side="left", padx=8)
        tk.Label(controls, text="1280x720 | 4x MSAA").pack(side="right")
        camera_controls = tk.LabelFrame(self.root, text="Camera position and look-at point [m]", padx=8, pady=4)
        camera_controls.pack(fill="x", padx=8)
        camera_id = self.sim.model.camera("overhead").id
        camera_values = [*self.sim.model.cam_pos[camera_id], *self.sim.camera_look_at, self.sim.model.cam_fovy[camera_id]]
        self.camera_variables = [tk.DoubleVar(value=float(value)) for value in camera_values]
        for name, variable in zip(["x", "y", "z", "look_x", "look_y", "look_z", "fovy"], self.camera_variables):
            tk.Label(camera_controls, text=name).pack(side="left")
            tk.Entry(camera_controls, textvariable=variable, width=7).pack(side="left", padx=3)
        tk.Button(camera_controls, text="Apply camera", command=self.apply_camera).pack(side="left", padx=8)
        tk.Button(camera_controls, text="Robot head camera" if self.is_kit else "Official head camera", command=self.restore_default_camera).pack(side="left")
        if self.is_kit:
            tk.Button(camera_controls, text="Kit overview", command=self.restore_kit_camera).pack(side="left", padx=5)
        self.canvas = tk.Canvas(self.root, width=width, height=height, highlightthickness=0)
        self.canvas.pack()
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.bind("<Button-1>", self._click)
        tk.Label(self.root, textvariable=self.message, anchor="w", padx=10).pack(fill="x")
        tk.Label(self.root, textvariable=self.feedback, anchor="w", padx=10, pady=6).pack(fill="x")
        if self.is_kit:
            tk.Label(self.root, text="Electronic Drum Set by SINNIK | CC BY 4.0 | Converted for MuJoCo", anchor="w", padx=10).pack(fill="x")
        self.frame = self.camera.capture(self.sim.data)
        self.selected_pixel: tuple[int, int] | None = None
        self.closed = False
        self.error: BaseException | None = None
        self._next_tick = time.monotonic()
        self._paint()

    def _click(self, event) -> None:
        if self.reference_actions is not None:
            self.message.set("Reference controller is running. Wait or cancel the motion.")
            return
        if self.is_kit:
            from .drum_asset import pick_pad

            selected = pick_pad(event.x, event.y, self.frame, self.camera.width, self.camera.height, self.sim.drum_asset["pads"])
            if selected is None:
                self.message.set("Select a visible drum-pad surface.")
                return
            pad, surface = selected
            target = surface + np.array(pad["normal"]) * self.target_clearance
        else:
            surface = self.camera.unproject(event.x, event.y, self.frame)
            target = surface + [0, 0, self.target_clearance]
        result = self.sim.start_reach(self.arm.get(), target)
        if result.accepted:
            self.selected_pixel = (event.x, event.y)
            self.message.set(f"Target [m]: {np.round(target, 4).tolist()} | Planner: {self.sim.planner_backend}")
        else:
            self.message.set(result.reason)
        # IK 规划期间不推进物理仿真，执行轨迹使用固定仿真步长。
        self._next_tick = time.monotonic()

    def cancel(self) -> None:
        self.reference_actions = None
        self.sim.cancel()
        self.message.set("Motion cancelled. Holding the current position.")

    def apply_camera(self) -> None:
        values = [variable.get() for variable in self.camera_variables]
        self.sim.set_camera(np.array(values[:3]), np.array(values[3:6]), values[6])
        self.selected_pixel = None
        self._paint()
        self.message.set("Camera updated. Select a target in the current image.")

    def restore_default_camera(self) -> None:
        for variable, value in zip(self.camera_variables, self.default_camera):
            variable.set(value)
        self.apply_camera()

    def restore_kit_camera(self) -> None:
        values = [1.75, 1.5, 1.45, 0.30, 0, 0.55, 52]
        for variable, value in zip(self.camera_variables, values):
            variable.set(value)
        self.apply_camera()

    def reset(self) -> None:
        self.reference_actions = None
        if self.drum_env is not None:
            self.drum_env.reset(seed=0)
        else:
            self.sim.reset()
        self.selected_pixel = None
        self.message.set("Simulation reset. Click the tabletop to select a target.")

    def start_reference_pickup(self) -> None:
        if not isinstance(self.drum_env, StickPickupEnv):
            raise RuntimeError("Reference pickup requires the pickup scene")
        if not self.drum_env.active:
            self.message.set("Reset simulation before starting another pickup.")
            return
        self.reference_actions = self.drum_env.reference_pickup_actions()
        self.message.set("Reference controller: lower, close the gripper, lift, and hold. No learning is running.")

    def start_reference_strike(self) -> None:
        if not isinstance(self.drum_env, ResidualDrumStrikeEnv):
            raise RuntimeError("Reference strike requires the strike scene")
        if self.drum_env.env.elapsed > 0:
            self.message.set("Reset simulation before starting another strike.")
            return

        def actions():
            for _ in range(101):
                action = np.zeros(8, dtype=np.float32)
                action[7] = -1
                yield action

        self.reference_actions = actions()
        self.message.set("Executing the nominal strike reference with zero arm residual. No learning is running.")

    def _paint(self) -> None:
        self.frame = self.camera.capture(self.sim.data)
        image = Image.fromarray(self.frame.rgb)
        draw = ImageDraw.Draw(image)
        if self.selected_pixel is not None:
            u, v = self.selected_pixel
            draw.ellipse((u - 7, v - 7, u + 7, v + 7), outline="#ffb62e", width=3)
        for arm, color in [("right", "#45e7ff"), ("left", "#a6ff60")]:
            u, v = self.camera.project(self.sim.tcp_position(arm), self.frame)
            draw.line((u - 6, v, u + 6, v), fill=color, width=2)
            draw.line((u, v - 6, u, v + 6), fill=color, width=2)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.itemconfigure(self.image_item, image=self.photo)
        status = self.sim.status
        labels = {"idle": "Idle", "running": "Running", "succeeded": "Reached", "failed": "Failed", "cancelled": "Cancelled"}
        position = np.round(self.sim.tcp_position(self.arm.get()), 4).tolist()
        self.feedback.set(
            f"State: {labels[status.state]} | TCP [m]: {position} | "
            f"Target error: {status.error_m * 1000:.1f} mm | Simulation time: {self.sim.data.time:.2f} s"
        )
        if status.reason:
            self.message.set(status.reason)
        if isinstance(self.drum_env, StickPickupEnv):
            info = self.drum_env.info()
            self.feedback.set(self.feedback.get() + f" | Finger contacts: {info['gripper_contacts']}/2 | Stick lift: {info['lift_height_m'] * 1000:.1f} mm")
        elif isinstance(self.drum_env, ResidualDrumStrikeEnv):
            info = self.drum_env.env._info()
            self.feedback.set(self.feedback.get() + f" | Impacts: {info['hit_count']} | Grasp retained: {info['grasped']} | Task success: {info['is_success']}")

    def _tick(self) -> None:
        now = time.monotonic()
        if now >= self._next_tick:
            frame_period = 1 / 30
            if self.reference_actions is not None:
                frame_period = 0.02
                action = next(self.reference_actions, None)
                if action is None:
                    self.reference_actions = None
                else:
                    _, _, terminated, truncated, info = self.drum_env.step(action)
                    if terminated or truncated:
                        self.reference_actions = None
                        self.message.set(
                            f"Reference {self.reference_label} completed | Task success: {info['is_success']}"
                        )
            else:
                substeps = round(frame_period / self.sim.model.opt.timestep)
                for _ in range(substeps):
                    self.sim.step()
            self._paint()
            self._next_tick += frame_period
            if now - self._next_tick > 0.2:
                self._next_tick = now
        self.root.after(2, self._tick)

    def _callback_error(self, exc_type, exc_value, traceback) -> None:
        self.error = exc_value
        self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.drum_env is not None:
            self.drum_env.close()
        else:
            self.sim.close()
        self.camera.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.after(1, self._tick)
        try:
            self.root.mainloop()
        finally:
            self.close()
        if self.error is not None:
            raise self.error

    def schedule_smoke_test(self) -> None:
        started = time.monotonic()

        def select_target():
            if self.is_kit:
                rejected_pad = next(pad for pad in self.sim.drum_asset["pads"] if pad["source_node"] == "Object_33")
                self.arm.set("right")
                rejected_pixel = self.camera.project(np.array(rejected_pad["center"]), self.frame)
                self.canvas.event_generate("<Button-1>", x=rejected_pixel[0], y=rejected_pixel[1])
                if self.closed or self.sim.status.state != "idle" or "IK" not in self.message.get():
                    raise RuntimeError("GUI did not reject the infeasible IK request while remaining ready")
                self.arm.set("left")
                pad = next(pad for pad in self.sim.drum_asset["pads"] if pad["source_node"] == "Object_28")
                pixel = self.camera.project(np.array(pad["center"]), self.frame)
                self.canvas.event_generate("<Button-1>", x=pixel[0], y=pixel[1])
                self.root.after(100, check_execution)
                return
            if isinstance(self.drum_env, (StickPickupEnv, ResidualDrumStrikeEnv)):
                self.reference_button.invoke()
                self.root.after(100, check_execution)
                return
            pixel = self.camera.project(np.array([0.43, -0.24, self.sim.table_height]), self.frame)
            self.canvas.event_generate("<Button-1>", x=pixel[0], y=pixel[1])
            self.root.after(100, check_execution)

        def check_execution():
            if isinstance(self.drum_env, ResidualDrumStrikeEnv):
                info = self.drum_env.env._info()
                if not self.drum_env.env._episode_active:
                    if not info["hit_count"] or not info["grasped"]:
                        raise RuntimeError(f"GUI strike did not demonstrate retained-grasp contact: {info}")
                    print(json.dumps({"state": "strike_contact_demonstrated", **info}))
                    self.close()
                elif time.monotonic() - started > 25:
                    raise RuntimeError(f"GUI strike check did not complete: {info}")
                else:
                    self.root.after(100, check_execution)
                return
            if isinstance(self.drum_env, StickPickupEnv):
                info = self.drum_env.info()
                if info["is_success"]:
                    print(json.dumps({"state": "pickup_succeeded", **info}))
                    self.close()
                elif not self.drum_env.active or time.monotonic() - started > 25:
                    raise RuntimeError(f"GUI pickup check did not complete: {info}")
                else:
                    self.root.after(100, check_execution)
                return
            status = self.sim.status
            if status.state == "succeeded":
                print(json.dumps({"state": status.state, "error_m": status.error_m, "sim_time": self.sim.data.time}))
                self.close()
            elif status.state == "failed" or time.monotonic() - started > 20:
                raise RuntimeError(f"GUI reach check did not complete: {status}")
            else:
                self.root.after(100, check_execution)

        self.root.after(200, select_target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Click a calibrated head-camera image to execute OpenArm reach")
    parser.add_argument("--smoke-test", action="store_true", help="Click the real GUI, verify physical reach, and close")
    parser.add_argument("--scene", choices=["reach", "drum", "pickup", "strike", "kit"], default="reach", help="Choose a task or the imported electronic kit")
    parser.add_argument("--drum-asset", type=Path)
    parser.add_argument("--planner", choices=["curobo", "mink"], default="curobo")
    args = parser.parse_args()
    demo = ReachDemo(task=args.scene, planner=args.planner, drum_asset=args.drum_asset)
    if args.smoke_test:
        demo.schedule_smoke_test()
    demo.run()


if __name__ == "__main__":
    main()
