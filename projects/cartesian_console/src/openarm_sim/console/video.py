import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from ..camera import OverheadCamera
from ..compute_guard import check_resources
from .session import RobotWorker, SyncSession


def record(output: Path, planner: str, scene: Path | None):
    output.mkdir(parents=True, exist_ok=True)
    worker = RobotWorker(scene=scene)
    session = None
    cameras = []
    try:
        if not worker.ready.wait(10):
            raise TimeoutError("Demo plant did not become ready")
        worker.read()
        session = SyncSession(worker, planner, scene)
        for sim in (session.twin, session.preview):
            sim.set_camera(np.array([1.55, 1.15, 1.25]), np.array([0.25, 0, 0.25 if sim.tabletop_mount else 0.48]), 48)
            cameras.append(OverheadCamera(sim.model, 640, 360, sim.table_height))
        initial = session.sync().positions["right"].copy()
        target = np.array([0.43, -0.24, 0.44])
        if scene is not None:
            target = np.array([0.43, -0.34, 0.56])
        plan = session.prepare("right", target)
        if plan is None:
            raise RuntimeError(session.rejection_reason)
        trace = []
        started = time.monotonic()
        preview_duration = min(plan.trajectory.duration, 4.0)
        execution_started = None
        frame_index = 0
        with imageio.get_writer(output / "sync_demo.mp4", fps=20, codec="libx264", macro_block_size=1,
                                output_params=["-movflags", "+faststart"]) as writer:
            while True:
                check_resources()
                elapsed = time.monotonic() - started
                if elapsed > 90:
                    raise TimeoutError("Demo recording exceeded 90 seconds")
                if elapsed < preview_duration:
                    phase = "PREVIEW ONLY - robot holds its position"
                    session.show_preview(elapsed / preview_duration * plan.trajectory.duration)
                else:
                    if execution_started is None:
                        session.execute()
                        execution_started = time.monotonic()
                    session.show_preview(time.monotonic() - execution_started)
                    phase = "EXECUTING - live twin follows measured joints"
                feedback = session.sync()
                trace.append({"elapsed_s": elapsed, "state": worker.state,
                              "joint_positions_rad": feedback.positions["right"].tolist(),
                              "sim_tcp_m": session.twin.tcp_position("right").tolist()})
                frames = [camera.capture(sim.data).rgb for camera, sim in zip(cameras, (session.twin, session.preview))]
                canvas = Image.new("RGB", (1280, 432), "#14202c")
                canvas.paste(Image.fromarray(np.concatenate(frames, axis=1)), (0, 40))
                draw = ImageDraw.Draw(canvas)
                draw.text((16, 12), "LIVE JOINT FEEDBACK | Independent MuJoCo robot", fill="white")
                draw.text((656, 12), f"TRAJECTORY PREVIEW | {planner}", fill="white")
                draw.text((16, 410), f"{phase} | Target [m]: {target.tolist()}", fill="white")
                writer.append_data(np.asarray(canvas))
                frame_index += 1
                if worker.state == "completed":
                    break
                time.sleep(max(0.0, started + frame_index / 20 - time.monotonic()))
        session.save(output / "targets.json", "demo_reach")
        final = session.sync()
        error = float(np.linalg.norm(session.twin.tcp_position("right") - target))
        if error > 0.015:
            raise RuntimeError(f"Demo TCP error exceeds 0.015 m: {error}")
        report = {"backend": "independent_mujoco_physics", "planner": planner, "target_m": target.tolist(),
                  "initial_joint_positions_rad": initial.tolist(), "final_joint_positions_rad": final.positions["right"].tolist(),
                  "tcp_error_m": error, "trajectory_duration_s": plan.trajectory.duration, "execution": worker.report,
                  "trace": trace}
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "trace"}), flush=True)
    finally:
        if session is not None:
            session.close()
        else:
            worker.close()
        for camera in cameras:
            camera.close()
