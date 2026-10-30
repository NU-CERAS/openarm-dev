import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import tkinter as tk
import uuid

import numpy as np

from ..camera import OverheadCamera
from ..compute_guard import budgeted
from ..paths import RUNS
from .session import RobotWorker, SyncSession
from .launch import stop_backend


@budgeted("openarm_sim.console.doctor")
def main() -> None:
    parser = argparse.ArgumentParser(description="Check the console environment using actual MuJoCo planning and execution")
    parser.add_argument("--gui", action="store_true", help="Also exercise both demo windows, preview, execution and automatic sync")
    parser.add_argument("--hardware", action="store_true", help="Verify the CAN Python binding without opening any bus")
    args = parser.parse_args()
    for executable in (("systemd-run", "systemctl", "nice") if sys.platform == "linux" else ("nice",)):
        if shutil.which(executable) is None:
            raise FileNotFoundError(f"Required executable not found: {executable}")
    if sys.platform == "linux":
        subprocess.run(["systemctl", "--user", "show-environment"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    if sys.platform == "linux" and not os.environ.get("DISPLAY"):
        raise RuntimeError("A graphical desktop with DISPLAY is required for the Tk console")
    root = tk.Tk()
    root.withdraw()
    root.update()
    root.destroy()
    namespace = f"/console_check_{uuid.uuid4().hex[:10]}"
    output = RUNS / "checks" / str(time.time_ns())
    output.mkdir(parents=True)
    backend_log = (output / "backend.log").open("w")
    process = subprocess.Popen([sys.executable, "-m", "openarm_sim.console.ros_backend", "--namespace", namespace],
                               stdout=backend_log, stderr=subprocess.STDOUT)
    worker = RobotWorker("ros", ros_namespace=namespace)
    session = None
    try:
        if not worker.ready.wait(10):
            raise TimeoutError("MuJoCo robot did not become ready")
        session = SyncSession(worker, "ompl")
        errors = []
        for target in ([40, -25, -30], [42, -25, -28]):
            measured = session.sync()
            plan = session.prepare_relative("right", target)
            if plan is None:
                raise RuntimeError(session.rejection_reason)
            np.testing.assert_allclose(plan.trajectory.positions[0], measured.positions["right"], atol=0.002)
            session.execute()
            deadline = time.monotonic() + 30
            while worker.completed_plan is not plan:
                worker.read()
                if time.monotonic() > deadline:
                    raise TimeoutError("Native MuJoCo reach exceeded 30 seconds")
                time.sleep(0.01)
            session.sync()
            error = float(np.linalg.norm(session.preview.tcp_position("right") - plan.target))
            if error > 0.015:
                raise RuntimeError(f"MuJoCo TCP target error exceeds 0.015 m: {error}")
            errors.append(error)
        with OverheadCamera(session.preview.model, 640, 480, session.preview.table_height) as camera:
            frame = camera.capture(session.preview.data)
            if frame.rgb.shape != (480, 640, 3):
                raise ValueError("Renderer returned an unexpected image shape")
    finally:
        try:
            session.close() if session is not None else worker.close()
        finally:
            stop_backend(process)
            backend_log.close()
    if args.hardware:
        import openarm_can

        print(f"CAN binding: {openarm_can.__file__}", flush=True)
    if args.gui:
        subprocess.run([sys.executable, "-m", "openarm_sim.console.launch", "--target", "demo", "--show-plant",
                        "--namespace", namespace, "--smoke-test",
                        "--targets", str(output / "targets.json")], check=True)
    print(json.dumps({"console_check": "passed", "python": sys.executable,
                      "consecutive_reaches": 2, "transport": "ros2", "tcp_errors_m": errors, "gui_checked": args.gui,
                      "hardware_binding_checked": args.hardware}), flush=True)


if __name__ == "__main__":
    main()
