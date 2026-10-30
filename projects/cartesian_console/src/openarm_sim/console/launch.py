import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from openarm_tools.ros_execution import RosExecutionClient

from ..compute_guard import budgeted
from ..paths import RUNS


def stop_backend(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise TimeoutError("ROS execution backend did not stop") from None
    if process.returncode not in (0, -signal.SIGINT):
        raise RuntimeError(f"ROS execution backend exited with code {process.returncode}")


@budgeted("openarm_sim.console.launch")
def main():
    parser = argparse.ArgumentParser(description="Start a ROS 2 execution backend and the Cartesian Console")
    parser.add_argument("--target", choices=["demo", "can", "remote"], default="demo")
    parser.add_argument("--namespace", help="ROS namespace; owned demo launches use a unique namespace by default")
    parser.add_argument("--hardware-config", type=Path)
    parser.add_argument("--supported", action="store_true")
    parser.add_argument("--show-plant", action="store_true")
    parser.add_argument("--drum-asset", type=Path)
    args, console_args = parser.parse_known_args()
    if args.namespace is None:
        args.namespace = f"/openarm_demo_{uuid.uuid4().hex[:10]}" if args.target == "demo" else ""
    if args.target != "demo" and (args.show_plant or "--smoke-test" in console_args):
        raise ValueError("Automatic GUI checks and the plant window require the demo target")
    if args.target == "can" and (args.hardware_config is None or not args.supported):
        raise ValueError("CAN requires --hardware-config and --supported")
    if args.target == "can":
        from .session import hardware_configuration

        hardware_configuration(args.hardware_config)
    with ExitStack() as resources:
        if args.target != "remote":
            directory = RUNS / "sync/ros" / str(time.time_ns())
            directory.mkdir(parents=True)
            log = resources.enter_context((directory / "backend.log").open("w"))
            command = [sys.executable, "-u", "-m", "openarm_sim.console.ros_backend", "--backend", args.target,
                       "--namespace", args.namespace]
            if args.show_plant:
                command.append("--show-plant")
            if args.hardware_config:
                command += ["--hardware-config", str(args.hardware_config)]
            if args.supported:
                command.append("--supported")
            if args.drum_asset:
                command += ["--drum-asset", str(args.drum_asset)]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            resources.callback(stop_backend, process)
            print(json.dumps({"execution_backend_log": str(directory / "backend.log")}), flush=True)
        with ExitStack() as connection:
            client = RosExecutionClient(args.namespace)
            connection.callback(client.close)
            client.wait_ready(timeout=15)
        command = [sys.executable, "-u", "-m", "openarm_sim.console.app", "--namespace", args.namespace, *console_args]
        if args.target == "can":
            command.append("--physical")
        if args.drum_asset:
            command += ["--drum-asset", str(args.drum_asset)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
