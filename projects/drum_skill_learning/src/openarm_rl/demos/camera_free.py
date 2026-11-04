import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from openarm_sim.compute_guard import budgeted
from openarm_sim.console.launch import stop_backend
from openarm_tools.ros_execution import RosExecutionClient

from ..paths import RUNS


@budgeted("openarm_rl.demos.camera_free")
def main():
    parser = argparse.ArgumentParser(description="Start a ROS MuJoCo robot and preview or execute an exported drum skill")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tool", default="strike_snare")
    parser.add_argument("--speed", type=float, default=.6)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-experimental", action="store_true")
    args = parser.parse_args()
    namespace = f"/drum_demo_{uuid.uuid4().hex[:10]}"
    output = RUNS / "demos" / str(time.time_ns())
    output.mkdir(parents=True)
    document = json.loads(args.manifest.read_text())
    scene_path = Path(document["scene_directory"])
    if not scene_path.is_absolute():
        scene_path = args.manifest.resolve().parent / scene_path
    metadata = json.loads((scene_path / "scene.json").read_text())
    layout = None
    if "annotations" in metadata["scene"]:
        layout = scene_path / "layout.json"
        if not layout.is_file():
            raise FileNotFoundError("Annotated demo requires its original layout.json")
    with ExitStack() as resources:
        log = resources.enter_context((output / "backend.log").open("w"))
        command = [sys.executable, "-u", "-m", "openarm_sim.console.ros_backend", "--model", "opendrummer", "--namespace", namespace]
        if layout:
            command += ["--layout", str(layout)]
        backend = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        resources.callback(stop_backend, backend)
        client = RosExecutionClient(namespace)
        try:
            client.wait_ready(timeout=15)
        finally:
            client.close()
        console = None
        if args.gui:
            command = [sys.executable, "-m", "openarm_sim.console.app", "--model", "opendrummer", "--namespace", namespace]
            if layout:
                command += ["--layout", str(layout)]
            console = subprocess.Popen(command)
            resources.callback(stop_backend, console)
        mapping = Path(__file__).resolve().parents[3] / "configs/deployment/drum_joint_mapping.example.json"
        command = [sys.executable, "-m", "openarm_rl.skills.cli", "--manifest", str(args.manifest.resolve()), "--backend", "ros",
                   "--namespace", namespace, "--mapping", str(mapping), "--tool", args.tool, "--speed", str(args.speed),
                   "--result", str(output / "result.json")]
        if args.execute:
            command.append("--execute")
        if args.allow_experimental:
            command.append("--allow-experimental")
        subprocess.run(command, check=True)
        print(json.dumps({"result": str(output / "result.json"), "backend_log": str(output / "backend.log")}), flush=True)
        if console is not None:
            console.wait()


if __name__ == "__main__":
    main()
