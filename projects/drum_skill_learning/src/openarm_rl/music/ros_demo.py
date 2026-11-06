import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import uuid

from openarm_sim.compute_guard import budgeted
from openarm_sim.console.launch import stop_backend

from ..deployment.rhythm_actor import RhythmActorRuntime
from ..deployment.ros_music import RosMusicTool
from ..paths import PROJECT_ROOT


@budgeted("openarm_rl.music.ros_demo")
def main():
    parser = argparse.ArgumentParser(description="Start an owned ROS MuJoCo robot and explicitly execute MIDI playback")
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--midi", type=Path, required=True)
    parser.add_argument("--percussion-mapping", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        raise ValueError("Demo repeat count must be between one and ten")
    args.output.mkdir(parents=True, exist_ok=False)
    namespace = f"/midi_demo_{uuid.uuid4().hex[:10]}"
    with ExitStack() as resources:
        log = resources.enter_context((args.output / "backend.log").open("w"))
        command = [sys.executable, "-u", "-m", "openarm_sim.console.ros_backend", "--model", "opendrummer", "--namespace", namespace]
        layout = args.scene_directory / "layout.json"
        if layout.is_file():
            command += ["--layout", str(layout)]
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        resources.callback(stop_backend, process)
        runtime = RhythmActorRuntime(args.actor, args.scene_directory, args.profiles, args.midi, mapping=args.percussion_mapping)
        tool = RosMusicTool(namespace, PROJECT_ROOT / "configs/deployment/drum_joint_mapping.example.json")
        resources.callback(tool.close)
        results = []
        for index in range(args.repeats):
            try:
                results.append(tool.perform(runtime, args.execute))
            finally:
                (args.output / f"trace_{index:02d}.json").write_text(json.dumps(runtime.trace, indent=2))
        (args.output / "result.json").write_text(json.dumps({"namespace": namespace, "performances": results}, indent=2))
        print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
