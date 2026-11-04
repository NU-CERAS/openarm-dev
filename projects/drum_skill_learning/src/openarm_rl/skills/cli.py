import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

from openarm_sim.compute_guard import budgeted

from .registry import SKILLS
from .runtime import SimDrumTools


@budgeted("openarm_rl.skills.cli")
def main():
    parser = argparse.ArgumentParser(description="Inspect or execute exported camera-free drum tools")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", choices=["sim", "ros"], default="sim")
    parser.add_argument("--namespace", default="")
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--physical", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-experimental", action="store_true")
    parser.add_argument("--tool", choices=[s.name for s in SKILLS.values()])
    parser.add_argument("--speed", type=float, default=.6)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    tools = SimDrumTools(args.manifest, args.allow_experimental)
    if args.list:
        print(json.dumps({"tools": tools.schemas(), "availability": tools.document["targets"]}), flush=True)
        return
    with ExitStack() as resources:
        ros = None
        if args.backend == "ros":
            if args.mapping is None:
                raise ValueError("ROS execution requires an explicit joint mapping")
            from ..deployment.ros_drum import RosDrumTool

            ros = RosDrumTool(args.namespace, args.mapping, args.physical)
            resources.callback(ros.close)

        def invoke(name, speed, execute):
            skill = next(s for s in SKILLS.values() if s.name == name)
            checkpoint = tools.checkpoint(skill.target)
            if ros is not None:
                return ros.execute(checkpoint, speed, execute)
            if not execute:
                from ..deployment.drum_actor import DrumActor

                actor = DrumActor(checkpoint)
                return {"tool": name, "execute": False, "ready_joints_rad": actor.reference.positions[0].tolist(),
                        "duration_s": actor.reference.duration}
            return tools.call(name, speed)

        if args.serve:
            print(json.dumps({"tools": tools.schemas(), "execution_enabled": args.execute}), flush=True)
            for line in sys.stdin:
                request = json.loads(line)
                execute = bool(request.get("execute", False))
                if execute and not args.execute:
                    raise ValueError("Start with --execute to enable robot execution")
                arguments = request.get("arguments", {})
                if arguments.get("timing_offset_s", 0) != 0:
                    raise ValueError("Timing offset is fixed at zero")
                result = invoke(request["tool"], arguments.get("target_speed_m_s", .6), execute)
                print(json.dumps({"id": request.get("id"), "result": result}), flush=True)
        elif args.tool:
            result = invoke(args.tool, args.speed, args.execute)
            if args.result:
                from openarm_tools.recording.storage import write_json

                write_json(args.result, result)
            print(json.dumps(result), flush=True)
        else:
            raise ValueError("Specify --list, --tool or --serve")


if __name__ == "__main__":
    main()
