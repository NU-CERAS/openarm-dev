import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import signal
import time

import jsonlines
import mujoco
import numpy as np

from ..compute_guard import budgeted, check_resources, interrupt
from ..execution import ArmCommand
from ..paths import RUNS
from ..simulation import ReachSimulation
from .adapter import PolicyAdapter, PolicySettings, SimulationPolicyInterface
from .can import CanPolicyInterface, GripperCalibration


def load_records(path: Path, episode: int, mode: str) -> tuple[list[dict], ArmCommand]:
    with jsonlines.open(path) as reader:
        records = [record for record in reader if record["episode"] == episode]
    if not records or [record["step"] for record in records] != list(range(len(records))):
        raise ValueError("Policy trace must contain one complete episode with consecutive steps starting at zero")
    first = records[0]
    if "state" in first:
        state = np.asarray(first["state"], dtype=float)
        if state.shape != (9,):
            raise ValueError("Visual pickup trace requires its nine-dimensional robot state")
        initial = ArmCommand(state[:7], float(state[7]))
    else:
        state = np.asarray(first["oracle_observation"], dtype=float)
        if state.shape != (51,):
            raise ValueError("Strike trace requires its 51-dimensional oracle state")
        initial = ArmCommand(state[:7], float(state[14]))
    for record in records:
        key = "command" if mode == "absolute" else "action"
        values = np.asarray(record[key], dtype=float)
        if values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError("Recorded policy actions must contain eight finite values")
        if mode != "absolute" and np.any(np.abs(values) > 1):
            raise ValueError("Recorded normalized actions exceed [-1, 1]")
        record["policy_action"] = values
        if mode == "strike_residual":
            state = np.asarray(record["oracle_observation"], dtype=float)
            if state.shape != (51,) or not np.isfinite(state).all():
                raise ValueError("Invalid recorded oracle strike state")
            record["nominal"] = state[39:46]
        elif mode == "pickup_residual":
            reference = np.asarray(record["reference"], dtype=float)
            if reference.shape != (8,) or not np.isfinite(reference).all():
                raise ValueError("Pickup residual traces require eight nominal ACT command values")
            record["nominal"] = reference
    return records, initial


@budgeted("openarm_sim.deployment.replay")
def main() -> None:
    parser = argparse.ArgumentParser(description="Replay policy commands through the shared sim/CAN execution adapter")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--mode", choices=["absolute", "joint_delta", "pickup_residual", "strike_residual"], required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--backend", choices=["sim", "can"], default="sim")
    parser.add_argument("--hardware-config", type=Path)
    parser.add_argument("--gripper-config", type=Path)
    parser.add_argument("--initialize-sim", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--supported", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=RUNS / "deployment/replay.json")
    args = parser.parse_args()
    if not np.isfinite(args.hold_seconds) or args.hold_seconds < 0:
        raise ValueError("Final hold duration must be finite and nonnegative")
    records, initial = load_records(args.trace, args.episode, args.mode)
    if args.initialize_sim and args.backend != "sim":
        raise ValueError("Recorded starting positions can be initialized only in simulation")
    if args.backend == "can":
        if args.hardware_config is None or args.gripper_config is None:
            raise ValueError("CAN policy replay requires reviewed arm and gripper configuration paths")
        GripperCalibration(json.loads(args.gripper_config.read_text()))
        if args.execute and not args.supported:
            raise ValueError("Support both arms before CAN policy execution and supply --supported")
    if not args.execute:
        print(json.dumps({"executed": False, "backend": args.backend, "mode": args.mode,
                          "episode": args.episode, "commands": len(records), "period_s": 0.02,
                          "required_start_joint_positions_rad": initial.position.tolist(),
                          "required_start_gripper_m": initial.gripper}, indent=2))
        return
    history = []
    signal.signal(signal.SIGTERM, interrupt)
    with ExitStack() as resources:
        if args.backend == "sim":
            sim = resources.enter_context(ReachSimulation(tabletop_mount=True))
            sim.reset_zero()
            if args.initialize_sim:
                sim.data.qpos[sim.qpos_ids[args.arm]] = initial.position
                sim._desired[sim.actuator_ids[args.arm]] = initial.position
                for index in (1, 2):
                    joint = sim.model.joint(f"openarm_{args.arm}_finger_joint{index}").id
                    sim.data.qpos[sim.model.jnt_qposadr[joint]] = initial.gripper
                sim._desired[sim.model.actuator(f"{args.arm}_finger1_ctrl").id] = initial.gripper
                mujoco.mj_forward(sim.model, sim.data)
            interface = SimulationPolicyInterface(sim, args.arm)
            limits = sim.model.jnt_range[sim.joint_ids[args.arm]].copy()
            settings = PolicySettings(mode=args.mode)
            clock = lambda: float(sim.data.time)
        else:
            interface = CanPolicyInterface(args.hardware_config, args.gripper_config, args.arm)
            resources.callback(interface.close)
            limits = interface.limits
            settings = PolicySettings(mode=args.mode, max_velocity_rad_s=interface.cfg["max_speed"],
                                      feedback_timeout_s=interface.cfg["feedback_timeout"],
                                      max_tracking_error_rad=interface.cfg["max_tracking_error"])
            clock = time.monotonic
        adapter = resources.enter_context(PolicyAdapter(interface, limits, settings, clock))
        measured = adapter.read()
        if np.max(np.abs(measured.position - initial.position)) > 0.03 or abs(measured.gripper - initial.gripper) > 0.003:
            raise RuntimeError("Robot feedback differs from the recorded policy starting position")
        started = clock()
        for index, record in enumerate(records):
            check_resources()
            if args.backend == "can":
                time.sleep(max(0.0, started + index * settings.period_s - clock()))
            command = adapter.step(record["policy_action"], clock(), record.get("nominal"))
            history.append({"step": index, "joint_positions_rad": command.position.tolist(), "gripper_m": command.gripper})
            if args.backend == "sim":
                for _ in range(round(settings.period_s / sim.model.opt.timestep)):
                    sim.step()
        hold_steps = int(np.ceil(args.hold_seconds / settings.period_s))
        for index in range(hold_steps):
            check_resources()
            if args.backend == "can":
                time.sleep(max(0.0, started + (len(records) + index) * settings.period_s - clock()))
            adapter.hold(clock())
            if args.backend == "sim":
                for _ in range(round(settings.period_s / sim.model.opt.timestep)):
                    sim.step()
        final = adapter.read()
        report = {"backend": args.backend, "settings": asdict(settings), "episode": args.episode,
                  "commands": len(history), "history": history, "executed": True,
                  "hold_steps": hold_steps,
                  "final_joint_positions_rad": final.position.tolist(), "final_gripper_m": final.gripper,
                  "max_final_tracking_error_rad": float(np.max(np.abs(final.position - adapter.decoder.previous.position)))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
