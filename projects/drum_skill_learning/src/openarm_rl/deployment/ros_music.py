import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time

import numpy as np

from openarm_tools.recording.storage import lock
from openarm_tools.ros_execution import trajectory_message

from .rhythm_actor import RhythmActorRuntime
from .ros_drum import JointMapping, RosDrumTool
from ..skills.runtime import prepare_transfers


class RosMusicTool(RosDrumTool):
    def perform(self, runtime, execute=False):
        if not self.reservation.acquire(blocking=False):
            raise RuntimeError("Another robot action is executing")
        streaming = False
        try:
            with ExitStack() as resources:
                if execute:
                    resources.enter_context(lock(f"ros-command:{self.namespace}"))
                mapping = JointMapping(self.mapping_document, runtime.limits, self.physical)
                def stop():
                    runtime.cancel()
                    if streaming:
                        snapshot = self.client.read()
                        q, _ = mapping.measured(snapshot)
                        bounds = runtime.controller.bounds
                        hold = np.clip(runtime.servo.position_target(q, q), bounds[:, 0], bounds[:, 1])
                        self.publish(mapping, hold, np.zeros(14))
                        time.sleep(.12)
                resources.callback(stop)
                if self.physical:
                    for key, source in (("robot_source_sha256", "robot"), ("import_config_sha256", "import"), ("layout_sha256", "layout")):
                        if self.mapping_document[key] != runtime.identity.get(source, ""):
                            raise ValueError(f"Reviewed MIDI mapping identity differs: {key}")
                else:
                    if not self.parameters.wait_for_services(timeout_sec=2):
                        raise RuntimeError("ROS model identity service is unavailable")
                    values = self.client.wait_future(self.parameters.get_parameters(["robot_model", "layout_sha256"]), time.monotonic()+2).values
                    if values[0].string_value != "opendrummer" or values[1].string_value != runtime.identity.get("layout", ""):
                        raise ValueError("ROS execution model/layout differs from the MIDI scene")
                current, dq = mapping.measured(self.client.read())
                transfers = prepare_transfers(runtime.model, runtime.controller, current, runtime.reference.initial)
                result = {"execute": execute, "target_events": len(runtime.score.goals),
                          "duration_s": runtime.reference.duration, "event_assignments": runtime.reference.assignment,
                          "measured_start_rad": current.tolist()}
                if not execute:
                    return result
                self.cancelled.clear()
                for arm, trajectory in transfers:
                    times = np.r_[np.arange(0, trajectory.ends[-1], .02), trajectory.ends[-1]]
                    samples = [mapping.command(*trajectory.sample(t)) for t in times]
                    selected = slice(0, 7) if arm == "left" else slice(7, 14)
                    message = trajectory_message(arm, times, np.array([q[selected] for q, _ in samples]),
                                                 np.array([v[selected] for _, v in samples]))
                    self.client.execute(arm, message, self.cancelled, lambda _: None)
                deadline = time.monotonic()+2
                while True:
                    snapshot = self.client.read()
                    q, dq = mapping.measured(snapshot)
                    if np.max(np.abs(q-runtime.reference.initial)) < .03 and np.max(np.abs(dq)) < .08:
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError("MIDI preparation did not settle")
                    time.sleep(.01)
                started = time.monotonic()
                runtime.begin(q, dq, snapshot.received_at, started)
                tick, previous, periods, ages = 0, None, [], []
                streaming = True
                while runtime.running:
                    if self.cancelled.is_set():
                        raise InterruptedError("MIDI playback cancelled")
                    snapshot = self.client.read(wait=False)
                    q, dq = mapping.measured(snapshot)
                    now = time.monotonic()
                    if previous is not None:
                        periods.append(now-previous)
                    previous = now
                    ages.append(now-snapshot.received_at)
                    command = runtime.step(q, dq, snapshot.received_at, now)
                    if command is None:
                        break
                    self.publish(mapping, *command)
                    tick += 1
                    time.sleep(max(0., started+tick*.01-time.monotonic()))
                snapshot = self.client.read()
                final, dq = mapping.measured(snapshot)
                return {**result, "encoder_verified": True, "command_count": tick,
                        "max_control_period_s": max(periods, default=0), "max_feedback_age_s": max(ages, default=0),
                        "final_joints_rad": final.tolist(), "hit_verification": "external_MIDI_or_operator_required"}
        finally:
            runtime.cancel()
            self.reservation.release()


def main():
    parser = argparse.ArgumentParser(description="Preview or explicitly execute a MIDI actor through the shared ROS interface")
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--midi", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--percussion-mapping", type=Path)
    parser.add_argument("--tempo-scale", type=float, default=1.)
    parser.add_argument("--namespace", default="")
    parser.add_argument("--physical", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    runtime = RhythmActorRuntime(args.actor, args.scene_directory, args.profiles, args.midi,
                                 mapping=args.percussion_mapping, tempo_scale=args.tempo_scale)
    tool = RosMusicTool(args.namespace, args.mapping, args.physical)
    try:
        print(json.dumps(tool.perform(runtime, args.execute)), flush=True)
    finally:
        tool.close()


if __name__ == "__main__":
    main()
