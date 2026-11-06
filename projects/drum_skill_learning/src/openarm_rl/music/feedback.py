import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .score import MATCH_HORIZON_S, ContactMatcher, ContactScore


class VelocityCalibration:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.document = json.loads(self.path.read_text())
        if self.document["schema"] != "openarm.drum_velocity_calibration.v1":
            raise ValueError("Unsupported strike velocity calibration")
        self.curves = {}
        for target, rows in self.document["drums"].items():
            values = np.asarray(rows, dtype=float)
            if (values.ndim != 2 or values.shape[1] != 2 or len(values) < 3 or not np.isfinite(values).all()
                    or np.any(values[:, 0] <= 0) or np.any(np.diff(values[:, 0]) <= 0)
                    or np.any(np.diff(values[:, 1]) < 0) or np.any(values[:, 1] < 1) or np.any(values[:, 1] > 127)):
                raise ValueError(f"Calibration requires increasing measured speeds and monotonic MIDI velocities: {target}")
            self.curves[target] = values

    def velocity(self, target: str, downward_speed: float) -> float:
        if not np.isfinite(downward_speed) or downward_speed < 0:
            raise ValueError("Measured strike speed must be finite and nonnegative")
        values = self.curves[target]
        return float(np.interp(downward_speed, values[:, 0], values[:, 1]))

    def speed(self, target: str, velocity: int) -> float:
        values = self.curves[target]
        if np.any(np.diff(values[:, 1]) <= 0) or not values[0, 1] <= velocity <= values[-1, 1]:
            raise ValueError("Requested intensity is outside an invertible measured calibration")
        return float(np.interp(velocity, values[:, 1], values[:, 0]))


def compare_performance(target_midi: Path, performed_midi: Path, mapping=None, latency_s=0.) -> dict:
    if not np.isfinite(latency_s):
        raise ValueError("Measured latency must be finite")
    target = ContactScore.read(target_midi, mapping)
    performed = ContactScore.read(performed_midi, mapping, allow_empty=True)
    matcher = ContactMatcher(target.goals)
    errors = []
    for event in performed.goals:
        matched = matcher.hit(event.target, event.time_s-latency_s, "external")
        matcher.actual[-1]["midi_velocity"] = event.velocity
        if matched:
            goal = target.goals[matcher.actual[-1]["matched_event_id"]]
            errors.append(abs(event.velocity-goal.velocity))
    matcher.expire(target.duration_s+MATCH_HORIZON_S+1e-6)
    return {**matcher.report(detailed=True), "velocity_mae": float(np.mean(errors)) if errors else None,
            "latency_correction_s": latency_s, "feedback_source": "recorded_performed_MIDI",
            "target_sha256": target.source_sha256, "performed_sha256": performed.source_sha256}


def main():
    parser = argparse.ArgumentParser(description="Evaluate performed electronic-drum MIDI or compile measured velocity calibration")
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--target-midi", type=Path, required=True)
    compare.add_argument("--performed-midi", type=Path, required=True)
    compare.add_argument("--mapping", type=Path)
    compare.add_argument("--latency-seconds", type=float, default=0.)
    compare.add_argument("--output", type=Path, required=True)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--observations", type=Path, required=True)
    calibrate.add_argument("--measurement-source", choices=["physical_electronic_kit", "native_simulator"], required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "compare":
        report = compare_performance(args.target_midi, args.performed_midi, args.mapping, args.latency_seconds)
    else:
        drums = {}
        with args.observations.open(newline="") as source:
            for row in csv.DictReader(source):
                drums.setdefault(row["target"], []).append([float(row["downward_speed_m_s"]), float(row["midi_velocity"])])
        if not drums:
            raise ValueError("Calibration CSV contains no measurements")
        report = {"schema": "openarm.drum_velocity_calibration.v1", "measurement_source": args.measurement_source,
                  "observations_sha256": hashlib.sha256(args.observations.read_bytes()).hexdigest(),
                  "drums": {target: sorted(rows) for target, rows in drums.items()}}
    args.output.write_text(json.dumps(report, indent=2))
    if args.command == "calibrate":
        VelocityCalibration(args.output)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
