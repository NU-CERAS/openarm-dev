import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .score import MATCH_HORIZON_S, MATCH_TOLERANCE_S, ContactMatcher, ContactScore


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


def compare_performance(target_midi: Path, performed_midi: Path, mapping=None, latency_s=0., *,
                        score_start_s=None, intensity="auto", require_alignment=False) -> dict:
    """Match measured hits after explicit clock alignment, with inspectable rewards.

    ``score_start_s`` is target MIDI time zero on the performed MIDI timeline.
    Positive latency is subtracted *after* that offset. Never align the first hit
    automatically: doing so would erase its real timing error.
    """
    if not np.isfinite(latency_s):
        raise ValueError("Measured latency must be finite")
    if intensity not in ("auto", "off", "on"):
        raise ValueError("Intensity mode must be auto, off, or on")
    target = ContactScore.read(target_midi, mapping)
    performed = ContactScore.read(performed_midi, mapping, allow_empty=True)
    metadata_path = Path(str(performed_midi)+".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if not isinstance(metadata, dict) or (metadata_path.exists() and
            (metadata.get("schema") != "openarm.performed_midi.v1"
             or metadata.get("performed_sha256") != performed.source_sha256)):
        raise ValueError("Performed MIDI metadata does not match this file's schema or SHA256")
    if score_start_s is None:
        score_start_s = metadata.get("score_start_s")
    if score_start_s is None:
        if require_alignment or metadata.get("measurement_source") == "live_MIDI_input":
            raise ValueError("Specify --score-start-seconds: target MIDI time zero on the recording clock; use 0 only for already aligned files")
        score_start_s = 0.
    if not isinstance(score_start_s, (float, int)) or not np.isfinite(score_start_s):
        raise ValueError("Score start offset must be finite")
    placeholder = metadata.get("intensity") == "uncalibrated_placeholder"
    if placeholder and intensity == "on":
        raise ValueError("Uncalibrated simulated velocity is a placeholder and cannot be scored for intensity")
    score_intensity = intensity != "off" and not placeholder
    matcher = ContactMatcher(target.goals)
    errors = []
    for event in performed.goals:
        matched = matcher.hit(event.target, event.time_s-score_start_s-latency_s, "external")
        measured = matcher.actual[-1]
        measured["recorded_time_s"] = event.time_s
        measured["midi_velocity"] = event.velocity if not placeholder else None
        components = {"matched": 1. if matched else 0., "timing": 0., "intensity": 0.,
                      "missed": 0., "extra": 0. if matched else -.5}
        if matched:
            goal = target.goals[measured["matched_event_id"]]
            components["timing"] = -.25*abs(measured["time_s"]-goal.time_s)/MATCH_TOLERANCE_S
            if score_intensity:
                errors.append(abs(event.velocity-goal.velocity))
                components["intensity"] = -.25*errors[-1]/126
        measured["reward_components"] = components
        measured["reward"] = sum(components.values())
    matcher.expire(target.duration_s+MATCH_HORIZON_S+1e-6)
    report = matcher.report(detailed=True)
    matched_hits = {hit["matched_event_id"]: hit for hit in matcher.actual if hit["matched_event_id"] is not None}
    for event in report["events"]:
        hit = matched_hits.get(event["event_id"])
        event["reward_components"] = hit["reward_components"] if hit else {
            "matched": 0., "timing": 0., "intensity": 0., "missed": -2., "extra": 0.}
        event["reward"] = sum(event["reward_components"].values())
    reward_rows = report["events"]+[hit for hit in matcher.actual if hit["matched_event_id"] is None]
    totals = {name: sum(row["reward_components"][name] for row in reward_rows)
              for name in ("matched", "timing", "intensity", "missed", "extra")}
    return {**report, "schema": "openarm.midi_feedback.v1",
            "velocity_mae": float(np.mean(errors)) if errors else None,
            "intensity_scored": score_intensity,
            "intensity_source": metadata.get("intensity", "external_MIDI_velocity_unverified"),
            "reward_components": totals, "total_reward": sum(totals.values()),
            "reward_definition": {"matched": 1., "timing_absolute_error_over_tolerance": -.25,
                                  "intensity_absolute_error_over_126": -.25, "missed": -2., "extra": -.5},
            "matching_tolerance_s": MATCH_TOLERANCE_S,
            "score_start_s": score_start_s, "latency_correction_s": latency_s,
            "alignment": "aligned_time = recorded_time - score_start_s - latency_correction_s",
            "feedback_source": metadata.get("measurement_source", "external_MIDI_unverified"),
            "target_sha256": target.source_sha256, "performed_sha256": performed.source_sha256}


def main():
    parser = argparse.ArgumentParser(description="Evaluate performed electronic-drum MIDI or compile measured velocity calibration")
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--target-midi", type=Path, required=True)
    compare.add_argument("--performed-midi", type=Path, required=True)
    compare.add_argument("--mapping", type=Path)
    compare.add_argument("--latency-seconds", type=float, default=0.)
    compare.add_argument("--score-start-seconds", type=float)
    compare.add_argument("--intensity", choices=["auto", "off", "on"], default="auto")
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
        report = compare_performance(args.target_midi, args.performed_midi, args.mapping, args.latency_seconds,
                                     score_start_s=args.score_start_seconds, intensity=args.intensity,
                                     require_alignment=True)
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
