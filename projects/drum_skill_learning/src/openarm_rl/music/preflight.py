import argparse
import hashlib
import json
from pathlib import Path

import mido
import mujoco
import numpy as np

from openarm_sim.compute_guard import budgeted

from ..paths import PROJECT_ROOT
from ..skills.registry import SKILLS
from .reference import PerformanceReference, ScoreFeasibilityError
from .score import ContactScore


def inspect_performance(midi_file: Path, scene_directory: Path, profiles: Path,
                        mapping: Path | None = None, tempo_scale: float = 1.) -> dict:
    if not np.isfinite(tempo_scale) or tempo_scale <= 0:
        raise ValueError("Tempo scale must be positive and finite")
    mapping = PROJECT_ROOT/"configs/music/percussion.json" if mapping is None else Path(mapping)
    document = json.loads(mapping.read_text())
    if document["schema"] != "openarm.percussion_mapping.v1" or not 0 <= document["channel"] <= 15:
        raise ValueError("Invalid percussion mapping")
    notes = {int(note): target for note, target in document["notes"].items()}
    if any(target not in SKILLS for target in notes.values()):
        raise ValueError("Mapping contains an unsupported physical drum")
    midi = mido.MidiFile(midi_file)
    if midi.type not in (0, 1):
        raise ValueError("Synchronous MIDI type 0 or 1 is required")
    elapsed, events = 0., []
    for message in midi:
        elapsed += message.time/tempo_scale
        if message.type == "note_on" and message.velocity > 0 and message.channel == document["channel"]:
            target = notes.get(message.note)
            available = target is not None and any((Path(profiles)/name).is_file() for name in
                (f"{target}.json", f"{target}.left.json", f"{target}.right.json"))
            events.append({"event_id": len(events), "time_s": elapsed, "note": message.note,
                "velocity": message.velocity, "target": target,
                "status": "unmapped_note" if target is None else "missing_reference" if not available else "pending"})
    metadata = json.loads((Path(scene_directory)/"scene.json").read_text())
    report = {"schema": "openarm.midi_preflight.v1", "midi_sha256": hashlib.sha256(Path(midi_file).read_bytes()).hexdigest(),
        "mapping_sha256": hashlib.sha256(mapping.read_bytes()).hexdigest(), "tempo_scale": tempo_scale,
        "original_kit_positions": "layout_sha256" not in metadata["scene"],
        "duration_s": midi.length/tempo_scale, "target_events": len(events), "events": events,
        "executable": False, "issues": []}
    if not events:
        report["issues"].append({"reason": "no_percussion_events"})
    report["issues"].extend({"event_id": event["event_id"], "time_s": event["time_s"], "note": event["note"],
        "target": event["target"], "reason": event["status"]} for event in events if event["status"] != "pending")
    if report["issues"]:
        return report
    score = ContactScore.read(midi_file, mapping, tempo_scale)
    model = mujoco.MjModel.from_binary_path(str(Path(scene_directory)/"model.mjb"))
    try:
        reference = PerformanceReference(model, metadata, score, profiles)
    except ScoreFeasibilityError as error:
        report["issues"].append({"reason": "trajectory_infeasible", "time_s": error.time_s,
            "target": error.target, "arm": error.arm, "detail": str(error)})
        return report
    for event in events:
        event.update(status="scheduled", arm=reference.assignment[event["event_id"]])
    report.update(executable=True, performance_duration_s=reference.duration, preroll_s=reference.preroll)
    return report


@budgeted("openarm_rl.music.preflight")
def main():
    parser = argparse.ArgumentParser(description="Check every MIDI percussion event against the original robot scene")
    parser.add_argument("--midi", type=Path, required=True)
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--tempo-scale", type=float, default=1.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = inspect_performance(args.midi, args.scene_directory, args.profiles, args.mapping, args.tempo_scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "events"}), flush=True)
    if not report["executable"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
