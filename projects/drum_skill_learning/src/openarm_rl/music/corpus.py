import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import mido
import mujoco
import numpy as np

from openarm_sim.compute_guard import budgeted, check_resources

from ..paths import PROJECT_ROOT
from .reference import PerformanceReference
from .score import ContactScore


SOURCE_URL = "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip"
SOURCE_SHA256 = "651cbc524ffb891be1a3e46d89dc82a1cecb09a57c748c7b45b844c4841dcc1e"
SOURCE_PAGE = "https://magenta.tensorflow.org/datasets/groove"


def percussion_events(raw: bytes, channel: int = 9) -> list[dict]:
    midi = mido.MidiFile(file=io.BytesIO(raw))
    if midi.type not in (0, 1):
        raise ValueError("Corpus requires synchronous MIDI files")
    elapsed, events = 0., []
    for message in midi:
        elapsed += message.time
        if message.type == "note_on" and message.channel == channel and message.velocity > 0:
            events.append({"time_s": elapsed, "note": message.note, "velocity": message.velocity})
    return events


def write_segment(path: Path, events: list[dict], start: float, tempo_scale: float, duration_s: float | None = None) -> None:
    midi = mido.MidiFile(ticks_per_beat=9600)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000))
    messages = []
    for event in events:
        tick = round(mido.second2tick((event["time_s"]-start)/tempo_scale, 9600, 500000))
        messages.extend([(tick, mido.Message("note_on", channel=9, note=event["note"], velocity=event["velocity"])),
                         (tick+1, mido.Message("note_off", channel=9, note=event["note"], velocity=0))])
    previous = 0
    for tick, message in sorted(messages, key=lambda row: row[0]):
        track.append(message.copy(time=tick-previous))
        previous = tick
    if duration_s is not None:
        if not np.isfinite(duration_s) or duration_s < 0:
            raise ValueError("MIDI segment duration must be nonnegative and finite")
        final = max(previous, round(mido.second2tick(duration_s/tempo_scale, 9600, 500000)))
        track.append(mido.MetaMessage("end_of_track", time=final-previous))
    midi.save(path)


def read_manifest(path: Path) -> dict:
    document = json.loads(Path(path).read_text())
    if document["schema"] != "openarm.midi_corpus.v1":
        raise ValueError("Unsupported MIDI corpus schema")
    groups, hashes, identifiers = {}, {}, set()
    for clip in document["clips"]:
        if clip["split"] not in ("train", "validation", "test"):
            raise ValueError("Unknown corpus split")
        if clip["id"] in identifiers:
            raise ValueError("Corpus clip identifiers must be unique")
        identifiers.add(clip["id"])
        for key, index in (("source_id", groups), ("source_sha256", hashes)):
            previous = index.setdefault(clip[key], clip["split"])
            if previous != clip["split"]:
                raise ValueError("A source performance crosses corpus splits")
        directory = Path(path).parent.resolve()
        midi = (directory / clip["midi"]).resolve()
        if not midi.is_relative_to(directory):
            raise ValueError("Corpus MIDI must be inside its prepared directory")
        if hashlib.sha256(midi.read_bytes()).hexdigest() != clip["midi_sha256"]:
            raise ValueError(f"Corpus MIDI checksum differs: {clip['id']}")
    if not document["clips"]:
        raise ValueError("Corpus contains no accepted clips")
    training_targets = {target for clip in document["clips"] if clip["split"] == "train" for target in clip["targets"]}
    heldout_targets = {target for clip in document["clips"] if clip["split"] != "train" for target in clip["targets"]}
    if not heldout_targets <= training_targets:
        raise ValueError(f"Held-out drums lack training coverage: {sorted(heldout_targets-training_targets)}")
    return document


@budgeted("openarm_rl.music.corpus")
def main():
    parser = argparse.ArgumentParser(description="Prepare provenance-tracked native-feasible Groove MIDI segments")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--whole-files", action="store_true")
    parser.add_argument("--mapping", type=Path, default=PROJECT_ROOT/"configs/music/groove.json")
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2.)
    parser.add_argument("--stride", type=float, default=1.)
    parser.add_argument("--tempo-scale", type=float, default=.2)
    parser.add_argument("--minimum-events", type=int, default=2)
    parser.add_argument("--max-per-split", type=int, default=8)
    parser.add_argument("--max-per-source", type=int, default=1)
    parser.add_argument("--max-per-target", type=int, default=3)
    parser.add_argument("--max-candidates-per-split", type=int, default=2000)
    args = parser.parse_args()
    if (not np.isfinite([args.seconds, args.stride, args.tempo_scale]).all()
            or min(args.seconds, args.stride, args.tempo_scale, args.minimum_events,
                   args.max_per_split, args.max_per_source, args.max_per_target, args.max_candidates_per_split) <= 0):
        raise ValueError("Corpus durations and counts must be positive and finite")
    if args.download:
        if args.archive.exists():
            raise FileExistsError(args.archive)
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(SOURCE_URL, args.archive)
    if hashlib.sha256(args.archive.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Groove archive does not match its published SHA256")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "clips").mkdir()
    mapping = args.mapping
    supported = {int(note) for note in json.loads(mapping.read_text())["notes"]}
    note_targets = {int(note): target for note, target in json.loads(mapping.read_text())["notes"].items()}
    model = mujoco.MjModel.from_binary_path(str(args.scene_directory / "model.mjb"))
    metadata = json.loads((args.scene_directory / "scene.json").read_text())
    accepted, rejected = [], []
    counts = {split: 0 for split in ("train", "validation", "test")}
    attempts = dict(counts)
    source_counts = {}
    target_counts = {split: {} for split in counts}
    document = {"schema": "openarm.midi_corpus.v1", "source": SOURCE_PAGE, "archive_url": SOURCE_URL,
                "archive_sha256": SOURCE_SHA256, "license": "CC-BY-4.0",
                "attribution": "Gillick, Roberts, Engel, Eck and Bamman. Learning to Groove with Inverse Sequence Transformations. ICML 2019.",
                "mapping": "mapping.json", "tempo_scale": args.tempo_scale,
                "window_seconds": args.seconds, "stride_seconds": args.stride,
                "whole_performances": args.whole_files,
                "selection": "supported notes and compiled trajectory feasibility; original events preserved",
                "clips": accepted}
    (args.output / "mapping.json").write_text(mapping.read_text())
    with ZipFile(args.archive) as archive:
        with archive.open("groove/info.csv") as source:
            rows = list(csv.DictReader(io.TextIOWrapper(source, encoding="utf-8")))
        rng = np.random.default_rng(20261103)
        for index in rng.permutation(len(rows)):
            row = rows[index]
            split = row["split"]
            if counts[split] >= args.max_per_split or attempts[split] >= args.max_candidates_per_split:
                continue
            raw = archive.read("groove/"+row["midi_filename"])
            events = percussion_events(raw)
            if not events:
                continue
            source_duration = mido.MidiFile(file=io.BytesIO(raw)).length
            starts = [0.] if args.whole_files else np.arange(0, events[-1]["time_s"]+1e-9, args.stride)
            for start in starts:
                if source_counts.get(row["id"], 0) >= args.max_per_source:
                    break
                if counts[split] >= args.max_per_split or attempts[split] >= args.max_candidates_per_split:
                    break
                selected = events if args.whole_files else [event for event in events if start <= event["time_s"] < start+args.seconds]
                if len(selected) < args.minimum_events:
                    continue
                attempts[split] += 1
                clip = {"id": f"{split}_{attempts[split]:06d}", "source_id": row["id"],
                        "source_sha256": hashlib.sha256(raw).hexdigest(), "source_midi": row["midi_filename"],
                        "split": split, "start_s": float(start), "end_s": float(source_duration if args.whole_files else start+args.seconds),
                        "whole_performance": args.whole_files,
                        "events": len(selected), "notes": sorted({event["note"] for event in selected}),
                        "source_events": selected}
                unknown = sorted(set(clip["notes"])-supported)
                if unknown:
                    rejected.append({**clip, "reason": "unsupported_articulation_or_foot_event", "unsupported_notes": unknown})
                    continue
                targets = {note_targets[note] for note in clip["notes"]}
                if all(target_counts[split].get(target, 0) >= args.max_per_target for target in targets):
                    rejected.append({**clip, "reason": "target_sampling_quota"})
                    continue
                midi = args.output / "clips" / f"{clip['id']}.mid"
                write_segment(midi, selected, float(start), args.tempo_scale,
                              duration_s=source_duration if args.whole_files else None)
                score = ContactScore.read(midi, mapping)
                check_resources()
                try:
                    reference = PerformanceReference(model, metadata, score, args.profiles)
                except ValueError as error:
                    rejected.append({**clip, "reason": "trajectory_infeasible", "detail": str(error)})
                    midi.unlink()
                    continue
                clip.update(midi=str(midi.relative_to(args.output)), midi_sha256=hashlib.sha256(midi.read_bytes()).hexdigest(),
                            duration_s=reference.duration, tempo_scale=args.tempo_scale,
                            targets=sorted({goal.target for goal in score.goals}))
                accepted.append(clip)
                counts[split] += 1
                source_counts[row["id"]] = source_counts.get(row["id"], 0)+1
                for target in targets:
                    target_counts[split][target] = target_counts[split].get(target, 0)+1
                print(json.dumps({"accepted": clip["id"], "counts": counts, "source_id": clip["source_id"]}), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(document, indent=2))
    (args.output / "rejections.json").write_text(json.dumps(rejected, indent=2))
    (args.output / "summary.json").write_text(json.dumps({"accepted": counts, "examined": attempts,
        "rejected": len(rejected), "original_files": len(rows), "tempo_scale": args.tempo_scale,
        "target_counts": target_counts}, indent=2))
    read_manifest(args.output / "manifest.json")
    if min(counts.values()) == 0:
        raise RuntimeError("Corpus requires accepted clips in every split; inspect summary and rejection records")
    print(json.dumps({"accepted": counts, "examined": attempts, "rejected": len(rejected)}), flush=True)


if __name__ == "__main__":
    main()
