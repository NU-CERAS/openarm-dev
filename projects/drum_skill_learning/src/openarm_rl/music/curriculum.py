import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from openarm_sim.compute_guard import budgeted, check_resources

from ..paths import PROJECT_ROOT
from .corpus import read_manifest, write_segment
from .preflight import inspect_performance


def composition(index: int, seed: int, sections: int = 2) -> tuple[list[dict], float]:
    rng = np.random.default_rng(seed)
    pairs = [(38, 43), (48, 47), (38, 47), (48, 43)]
    interval = float(rng.uniform(.58, .95))
    phase = float(rng.uniform(.02, .20))
    events = []
    for section in range(sections):
        pair = pairs[(index+section*(1+index % 3)) % len(pairs)]
        start = .4+section*4.5
        for beat in range(3):
            for arm, note in enumerate(pair):
                time_s = start+beat*interval+arm*phase
                events.append({"time_s": time_s, "note": note, "velocity": int(rng.integers(60, 113))})
    return sorted(events, key=lambda event: event["time_s"]), sections*4.5


@budgeted("openarm_rl.music.curriculum")
def main():
    parser = argparse.ArgumentParser(description="Generate provenance-labelled complete two-hand MIDI exercises at original drum positions")
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--training-patterns", type=int, default=12)
    parser.add_argument("--heldout-patterns", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.training_patterns, args.heldout_patterns) < 4:
        raise ValueError("Every split requires at least four generated arrangements")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"clips").mkdir()
    mapping = PROJECT_ROOT/"configs/music/percussion.json"
    (args.output/"mapping.json").write_text(mapping.read_text())
    clips, rejected = [], []
    for split_index, (split, count) in enumerate((("train", args.training_patterns), ("validation", args.heldout_patterns), ("test", args.heldout_patterns))):
        for index in range(count):
            check_resources()
            seed = 202611040+10000*split_index+index
            events, duration = composition(index, seed)
            identifier = f"{split}_{index:03d}"
            midi = args.output/"clips"/f"{identifier}.mid"
            write_segment(midi, events, 0., 1., duration_s=duration)
            report = inspect_performance(midi, args.scene_directory, args.profiles, mapping)
            (args.output/f"{identifier}_preflight.json").write_text(json.dumps(report, indent=2))
            if not report["executable"]:
                rejected.append({"id": identifier, "issues": report["issues"]})
                print(json.dumps({"pattern": identifier, "executable": False, "issues": report["issues"]}), flush=True)
                continue
            recipe = {"kind": "generated_two_hand_exercise", "seed": seed, "events": events, "duration_s": duration}
            clips.append({"id": identifier, "source_id": f"generated/{split}/{index}", "split": split,
                "source_sha256": hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest(),
                "midi": str(midi.relative_to(args.output)), "midi_sha256": report["midi_sha256"],
                "targets": sorted({event["target"] for event in report["events"]}),
                "events": len(events), "duration_s": report["performance_duration_s"],
                "whole_performance": True, "tempo_scale": 1., "recipe": recipe})
            print(json.dumps({"pattern": identifier, "executable": True, "events": len(events)}), flush=True)
    document = {"schema": "openarm.midi_corpus.v1", "source": "generated_exercises",
        "human_performances": False, "mapping": "mapping.json", "tempo_scale": 1.,
        "whole_performances": True, "clips": clips}
    (args.output/"manifest.json").write_text(json.dumps(document, indent=2))
    (args.output/"rejections.json").write_text(json.dumps(rejected, indent=2))
    counts = {split: sum(clip["split"] == split for clip in clips) for split in ("train", "validation", "test")}
    (args.output/"summary.json").write_text(json.dumps({"accepted": counts, "rejected": len(rejected),
        "source": "generated_exercises", "human_performances": False}, indent=2))
    if any(count == 0 for count in counts.values()):
        raise RuntimeError("Complete generated MIDI coverage is missing a split")
    read_manifest(args.output/"manifest.json")


if __name__ == "__main__":
    main()
