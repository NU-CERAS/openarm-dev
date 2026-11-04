import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

import numpy as np

from openarm_sim.compute_guard import budgeted, check_resources

from .arrangement import CANONICAL_MAPPING, arrange_midi
from .corpus import SOURCE_PAGE, read_manifest


def heldout_supported(clip: dict, training_targets: set) -> bool:
    return clip["split"] == "train" or set(clip["targets"]) <= training_targets


@budgeted("openarm_rl.music.arranged_corpus")
def main():
    parser = argparse.ArgumentParser(description="Build a training corpus of complete Groove performances arranged for the calibrated kit")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--train", type=int, default=32)
    parser.add_argument("--validation", type=int, default=8)
    parser.add_argument("--test", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=30.)
    parser.add_argument("--minimum-events", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"clips").mkdir()
    shutil.copyfile(CANONICAL_MAPPING, args.output/"mapping.json")
    clips, rejected, training_targets = [], [], set()
    with ZipFile(args.archive) as archive:
        with archive.open("groove/info.csv") as source:
            rows = list(csv.DictReader(io.TextIOWrapper(source, encoding="utf-8")))
        order = np.random.default_rng(args.seed).permutation(len(rows))
        for split, quota in (("train", args.train), ("validation", args.validation), ("test", args.test)):
            accepted = 0
            for index in order:
                row = rows[index]
                if accepted >= quota:
                    break
                if row["split"] != split or float(row["duration"]) > args.max_seconds:
                    continue
                check_resources()
                identifier = f"{split}_{accepted:04d}"
                work = args.output/"work"/str(index)
                work.mkdir(parents=True)
                raw = archive.read("groove/"+row["midi_filename"])
                source_midi = work/"source.mid"
                source_midi.write_bytes(raw)
                try:
                    arranged, report = arrange_midi(source_midi, args.scene_directory, args.profiles, work)
                except ValueError as error:
                    rejected.append({"source_id": row["id"], "reason": str(error)})
                    continue
                clip = {"id": identifier, "source_id": row["id"], "split": split, "source_midi": row["midi_filename"],
                        "source_sha256": hashlib.sha256(raw).hexdigest(), "events": report["kept_events"],
                        "stick_coverage": report["stick_coverage"], "duration_s": report["performance_duration_s"],
                        "targets": sorted(report["kept_by_target"]), "whole_performance": True, "tempo_scale": 1.}
                if report["kept_events"] < args.minimum_events or not heldout_supported(clip, training_targets):
                    rejected.append({"source_id": row["id"], "reason": "too_few_events_or_uncovered_drum"})
                    continue
                midi = args.output/"clips"/f"{identifier}.mid"
                shutil.copyfile(arranged, midi)
                clip.update(midi=str(midi.relative_to(args.output)), midi_sha256=hashlib.sha256(midi.read_bytes()).hexdigest())
                clips.append(clip)
                accepted += 1
                if split == "train":
                    training_targets.update(clip["targets"])
                print(json.dumps({"accepted": identifier, "events": clip["events"], "coverage": clip["stick_coverage"]}), flush=True)
    shutil.rmtree(args.output/"work")
    document = {"schema": "openarm.midi_corpus.v1", "source": SOURCE_PAGE, "license": "CC-BY-4.0",
                "attribution": "Gillick, Roberts, Engel, Eck and Bamman. Learning to Groove with Inverse Sequence Transformations. ICML 2019.",
                "mapping": "mapping.json", "tempo_scale": 1., "whole_performances": True,
                "selection": "complete performances arranged by openarm_rl.music.arrangement", "clips": clips}
    (args.output/"manifest.json").write_text(json.dumps(document, indent=2))
    (args.output/"rejections.json").write_text(json.dumps(rejected, indent=2))
    read_manifest(args.output/"manifest.json")


if __name__ == "__main__":
    main()
