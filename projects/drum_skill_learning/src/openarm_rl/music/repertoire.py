import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from openarm_sim.compute_guard import budgeted

from ..envs.rhythmic import RhythmicDrummingEnv
from .arrangement import CANONICAL_MAPPING, arrange_midi


def source_metrics(arrangement: dict, played: dict) -> dict:
    """Score physical hits against the stick part of the source file, not only the arranged score."""
    precision = played["correct_hits"]/max(1, played["actual_hits"])
    recall = played["correct_hits"]/max(1, arrangement["stick_events"])
    return {"source_precision": precision, "source_recall": recall,
            "source_f1": 2*precision*recall/max(1e-12, precision+recall),
            "faithful_coverage": (arrangement["kept_events"]-arrangement["substituted_events"])/max(1, arrangement["stick_events"])}


def perform_file(job: dict) -> dict:
    output = Path(job["output"])
    try:
        midi, arrangement = arrange_midi(job["midi"], job["scene"], job["profiles"], output)
        with RhythmicDrummingEnv(job["scene"], job["profiles"], midi, mapping=CANONICAL_MAPPING, score_only=True,
                                 domain_range=job["domain_range"], position_range=job["position_range"]) as env:
            env.reset(seed=job["seed"])
            while True:
                _, _, done, truncated, info = env.step(env.reference_action())
                if done or truncated:
                    break
    except ValueError as error:
        return {"id": job["id"], "status": "not_arranged", "reason": str(error)}
    played = {key: info[key] for key in ("target_events", "actual_hits", "correct_hits", "missed_hits", "extra_hits", "f1",
                                         "timing_mae_s", "timing_p95_s", "unsafe_contact", "failure", "peak_force_N",
                                         "max_penetration_m", "is_success")}
    summary = {key: arrangement[key] for key in ("source_events", "stick_events", "kept_events", "substituted_events",
                                                 "stick_coverage", "dropped_by_reason", "kept_by_target", "duration_s")}
    row = {"id": job["id"], "status": "performed", "style": job["style"], "bpm": job["bpm"], **summary, **played,
           **source_metrics(arrangement, played)}
    (output/"result.json").write_text(json.dumps(row, indent=2))
    print(json.dumps({key: row[key] for key in ("id", "bpm", "stick_coverage", "f1", "source_f1", "timing_mae_s", "failure")}), flush=True)
    return row


@budgeted("openarm_rl.music.repertoire")
def main():
    parser = argparse.ArgumentParser(description="Arrange and physically perform Groove MIDI recordings with the reference controller")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--files", type=int, default=24)
    parser.add_argument("--max-seconds", type=float, default=60.)
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--domain-range", type=float, default=0.)
    parser.add_argument("--position-range", type=float, default=0.)
    parser.add_argument("--seed", type=int, default=800000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"midi").mkdir()
    with ZipFile(args.archive) as archive:
        with archive.open("groove/info.csv") as source:
            rows = [row for row in csv.DictReader(io.TextIOWrapper(source, encoding="utf-8"))
                    if row["split"] == args.split and float(row["duration"]) <= args.max_seconds]
        order = np.random.default_rng(args.seed).permutation(len(rows))[:args.files]
        jobs = []
        for index in order:
            row = rows[index]
            identifier = row["id"].replace("/", "_")
            midi = args.output/"midi"/f"{identifier}.mid"
            midi.write_bytes(archive.read("groove/"+row["midi_filename"]))
            jobs.append({"id": identifier, "midi": str(midi), "scene": str(args.scene_directory), "profiles": str(args.profiles),
                         "output": str(args.output/identifier), "style": row["style"], "bpm": float(row["bpm"]), "seed": args.seed,
                         "domain_range": args.domain_range, "position_range": args.position_range})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(perform_file, jobs))
    performed = [row for row in results if row["status"] == "performed"]
    means = {key: float(np.mean([row[key] for row in performed])) for key in
             ("stick_coverage", "faithful_coverage", "f1", "source_f1", "source_recall", "source_precision")} if performed else {}
    summary = {"schema": "openarm.midi_repertoire.v1", "split": args.split, "files": len(results), "performed": len(performed),
               "controller": "native_reference_teacher", "domain_range": args.domain_range, "position_range": args.position_range,
               "unsafe_files": sum(row["unsafe_contact"] for row in performed),
               "timing_mae_s": float(np.mean([row["timing_mae_s"] for row in performed if row["timing_mae_s"] is not None])) if performed else None,
               **means, "results": results}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}), flush=True)


if __name__ == "__main__":
    main()
