import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from openarm_sim.compute_guard import budgeted

from ..deployment.rhythm_actor import ReferenceRuntime
from ..envs.rhythmic import RhythmicDrummingEnv
from .arrangement import CANONICAL_MAPPING, arrange_midi
from .preflight import inspect_performance
from .reference import profile_hashes

# Required conditions gate the robot; the sweep only measures how far the kit may sit from the model.
REQUIRED = (("nominal", 0., 0.), ("deployment", .05, .002))
SWEEP = (("kit_5mm", .05, .005), ("kit_10mm", .05, .01), ("kit_20mm", .05, .02), ("model_10pct", .1, .002))


def perform_once(job):
    """Run the runtime that drives the robot against contact physics, including its own stop checks."""
    condition, required, domain, position, seed, scene, profiles, midi, mapping, tempo = job
    runtime = ReferenceRuntime(scene, profiles, midi, mapping, tempo)
    failure = None
    with RhythmicDrummingEnv(scene, profiles, midi, mapping=mapping, tempo_scale=tempo, score_only=True,
                             domain_range=domain, position_range=position) as env:
        env.reset(seed=seed)
        info = env.info()
        try:
            sampled_at, q, dq = env.sensor_cache
            runtime.begin(q, dq, sampled_at, env.elapsed)
            while env.active:
                sampled_at, q, dq = env.sensor_cache
                if runtime.step(q, dq, sampled_at, env.elapsed) is None:
                    break
                _, _, _, _, info = env.step(runtime.last_action)
        except RuntimeError as error:
            failure = str(error)
        report = env.matcher.report(detailed=True)
        completed = env.elapsed >= env.reference.duration-.011
    peak = max((hit.get("peak_force_N", 0.) for hit in report["hits"]), default=0.)
    errors = np.abs(report.get("timing_errors_s") or [0.])
    return {"condition": condition, "required": required, "seed": seed, "f1": report["f1"],
            "correct_hits": report["correct_hits"], "missed_hits": report["missed_hits"], "extra_hits": report["extra_hits"],
            "is_success": bool(info["is_success"]) and failure is None, "completed": bool(completed),
            "failure": failure or info.get("failure"), "unsafe_contact": bool(info["unsafe_contact"]),
            "peak_force_N": float(peak), "timing_mae_s": float(np.mean(errors)), "timing_max_s": float(np.max(errors))}


def judge(runs):
    return all(run["is_success"] and run["completed"] and run["failure"] is None for run in runs if run["required"])


def summarize(runs):
    summary = {}
    for name in dict.fromkeys(run["condition"] for run in runs):
        rows = [run for run in runs if run["condition"] == name]
        summary[name] = {"runs": len(rows), "success": float(np.mean([run["is_success"] and run["completed"] for run in rows])),
                         "f1_min": min(run["f1"] for run in rows), "unsafe": sum(run["unsafe_contact"] for run in rows),
                         "peak_force_N": max(run["peak_force_N"] for run in rows),
                         "timing_max_s": max(run["timing_max_s"] for run in rows)}
    return summary


def rehearse(scene_directory, profiles, midi, mapping=None, tempo_scale=1., output=None, seeds=8, sweep=True, workers=6):
    scene_directory, profiles, midi = str(Path(scene_directory).resolve()), str(Path(profiles).resolve()), str(Path(midi).resolve())
    mapping = str(Path(CANONICAL_MAPPING if mapping is None else mapping).resolve())
    jobs = [("nominal", True, 0., 0., 700000, scene_directory, profiles, midi, mapping, tempo_scale)]
    for required, conditions in ((True, REQUIRED[1:]), (False, SWEEP if sweep else ())):
        jobs += [(name, required, domain, position, 700000+seed, scene_directory, profiles, midi, mapping, tempo_scale)
                 for name, domain, position in conditions for seed in range(seeds)]
    if workers > 1:
        with ProcessPoolExecutor(min(workers, len(jobs))) as pool:
            runs = list(pool.map(perform_once, jobs))
    else:
        runs = [perform_once(job) for job in jobs]
    runtime = ReferenceRuntime(scene_directory, profiles, midi, mapping, tempo_scale)
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = {"schema": "openarm.midi_rehearsal.v1", "passed": judge(runs), "summary": summarize(runs), "runs": runs,
              "inputs": {"scene_directory": scene_directory, "profiles": profiles, "midi": midi, "mapping": mapping, "tempo_scale": tempo_scale},
              "midi_sha256": digest(midi), "mapping_sha256": digest(mapping), "model_sha256": digest(Path(scene_directory)/"model.mjb"),
              "profiles_sha256": profile_hashes(Path(profiles)), "scene_identity": runtime.identity,
              "duration_s": runtime.reference.duration, "target_events": len(runtime.score.goals),
              "ready_pose_rad": runtime.reference.initial.tolist(), "runtime_limits": runtime.limits}
    if output is not None:
        Path(output).write_text(json.dumps(report, indent=2))
    return report


@budgeted("openarm_rl.music.rehearse")
def main():
    parser = argparse.ArgumentParser(description="Qualify one MIDI file for the robot by playing it through the deployed runtime in simulation")
    parser.add_argument("--midi", type=Path, required=True)
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--tempo-scale", type=float, default=1.)
    parser.add_argument("--arrange", action="store_true", help="Fold General MIDI notes onto the kit and report every changed or dropped note")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--no-sweep", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    midi, mapping, tempo = args.midi, args.mapping, args.tempo_scale
    if args.arrange:
        midi, arrangement = arrange_midi(midi, args.scene_directory, args.profiles, args.output/"arrangement", mapping, tempo)
        mapping, tempo = CANONICAL_MAPPING, 1.
        print(json.dumps({key: value for key, value in arrangement.items() if key not in ("kept", "dropped")}), flush=True)
    preflight = inspect_performance(midi, args.scene_directory, args.profiles, mapping, tempo)
    (args.output/"preflight.json").write_text(json.dumps(preflight, indent=2))
    if not preflight["executable"]:
        raise SystemExit(f"MIDI cannot be scheduled; inspect {args.output/'preflight.json'}")
    report = rehearse(args.scene_directory, args.profiles, midi, mapping, tempo, args.output/"rehearsal.json",
                      args.seeds, not args.no_sweep, args.workers)
    print(json.dumps({"passed": report["passed"], "summary": report["summary"]}, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
