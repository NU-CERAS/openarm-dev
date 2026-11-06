import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import partial
from itertools import product
import json
from pathlib import Path

import numpy as np

from openarm_sim.compute_guard import budgeted, check_resources

from ..control.reference import JointReference
from ..envs.rhythmic import RhythmicDrummingEnv
from ..music.corpus import write_segment
from ..music.reference import ScoreFeasibilityError
from ..paths import PROJECT_ROOT


STROKE_GRID = tuple(product((.03, .05, .08), (.05, .07, .10), (.01, .03)))
BANK_SPEEDS = (.35, .45, .6, .75, .85)
PERFORMANCE_SPEED = .6
BURST_HITS = 8
SPEED_BAND = .05
TIMING_SPREAD_S = .008


def select_stroke(candidates: list[dict], speed: float) -> dict:
    """Shortest qualified stroke among those nearest the requested impact speed."""
    nearest = min(abs(row["speed_m_s"]-speed) for row in candidates)
    return min((row for row in candidates if abs(row["speed_m_s"]-speed) <= nearest+SPEED_BAND),
               key=lambda row: (row["duration_s"], abs(row["speed_m_s"]-speed)))


def shared_ready(candidates: list[dict], speed: float) -> list[dict]:
    """A speed bank blends strokes, so every entry keeps the ready pose of the stroke chosen for `speed`."""
    anchor = select_stroke(candidates, speed)["reference"]["positions_rad"][0]
    return [row for row in candidates if row["reference"]["positions_rad"][0] == anchor]


def compact_reference(reference: JointReference, approach: float, retract: float, settle: float) -> JointReference:
    if len(reference.times) != 7:
        raise ValueError("Rhythmic compaction requires a seven-knot calibrated wrist stroke")
    downstroke, hold = np.diff(reference.times)[2:4]
    times = np.cumsum([0., approach, downstroke, hold, retract, settle])
    return JointReference(times, reference.positions[1:], reference.wrist,
        approach+reference.impact_time-reference.times[2], reference.contact_multiplier,
        reference.compliance_seconds, reference.velocities[1:])


def measure_burst(scene: Path, profiles: Path, midi: Path) -> dict:
    env = RhythmicDrummingEnv(scene, profiles, midi, score_only=True)
    with env:
        env.reset(seed=271828)
        while True:
            _, _, done, truncated, info = env.step(env.reference_action())
            if done or truncated:
                return info


def calibrate(scene: Path, source: Path, output: Path, maximum_period: float, transfer_velocity: float, transfer_acceleration: float) -> dict:
    profile = json.loads(source.read_text())
    profile["transfer_limits"] = {"velocity_rad_s": transfer_velocity, "acceleration_rad_s2": transfer_acceleration}
    target = profile["target"]
    directory = output/source.stem
    directory.mkdir()
    probe_profiles = directory/"profiles"
    probe_profiles.mkdir()
    mapping = json.loads((PROJECT_ROOT/"configs/music/percussion.json").read_text())
    note = next(int(note) for note, name in mapping["notes"].items() if name == target)
    midi = directory/"probe.mid"
    unique = {json.dumps(row, sort_keys=True): row for row in profile["reference_bank"]["references"]}
    measurements, candidates = [], []
    for document, (approach, retract, settle) in product(unique.values(), STROKE_GRID):
        reference = compact_reference(JointReference.load(document), approach, retract, settle)
        if reference.duration > maximum_period+1e-8:
            continue
        check_resources()
        probe = deepcopy(profile)
        probe["reference_bank"] = {"schema": "openarm.speed_conditioned_strike.v1", "speeds_m_s": [.35, .85],
            "references": [reference.document(), reference.document()]}
        (probe_profiles/f"{target}.json").write_text(json.dumps(probe))
        # A repeated burst qualifies the period; one isolated stroke cannot show rearming or residual motion.
        period = reference.duration+.002
        write_segment(midi, [{"time_s": .5+index*period, "note": note, "velocity": 96} for index in range(BURST_HITS)], 0., 1.)
        row = {"approach_s": approach, "retract_s": retract, "settle_s": settle, "duration_s": reference.duration}
        try:
            info = measure_burst(scene, probe_profiles, midi)
        except ScoreFeasibilityError as error:
            measurements.append({**row, "valid": False, "failure": str(error)})
            continue
        errors = np.asarray(info["timing_errors_s"])
        valid = (info["is_success"] and info["correct_hits"] == BURST_HITS and info["extra_hits"] == 0
                 and info["peak_force_N"] <= 22 and info["max_penetration_m"] <= .002 and float(np.std(errors)) <= TIMING_SPREAD_S)
        speed = float(np.mean([hit["downward_speed_m_s"] for hit in info["hits"]])) if info["hits"] else 0.
        row.update(valid=bool(valid), speed_m_s=speed, measurements=info)
        if valid:
            reference.impact_time += float(np.mean(errors))
            row["reference"] = reference.document()
            candidates.append(row)
        measurements.append(row)
    (directory/"calibration.json").write_text(json.dumps(measurements, indent=2))
    if not candidates:
        result = {"target": target, "arm": profile["arm"], "valid_candidates": 0, "maximum_period_s": maximum_period}
        print(json.dumps(result), flush=True)
        return result
    bank = shared_ready(candidates, PERFORMANCE_SPEED)
    selected = [select_stroke(bank, speed) for speed in BANK_SPEEDS]
    profile["reference_bank"] = {"schema": "openarm.speed_conditioned_strike.v1", "speeds_m_s": list(BANK_SPEEDS),
        "references": [row["reference"] for row in selected], "calibration_seed": 271828,
        "calibration_speed_m_s": [row["speed_m_s"] for row in selected],
        "repeat_period_s": [row["duration_s"]+.002 for row in selected]}
    destination = output/"profiles"/source.name
    destination.write_text(json.dumps(profile, indent=2))
    result = {"target": target, "arm": profile["arm"], "valid_candidates": len(candidates),
        "maximum_period_s": maximum_period, "measured_speeds_m_s": profile["reference_bank"]["calibration_speed_m_s"],
        "durations_s": [row["duration_s"] for row in selected]}
    print(json.dumps(result), flush=True)
    return result


@budgeted("openarm_rl.planning.rhythmic_references")
def main():
    parser = argparse.ArgumentParser(description="Measure repeated-stroke references at the shortest safe period in the original drum scene")
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, nargs="+", required=True)
    parser.add_argument("--maximum-period", type=float, default=.32)
    parser.add_argument("--transfer-velocity", type=float, default=2.)
    parser.add_argument("--transfer-acceleration", type=float, default=8.)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.maximum_period) or not .1 <= args.maximum_period <= 1.:
        raise ValueError("Maximum stroke period must lie in [0.1, 1.0] seconds")
    if not 0 < args.transfer_velocity <= 4 or not 0 < args.transfer_acceleration <= 120:
        raise ValueError("Transfer velocity/acceleration require positive bounded values")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"profiles").mkdir()
    measure = partial(calibrate, args.scene_directory, output=args.output, maximum_period=args.maximum_period,
                      transfer_velocity=args.transfer_velocity, transfer_acceleration=args.transfer_acceleration)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(measure, args.profiles))
    (args.output/"summary.json").write_text(json.dumps(results, indent=2))
    if any(row["valid_candidates"] == 0 for row in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
