import argparse
from collections import Counter
from functools import cache
import hashlib
import io
import json
from pathlib import Path

import mido
import mujoco
import numpy as np

from openarm_drummer.model import HoldController
from openarm_sim.compute_guard import budgeted

from ..paths import PROJECT_ROOT
from ..skills.registry import SKILLS
from .corpus import percussion_events, write_segment
from .reference import PerformanceReference, ScoreFeasibilityError, TransferPlanner, load_strokes
from .score import ContactScore


PROJECT_MAPPING = PROJECT_ROOT/"configs/music/general_midi.json"
CANONICAL_MAPPING = PROJECT_ROOT/"configs/music/percussion.json"
TARGET_WEIGHTS = {"snare": 1., "rack_tom_left": .8, "rack_tom_right": .8, "floor_tom": .8,
                  "cymbal_left": .7, "cymbal_right": .7, "hi_hat": .5}
FLAM_WINDOW_S = .03
SUBSTITUTION_COST = .3
TRANSFER_COST_PER_S = .05
CROSS_HAND_COST = .15
BEAM_WIDTH = 64
MAX_PATH_REPAIRS = 200


def read_source_events(path: Path, tempo_scale: float = 1., channel: int = 9) -> tuple[list[dict], float]:
    if not np.isfinite(tempo_scale) or tempo_scale <= 0:
        raise ValueError("Tempo scale must be positive and finite")
    raw = Path(path).read_bytes()
    events = [{**event, "time_s": event["time_s"]/tempo_scale} for event in percussion_events(raw, channel)]
    if not events:
        raise ValueError("MIDI file has no percussion events on the selected channel")
    return events, mido.MidiFile(file=io.BytesIO(raw)).length/tempo_scale


def event_weight(target: str, velocity: int) -> float:
    return TARGET_WEIGHTS[target]*(.5+.5*velocity/127)


def fold_events(events: list[dict], document: dict) -> dict:
    """Fold General MIDI articulations onto kit pieces; feet and unknown instruments have no stick stroke."""
    if document["schema"] != "openarm.percussion_arrangement.v1":
        raise ValueError("Unsupported percussion arrangement mapping")
    notes = {int(note): targets for note, targets in document["notes"].items()}
    if any(target not in SKILLS for targets in notes.values() for target in targets):
        raise ValueError("Arrangement mapping contains an unsupported physical drum")
    foot = set(document["foot_notes"])
    folded, dropped = [], []
    for event in sorted(events, key=lambda row: row["time_s"]):
        if event["note"] in foot:
            dropped.append({**event, "reason": "foot_event"})
        elif event["note"] not in notes:
            dropped.append({**event, "reason": "no_kit_piece"})
        else:
            targets = notes[event["note"]]
            row = {**event, "targets": targets, "weight": event_weight(targets[0], event["velocity"])}
            flam = next((previous for previous in reversed(folded) if row["time_s"]-previous["time_s"] <= FLAM_WINDOW_S
                         and previous["targets"][0] == targets[0]), None)
            if flam is None:
                folded.append(row)
                continue
            softer, louder = sorted((flam, row), key=lambda item: item["velocity"])
            folded[folded.index(flam)] = louder
            dropped.append({key: softer[key] for key in ("time_s", "note", "velocity")} | {"reason": "merged_flam"})
    return {"events": folded, "dropped": dropped}


class StrokeTable:
    """Per-hand timing: `strokes` maps (target, arm) to (impact lead, stroke duration) in seconds."""
    def __init__(self, strokes: dict, transfer):
        self.strokes, self.transfer_seconds, self.cache = strokes, transfer, {}

    def primary(self):
        strokes = {key: value for key, value in self.strokes.items() if SKILLS[key[0]].arm == key[1]}
        return StrokeTable(strokes, self.transfer_seconds)

    def arms(self, target):
        return [arm for candidate, arm in self.strokes if candidate == target]

    def transfer(self, first, second, arm):
        key = first, second, arm
        if key not in self.cache:
            self.cache[key] = self.transfer_seconds(first, second, arm)
        return self.cache[key]


def schedule(events: list[dict], table: StrokeTable) -> dict:
    """Beam search that keeps the heaviest playable subset under each hand's stroke and transfer timing."""
    ordered = sorted(events, key=lambda row: (row["time_s"], -row["weight"]))
    idle = (None, -np.inf)
    beam = [(0., idle, idle, None)]
    for event in ordered:
        expansions = {}
        for state in beam:
            cost, left, right, history = state
            for rank, target in enumerate(event["targets"]):
                for arm in table.arms(target):
                    lead, duration = table.strokes[(target, arm)]
                    previous, busy = left if arm == "left" else right
                    start = event["time_s"]-lead
                    delay = 0. if previous in (None, target) else table.transfer(previous, target, arm)
                    if start-delay < busy-1e-8:
                        continue
                    lane = (target, start+duration)
                    choice = (lane, right) if arm == "left" else (left, lane)
                    total = (cost+SUBSTITUTION_COST*rank*event["weight"]+TRANSFER_COST_PER_S*delay
                             + CROSS_HAND_COST*event["weight"]*(arm != SKILLS[target].arm))
                    key = choice[0][0], round(choice[0][1], 4), choice[1][0], round(choice[1][1], 4)
                    if key not in expansions or total < expansions[key][0]:
                        expansions[key] = (total, *choice, (history, event, target, arm))
            key = left[0], round(left[1], 4), right[0], round(right[1], 4)
            total = cost+event["weight"]
            if key not in expansions or total < expansions[key][0]:
                expansions[key] = (total, left, right, (history, event, None, None))
        beam = sorted(expansions.values(), key=lambda row: row[0])[:BEAM_WIDTH]
    kept, dropped = [], []
    history = beam[0][3]
    while history is not None:
        history, event, target, arm = history
        source = {key: event[key] for key in ("time_s", "note", "velocity")}
        if target is None:
            available = any(table.arms(candidate) for candidate in event["targets"])
            dropped.append({**source, "weight": event["weight"], "reason": "hand_busy" if available else "no_calibrated_stroke"})
        else:
            row = {**source, "target": target, "arm": arm, "weight": event["weight"]}
            if target != event["targets"][0]:
                row["substituted_from"] = event["targets"][0]
            kept.append(row)
    return {"kept": kept[::-1], "dropped": dropped[::-1]}


def write_arrangement(path: Path, kept: list[dict], duration_s: float) -> None:
    canonical = {target: int(note) for note, target in json.loads(CANONICAL_MAPPING.read_text())["notes"].items()}
    write_segment(path, [{"time_s": row["time_s"], "note": canonical[row["target"]], "velocity": row["velocity"]}
                         for row in kept], 0., 1., duration_s=duration_s)


def stroke_table(model, metadata, profiles: Path, planner: TransferPlanner) -> StrokeTable:
    _, references, _ = load_strokes(metadata, profiles, list(SKILLS))
    strokes = {key: (reference.impact_time, reference.duration) for key, reference in references.items()}

    def transfer(first, second, arm):
        selected = slice(0, 7) if arm == "left" else slice(7, 14)
        initial, ready = np.zeros(14), np.zeros(14)
        initial[selected] = references[(first, arm)].positions[-1, selected]
        ready[selected] = references[(second, arm)].positions[0, selected]
        try:
            return float(planner.plan(initial, ready, arm).ends[-1])
        except ValueError:
            return np.inf

    return StrokeTable(strokes, transfer)


@cache
def arrangement_context(scene_directory: str, profiles: str):
    """Model, planner and transfer times are shared by every file arranged for one scene and profile set."""
    model = mujoco.MjModel.from_binary_path(str(Path(scene_directory)/"model.mjb"))
    metadata = json.loads((Path(scene_directory)/"scene.json").read_text())
    planner = TransferPlanner.for_profiles(model, metadata, profiles)
    return model, metadata, planner, stroke_table(model, metadata, profiles, planner)


def compile_arrangement(arranged, kept, duration, model, metadata, profiles, planner) -> tuple[list, list, float]:
    """Drop the lightest nearby note at each reported path failure until the two-arm path compiles."""
    kept, dropped = list(kept), []
    for _ in range(MAX_PATH_REPAIRS):
        write_arrangement(arranged, kept, duration)
        try:
            reference = PerformanceReference(model, metadata, ContactScore.read(arranged, CANONICAL_MAPPING),
                                             profiles, planner=planner)
            return kept, dropped, reference.duration
        except ScoreFeasibilityError as error:
            if len(kept) == 1:
                raise
            time_s = error.time_s if error.time_s is not None else kept[0]["time_s"]
            nearby = [row for row in kept if abs(row["time_s"]-time_s) <= .5] or kept
            victim = min(nearby, key=lambda row: (row["weight"], abs(row["time_s"]-time_s)))
            kept.remove(victim)
            dropped.append({**{key: victim[key] for key in ("time_s", "note", "velocity")},
                            "reason": "path_infeasible", "detail": str(error)})
    raise RuntimeError("Arrangement repair exceeded its limit; inspect stroke and transfer profiles")


def arrange_midi(source: Path, scene_directory: Path, profiles: Path, output: Path, mapping: Path | None = None,
                 tempo_scale: float = 1.) -> tuple[Path, dict]:
    """Write `arranged.mid` and `arrangement.json`; every kept note compiles to a checked two-arm path."""
    document = json.loads(Path(PROJECT_MAPPING if mapping is None else mapping).read_text())
    events, duration = read_source_events(source, tempo_scale, document["channel"])
    folded = fold_events(events, document)
    model, metadata, planner, table = arrangement_context(str(scene_directory), str(profiles))
    Path(output).mkdir(parents=True, exist_ok=True)
    arranged = Path(output)/"arranged.mid"
    best = None
    # A cross-hand stroke can park one arm in the other's path, so a failed plan is retried with primary hands.
    for candidate in (table, table.primary()):
        result = schedule(folded["events"], candidate)
        if not result["kept"]:
            continue
        kept, repaired, performance_duration = compile_arrangement(arranged, result["kept"], duration, model, metadata, profiles, planner)
        if best is None or sum(row["weight"] for row in kept) > sum(row["weight"] for row in best[0]):
            best = kept, result["dropped"]+repaired, performance_duration
        if not repaired:
            break
    if best is None:
        raise ValueError("No percussion event has a calibrated stick stroke")
    kept, scheduled_drops, performance_duration = best
    write_arrangement(arranged, kept, duration)
    dropped = folded["dropped"]+scheduled_drops
    stick = len(events)-sum(row["reason"] in ("foot_event", "no_kit_piece") for row in dropped)
    report = {"schema": "openarm.midi_arrangement.v1", "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
              "tempo_scale": tempo_scale, "duration_s": duration, "source_events": len(events), "stick_events": stick,
              "kept_events": len(kept), "substituted_events": sum("substituted_from" in row for row in kept),
              "stick_coverage": len(kept)/max(1, stick), "dropped_by_reason": dict(Counter(row["reason"] for row in dropped)),
              "kept_by_target": dict(Counter(row["target"] for row in kept)),
              "performance_duration_s": performance_duration, "kept": kept,
              "dropped": sorted(dropped, key=lambda row: row["time_s"])}
    (Path(output)/"arrangement.json").write_text(json.dumps(report, indent=2))
    return arranged, report


@budgeted("openarm_rl.music.arrangement")
def main():
    parser = argparse.ArgumentParser(description="Arrange a complete MIDI drum part for the available sticks and calibrated strokes")
    parser.add_argument("--midi", type=Path, required=True)
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--tempo-scale", type=float, default=1.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    _, report = arrange_midi(args.midi, args.scene_directory, args.profiles, args.output, args.mapping, args.tempo_scale)
    print(json.dumps({key: value for key, value in report.items() if key not in ("kept", "dropped")}), flush=True)


if __name__ == "__main__":
    main()
