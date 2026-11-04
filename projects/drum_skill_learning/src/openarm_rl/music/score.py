from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import mido
import numpy as np

from ..paths import PROJECT_ROOT
from ..skills.registry import SKILLS


MATCH_TOLERANCE_S = .06
# A contact is confirmed after its onset, so a goal stays open briefly past the matching window.
CONFIRMATION_GRACE_S = .02
MATCH_HORIZON_S = MATCH_TOLERANCE_S+CONFIRMATION_GRACE_S


@dataclass(frozen=True)
class ContactGoal:
    event_id: int
    time_s: float
    target: str
    velocity: int
    note: int


class ContactScore:
    def __init__(self, goals, source_sha256, tempo_scale=1., allow_empty=False, duration_s=None):
        self.goals = tuple(goals)
        if (not self.goals and not allow_empty) or any(g.target not in SKILLS or g.time_s < 0 or not np.isfinite(g.time_s)
                                 or not 1 <= g.velocity <= 127 for g in self.goals):
            raise ValueError("Score requires finite timed contacts on supported drums")
        if any(a.time_s > b.time_s for a, b in zip(self.goals, self.goals[1:])):
            raise ValueError("Score events must be ordered by absolute time")
        if [g.event_id for g in self.goals] != list(range(len(self.goals))):
            raise ValueError("Score event identifiers must be consecutive")
        self.source_sha256, self.tempo_scale = source_sha256, tempo_scale
        self.times = [goal.time_s for goal in self.goals]
        self.duration_s = float(duration_s if duration_s is not None else max((goal.time_s for goal in self.goals), default=0.))
        if not np.isfinite(self.duration_s) or self.duration_s < max((goal.time_s for goal in self.goals), default=0.)-1e-8:
            raise ValueError("Score duration must include every percussion event")

    @classmethod
    def read(cls, path, mapping=None, tempo_scale=1., allow_empty=False):
        if not np.isfinite(tempo_scale) or tempo_scale <= 0:
            raise ValueError("Tempo scale must be positive and finite")
        path = Path(path)
        mapping = PROJECT_ROOT / "configs/music/percussion.json" if mapping is None else Path(mapping)
        document = json.loads(mapping.read_text())
        if document["schema"] != "openarm.percussion_mapping.v1" or not 0 <= document["channel"] <= 15:
            raise ValueError("Invalid MIDI percussion mapping")
        notes = {int(note): target for note, target in document["notes"].items()}
        if any(target not in SKILLS for target in notes.values()):
            raise ValueError("Mapping contains an unsupported physical drum")
        midi = mido.MidiFile(path)
        if midi.type not in (0, 1):
            raise ValueError("Synchronous MIDI type 0 or 1 is required")
        elapsed, goals, unsupported = 0., [], set()
        for message in midi:
            elapsed += message.time / tempo_scale
            if message.type != "note_on" or message.velocity == 0 or message.channel != document["channel"]:
                continue
            if message.note not in notes:
                unsupported.add(message.note)
                continue
            goals.append(ContactGoal(len(goals), elapsed, notes[message.note], message.velocity, message.note))
        if unsupported:
            raise ValueError(f"Unmapped percussion notes require an explicit mapping or arrangement: {sorted(unsupported)}")
        return cls(goals, hashlib.sha256(path.read_bytes()).hexdigest(), tempo_scale, allow_empty,
                   duration_s=midi.length/tempo_scale)

    def document(self):
        return {"schema": "openarm.contact_score.v1", "midi_sha256": self.source_sha256,
                "tempo_scale": self.tempo_scale, "duration_s": self.duration_s,
                "goals": [asdict(goal) for goal in self.goals]}

    def future(self, time_s, count=8):
        rows = np.zeros((count, len(SKILLS) + 3), np.float32)
        targets = list(SKILLS)
        first = bisect_left(self.times, time_s-MATCH_TOLERANCE_S)
        future = self.goals[first:first+count]
        for row, goal in zip(rows, future):
            row[targets.index(goal.target)] = 1
            row[-3:] = [goal.time_s - time_s, goal.velocity / 127, 1]
        return rows.ravel()


class ContactMatcher:
    def __init__(self, goals, tolerance=MATCH_TOLERANCE_S):
        self.goals, self.tolerance = tuple(goals), tolerance
        self.times = [goal.time_s for goal in self.goals]
        self.matched, self.missed, self.actual = {}, set(), []
        self.expired_until = 0

    def hit(self, target, time_s, arm):
        first, last = bisect_left(self.times, time_s-self.tolerance), bisect_right(self.times, time_s+self.tolerance)
        candidates = [g for g in self.goals[first:last] if g.event_id not in self.matched and g.event_id not in self.missed
                      and g.target == target and abs(g.time_s - time_s) <= self.tolerance]
        goal = min(candidates, key=lambda g: abs(g.time_s-time_s)) if candidates else None
        event = {"target": target, "time_s": time_s, "arm": arm,
                 "matched_event_id": None if goal is None else goal.event_id}
        self.actual.append(event)
        if goal is not None:
            self.matched[goal.event_id] = time_s - goal.time_s
        return goal is not None

    def expire(self, time_s):
        limit = bisect_left(self.times, time_s-self.tolerance-CONFIRMATION_GRACE_S)
        expired = {g.event_id for g in self.goals[self.expired_until:limit] if g.event_id not in self.matched} - self.missed
        self.expired_until = max(self.expired_until, limit)
        self.missed.update(expired)
        return len(expired)

    def active(self, time_s):
        return self.goals[bisect_left(self.times, time_s-self.tolerance):bisect_right(self.times, time_s+self.tolerance)]

    def report(self, detailed=False):
        tp, count, actual = len(self.matched), len(self.goals), len(self.actual)
        precision, recall = tp / max(1, actual), tp / max(1, count)
        errors = np.abs(list(self.matched.values()))
        per_drum = {}
        for target in sorted({g.target for g in self.goals} | {hit["target"] for hit in self.actual}):
            expected = sum(g.target == target for g in self.goals)
            hits = [hit for hit in self.actual if hit["target"] == target]
            matched = sum(hit["matched_event_id"] is not None for hit in hits)
            p, r = matched/max(1, len(hits)), matched/max(1, expected)
            per_drum[target] = {"target_events": expected, "actual_hits": len(hits), "correct_hits": matched,
                                "precision": p, "recall": r, "f1": 2*p*r/max(1e-12, p+r)}
        report = {"target_events": count, "actual_hits": actual, "correct_hits": tp, "per_drum": per_drum,
                "missed_hits": count-tp, "extra_hits": actual-tp, "precision": precision, "recall": recall,
                "f1": 2*precision*recall/max(1e-12, precision+recall),
                "timing_mae_s": float(errors.mean()) if len(errors) else None,
                "timing_p95_s": float(np.quantile(errors, .95)) if len(errors) else None,
                "hits": self.actual, "timing_errors_s": list(self.matched.values())}
        if detailed:
            hits = {hit["matched_event_id"]: hit for hit in self.actual if hit["matched_event_id"] is not None}
            report["events"] = [{**asdict(goal),
                "status": "hit" if goal.event_id in hits else "missed" if goal.event_id in self.missed else "pending",
                "actual_time_s": hits[goal.event_id]["time_s"] if goal.event_id in hits else None,
                "timing_error_s": self.matched.get(goal.event_id),
                "actual_arm": hits[goal.event_id]["arm"] if goal.event_id in hits else None,
                "measured_midi_velocity": hits[goal.event_id].get("midi_velocity") if goal.event_id in hits else None}
                for goal in self.goals]
        return report
