import json

import mido
import pytest

from openarm_rl.music.arrangement import (PROJECT_MAPPING, StrokeTable, fold_events, read_source_events,
                                          schedule, write_arrangement)
from openarm_rl.music.arranged_corpus import heldout_supported
from openarm_rl.music.reference import PerformanceReference
from openarm_rl.music.repertoire import source_metrics
from openarm_rl.music.score import ContactScore


MAPPING = json.loads(PROJECT_MAPPING.read_text())


def table(period=.2, transfer=.3, pairs=(("snare", "left"), ("hi_hat", "left"), ("floor_tom", "right"), ("cymbal_right", "right"))):
    return StrokeTable({pair: (.1, period) for pair in pairs}, lambda first, second, arm: transfer)


def event(time_s, note, velocity=96):
    return {"time_s": time_s, "note": note, "velocity": velocity}


def test_general_midi_folding_separates_foot_notes_from_stick_notes():
    folded = fold_events([event(0., 36), event(.1, 44), event(.2, 42), event(.3, 22), event(.4, 40), event(.5, 56)], MAPPING)
    assert [row["reason"] for row in folded["dropped"]] == ["foot_event", "foot_event", "no_kit_piece"]
    assert [row["targets"][0] for row in folded["events"]] == ["hi_hat", "hi_hat", "snare"]


def test_near_simultaneous_hits_on_one_drum_merge_into_the_loudest():
    folded = fold_events([event(1., 38, 40), event(1.012, 38, 110), event(1.2, 38, 60)], MAPPING)
    assert [(row["time_s"], row["velocity"]) for row in folded["events"]] == [(1.012, 110), (1.2, 60)]
    assert [row["reason"] for row in folded["dropped"]] == ["merged_flam"]


def test_schedule_keeps_every_event_when_stroke_period_allows():
    folded = fold_events([event(.2*index, 38) for index in range(8)], MAPPING)
    result = schedule(folded["events"], table())
    assert len(result["kept"]) == 8 and not result["dropped"]
    assert {row["arm"] for row in result["kept"]} == {"left"}


def test_schedule_drops_only_hits_faster_than_the_hand():
    folded = fold_events([event(.1*index, 38) for index in range(8)], MAPPING)
    result = schedule(folded["events"], table())
    assert [row["time_s"] for row in result["kept"]] == pytest.approx([0., .2, .4, .6])
    assert {row["reason"] for row in result["dropped"]} == {"hand_busy"}


def test_snare_keeps_its_hand_and_hi_hat_moves_to_the_free_arm():
    folded = fold_events([event(0., 38), event(0., 42)], MAPPING)
    result = schedule(folded["events"], table())
    played = {row["target"]: row["arm"] for row in result["kept"]}
    assert played == {"snare": "left", "cymbal_right": "right"}
    assert [row["substituted_from"] for row in result["kept"] if row["target"] == "cymbal_right"] == ["hi_hat"]


def test_third_simultaneous_hit_exceeds_two_sticks():
    folded = fold_events([event(0., 38), event(0., 43), event(0., 42)], MAPPING)
    result = schedule(folded["events"], table())
    assert sorted(row["target"] for row in result["kept"]) == ["floor_tom", "snare"]
    assert [row["reason"] for row in result["dropped"]] == ["hand_busy"]


def test_transfer_time_blocks_a_drum_change_that_a_repeat_would_allow():
    folded = fold_events([event(0., 38), event(.25, 42)], MAPPING)
    pairs = (("snare", "left"), ("hi_hat", "left"))
    assert len(schedule(folded["events"], table(transfer=.3, pairs=pairs))["kept"]) == 1
    assert len(schedule(folded["events"], table(transfer=.02, pairs=pairs))["kept"]) == 2


def test_missing_kit_piece_is_reported_as_unavailable():
    folded = fold_events([event(0., 48)], MAPPING)
    result = schedule(folded["events"], table(pairs=(("snare", "left"),)))
    assert [row["reason"] for row in result["dropped"]] == ["no_calibrated_stroke"]


def test_arranged_midi_round_trips_through_the_strict_score(tmp_path):
    folded = fold_events([event(.5, 38, 70), event(1., 42, 100)], MAPPING)
    kept = schedule(folded["events"], table(transfer=.02))["kept"]
    path = tmp_path/"arranged.mid"
    write_arrangement(path, kept, 2.)
    score = ContactScore.read(path)
    assert [(goal.target, goal.velocity) for goal in score.goals] == [("snare", 70), ("hi_hat", 100)]
    assert [goal.time_s for goal in score.goals] == pytest.approx([.5, 1.], abs=1e-3)
    assert score.duration_s == pytest.approx(2., abs=1e-3)


def test_source_reader_applies_tempo_scale_to_percussion_only(tmp_path):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.Message("note_on", channel=0, note=60, velocity=90, time=0))
    track.append(mido.Message("note_on", channel=9, note=38, velocity=90, time=480))
    path = tmp_path/"song.mid"
    midi.save(path)
    events, duration = read_source_events(path, .5)
    assert [row["time_s"] for row in events] == pytest.approx([1.])
    assert duration == pytest.approx(1.)


class CountingMotion:
    def __init__(self, value):
        self.value, self.calls = value, 0

    def sample(self, elapsed):
        self.calls += 1
        import numpy as np
        return np.full(14, self.value), np.zeros(14)


def test_reference_sampling_touches_one_motion_per_arm():
    import numpy as np
    reference = PerformanceReference.__new__(PerformanceReference)
    reference.initial = np.zeros(14)
    motions = [CountingMotion(float(index)) for index in range(100)]
    reference.lanes = {"left": [(index, index+.5, motion, True) for index, motion in enumerate(motions)], "right": []}
    reference.index_lanes()
    q, _ = reference.sample(70.25)
    assert q[0] == 70. and sum(motion.calls for motion in motions) == 1
    q, dq = reference.sample(70.75)
    assert q[0] == 70. and not dq.any()
    assert reference.sample(-1.)[0][0] == 0.


def test_repertoire_metrics_score_against_source_stick_events():
    arrangement = {"stick_events": 10, "kept_events": 8, "substituted_events": 2, "source_events": 14}
    played = {"correct_hits": 6, "actual_hits": 7, "target_events": 8, "timing_mae_s": .01, "unsafe_contact": False}
    metrics = source_metrics(arrangement, played)
    assert metrics["source_recall"] == pytest.approx(.6)
    assert metrics["source_precision"] == pytest.approx(6/7)
    assert metrics["source_f1"] == pytest.approx(2*.6*(6/7)/(.6+6/7))
    assert metrics["faithful_coverage"] == pytest.approx(.6)


def test_schedule_uses_the_primary_hand_when_both_can_play():
    folded = fold_events([event(0., 48)], MAPPING)
    both = table(pairs=(("rack_tom_left", "right"), ("rack_tom_left", "left")))
    assert schedule(folded["events"], both)["kept"][0]["arm"] == "left"


def test_cross_hand_stroke_remains_available_when_the_primary_hand_is_busy():
    folded = fold_events([event(0., 38), event(0., 48)], MAPPING)
    pairs = (("snare", "left"), ("rack_tom_left", "left"), ("rack_tom_left", "right"))
    played = {row["target"]: row["arm"] for row in schedule(folded["events"], table(pairs=pairs))["kept"]}
    assert played == {"snare": "left", "rack_tom_left": "right"}


def test_primary_table_removes_cross_hand_strokes():
    both = table(pairs=(("rack_tom_left", "right"), ("rack_tom_left", "left"), ("snare", "left")))
    assert set(both.primary().strokes) == {("rack_tom_left", "left"), ("snare", "left")}


def test_arranged_corpus_keeps_heldout_drums_inside_training_coverage():
    assert heldout_supported({"targets": ["snare"], "split": "test"}, {"snare", "floor_tom"})
    assert not heldout_supported({"targets": ["hi_hat"], "split": "test"}, {"snare"})
    assert heldout_supported({"targets": ["hi_hat"], "split": "train"}, set())
