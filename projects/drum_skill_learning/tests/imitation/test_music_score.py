import mido
import pytest

from openarm_rl.music.reference import SETTLE_TAIL_S
from openarm_rl.music.score import ContactGoal, ContactMatcher, ContactScore


def test_midi_tempo_changes_simultaneous_events_and_velocity_zero(tmp_path):
    midi = mido.MidiFile(ticks_per_beat=480)
    tempo, notes = mido.MidiTrack(), mido.MidiTrack()
    midi.tracks.extend((tempo, notes))
    tempo.extend([mido.MetaMessage("set_tempo", tempo=500000), mido.MetaMessage("set_tempo", tempo=1000000, time=480)])
    notes.extend([mido.Message("note_on", channel=9, note=38, velocity=90, time=480),
                  mido.Message("note_on", channel=9, note=43, velocity=80, time=0),
                  mido.Message("note_on", channel=9, note=38, velocity=0, time=0),
                  mido.Message("note_on", channel=9, note=38, velocity=70, time=480)])
    path = tmp_path / "tempo.mid"
    midi.save(path)
    score = ContactScore.read(path)
    assert [g.time_s for g in score.goals] == [.5, .5, 1.5]
    assert [g.velocity for g in score.goals] == [90, 80, 70]
    assert [g.time_s for g in ContactScore.read(path, tempo_scale=2).goals] == [.25, .25, .75]
    notes.append(mido.Message("note_on", channel=9, note=36, velocity=90))
    midi.save(path)
    with pytest.raises(ValueError, match="Unmapped percussion notes"):
        ContactScore.read(path)


def test_contact_matching_counts_each_goal_and_contact_once():
    goals = [ContactGoal(0, .5, "snare", 90, 38), ContactGoal(1, 1., "floor_tom", 90, 43)]
    matcher = ContactMatcher(goals)
    assert matcher.hit("snare", .51, "left")
    assert not matcher.hit("snare", .52, "left")
    assert matcher.expire(1.1) == 1
    assert matcher.expire(1.2) == 0
    report = matcher.report()
    assert report["correct_hits"] == 1
    assert report["extra_hits"] == 1
    assert report["missed_hits"] == 1
    assert report["f1"] == .5
    detailed = matcher.report(detailed=True)
    assert [event["status"] for event in detailed["events"]] == ["hit", "missed"]
    assert detailed["events"][0]["actual_arm"] == "left"
    assert detailed["events"][0]["timing_error_s"] == pytest.approx(.01)
    assert detailed["events"][1]["actual_time_s"] is None


def test_complete_midi_preserves_final_rest_and_tempo_scale(tmp_path):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.extend([mido.Message("note_on", channel=9, note=38, velocity=80),
                  mido.Message("note_off", channel=9, note=38, time=240),
                  mido.MetaMessage("end_of_track", time=1680)])
    midi.save(tmp_path/"rest.mid")
    assert ContactScore.read(tmp_path/"rest.mid").duration_s == pytest.approx(2.)
    assert ContactScore.read(tmp_path/"rest.mid", tempo_scale=.5).duration_s == pytest.approx(4.)


def test_contact_confirmed_after_window_still_matches_its_onset():
    goal = ContactGoal(0, 1., "snare", 96, 38)
    matcher = ContactMatcher([goal])
    assert matcher.expire(1.065) == 0
    assert matcher.hit("snare", 1.055, "left")
    assert matcher.report()["missed_hits"] == 0


def test_expired_goal_is_counted_once_after_confirmation_grace():
    matcher = ContactMatcher([ContactGoal(0, 1., "snare", 96, 38)])
    assert matcher.expire(1.2) == 1
    assert matcher.expire(1.3) == 0
    assert not matcher.hit("snare", 1.21, "left")


def test_score_future_matches_linear_selection_for_long_scores():
    goals = [ContactGoal(index, .25*index, "snare", 96, 38) for index in range(40)]
    score = ContactScore(goals, "test")
    rows = score.future(2.03, count=2).reshape(2, -1)
    assert rows[0, -3] == pytest.approx(2.-2.03)
    assert rows[1, -3] == pytest.approx(2.25-2.03)


def test_performance_leaves_settling_time_after_the_final_stroke():
    assert SETTLE_TAIL_S >= .4
