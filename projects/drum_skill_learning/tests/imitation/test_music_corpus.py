import hashlib
import json

import mido
import pytest

from openarm_rl.music.corpus import percussion_events, read_manifest, write_segment
from openarm_rl.music.feedback import compare_performance
from openarm_rl.music.score import ContactScore


def test_segment_preserves_event_timing_velocity_and_relative_intervals(tmp_path):
    events = [{"time_s": 2.1, "note": 38, "velocity": 32}, {"time_s": 2.4, "note": 43, "velocity": 110}]
    path = tmp_path/"segment.mid"
    write_segment(path, events, 2., .5)
    score = ContactScore.read(path)
    assert [goal.time_s for goal in score.goals] == pytest.approx([.2, .8])
    assert [goal.velocity for goal in score.goals] == [32, 110]
    parsed = percussion_events(path.read_bytes())
    assert [row["note"] for row in parsed] == [38, 43]
    write_segment(tmp_path/"complete.mid", events, 2., .5, duration_s=2.)
    assert ContactScore.read(tmp_path/"complete.mid").duration_s == pytest.approx(4.)


def test_manifest_rejects_source_leakage_and_changed_midi(tmp_path):
    midi = tmp_path/"clip.mid"
    write_segment(midi, [{"time_s": 0., "note": 38, "velocity": 80}], 0., 1.)
    clip = {"id": "a", "split": "train", "source_id": "performance1", "source_sha256": hashlib.sha256(midi.read_bytes()).hexdigest(),
            "targets": ["snare"],
            "midi": "clip.mid", "midi_sha256": hashlib.sha256(midi.read_bytes()).hexdigest()}
    manifest = tmp_path/"manifest.json"
    document = {"schema": "openarm.midi_corpus.v1", "clips": [clip]}
    manifest.write_text(json.dumps(document))
    assert len(read_manifest(manifest)["clips"]) == 1
    document["clips"].append({**clip, "id": "b", "split": "test"})
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="crosses"):
        read_manifest(manifest)
    document["clips"] = [clip]
    manifest.write_text(json.dumps(document))
    midi.write_bytes(b"changed contents")
    with pytest.raises(ValueError, match="checksum"):
        read_manifest(manifest)


def test_performed_midi_feedback_has_timing_velocity_and_duplicate_penalties(tmp_path):
    target, performed = tmp_path/"target.mid", tmp_path/"performed.mid"
    write_segment(target, [{"time_s": .5, "note": 38, "velocity": 80}], 0., 1.)
    write_segment(performed, [{"time_s": .51, "note": 38, "velocity": 90},
                              {"time_s": .52, "note": 38, "velocity": 90}], 0., 1.)
    result = compare_performance(target, performed)
    assert result["correct_hits"] == 1 and result["extra_hits"] == 1
    assert result["velocity_mae"] == 10
    assert result["timing_mae_s"] == pytest.approx(.01)
    silent = mido.MidiFile()
    silent.tracks.append(mido.MidiTrack())
    silent.save(tmp_path/"silence.mid")
    silence = compare_performance(target, tmp_path/"silence.mid")
    assert silence["actual_hits"] == 0 and silence["missed_hits"] == 1
