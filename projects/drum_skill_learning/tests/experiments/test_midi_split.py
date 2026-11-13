import hashlib

import mido

from openarm_rl.experiments.compare import musical_score_sha256


def write_score(path, *, ticks=480, title="original", second_onset_s=.5, second_velocity=95, release_ticks=10):
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks)
    second_tick = round(mido.second2tick(second_onset_s, ticks, 500000))
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage("track_name", name=title),
        mido.MetaMessage("set_tempo", tempo=500000),
        mido.Message("note_on", channel=9, note=38, velocity=90),
        mido.Message("note_off", channel=9, note=38, velocity=0, time=release_ticks),
        mido.Message("note_on", channel=9, note=38, velocity=second_velocity, time=second_tick-release_ticks),
        mido.Message("note_off", channel=9, note=38, velocity=0, time=release_ticks),
    ]))
    midi.save(path)
    return path


def test_metadata_and_note_off_changes_do_not_create_held_out_musical_score(tmp_path):
    original = write_score(tmp_path/"train.mid")
    renamed = write_score(tmp_path/"renamed.mid", title="nominally held out", release_ticks=50)
    assert hashlib.sha256(original.read_bytes()).hexdigest() != hashlib.sha256(renamed.read_bytes()).hexdigest()
    assert musical_score_sha256(original) == musical_score_sha256(renamed)


def test_equivalent_tick_resolution_and_nonpercussion_track_are_same_target(tmp_path):
    original = write_score(tmp_path/"train.mid")
    reencoded = write_score(tmp_path/"reencoded.mid", ticks=960)
    midi = mido.MidiFile(reencoded)
    midi.tracks.append(mido.MidiTrack([
        mido.Message("note_on", channel=0, note=60, velocity=60),
        mido.Message("note_off", channel=0, note=60, time=480),
    ]))
    midi.save(reencoded)
    assert musical_score_sha256(original) == musical_score_sha256(reencoded)


def test_changed_contact_timing_or_intensity_is_a_different_musical_target(tmp_path):
    original = write_score(tmp_path/"train.mid")
    retimed = write_score(tmp_path/"retimed.mid", second_onset_s=.6)
    louder = write_score(tmp_path/"louder.mid", second_velocity=110)
    assert musical_score_sha256(original) != musical_score_sha256(retimed)
    assert musical_score_sha256(original) != musical_score_sha256(louder)
