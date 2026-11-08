import json
from pathlib import Path
from types import SimpleNamespace

import mido

from openarm_rl.control.reference import JointReference
from openarm_rl.envs.rhythmic import RhythmicDrummingEnv
from openarm_rl.music.perform import reference_playback_score_only
from openarm_rl.planning.rhythmic_references import compact_reference


def test_reference_teacher_uses_the_qualified_command_path():
    # Rhythmic calibration and the hardware ReferenceRuntime both execute strokes through the
    # score-only command path; reference playback without an actor must match them.
    assert reference_playback_score_only(SimpleNamespace(score_only=False, actor=None))
    assert reference_playback_score_only(SimpleNamespace(score_only=True, actor=None))
    # An actor keeps the observation/command schema it was trained with.
    assert not reference_playback_score_only(SimpleNamespace(score_only=False, actor=Path("actor.pt")))
    assert reference_playback_score_only(SimpleNamespace(score_only=True, actor=Path("actor.pt")))


def test_compact_stroke_followed_by_a_rest_completes_in_reference_playback(rhythm_options, tmp_path):
    profile = json.loads((Path(rhythm_options["profiles"]) / "snare.json").read_text())
    stroke = JointReference.load(profile["reference_bank"]["references"][0])
    compact = compact_reference(stroke, approach=.08, retract=.05, settle=.01).document()
    profile["reference_bank"]["references"] = [compact, compact]
    (tmp_path / "snare.json").write_text(json.dumps(profile))
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    # Two isolated hits one second apart: each compact stroke ends in a held rest.
    track.extend([mido.Message("note_on", channel=9, note=38, velocity=90),
                  mido.Message("note_off", channel=9, note=38, velocity=0, time=1),
                  mido.Message("note_on", channel=9, note=38, velocity=90, time=1919),
                  mido.Message("note_off", channel=9, note=38, velocity=0, time=1)])
    midi.save(tmp_path / "score.mid")
    args = SimpleNamespace(score_only=False, actor=None)
    with RhythmicDrummingEnv(rhythm_options["scene_directory"], tmp_path, tmp_path / "score.mid",
                             score_only=reference_playback_score_only(args)) as env:
        env.reset(seed=1)
        while env.active:
            env.step(env.reference_action())
        info = env.info()
    assert info["correct_hits"] == 2 and info["missed_hits"] == 0 and info["extra_hits"] == 0, info
    assert not info["unsafe_contact"] and info["failure"] is None and info["is_success"], info
