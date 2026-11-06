import numpy as np
import pytest

from openarm_rl.recording.music_audio import RATE, VOICES, drum_voice, mix_contacts


@pytest.mark.parametrize("target", sorted(VOICES))
def test_every_voice_decays_before_it_ends(target):
    pulse = drum_voice(target, 3)
    assert np.max(np.abs(pulse[-50:])) < .01*np.max(np.abs(pulse))


def test_contact_lands_at_its_performance_time():
    samples = mix_contacts([{"target": "snare", "time_s": 1.}], preroll=.5, duration=3.)
    onset = int(np.argmax(np.abs(samples) > .01*np.max(np.abs(samples))))
    assert onset/RATE == pytest.approx(1.5, abs=.005)


def test_backing_is_mixed_only_when_supplied():
    hits = [{"target": "snare", "time_s": .2}]
    backing = [{"target": "kick", "time_s": 1., "midi_velocity": 100}]
    late = slice(int(1.*RATE), int(1.2*RATE))
    assert np.max(np.abs(mix_contacts(hits, 0., 2.)[late])) < 1e-6
    assert np.max(np.abs(mix_contacts(hits, 0., 2., backing)[late])) > .01


def test_kick_is_not_a_robot_contact_voice_by_accident():
    assert drum_voice("kick", 0).shape[0] > 0
    assert "kick" in VOICES
