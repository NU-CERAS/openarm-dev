import json

import mido
import mujoco
import pytest

from openarm_rl.envs.camera_free_strike import CameraFreeStrikeEnv
from openarm_rl.planning.scene import prepare_scene
from openarm_rl.planning.strikes import plan_center
from openarm_rl.skills.registry import SKILLS


@pytest.fixture(scope="session")
def native_strike_fixture(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-camera-free")
    prepare_scene(directory / "scene")
    model = mujoco.MjModel.from_binary_path(str(directory / "scene/model.mjb"))
    metadata = json.loads((directory / "scene/scene.json").read_text())
    profile = plan_center(model, metadata["robot"], metadata["scene"], SKILLS["snare"])
    assert profile["status"] == "ready", profile
    (directory / "snare.json").write_text(json.dumps(profile))
    return directory


@pytest.fixture(scope="session")
def rhythm_options(native_strike_fixture, tmp_path_factory):
    output = tmp_path_factory.mktemp("rhythm")
    profile = json.loads((native_strike_fixture / "snare.json").read_text())
    profile["configuration"]["retract_seconds"] = .28
    (output / "source.json").write_text(json.dumps(profile))
    with CameraFreeStrikeEnv(native_strike_fixture / "scene", output / "source.json",
                             position_range=0, domain_range=0) as env:
        env.reset(seed=1)
        while True:
            _, _, done, truncated, info = env.step(env.reference_action())
            if done or truncated:
                break
        assert info["hit_count"] == 1 and not info["unsafe_contact"]
        assert info["hits"][0]["contact_duration_s"] <= .12
        reference = env.reference.document()
        reference["impact_time_s"] = info["hits"][0]["onset_time_s"]
        profile = dict(env.profile)
        profile["reference_bank"] = {"schema": "openarm.speed_conditioned_strike.v1",
                                     "speeds_m_s": [.35, .85], "references": [reference, reference]}
    (output / "snare.json").write_text(json.dumps(profile))
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.extend([mido.Message("note_on", channel=9, note=38, velocity=90),
                  mido.Message("note_off", channel=9, note=38, velocity=0, time=1),
                  mido.Message("note_on", channel=9, note=38, velocity=90, time=3839)])
    midi.save(output / "score.mid")
    return {"scene_directory": str(native_strike_fixture / "scene"), "profiles": str(output),
            "midi_file": str(output / "score.mid")}
