import pytest

from openarm_rl.envs import multiscore
from openarm_rl.envs.multiscore import MultiScoreDrummingEnv


class FakeEnv:
    observation_space = action_space = None

    def __init__(self, midi_file, source=None, **_):
        self.midi_file, self.closed = midi_file, False

    def restore_nominal(self):
        pass

    def reset(self, seed=None):
        return {}, {}

    def close(self):
        self.closed = True


@pytest.fixture
def corpus(monkeypatch, tmp_path):
    clips = [{"id": f"c{index}", "source_id": str(index), "split": "train", "midi": f"{index}.mid"} for index in range(4)]
    monkeypatch.setattr(multiscore, "read_manifest", lambda path: {"tempo_scale": 1., "mapping": "m.json", "clips": clips})
    built = []

    def build(**options):
        built.append(FakeEnv(**options))
        return built[-1]

    monkeypatch.setattr(multiscore, "RhythmicDrummingEnv", build)
    return tmp_path/"manifest.json", built


def test_returning_to_a_clip_reuses_its_compiled_environment(corpus):
    manifest, built = corpus
    env = MultiScoreDrummingEnv(manifest)
    for clip in ("c1", "c0", "c1", "c0"):
        env.reset(options={"clip_id": clip})
    assert len(built) == 2


def test_clip_cache_closes_the_least_recent_environment_at_its_limit(corpus):
    manifest, built = corpus
    env = MultiScoreDrummingEnv(manifest, clip_cache=2)
    for clip in ("c1", "c2", "c0"):
        env.reset(options={"clip_id": clip})
    assert sum(item.closed for item in built) == 2
    assert not env.env.closed


def test_close_releases_every_cached_environment(corpus):
    manifest, built = corpus
    env = MultiScoreDrummingEnv(manifest)
    env.reset(options={"clip_id": "c1"})
    env.close()
    assert all(item.closed for item in built)
