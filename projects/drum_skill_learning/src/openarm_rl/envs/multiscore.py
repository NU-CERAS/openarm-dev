from pathlib import Path

import gymnasium as gym
import numpy as np

from ..music.corpus import read_manifest
from .rhythmic import RhythmicDrummingEnv


class MultiScoreDrummingEnv(gym.Env):
    metadata = RhythmicDrummingEnv.metadata

    def __init__(self, corpus_manifest, corpus_split="train", clip_cache=32, **options):
        self.manifest = Path(corpus_manifest)
        document = read_manifest(self.manifest)
        self.tempo_scale = document["tempo_scale"]
        self.clips = [clip for clip in document["clips"] if clip["split"] == corpus_split]
        if not self.clips:
            raise ValueError(f"No clips for corpus split {corpus_split}")
        self.options = {**options, "mapping": self.manifest.parent/document["mapping"], "score_only": True}
        if clip_cache < 1:
            raise ValueError("Clip cache must hold at least one environment")
        # Compiling a clip costs as much as hundreds of control steps; workers revisit clips throughout training.
        self.clip_cache, self.cached = clip_cache, {}
        self.clip = self.clips[0]
        self.env = self.select(self.clip)
        self.observation_space, self.action_space = self.env.observation_space, self.env.action_space

    def create(self, clip):
        source = next(iter(self.cached.values()), None)
        if source is not None:
            source.restore_nominal()
        return RhythmicDrummingEnv(**self.options, midi_file=self.manifest.parent/clip["midi"], source=source)

    def select(self, clip):
        environment = self.cached.pop(clip["id"], None) or self.create(clip)
        self.cached[clip["id"]] = environment
        while len(self.cached) > self.clip_cache:
            self.cached.pop(next(iter(self.cached))).close()
        return environment

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "env"), name)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options is None:
            clip = self.clips[int(self.np_random.integers(len(self.clips)))]
        else:
            if set(options) != {"clip_id"}:
                raise ValueError("Multi-score reset accepts only a clip_id")
            choices = [clip for clip in self.clips if clip["id"] == options["clip_id"]]
            if len(choices) != 1:
                raise ValueError("Requested clip is outside this corpus split")
            clip = choices[0]
        if clip["id"] != self.clip["id"]:
            self.env = self.select(clip)
            self.clip = clip
        observation, info = self.env.reset(seed=seed)
        return observation, self.annotate(info)

    def annotate(self, info):
        return {**info, "clip_id": self.clip["id"], "source_id": self.clip["source_id"],
                "corpus_split": self.clip["split"], "tempo_scale": self.tempo_scale}

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, self.annotate(info)

    def render(self):
        return self.env.render()

    def close(self):
        for environment in self.cached.values():
            environment.close()
        self.cached.clear()


def make_rhythm_env(**options):
    return MultiScoreDrummingEnv(**options) if "corpus_manifest" in options else RhythmicDrummingEnv(**options)
