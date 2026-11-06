from copy import deepcopy

import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecNormalize

from .asymmetric import AsymmetricNetworks, AsymmetricStrikePolicy
from ..music.observations import score_observation_scales


class ScoreNormalize(VecNormalize):
    def floor_statistics(self):
        scales = score_observation_scales()
        for name in self.obs_rms:
            stats = self.obs_rms[name]
            floor = np.full_like(stats.var, .01**2)
            floor[:269] = scales**2
            stats.var = np.maximum(stats.var, floor)
            stats.mean[189:269] = 0
            stats.var[189:269] = 1

    def normalize_obs(self, observation):
        self.floor_statistics()
        return super().normalize_obs(observation)


class ScoreFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, sum(int(np.prod(s.shape)) for s in observation_space.spaces.values()))

    def forward(self, observations):
        return torch.cat((observations["actor"], observations["critic"], observations["nominal"]), dim=-1)


class ScoreNetworks(AsymmetricNetworks):
    def __init__(self, actor_dim, critic_dim):
        super().__init__(actor_dim, critic_dim)
        self.critic_dim = critic_dim
        self.latent_dim_pi = 142

    def forward_actor(self, features):
        return torch.cat((self.policy_net(features[..., :self.actor_dim]), features[..., -14:]), dim=-1)

    def forward_critic(self, features):
        return self.value_net(features[..., self.actor_dim:self.actor_dim+self.critic_dim])


class RhythmicPolicy(AsymmetricStrikePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.no_grad():
            self.action_net.weight[:, -14:] = torch.eye(14, device=self.device)

    def make_features_extractor(self):
        return ScoreFeatures(self.observation_space)

    def _build_mlp_extractor(self):
        self.mlp_extractor = ScoreNetworks(self.observation_space["actor"].shape[0],
                                           self.observation_space["critic"].shape[0]).to(self.device)


class RhythmActor(nn.Module):
    def __init__(self, policy, normalizer):
        super().__init__()
        self.network = deepcopy(policy.mlp_extractor.policy_net).cpu()
        self.output = deepcopy(policy.action_net).cpu()
        stats = normalizer.obs_rms["actor"]
        self.register_buffer("mean", torch.tensor(stats.mean, dtype=torch.float64))
        self.register_buffer("variance", torch.tensor(stats.var, dtype=torch.float64))
        self.epsilon, self.clip = normalizer.epsilon, normalizer.clip_obs

    def forward(self, actor, nominal):
        state = torch.clamp((actor.to(torch.float64)-self.mean)/torch.sqrt(self.variance+self.epsilon), -self.clip, self.clip)
        return torch.tanh(self.output(torch.cat((self.network(state.to(torch.float32)), nominal), dim=-1)))
