import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from .policies import SquashedResidualPolicy


class ObservationFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, sum(int(np.prod(s.shape)) for s in observation_space.spaces.values()))

    def forward(self, observations):
        return torch.cat((observations["actor"], observations["critic"]), dim=-1)


class AsymmetricNetworks(nn.Module):
    def __init__(self, actor_dim, critic_dim):
        super().__init__()
        self.actor_dim = actor_dim
        self.latent_dim_pi = self.latent_dim_vf = 128
        self.policy_net = nn.Sequential(nn.Linear(actor_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        self.value_net = nn.Sequential(nn.Linear(critic_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())

    def forward_actor(self, features):
        return self.policy_net(features[..., :self.actor_dim])

    def forward_critic(self, features):
        return self.value_net(features[..., self.actor_dim:])

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class AsymmetricStrikePolicy(SquashedResidualPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = ObservationFeatures
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = AsymmetricNetworks(self.observation_space["actor"].shape[0], self.observation_space["critic"].shape[0]).to(self.device)


class DeployableActor(nn.Module):
    def __init__(self, policy, normalizer):
        super().__init__()
        from copy import deepcopy

        self.network = deepcopy(policy.mlp_extractor.policy_net).cpu()
        self.output = deepcopy(policy.action_net).cpu()
        stats = normalizer.obs_rms["actor"]
        self.register_buffer("mean", torch.tensor(stats.mean, dtype=torch.float64))
        self.register_buffer("variance", torch.tensor(stats.var, dtype=torch.float64))
        self.epsilon = normalizer.epsilon
        self.clip = normalizer.clip_obs

    def forward(self, observation):
        normalized = torch.clamp((observation.to(self.mean.dtype) - self.mean) / torch.sqrt(self.variance + self.epsilon), -self.clip, self.clip)
        return torch.tanh(self.output(self.network(normalized.to(torch.float32))))
