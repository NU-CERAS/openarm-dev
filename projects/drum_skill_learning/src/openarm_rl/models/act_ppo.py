from copy import deepcopy

import torch
from torch import nn

from .act import ACTConfig, ActionChunkTransformer
from .policies import SquashedResidualPolicy
from .rhythmic import ScoreFeatures


class DifferentiableACTActor(nn.Module):
    def __init__(self, specification):
        super().__init__()
        config = ACTConfig(**specification["config"])
        if (config.state_dim != 283 or config.action_dim != 14 or not config.relative_commands
                or config.action_semantics != "normalized_episode_anchor_residual" or config.vision):
            raise ValueError("ACT PPO requires encoder-anchor score control")
        self.act = ActionChunkTransformer(config,
            specification["state_mean"], specification["state_std"],
            specification["action_mean"], specification["action_std"])

    def forward(self, state):
        prediction, _, _ = self.act(state)
        commands = prediction[:, 0]*self.act.action_std+self.act.action_mean
        commands = commands+torch.tanh(state[:, -14:])
        return torch.atanh(commands.clamp(-.999, .999))


class ACTScoreNetworks(nn.Module):
    def __init__(self, actor_dim, critic_dim, specification):
        super().__init__()
        if actor_dim != 269 or critic_dim != 344:
            raise ValueError("ACT PPO requires score-only actor and asymmetric critic observations")
        self.actor_dim, self.critic_dim = actor_dim, critic_dim
        self.latent_dim_pi, self.latent_dim_vf = 14, 128
        self.policy_net = DifferentiableACTActor(specification)
        self.value_net = nn.Sequential(nn.Linear(critic_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())

    def forward_actor(self, features):
        return self.policy_net(torch.cat((features[..., :self.actor_dim], features[..., -14:]), dim=-1))

    def forward_critic(self, features):
        return self.value_net(features[..., self.actor_dim:self.actor_dim+self.critic_dim])

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class ACTRhythmPolicy(SquashedResidualPolicy):
    def __init__(self, *args, act_specification, actor_update_scope="all", **kwargs):
        if actor_update_scope not in ("all", "head"):
            raise ValueError("Actor update scope must be all or head")
        self.act_specification = act_specification
        self.actor_update_scope = actor_update_scope
        kwargs["features_extractor_class"] = ScoreFeatures
        super().__init__(*args, **kwargs)
        self.action_net = nn.Identity()
        if actor_update_scope == "head":
            for name, parameter in self.mlp_extractor.policy_net.act.named_parameters():
                parameter.requires_grad_(name.startswith("action_head."))
        self.optimizer = self.optimizer_class([parameter for parameter in self.parameters() if parameter.requires_grad],
                                               lr=self.optimizer.param_groups[0]["lr"],
                                               **self.optimizer_kwargs)

    def make_features_extractor(self):
        return ScoreFeatures(self.observation_space)

    def _build_mlp_extractor(self):
        self.mlp_extractor = ACTScoreNetworks(self.observation_space["actor"].shape[0],
            self.observation_space["critic"].shape[0], self.act_specification).to(self.device)

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), "act_specification": self.act_specification,
                "actor_update_scope": self.actor_update_scope}


class ACTRhythmActor(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.network = deepcopy(policy.mlp_extractor.policy_net).cpu()
        self.register_buffer("mean", torch.zeros(269))

    def forward(self, actor, nominal):
        return torch.tanh(self.network(torch.cat((actor, nominal), dim=-1)))
