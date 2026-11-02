import torch
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class SquashedResidualPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_dist = SquashedDiagGaussianDistribution(self.action_space.shape[0])
        self._squash_output = True
        torch.nn.init.zeros_(self.action_net.weight)
        torch.nn.init.zeros_(self.action_net.bias)

    def evaluate_actions(self, obs, actions):
        values, log_probability, _ = super().evaluate_actions(obs, actions)
        sampled = self.action_dist.sample()
        entropy = -self.action_dist.log_prob(sampled, self.action_dist.gaussian_actions)
        return values, log_probability, entropy


def load_residual_policy(path, **kwargs):
    return PPO.load(path, custom_objects={"policy_class": SquashedResidualPolicy}, **kwargs)
