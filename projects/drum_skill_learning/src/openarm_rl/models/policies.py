import torch
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class SquashedResidualPolicy(ActorCriticPolicy):
    def __init__(self, *args, sde_epsilon=1e-12, **kwargs):
        if not 0 < sde_epsilon <= 1e-6:
            raise ValueError("SDE epsilon must lie in (0, 1e-6]")
        self.sde_epsilon = sde_epsilon
        if kwargs.get("use_sde", False):
            kwargs["squash_output"] = True
        super().__init__(*args, **kwargs)
        if not self.use_sde:
            self.action_dist = SquashedDiagGaussianDistribution(self.action_space.shape[0])
        self._squash_output = True
        torch.nn.init.zeros_(self.action_net.weight)
        torch.nn.init.zeros_(self.action_net.bias)

    def _build(self, lr_schedule):
        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(self.action_space.shape[0],
                epsilon=self.sde_epsilon, **self.dist_kwargs)
        super()._build(lr_schedule)

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), "sde_epsilon": self.sde_epsilon}

    def evaluate_actions(self, obs, actions):
        values, log_probability, entropy = super().evaluate_actions(obs, actions)
        if self.use_sde:
            return values, log_probability, entropy
        sampled = self.action_dist.sample()
        entropy = -self.action_dist.log_prob(sampled, self.action_dist.gaussian_actions)
        return values, log_probability, entropy


def load_residual_policy(path, **kwargs):
    return PPO.load(path, custom_objects={"policy_class": SquashedResidualPolicy}, **kwargs)
