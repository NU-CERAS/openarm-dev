import gymnasium as gym
import numpy as np
import torch

from openarm_sim.act_residual import SquashedResidualPolicy


def test_entropy_gradient_increases_exploration_at_small_std():
    torch.manual_seed(42)
    policy = SquashedResidualPolicy(gym.spaces.Box(-np.inf, np.inf, (234,), dtype=np.float32),
                                    gym.spaces.Box(-1, 1, (8,), dtype=np.float32), lambda _: 1e-4,
                                    log_std_init=-3)
    observation = torch.zeros((4096, 234))
    with torch.no_grad():
        actions, _, _ = policy(observation)
    _, _, entropy = policy.evaluate_actions(observation, actions)
    (-entropy.mean()).backward()
    assert torch.isfinite(entropy).all()
    assert bool((policy.log_std.grad < -0.9).all())
