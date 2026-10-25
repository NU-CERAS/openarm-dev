from pathlib import Path

import numpy as np
import pytest
import torch

from openarm_sim.act_data import EpisodeDataset, collect_reference, proprioception
from openarm_sim.act_model import ACTConfig, ActionChunkTransformer, normalization
from openarm_sim.act_residual import ACTResidualVecEnv, SquashedResidualPolicy
from openarm_sim.warp_pickup import WarpPickupVecEnv, build_reset_bank


@pytest.fixture(scope="module")
def demonstration():
    directory = Path(".work/tests/act")
    directory.mkdir(parents=True, exist_ok=True)
    bank = directory / "resets.npz"
    build_reset_bank(bank, 4, 24)
    report = collect_reference(bank, directory / "episodes.h5", 4, 4, 24)
    assert report["successful"] > 0
    return directory


def test_actual_demonstration_chunk_padding_and_units(demonstration):
    dataset = EpisodeDataset(demonstration / "episodes.h5", 20)
    state, command, padding, image = dataset.batch(np.array([dataset.ends[0] - 1]), device="cpu")
    assert state.shape == (1, 57) and command.shape == (1, 20, 8)
    assert not padding[0, 0] and bool(padding[0, 1:].all())
    np.testing.assert_allclose(command[0, 0], dataset.episodes[0][2][-1])
    assert image is None and bool((command[..., 7] >= 0).all())
    assert bool((command[..., 7] <= 0.044001).all())


def test_act_cvae_real_data_backward_and_checkpoint(demonstration):
    dataset = EpisodeDataset(demonstration / "episodes.h5", 20)
    model = ActionChunkTransformer(ACTConfig(), *normalization(dataset))
    state, command, padding, image = dataset.batch(np.array([0, dataset.ends[0] - 1]), device="cpu")
    loss, reconstruction, kl = model.loss(state, command, padding)
    assert bool(torch.isfinite(loss)) and reconstruction > 0 and kl >= 0
    loss.backward()
    assert model.action_head.weight.grad is not None
    model.eval()
    expected = model.commands(state)
    model.save(demonstration / "teacher.pt")
    restored = ActionChunkTransformer.load(demonstration / "teacher.pt", device="cpu")
    torch.testing.assert_close(restored.commands(state), expected)


def test_student_proprioception_uses_only_robot_feedback_and_action_history(demonstration):
    dataset = EpisodeDataset(demonstration / "episodes.h5", 20)
    observed = proprioception(dataset.state)
    assert observed.shape == (len(dataset), 24)
    np.testing.assert_array_equal(observed[:, :16], dataset.state[:, :16])
    np.testing.assert_array_equal(observed[:, 16:], dataset.state[:, 34:42])


def test_frozen_act_zero_residual_matches_command_execution(demonstration):
    raw = WarpPickupVecEnv(2, demonstration / "resets.npz")
    env = ACTResidualVecEnv(raw, demonstration / "teacher.pt")
    try:
        observation = env.reset()
        current = raw.desired[:, raw.arm_aids].clone()
        expected = current + torch.clamp(env.nominal[:, :7] - current, -0.025, 0.025)
        expected = torch.clamp(expected, raw.limits[raw.arm_aids, 0], raw.limits[raw.arm_aids, 1])
        command = env.compose(np.zeros((2, 8), dtype=np.float32))
        torch.testing.assert_close(command[:, :7], expected)
        torch.testing.assert_close(command[:, 7], env.nominal[:, 7])
        assert observation.shape == (2, 234)
        assert all(not parameter.requires_grad for parameter in env.teacher.parameters())
        next_observation, reward, _, _ = env.step(np.zeros((2, 8), dtype=np.float32))
        assert np.isfinite(next_observation).all() and np.isfinite(reward).all()
    finally:
        env.close()


def test_squashed_residual_policy_log_probability_is_consistent(demonstration):
    raw = WarpPickupVecEnv(2, demonstration / "resets.npz")
    env = ACTResidualVecEnv(raw, demonstration / "teacher.pt")
    try:
        state = torch.as_tensor(env.reset(), device="cuda")
        policy = SquashedResidualPolicy(env.observation_space, env.action_space, lambda _: 0.0001,
                                        log_std_init=-1.6).cuda()
        action, value, log_probability = policy(state)
        _, reconstructed_log_probability, _ = policy.evaluate_actions(state, action.detach())
        assert bool((action.abs() < 1).all())
        torch.testing.assert_close(log_probability, reconstructed_log_probability)
        deterministic_action, _, _ = policy(state, deterministic=True)
        torch.testing.assert_close(deterministic_action, torch.zeros_like(deterministic_action))
        policy.optimizer.zero_grad(set_to_none=True)
        (-reconstructed_log_probability.mean() + value.square().mean()).backward()
        policy.optimizer.step()
        assert bool((policy.action_net.weight != 0).any())
        assert all(parameter.grad is None for parameter in env.teacher.parameters())
    finally:
        env.close()
