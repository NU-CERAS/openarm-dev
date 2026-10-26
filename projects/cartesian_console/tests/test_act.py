from pathlib import Path
import shutil

import numpy as np
import h5py
import pytest
import torch

from openarm_sim.act_data import EpisodeDataset, collect_reference, proprioception, visual_state
from openarm_sim.act_model import ACTConfig, ActionChunkTransformer, normalization
from openarm_sim.act_distill import merge_datasets, render_reference_images
from openarm_sim.act_residual import ACTResidualVecEnv, SquashedResidualPolicy
from openarm_sim.warp_pickup import WarpPickupVecEnv, build_reset_bank
from openarm_sim.paths import RUNS


@pytest.fixture(scope="module")
def demonstration():
    directory = RUNS / "tests/act"
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


def test_aggregation_preserves_physical_episode_commands_and_supervision(demonstration):
    source = demonstration / "episodes.h5"
    output = demonstration / "combined.h5"
    merge_datasets([source, source], output)
    with h5py.File(source) as original, h5py.File(output) as combined:
        count = len(original["episodes"])
        assert len(combined["episodes"]) == 2 * count
        assert combined.attrs["bank_sha256"] == original.attrs["bank_sha256"]
        assert combined.attrs["successful"] == 2 * count
        for index, episode in enumerate(original["episodes"].values()):
            copied = combined["episodes"][f"{index:06d}"]
            for name, dataset in episode.items():
                np.testing.assert_array_equal(copied[name][:], dataset[:])
            assert copied.attrs["supervision"] == "executed"
            assert copied.attrs["source_sha256"]
    dataset = EpisodeDataset(output, 20)
    assert len(dataset) == 2 * len(EpisodeDataset(source, 20))


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


def test_smoothed_residual_state_resets_between_physical_episodes(demonstration):
    raw = WarpPickupVecEnv(2, demonstration / "resets.npz")
    env = ACTResidualVecEnv(raw, demonstration / "teacher.pt", action_smoothing=0.8)
    try:
        env.reset()
        action = np.full((2, 8), 0.5, dtype=np.float32)
        expected = env.compose(action).clone()
        env.step_async(action)
        torch.testing.assert_close(env.executed, expected)
        torch.testing.assert_close(env.last_residual, torch.full_like(env.last_residual, 0.1))
        env.step_wait()
        env.reset()
        torch.testing.assert_close(env.last_residual, torch.zeros_like(env.last_residual))
    finally:
        env.close()


def test_visual_student_uses_rendered_rgb_and_robot_only_features(demonstration):
    shutil.copyfile(demonstration / "resets.npz", demonstration / "episodes_resets.npz")
    images_path = demonstration / "visual.h5"
    render_reference_images(demonstration / "episodes.h5", images_path, 1, width=320, height=180)
    dataset = EpisodeDataset(images_path, 20, student=True, student_state="position-clock")
    model = ActionChunkTransformer(ACTConfig(state_dim=9, hidden=32, layers=1, vision=True, relative_commands=True,
                                             image_width=320, image_height=180, image_feature_stage=3), *normalization(dataset, True))
    state, command, padding, images = dataset.batch(np.array([0]), device="cpu")
    assert state.shape == (1, 9)
    assert images.shape == (1, 3, 180, 320)
    loss, _, kl = model.distillation_loss(state, command, padding, images)
    assert kl == 0
    loss.backward()
    assert model.backbone[0].weight.grad is not None
    original = EpisodeDataset(demonstration / "episodes.h5", 20).state[:1].copy()
    changed = original.copy()
    changed[:, 16:34] += 12345
    changed[:, 42:] -= 12345
    np.testing.assert_array_equal(proprioception(original), proprioception(changed))
    np.testing.assert_array_equal(visual_state(original, 0.02, 9), visual_state(changed, 0.02, 9))
    model.eval()
    expected = model.commands(state, images)
    model.save(demonstration / "visual_student.pt")
    restored = ActionChunkTransformer.load(demonstration / "visual_student.pt", device="cpu")
    torch.testing.assert_close(restored.commands(state, images), expected)
