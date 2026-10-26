from pathlib import Path

import numpy as np
import pytest
import torch

from openarm_sim.warp_pickup import WarpPickupVecEnv, build_reset_bank
from openarm_sim.paths import RUNS


@pytest.fixture(scope="module")
def reset_bank():
    path = RUNS / "tests/warp-resets.npz"
    build_reset_bank(path, 8, 123)
    return path


def test_gpu_observation_matches_cpu_state_and_worlds_reset_independently(reset_bank):
    env = WarpPickupVecEnv(2, reset_bank)
    try:
        observation = env.reset()
        env.snapshot(0)
        env.cpu.initial_height = float(env.initial_height[0].item())
        env.cpu.last_action = np.zeros(8)
        np.testing.assert_allclose(observation[0], env.cpu._observation(), atol=2e-5)
        second = env.qpos[1].clone()
        env._reset_indices(np.array([0]))
        torch.testing.assert_close(env.qpos[1], second)
        for _ in range(4):
            observation, rewards, done, _ = env.step(np.zeros((2, 8), dtype=np.float32))
            assert np.isfinite(observation).all() and np.isfinite(rewards).all()
            assert not done.any()
    finally:
        env.close()


def test_gpu_reference_pickup_produces_physical_grasp_and_lift(reset_bank):
    env = WarpPickupVecEnv(1, reset_bank)
    try:
        env.reset()
        env.snapshot(0)
        env.sim._desired = env.desired[0].cpu().numpy().copy()
        env.cpu.initial_height = float(env.initial_height[0].item())
        success = False
        for action in env.cpu.reference_pickup_actions():
            _, _, done, infos = env.step(action[None])
            if done[0]:
                success = infos[0]["is_success"]
                break
            env.snapshot(0)
            env.sim._desired = env.desired[0].cpu().numpy().copy()
        assert success
        assert infos[0]["reward_terms"]["milestone"] == 3.0
        assert infos[0]["reward_terms"]["success"] == 50.0
        assert infos[0]["episode"]["r"] > 0
        assert infos[0]["max_lift_m"] > 0.10
    finally:
        env.close()
