from pathlib import Path

import numpy as np
import gc
import mujoco
import pytest
import torch

from openarm_sim.warp_pickup import WarpPickupVecEnv, build_reset_bank
from openarm_sim.paths import RUNS
from openarm_sim.simulation import ReachSimulation


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
        assert 0 <= infos[0]["reset_index"] < len(env.bank_qpos)
    finally:
        env.close()


def test_sequential_evaluation_resets_match_distinct_physical_bank_states(reset_bank):
    env = WarpPickupVecEnv(2, reset_bank, sequential_resets=True)
    try:
        env.set_options({"bank_offset": 3})
        env.reset()
        torch.testing.assert_close(env.reset_samples, torch.tensor([3, 4], device="cuda"))
        torch.testing.assert_close(env.qpos, env.bank_qpos[3:5])
        env.set_options({"bank_offset": 7})
        with pytest.raises(ValueError, match="offset"):
            env.reset()
    finally:
        env.close()


@pytest.mark.skipif(not (RUNS / "planning/openarm.yml").exists(), reason="Requires generated cuRobo assets")
def test_active_gpu_environment_survives_curobo_planner_lifetime(reset_bank):
    caller_stream = torch.cuda.current_stream().cuda_stream
    env = WarpPickupVecEnv(2, reset_bank)
    assert torch.cuda.current_stream().cuda_stream == caller_stream
    try:
        with ReachSimulation(planner="curobo") as planning:
            result = planning.start_reach("right", np.array([0.43, -0.24, 0.44]))
            assert result.accepted, result.reason
        del planning
        gc.collect()
        observation, reward, _, _ = env.step(np.zeros((2, 8), dtype=np.float32))
        assert torch.cuda.current_stream().cuda_stream == caller_stream
        assert np.isfinite(observation).all()
        assert np.isfinite(reward).all()
    finally:
        env.close()


def test_render_snapshot_matches_full_physics_snapshot_pixels(reset_bank):
    env = WarpPickupVecEnv(2, reset_bank)
    try:
        with mujoco.Renderer(env.model, height=180, width=320) as renderer:
            for _ in range(3):
                env.step(np.zeros((2, 8), dtype=np.float32))
                for world in (0, 1):
                    renderer.update_scene(env.snapshot(world), camera="overhead")
                    expected = renderer.render().copy()
                    renderer.update_scene(env.render_snapshot(world), camera="overhead")
                    np.testing.assert_array_equal(renderer.render(), expected)
    finally:
        env.close()
