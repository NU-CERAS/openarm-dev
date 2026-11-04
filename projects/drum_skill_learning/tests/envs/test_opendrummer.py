import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from openarm_rl.envs.opendrummer import OpenDrummerStrikeEnv, prepare_scene
from openarm_rl.training.vector_env import TrainingSubprocVecEnv


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    directory = tmp_path_factory.mktemp("drummer") / "scene"
    prepare_scene(directory)
    return directory


def test_gym_interface_and_repeatable_randomized_state(scene):
    with OpenDrummerStrikeEnv(scene) as env:
        check_env(env, skip_render_check=True)
        first, info = env.reset(seed=31)
        second, again = env.reset(seed=31)
        np.testing.assert_array_equal(first, second)
        assert info == again
        assert env.observation_space.contains(first)
        assert np.max(np.abs(info["kit_offset_m"])) <= 0.05
        assert np.linalg.norm(info["kit_offset_m"]) > 0.001


def test_physical_reference_impact_and_recovery(scene):
    with OpenDrummerStrikeEnv(scene, position_range=0, domain_range=0) as env:
        env.reset(seed=1)
        impacts = 0.0
        for _ in range(301):
            _, reward, done, truncated, info = env.step(np.zeros(1, np.float32))
            assert np.isfinite(reward)
            impacts += info["reward_terms"]["impact"]
            if done or truncated:
                break
        assert info["hit_count"] == 1, info
        assert info["recovered"], info
        assert not info["unsafe_contact"]
        assert info["hits"][0]["peak_force_N"] > 0
        assert info["hits"][0]["normal_impulse_Ns"] > 0
        assert impacts > 0


def test_no_contact_cannot_receive_hit_or_success_reward(scene):
    with OpenDrummerStrikeEnv(scene, position_range=0, domain_range=0) as env:
        env.reset(seed=1)
        env.model.geom_contype[:] = 0
        env.model.geom_conaffinity[:] = 0
        for _ in range(301):
            _, _, done, truncated, info = env.step(np.zeros(1, np.float32))
            assert info["reward_terms"]["impact"] == 0
            assert info["reward_terms"]["success"] == 0
            if done or truncated:
                break
        assert info["hit_count"] == 0
        assert not info["is_success"]
        assert not info["recovered"]
        assert info["failure"].startswith("Missed strike")


def test_residual_changes_only_joint7_command_and_rejects_bad_actions(scene):
    with OpenDrummerStrikeEnv(scene, position_range=0, domain_range=0) as env:
        env.reset(seed=3)
        env.step(np.ones(1, np.float32))
        assert env.reference.rows[-1]["commanded_joint_positions_rad"][6] != env.reference.staging[6]
        held = np.r_[0:6, 7:14]
        np.testing.assert_allclose(np.array(env.reference.rows[-1]["commanded_joint_positions_rad"])[held], env.reference.staging[held])
        with pytest.raises(ValueError, match="finite"):
            env.step(np.array([np.nan], np.float32))


def test_randomized_layouts_keep_ready_and_wrist_arc_feasible(scene):
    with OpenDrummerStrikeEnv(scene) as env:
        for seed in [*range(32), 700000, 700001, 800000, 800001]:
            observation, _ = env.reset(seed=seed)
            assert env.observation_space.contains(observation)
            assert not any(c.dist < -1e-5 for c in env.data.contact)
            assert len(env.reference.arc) == 101


def test_worker_failure_terminates_the_training_group(scene):
    env = make_vec_env(OpenDrummerStrikeEnv, n_envs=2,
                       env_kwargs={"scene_directory": scene, "position_range": 0, "domain_range": 0},
                       vec_env_cls=TrainingSubprocVecEnv, vec_env_kwargs={"start_method": "spawn"})
    try:
        env.reset()
        env.step_async(np.zeros((2, 1), np.float32))
        env.processes[0].terminate()
        env.processes[0].join(timeout=3)
        with pytest.raises(RuntimeError, match="workers exited"):
            env.step_wait()
    finally:
        env.close()
    assert all(not p.is_alive() for p in env.processes)


def test_consecutive_worker_resets_keep_the_full_wrist_stroke_reachable(scene):
    with OpenDrummerStrikeEnv(scene) as env:
        for episode in range(80):
            env.reset(seed=2 if episode == 0 else None)
            heights = [row["height_m"] for row in env.reference.arc]
            settings = env.reference_settings
            assert settings["ready_height_m"] > settings["approach_height_m"] > 0
            assert max(heights) >= settings["ready_height_m"] - 1e-8
            assert min(heights) <= -settings["command_depth_m"] + 1e-8
