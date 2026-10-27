import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from openarm_sim.drumming import DrumStrikeEnv


def test_drumming_env_obeys_gymnasium_interface():
    env = DrumStrikeEnv()
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_drumming_starts_after_physical_ik_approach():
    with DrumStrikeEnv() as env:
        observation, info = env.reset(seed=7)
        assert env.observation_space.contains(observation)
        assert info["approach_error_m"] < 0.01
        assert info["approach_sim_time"] > 0.5
        assert info["grasped"]
        assert env.sim.tcp_position("right")[2] > env.drum_top + 0.06
        action = np.zeros(8, dtype=np.float32)
        action[7] = -1
        _, _, terminated, truncated, info = env.step(action)
        assert info["hit_count"] == 0
        assert not info["is_success"]
        assert not terminated and not truncated


def test_real_drumstick_contact_produces_hit_event():
    with DrumStrikeEnv() as env:
        env.reset(seed=3)
        tip_offset = env.tip_position() - env.sim.tcp_position("right")
        target = env.drum_center + [0, 0, 0.015 - 0.012] - tip_offset
        goal = env.sim._solve("right", target, env.tool_orientation)
        assert goal is not None
        for _ in range(95):
            current = env.sim._desired[env.sim.actuator_ids["right"]]
            action = np.zeros(8, dtype=np.float32)
            action[:7] = np.clip((goal[env.sim.qpos_ids["right"]] - current) / env.action_scale, -1, 1)
            action[7] = -1
            _, _, terminated, truncated, info = env.step(action.astype(np.float32))
            if info["hit_count"] or terminated or truncated:
                break
        assert info["hit_count"] >= 1, info
        assert info["last_hit"]["normal_force_n"] > 0
        assert info["last_hit"]["downward_speed_m_s"] > 0


@pytest.mark.parametrize("seed", range(15))
def test_sampled_ik_initial_states_are_executable(seed):
    with DrumStrikeEnv() as env:
        observation, info = env.reset(seed=seed)
        assert np.isfinite(observation).all()
        assert info["approach_error_m"] < 0.01
