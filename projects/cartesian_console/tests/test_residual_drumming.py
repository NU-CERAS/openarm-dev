import numpy as np
import mujoco

from openarm_sim.residual_drumming import ResidualDrumStrikeEnv


def test_reference_kinematics_matches_full_mujoco_forward():
    with ResidualDrumStrikeEnv(randomization=0.0) as env:
        env.reset(seed=3)
        data = mujoco.MjData(env.sim.model)
        for time_s in np.linspace(0, 2, 11):
            data.qpos[:] = env.sim.data.qpos
            data.qpos[env.sim.qpos_ids["right"]] = env.reference.sample(time_s)
            mujoco.mj_forward(env.sim.model, data)
            expected = data.site_xpos[env.sim.site_ids["right"]] + env.tip_offset
            np.testing.assert_allclose(env._reference_tip(time_s), expected, atol=1e-12)


def test_nominal_reference_produces_real_contact_without_learning():
    with ResidualDrumStrikeEnv(randomization=0.0) as env:
        observation, info = env.reset(seed=3)
        assert env.observation_space.contains(observation)
        assert info["grasped"]
        for _ in range(101):
            action = np.zeros(8, dtype=np.float32)
            action[7] = -1
            observation, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(observation)
            assert np.isfinite(reward)
            assert np.isclose(reward, sum(info["reward_terms"].values()))
            if terminated or truncated:
                break
        assert info["hit_count"] > 0
        assert info["last_hit"]["normal_force_n"] > 0
        assert info["grasped"]
        assert not info["unsafe_contact"]


def test_randomized_reset_preserves_physical_grasp():
    with ResidualDrumStrikeEnv(randomization=0.1) as env:
        for seed in (0, 5, 11):
            observation, info = env.reset(seed=seed)
            assert env.observation_space.contains(observation)
            assert info["grasped"]
            assert 0.036 <= info["stick_mass_kg"] <= 0.044
