import json

import numpy as np

from openarm_rl.envs.camera_free_strike import CameraFreeStrikeEnv


def test_native_position_velocity_actions_and_encoder_history(native_strike_fixture, tmp_path):
    profile = json.loads((native_strike_fixture / "snare.json").read_text())
    profile["action_semantics"] = "absolute_wrist_position_velocity"
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    with CameraFreeStrikeEnv(native_strike_fixture / "scene", path, position_range=0, domain_range=0) as env:
        observation, _ = env.reset(seed=1)
        assert observation["actor"].shape == (480,)
        assert observation["critic"].shape == (76,)
        assert env.action_space.shape == (2,)
        before = observation["actor"].copy()
        env.center += [.001, 0, 0]
        np.testing.assert_array_equal(env.observation()["actor"], before)
        env.center -= [.001, 0, 0]
        while True:
            action = env.reference_action()
            _, reward, done, truncated, info = env.step(action)
            assert np.isfinite(reward)
            if done or truncated:
                break
        assert info["hit_count"] == 1, info
        assert info["recovered"], info
        velocity = np.array([row["joint_velocity_command_rad_s"] for row in env.trace])
        assert np.max(np.abs(velocity[:, env.skill.wrist])) <= .8*env.controller.velocity[env.skill.wrist] + 1e-6
        assert np.max(np.abs(np.diff(velocity[:, env.skill.wrist]))) <= env.acceleration[env.skill.wrist]*env.period + 1e-6
