import json

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from openarm_rl.envs.camera_free_strike import CameraFreeStrikeEnv
from openarm_rl.models.asymmetric import AsymmetricStrikePolicy
from openarm_rl.paths import ROOT


@pytest.fixture
def environment(native_strike_fixture):
    directory = native_strike_fixture
    with CameraFreeStrikeEnv(directory / "scene", directory / "snare.json", position_range=0, domain_range=0) as env:
        yield env


def test_actor_observation_and_reference_ignore_hidden_state(environment):
    env = environment
    first, _ = env.reset(seed=1)
    env.center += [.01, .01, 0]
    env.parameters[:] = [.9, 1.1, .95]
    second = env.observation()
    np.testing.assert_array_equal(first["actor"], second["actor"])
    assert not np.array_equal(first["critic"], second["critic"])
    policy = PPO(AsymmetricStrikePolicy, env, n_steps=8, batch_size=8, device="cpu")
    a, _ = policy.predict(first, deterministic=True)
    b, _ = policy.predict(second, deterministic=True)
    np.testing.assert_array_equal(a, b)
    tensor, _ = policy.policy.obs_to_tensor(first)
    changed = {"actor": tensor["actor"], "critic": tensor["critic"] + 1e3}
    with torch.no_grad():
        assert not torch.equal(policy.policy.predict_values(tensor), policy.policy.predict_values(changed))


def test_native_fixed_reference_hits_and_recovers(environment):
    env = environment
    env.reset(seed=1)
    for _ in range(400):
        _, reward, done, _, info = env.step(np.zeros(1, np.float32))
        assert np.isfinite(reward)
        if done:
            break
    assert info["hit_count"] == 1, info
    assert info["recovered"], info
    assert not info["unsafe_contact"], info


def test_no_contact_retracts_on_time_and_has_no_hit_reward(environment):
    env = environment
    env.reset(seed=1)
    env.model.geom_contype[:] = 0
    env.model.geom_conaffinity[:] = 0
    for _ in range(400):
        _, _, done, _, info = env.step(np.zeros(1, np.float32))
        assert info["reward_terms"]["impact"] == 0
        if done:
            break
    assert env.elapsed >= env.reference.duration
    assert info["hit_count"] == 0 and not info["is_success"]
    assert info["recovered"]


def test_gravity_compensation_uses_the_nominal_encoder_model(environment):
    env = environment
    env.reset(seed=2)
    q = env.data.qpos[env.controller.q].copy()
    before = env.servo.position_target(q, q)
    env.model.body_mass[:] *= 1.1
    after = env.servo.position_target(q, q)
    np.testing.assert_array_equal(before, after)


def test_constant_residual_cannot_hold_the_wrist_after_recovery(environment):
    env = environment
    env.reset(seed=3)
    for _ in range(400):
        _, _, done, _, info = env.step(np.array([.2], np.float32))
        if done:
            break
    assert env.reference.residual_weight(env.elapsed) == 0
    assert abs(env.command.offset) < 1e-6
    assert info["recovered"], info
