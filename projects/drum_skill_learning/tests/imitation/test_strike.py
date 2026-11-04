from contextlib import closing

import h5py
import json
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from openarm_rl.deployment.drum_actor import DrumActor
from openarm_rl.envs.camera_free_strike import CameraFreeStrikeEnv
from openarm_rl.imitation.strike import StrikeDemonstrations, initialize_ppo, run_expert
from openarm_rl.models.act import ACTConfig, ACTController, ActionChunkTransformer
from openarm_rl.models.asymmetric import AsymmetricStrikePolicy
from openarm_rl.training.camera_free import save_bundle


def test_native_live_demo_collection_and_position_velocity_act(native_strike_fixture, tmp_path):
    profile = json.loads((native_strike_fixture / "snare.json").read_text())
    profile["action_semantics"] = "absolute_wrist_position_velocity"
    profile["configuration"].update(ready_seconds=.10, approach_seconds=.20, retract_seconds=.28,
                                    recovery_seconds=.4, command_acceleration_rad_s2=120.)
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(json.dumps(profile))
    options = {"scene_directory": str(native_strike_fixture / "scene"),
               "profile_file": str(profile_file),
               "target": "snare", "position_range": 0, "domain_range": 0}
    path = tmp_path / "episodes.h5"
    report = run_expert(options, 1., 2, 200000, path, "live_demo")
    assert report["accepted"] == 2
    with h5py.File(path) as source:
        assert source.attrs["expert_source"] == "live_demo"
        assert source.attrs["teacher_contact_information"]
        assert not source.attrs["actor_oracle_inputs"]
        assert source["command"].shape == source["executed"].shape
    data = StrikeDemonstrations(path)
    config = ACTConfig(state_dim=480, action_dim=2, chunk=10, hidden=32, layers=1, heads=4)
    model = ActionChunkTransformer(config, data.state.mean(0), np.maximum(data.state.std(0), .01),
                                   data.command.mean(0), np.maximum(data.command.std(0), .01))
    state, command, padding = data.batch(np.array([0, data.ends[0] - 1]), config.chunk, "cpu")
    loss, _, _ = model.distillation_loss(state, command, padding, None)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.action_head.weight.grad is not None
    torch.optim.Adam(model.parameters(), lr=.001).step()
    model.eval()
    assert model.commands(state).shape == (2, 10, 2)
    controller = ACTController(model, 2, query_interval=1)
    assert controller.command(state).shape == (2, 2)
    act_file = tmp_path / "act.pt"
    model.save(act_file)
    with closing(VecNormalize(make_vec_env(CameraFreeStrikeEnv, n_envs=1, env_kwargs=options))) as normalizer:
        policy = PPO(AsymmetricStrikePolicy, normalizer, n_steps=128, batch_size=128,
                     n_epochs=1, device="cpu", seed=2)
        initialized = initialize_ppo(policy, normalizer, act_file, path, 2)
        assert initialized["samples"] == len(data.state)
        assert np.isfinite(initialized["final_loss"])
        policy.learn(128)
        bundle = tmp_path / "bundle"
        save_bundle(policy, bundle, options)
        normalizer.training = False
        actor = DrumActor(bundle)
        with CameraFreeStrikeEnv(**options) as env:
            observation, _ = env.reset(seed=1)
            q, dq = env.data.qpos[env.controller.q].copy(), env.data.qvel[env.controller.v].copy()
            action, _ = policy.predict(normalizer.normalize_obs(observation), deterministic=True)
            with torch.no_grad():
                exported = actor.actor(torch.from_numpy(observation["actor"]).unsqueeze(0)).numpy()[0]
            np.testing.assert_allclose(exported, action, atol=1e-6)
            actor.begin(q, dq, 1., 1., env.target_speed)
            command = actor.step(q, dq, 1., 1.)
            env.step(action)
            np.testing.assert_allclose(command, env.trace[-1]["joint_command_rad"], atol=1e-6)
            np.testing.assert_allclose(actor.output_velocity, env.trace[-1]["joint_velocity_command_rad_s"], atol=1e-6)
