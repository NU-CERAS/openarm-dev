from contextlib import closing
from dataclasses import asdict
import json

import mido
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from openarm_sim.annotation import DrumLayout

from openarm_rl.envs.camera_free_strike import CameraFreeStrikeEnv
from openarm_rl.envs.rhythmic import RhythmicDrummingEnv
from openarm_rl.deployment.rhythm_actor import RhythmActorRuntime
from openarm_rl.models.rhythmic import RhythmicPolicy, RhythmActor, ScoreNormalize
from openarm_rl.models.act_ppo import ACTRhythmPolicy
from openarm_rl.planning.scene import prepare_scene
from openarm_rl.training.rhythmic import collect, initialize, save_bundle, train_act, warmup_critic
from openarm_rl.music.preflight import inspect_performance


@pytest.fixture(scope="module")
def rhythm_options(native_strike_fixture, tmp_path_factory):
    output = tmp_path_factory.mktemp("rhythm")
    profile = json.loads((native_strike_fixture / "snare.json").read_text())
    profile["configuration"]["retract_seconds"] = .28
    (output / "source.json").write_text(json.dumps(profile))
    with CameraFreeStrikeEnv(native_strike_fixture / "scene", output / "source.json",
                             position_range=0, domain_range=0) as env:
        env.reset(seed=1)
        while True:
            _, _, done, truncated, info = env.step(env.reference_action())
            if done or truncated:
                break
        assert info["hit_count"] == 1 and not info["unsafe_contact"]
        assert info["hits"][0]["contact_duration_s"] <= .12
        reference = env.reference.document()
        reference["impact_time_s"] = info["hits"][0]["onset_time_s"]
        profile = dict(env.profile)
        profile["reference_bank"] = {"schema": "openarm.speed_conditioned_strike.v1",
                                     "speeds_m_s": [.35, .85], "references": [reference, reference]}
    (output / "snare.json").write_text(json.dumps(profile))
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.extend([mido.Message("note_on", channel=9, note=38, velocity=90),
                  mido.Message("note_off", channel=9, note=38, velocity=0, time=1),
                  mido.Message("note_on", channel=9, note=38, velocity=90, time=3839)])
    midi.save(output / "score.mid")
    return {"scene_directory": str(native_strike_fixture / "scene"), "profiles": str(output),
            "midi_file": str(output / "score.mid")}


def test_native_score_contacts_and_actor_excludes_privileged_measurements(rhythm_options):
    with RhythmicDrummingEnv(**rhythm_options) as env:
        before, _ = env.reset(seed=1)
        assert env.action_space.shape == (14,)
        assert before["critic"].shape == (372,)
        assert env.observation_space.contains(before)
        env.force[:] = 10
        after = env.observation()
        np.testing.assert_array_equal(after["actor"], before["actor"])
        assert not np.array_equal(after["critic"], before["critic"])
        while True:
            _, _, done, truncated, info = env.step(env.reference_action())
            if done or truncated:
                break
        assert info["target_events"] == info["correct_hits"] == 2, info
        assert info["f1"] == 1 and not info["unsafe_contact"], info
        assert info["timing_p95_s"] <= .06
        assert info["peak_force_N"] > 0
        for hit in info["hits"]:
            assert sum(contact["normal_force_N"] for contact in hit["contacts"]) >= env.cfg["hit_force_N"]
            for contact in hit["contacts"]:
                assert set(contact["geoms"]) == {f"{hit['arm']}_stick_tip_collision", f"{hit['target']}_head_collision"}
                assert np.isfinite(contact["position_m"]).all()
        assert info["max_penetration_m"] >= 0
        assert info["max_contact_duration_s"] > 0
        assert isinstance(info["recovered"], bool)
        json.dumps(info)


def test_music_video_requires_visible_original_kit(rhythm_options, tmp_path):
    with RhythmicDrummingEnv(**rhythm_options, render_mode="rgb_array") as env:
        metadata = env.scene
        layout = DrumLayout.from_scene(env.model, metadata["robot"], metadata["scene"])
    layout.save(tmp_path/"layout.json")
    prepare_scene(tmp_path/"annotated", layout_file=tmp_path/"layout.json")
    with pytest.raises(ValueError, match="visible original drum-kit"):
        RhythmicDrummingEnv(**{**rhythm_options, "scene_directory": tmp_path/"annotated"}, render_mode="rgb_array")


def test_complete_midi_preflight_reports_each_unmapped_note(rhythm_options, tmp_path):
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.extend([mido.Message("note_on", channel=9, note=38, velocity=80),
                  mido.Message("note_on", channel=9, note=36, velocity=90, time=480),
                  mido.Message("note_on", channel=9, note=36, velocity=90, time=480)])
    midi.save(tmp_path/"unsupported.mid")
    report = inspect_performance(tmp_path/"unsupported.mid", rhythm_options["scene_directory"], rhythm_options["profiles"])
    assert not report["executable"] and report["original_kit_positions"]
    assert report["target_events"] == 3
    assert [issue["time_s"] for issue in report["issues"]] == [.5, 1.]
    assert all(issue["note"] == 36 and issue["reason"] == "unmapped_note" for issue in report["issues"])


def test_full_joint_ppo_update_export_parity_and_critic_isolation(rhythm_options, tmp_path):
    with closing(VecNormalize(make_vec_env(RhythmicDrummingEnv, n_envs=1, env_kwargs=rhythm_options),
                              norm_obs_keys=["actor", "critic"])) as normalizer:
        policy = PPO(RhythmicPolicy, normalizer, n_steps=64, batch_size=64, n_epochs=1,
                     device="cpu", policy_kwargs={"log_std_init": -6.}, seed=3)
        policy.learn(64)
        assert policy._n_updates > 0
        normalizer.training = False
        with RhythmicDrummingEnv(**rhythm_options) as env:
            observation, _ = env.reset(seed=1)
            predicted = policy.predict(normalizer.normalize_obs(observation), deterministic=True)[0]
            observation["critic"] += 100
            np.testing.assert_allclose(policy.predict(normalizer.normalize_obs(observation), deterministic=True)[0], predicted)
            actor = RhythmActor(policy.policy, normalizer).eval()
            with torch.no_grad():
                exported = actor(torch.from_numpy(observation["actor"])[None], torch.from_numpy(observation["nominal"])[None]).numpy()[0]
            np.testing.assert_allclose(exported, predicted, atol=1e-6)
            save_bundle(policy, normalizer, tmp_path / "bundle", rhythm_options)
            actor_file = tmp_path / "bundle/actor.pt"
            runtime = RhythmActorRuntime(actor_file, rhythm_options["scene_directory"], rhythm_options["profiles"], rhythm_options["midi_file"])
            q, dq = env.data.qpos[env.controller.q].copy(), env.data.qvel[env.controller.v].copy()
            with pytest.raises(RuntimeError, match="stale"):
                runtime.begin(q, dq, 0., 1.)
            runtime.begin(q, dq, 1., 1.)
            for index in range(20):
                if index:
                    predicted = policy.predict(normalizer.normalize_obs(observation), deterministic=True)[0]
                motor, velocity = runtime.step(q, dq, 1.+index*.01, 1.+index*.01)
                observation, _, _, _, _ = env.step(predicted)
                np.testing.assert_allclose(motor, env.trace[-1]["motor_target_rad"], atol=1e-6)
                q, dq = env.data.qpos[env.controller.q].copy(), env.data.qvel[env.controller.v].copy()


def test_music_phase_initialization_has_matching_joint_state(rhythm_options):
    with RhythmicDrummingEnv(**rhythm_options, segment_seconds=2.) as env:
        starts = set()
        for seed in range(12):
            env.reset(seed=seed)
            starts.add(env.elapsed)
            q, dq = env.reference.sample(env.elapsed)
            np.testing.assert_allclose(env.data.qpos[env.controller.q], q)
            np.testing.assert_allclose(env.data.qvel[env.controller.v], dq)
            assert np.max(np.abs(dq)) < .08
            assert not any(contact.dist < -1e-4 for contact in env.data.contact)
        assert len(starts) > 1


def test_score_has_no_hit_reward_without_physical_contacts(rhythm_options):
    with RhythmicDrummingEnv(**rhythm_options) as env:
        env.reset(seed=1)
        env.model.geom_contype[:] = 0
        env.model.geom_conaffinity[:] = 0
        total = 0.
        while True:
            _, reward, done, truncated, info = env.step(env.reference_action())
            total += reward
            if done or truncated:
                break
        assert info["correct_hits"] == info["actual_hits"] == 0
        assert info["missed_hits"] == 2
        assert total <= -4.


def test_native_music_collection_relative_act_and_ppo_initialization(rhythm_options, tmp_path):
    data = collect(rhythm_options, tmp_path, 2)
    act = train_act(data, tmp_path, 2, "cpu", None)
    state = torch.tensor(np.c_[data[0][:2], data[2][:2]])
    commands = act.commands(state)
    assert commands.shape == (2, 10, 14)
    assert torch.isfinite(commands).all()
    assert act.config.relative_commands
    with closing(VecNormalize(make_vec_env(RhythmicDrummingEnv, n_envs=1, env_kwargs=rhythm_options),
                              norm_obs_keys=["actor", "critic"])) as normalizer:
        policy = PPO(RhythmicPolicy, normalizer, n_steps=64, batch_size=64, n_epochs=1, device="cpu")
        initialize(policy, normalizer, act, data, 2)
        policy.learn(64)
        assert policy._n_updates > 0


def test_delayed_encoder_prediction_is_bounded_and_actor_has_no_live_contact_state(rhythm_options):
    with RhythmicDrummingEnv(**rhythm_options, domain_range=.05) as env:
        env.reset(seed=1)
        env.feedback_period, env.feedback_delay = 2, 2
        for _ in range(12):
            observation, _, done, _, _ = env.step(env.reference_action())
            assert not done
            age = env.elapsed-env.sensor_cache[0]
            assert 0 <= age <= .03+1e-9
            before = observation["actor"].copy()
            env.force += 1
            np.testing.assert_array_equal(env.observation()["actor"], before)


def test_score_only_inputs_and_native_teacher(rhythm_options):
    options = {**rhythm_options, "score_only": True}
    with RhythmicDrummingEnv(**options) as env:
        observation, _ = env.reset(seed=1)
        assert observation["actor"].shape == (269,)
        assert observation["critic"].shape == (344,)
        q, _ = env.encoders()
        np.testing.assert_allclose(np.tanh(observation["nominal"]), env.encode(q), atol=1e-6)
        np.testing.assert_allclose(observation["actor"][168:189], env.centers.ravel())
        while True:
            _, _, done, _, info = env.step(env.reference_action())
            if done:
                break
        assert info["correct_hits"] == 2 and not info["unsafe_contact"], info
        assert info["actor_trajectory_inputs"] is False and info["motor_reference"] is False


@pytest.mark.parametrize("use_sde", [False, True])
def test_act_weights_initialize_ppo_exactly_and_receive_policy_gradients(rhythm_options, tmp_path, use_sde):
    options = {**rhythm_options, "score_only": True}
    data = collect(options, tmp_path, 2)
    with RhythmicDrummingEnv(**options) as env:
        bounds = env.controller.bounds.copy()
        observation, _ = env.reset(seed=1)
    act = train_act(data, tmp_path, 2, "cpu", None, score_only=True, joint_bounds=bounds, observation_jitter=.01)
    specification = {"config": asdict(act.config), **{name: getattr(act, name).tolist()
        for name in ("state_mean", "state_std", "action_mean", "action_std")}}
    with closing(ScoreNormalize(make_vec_env(RhythmicDrummingEnv, n_envs=1, env_kwargs=options),
                                norm_obs_keys=["critic"])) as normalizer:
        policy = PPO(ACTRhythmPolicy, normalizer, n_steps=32, batch_size=16, n_epochs=2,
                     device="cpu", use_sde=use_sde,
                     policy_kwargs={"act_specification": specification, "log_std_init": -9. if use_sde else -7., "actor_update_scope": "head"})
        policy.policy.mlp_extractor.policy_net.act.load_state_dict(act.state_dict())
        backbone = policy.policy.mlp_extractor.policy_net.act.state_projection.weight.detach().clone()
        initial_head = {name: value.detach().clone() for name, value in
                        policy.policy.mlp_extractor.policy_net.act.action_head.state_dict().items()}
        normalizer.training = False
        state = torch.tensor(np.r_[observation["actor"], observation["nominal"]])[None]
        expected = act.commands(state)[0, 0].numpy().clip(-.999, .999)
        predicted = policy.predict(normalizer.normalize_obs(observation), deterministic=True)[0]
        np.testing.assert_allclose(predicted, expected, atol=1e-6)
        if use_sde:
            policy.policy.reset_noise(1)
            sampled = policy.predict(normalizer.normalize_obs(observation), deterministic=False)[0]
            repeated = policy.predict(normalizer.normalize_obs(observation), deterministic=False)[0]
            np.testing.assert_array_equal(sampled, repeated)
            policy.policy.reset_noise(1)
            resampled = policy.predict(normalizer.normalize_obs(observation), deterministic=False)[0]
            assert np.isfinite(resampled).all() and not np.array_equal(sampled, resampled)
        observation["critic"] += 100
        np.testing.assert_allclose(policy.predict(normalizer.normalize_obs(observation), deterministic=True)[0], predicted, atol=1e-6)
        normalizer.training = True
        report = warmup_critic(policy, 32)
        assert report["actor_parameters_unchanged"] and report["actor_updates"] == 0
        assert report["environment_steps"] == 32 and policy.num_timesteps == 32
        np.testing.assert_allclose(policy.predict(normalizer.normalize_obs(observation), deterministic=True)[0], predicted, atol=1e-6)
        policy.learn(32, reset_num_timesteps=False)
        assert policy._n_updates > 0
        torch.testing.assert_close(policy.policy.mlp_extractor.policy_net.act.state_projection.weight, backbone, rtol=0, atol=0)
        assert any(not torch.equal(value, initial_head[name]) for name, value in
                   policy.policy.mlp_extractor.policy_net.act.action_head.state_dict().items())
        normalizer.training = False
        save_bundle(policy, normalizer, tmp_path/"act_ppo_bundle", options)
        loaded = PPO.load(tmp_path/"act_ppo_bundle/policy.zip", device="cpu")
        np.testing.assert_allclose(loaded.predict(normalizer.normalize_obs(observation), deterministic=True)[0],
                                   policy.predict(normalizer.normalize_obs(observation), deterministic=True)[0], atol=1e-6)
        policy.policy.save(tmp_path/"standalone_policy.pt")
        standalone = ACTRhythmPolicy.load(tmp_path/"standalone_policy.pt", device="cpu")
        np.testing.assert_allclose(standalone.predict(normalizer.normalize_obs(observation), deterministic=True)[0],
                                   policy.predict(normalizer.normalize_obs(observation), deterministic=True)[0], atol=1e-6)
        actor = torch.jit.load(str(tmp_path/"act_ppo_bundle/actor.pt"))
        with torch.no_grad():
            exported = actor(torch.tensor(observation["actor"])[None], torch.tensor(observation["nominal"])[None]).numpy()[0]
        np.testing.assert_allclose(exported, policy.predict(normalizer.normalize_obs(observation), deterministic=True)[0], atol=1e-6)
        with torch.no_grad():
            repeated = actor(torch.tensor(np.stack([observation["actor"]]*2)),
                             torch.tensor(np.stack([observation["nominal"]]*2))).numpy()
        np.testing.assert_allclose(repeated, np.stack([exported]*2), atol=1e-6)
        runtime = RhythmActorRuntime(tmp_path/"act_ppo_bundle/actor.pt", options["scene_directory"],
                                    options["profiles"], options["midi_file"])
        with RhythmicDrummingEnv(**options) as env:
            observation, _ = env.reset(seed=1)
            q, dq = env.data.qpos[env.controller.q].copy(), env.data.qvel[env.controller.v].copy()
            runtime.begin(q, dq, 1., 1.)
            for index in range(3):
                command, velocity = runtime.step(q, dq, 1.+index*.01, 1.+index*.01)
                observation, _, _, _, _ = env.step(runtime.last_action.copy())
                np.testing.assert_allclose(command, env.trace[-1]["motor_target_rad"], atol=1e-6)
                np.testing.assert_allclose(velocity, env.previous_velocity, atol=1e-6)
                q, dq = env.data.qpos[env.controller.q].copy(), env.data.qvel[env.controller.v].copy()


def test_force_penalty_uses_measured_contacts_without_changing_physics(rhythm_options):
    options = {**rhythm_options, "score_only": True}
    with RhythmicDrummingEnv(**options, force_penalty=0.) as baseline, RhythmicDrummingEnv(**options, force_penalty=.05) as penalized:
        baseline.reset(seed=3)
        penalized.reset(seed=3)
        difference = 0.
        while True:
            action = baseline.reference_action()
            _, reward, done, _, before = baseline.step(action)
            _, penalized_reward, penalized_done, _, after = penalized.step(action)
            np.testing.assert_array_equal(baseline.data.qpos, penalized.data.qpos)
            assert done == penalized_done
            difference += reward-penalized_reward
            if done:
                break
        assert before["correct_hits"] == after["correct_hits"] == 2
        assert after["contact_force_cost"] > 0
        assert difference == pytest.approx(.05*after["contact_force_cost"])


def test_score_environments_share_one_nominal_model(rhythm_options):
    with RhythmicDrummingEnv(**rhythm_options, domain_range=.1, position_range=.002) as first:
        nominal = first.nominal_mass.copy()
        first.reset(seed=11)
        assert not np.array_equal(first.model.body_mass, nominal)
        first.restore_nominal()
        with RhythmicDrummingEnv(**rhythm_options, source=first) as second:
            assert second.model is first.model and second.servo is first.servo
            assert np.array_equal(second.nominal_mass, nominal)
            assert np.array_equal(second.model.body_mass, nominal)


def test_stick_body_on_a_drum_head_counts_as_drum_contact(rhythm_options):
    with RhythmicDrummingEnv(**rhythm_options, score_only=True) as env:
        for arm in ("left", "right"):
            for part in ("tip", "taper", "shaft"):
                assert env.tips[env.model.geom(f"{arm}_stick_{part}_collision").id] == arm
