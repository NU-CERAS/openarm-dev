import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from openarm_sim.compute_guard import budgeted, check_resources

from ..envs.rhythmic import RhythmicDrummingEnv
from ..envs.multiscore import make_rhythm_env
from ..models.act import ACTConfig, ActionChunkTransformer
from ..models.rhythmic import RhythmicPolicy, RhythmActor, ScoreNormalize
from ..models.act_ppo import ACTRhythmPolicy, ACTRhythmActor
from ..planning.strikes import scene_identity
from ..music.examples import write_example
from ..music.corpus import read_manifest
from ..music.reference import profile_hashes
from ..music.observations import score_observation_scales
from ..music.perform import rollout
from ..recording.display import NativeDisplay
from .vector_env import TrainingSubprocVecEnv


def evaluate(options, output, policy=None, normalizer=None, act=None, episodes=8, seed=700000, video=False):
    rows = []
    if "corpus_manifest" in options:
        with make_rhythm_env(**options) as check:
            episodes = max(episodes, len(check.clips))
    for episode in range(episodes):
        with make_rhythm_env(**options, render_mode="rgb_array" if video and episode == 0 else None) as env:
            command = None
            if policy is not None:
                def command(observation):
                    return policy.predict(normalizer.normalize_obs(observation), deterministic=True)[0]
            if act is not None:
                def command(observation):
                    with torch.no_grad():
                        state = torch.tensor(np.r_[observation["actor"], observation["nominal"]], device=next(act.parameters()).device)[None]
                        return np.clip(act.commands(state)[0, 0].cpu().numpy(), -.999, .999)
            reset_options = {"clip_id": env.clips[episode % len(env.clips)]["id"]} if "corpus_manifest" in options else None
            result = rollout(env, Path(output) / f"episode_{episode:03d}", seed+episode, command,
                             video and episode == 0, reset_options=reset_options,
                             controller_label=("ACT + PPO" if policy._n_updates else "ACT INITIALIZATION") if policy is not None else
                                              ("ACT" if act is not None else "REFERENCE CONTROLLER"))
            rows.append(result)
    report = {"episodes": episodes, **{name: float(np.mean([r[name] for r in rows])) for name in
              ("f1", "precision", "recall", "correct_hits", "missed_hits", "extra_hits", "unsafe_contact",
               "is_success", "recovered", "max_contact_duration_s", "episode_return")},
              "results": rows}
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "evaluation.json").write_text(json.dumps(report, indent=2))
    return report


def collect(options, output, episodes):
    states, commands, nominal, critics, ends, outcomes = [], [], [], [], [], []
    with make_rhythm_env(**options) as env:
        accepted, consecutive_failures = 0, 0
        if "corpus_manifest" in options and episodes < len(env.clips):
            raise ValueError("Demonstration count must cover every training clip")
        for attempt in range(episodes*8):
            check_resources()
            reset_options = {"clip_id": env.clips[accepted % len(env.clips)]["id"]} if "corpus_manifest" in options else None
            observation, _ = env.reset(seed=200000+attempt, options=reset_options)
            rows = []
            while True:
                action = env.reference_action()
                rows.append((observation["actor"].copy(), action.copy(), observation["nominal"].copy(), observation["critic"].copy()))
                observation, _, done, truncated, info = env.step(action)
                if done or truncated:
                    break
            outcomes.append({"seed": 200000+attempt, "f1": info["f1"], "unsafe_contact": info["unsafe_contact"],
                             "is_success": info["is_success"], "clip_id": info.get("clip_id"),
                             "failure": info["failure"], "recovered": info["recovered"],
                             "max_contact_duration_s": info["max_contact_duration_s"]})
            if not info["is_success"]:
                consecutive_failures += 1
                if consecutive_failures >= 8:
                    (output/"collection.json").write_text(json.dumps({"accepted": accepted, "attempts": outcomes}, indent=2))
                    raise RuntimeError(f"Reference collection failed eight consecutive native episodes for {info.get('clip_id', 'score')}")
                continue
            consecutive_failures = 0
            for state, action, reference, critic in rows:
                states.append(state)
                commands.append(action)
                nominal.append(reference)
                critics.append(critic)
            ends.append(len(states))
            accepted += 1
            if accepted % 8 == 0 or accepted == episodes:
                print(json.dumps({"collection_accepted": accepted, "requested": episodes,
                                  "attempts": attempt+1, "clip_id": info.get("clip_id")}), flush=True)
            if accepted == episodes:
                break
        if accepted != episodes:
            (output / "collection.json").write_text(json.dumps(outcomes, indent=2))
            raise RuntimeError(f"Native score teacher collected {accepted}/{episodes} qualified episodes")
    with h5py.File(output / "demonstrations.h5", "w") as data:
        source = options.get("midi_file", options.get("corpus_manifest"))
        data.attrs.update(schema="openarm.rhythm_demonstrations.v1", actor_oracle_inputs=False,
                          source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                          actor_trajectory_inputs=not options.get("score_only", False),
                          proprio_anchor="episode_start_encoders" if options.get("score_only", False) else "timed_reference",
                          command_semantics="fourteen_normalized_joint_position_targets")
        for name, values in (("actor", states), ("action", commands), ("nominal", nominal), ("critic", critics), ("episode_ends", ends)):
            data.create_dataset(name, data=np.asarray(values), compression="gzip")
    (output / "collection.json").write_text(json.dumps({"accepted": accepted, "attempts": outcomes}, indent=2))
    return np.array(states), np.array(commands), np.array(nominal), np.array(critics), np.array(ends)


def augment_score_state(state, bounds, observation_jitter):
    noise = torch.randn((len(state), 14), device=state.device)*observation_jitter
    noise[:, [6, 13]] = 0
    for frame in range(4):
        selected = slice(frame*42, frame*42+14)
        state[:, selected] = torch.clamp(state[:, selected]+noise, bounds[:, 0], bounds[:, 1])
        state[:, frame*42+14:frame*42+28] += torch.randn((len(state), 14), device=state.device)*.3
        state[:, frame*42+28:frame*42+42] += torch.randn((len(state), 14), device=state.device)*.01
    return state


def train_act(data, output, steps, device, run, score_only=False, joint_bounds=None, observation_jitter=0.):
    torch.manual_seed(17)
    states, actions, nominal, critics, ends = data
    states = np.c_[states, nominal]
    config = ACTConfig(state_dim=states.shape[1], action_dim=14, chunk=10, hidden=128, layers=2,
                       relative_commands=True, action_semantics="normalized_episode_anchor_residual" if score_only else "normalized_joint_reference_residual",
                       rhythmic_time_features=score_only)
    residual = actions-np.tanh(nominal)
    state_floor = score_observation_scales(include_proprio=True) if score_only else .01
    state_mean, state_std = states.mean(0), np.maximum(states.std(0), state_floor)
    if score_only:
        state_mean[189:269], state_std[189:269] = 0., 1.
    model = ActionChunkTransformer(config, state_mean, state_std,
                                   residual.mean(0), np.maximum(residual.std(0), .005)).to(device)
    if observation_jitter and (not score_only or joint_bounds is None):
        raise ValueError("Encoder augmentation requires score-only joint bounds")
    bounds = torch.tensor(joint_bounds, dtype=torch.float32, device=device) if joint_bounds is not None else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rng = np.random.default_rng(17)
    probability = None
    if score_only:
        goals = states[:, 189:269].reshape(-1, 8, 10)
        imminent = np.any((np.abs(goals[:, :, 7]) < .4) & (goals[:, :, 9] > 0), axis=1)
        moving = np.max(np.abs(residual[:, [6, 13]]), axis=1) > .01
        probability = .2+imminent.astype(float)+2*moving.astype(float)
        probability /= probability.sum()
    model.train()
    for step in range(steps):
        indices = rng.choice(len(states), size=64, p=probability) if probability is not None else rng.integers(len(states), size=64)
        episode = np.searchsorted(ends, indices, side="right")
        offsets = indices[:, None]+np.arange(config.chunk)
        padding = offsets >= ends[episode, None]
        offsets = np.minimum(offsets, ends[episode, None]-1)
        state = torch.tensor(states[indices], dtype=torch.float32, device=device)
        if observation_jitter:
            state = augment_score_state(state, bounds, observation_jitter)
        targets = torch.tensor(actions[offsets], dtype=torch.float32, device=device)
        prediction, _, _ = model(state)
        expected = model.normalized_actions(targets, state)
        weights = torch.tensor([10., *([1.]*9)], device=device)[None, :, None]
        valid = torch.tensor(~padding, device=device)[:, :, None]*weights
        loss = ((prediction-expected).abs()*valid).sum()/(valid.sum()*14)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 100 == 0:
            check_resources()
            print(json.dumps({"ACT_step": step, "loss": float(loss.detach())}), flush=True)
            if run is not None:
                run.log({"ACT/step": step, "ACT/loss": float(loss.detach())})
    model.eval()
    model.save(output / "act.pt", actor_oracle_inputs=False, demonstration_schema="openarm.rhythm_demonstrations.v1")
    return model


def validate_source_run(source_run, options):
    previous = json.loads((source_run/"run_config.json").read_text())["environment"]
    for key in ("domain_range", "position_range", "score_only"):
        if previous[key] != options[key]:
            raise ValueError(f"Demonstration environment differs: {key}")
    for source in ("model.mjb", "scene.json"):
        if (Path(previous["scene_directory"])/source).read_bytes() != (Path(options["scene_directory"])/source).read_bytes():
            raise ValueError("Demonstrations belong to a different native model")
    if Path(previous["corpus_manifest"]).read_bytes() != Path(options["corpus_manifest"]).read_bytes():
        raise ValueError("Demonstrations belong to a different corpus")
    if profile_hashes(Path(previous["profiles"])) != profile_hashes(Path(options["profiles"])):
        raise ValueError("Demonstrations belong to different reference profiles")
    previous_calibration, calibration = previous.get("velocity_calibration"), options.get("velocity_calibration")
    if bool(previous_calibration) != bool(calibration):
        raise ValueError("Demonstration intensity configuration differs")
    if calibration and Path(previous_calibration).read_bytes() != Path(calibration).read_bytes():
        raise ValueError("Demonstration intensity measurements differ")


def reuse_demonstrations(source_run, options, output):
    validate_source_run(source_run, options)
    source = source_run/"demonstrations.h5"
    with h5py.File(source, "r") as data:
        if data.attrs["schema"] != "openarm.rhythm_demonstrations.v1" or data.attrs["actor_trajectory_inputs"]:
            raise ValueError("Require score-only native demonstrations")
        result = [data[name][:] for name in ("actor", "action", "nominal", "critic", "episode_ends")]
        attributes = dict(data.attrs)
    if result[0].shape[1] != 269 or result[1].shape[1] != 14 or result[-1][-1] != len(result[0]):
        raise ValueError("Demonstration observation or episode schema differs")
    start = 0
    for end in result[-1]:
        result[2][start:end] = result[2][start].copy()
        start = int(end)
    attributes["proprio_anchor"] = "episode_start_encoders"
    with h5py.File(output/"demonstrations.h5", "w") as target:
        target.attrs.update(attributes)
        for name, values in zip(("actor", "action", "nominal", "critic", "episode_ends"), result):
            target.create_dataset(name, data=values, compression="gzip")
    (output/"demonstration_source.json").write_text(json.dumps({"run": str(source_run),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "episodes": len(result[-1]),
        "nominal_input": "measured first-frame encoders repeated within each episode"}, indent=2))
    return result


def initialize(policy, normalizer, act, data, steps):
    states, actions, nominal, critics, ends = data
    for name, values in (("actor", states), ("critic", critics)):
        stats = normalizer.obs_rms[name]
        stats.mean, stats.var, stats.count = values.mean(0).astype(np.float64), values.var(0).astype(np.float64), len(values)
    normalized = normalizer.normalize_obs({"actor": states, "critic": critics, "nominal": nominal})
    features = torch.tensor(np.c_[normalized["actor"], normalized["critic"], nominal], device=policy.device)
    targets = []
    with torch.no_grad():
        for start in range(0, len(states), 128):
            batch = torch.tensor(np.c_[states[start:start+128], nominal[start:start+128]], device=policy.device)
            targets.append(act.commands(batch)[:, 0].clamp(-.999, .999))
    targets = torch.cat(targets)
    parameters = [*policy.policy.mlp_extractor.policy_net.parameters(), *policy.policy.action_net.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    rng = np.random.default_rng(0)
    for _ in range(steps):
        indices = rng.integers(len(states), size=256)
        prediction = torch.tanh(policy.policy.action_net(policy.policy.mlp_extractor.forward_actor(features[indices])))
        loss = torch.nn.functional.mse_loss(prediction, targets[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


class CriticWarmupCallback(BaseCallback):
    def _on_step(self):
        check_resources()
        return True


def warmup_critic(policy, steps, run=None):
    if not isinstance(policy.policy, ACTRhythmPolicy) or steps <= 0:
        raise ValueError("Critic warmup requires a score-only ACT policy and positive steps")
    actor = policy.policy.mlp_extractor.policy_net
    before = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
    log_std = policy.policy.log_std.detach().cpu().clone()
    _, callback = policy._setup_learn(steps, callback=CriticWarmupCallback(), reset_num_timesteps=False)
    parameters = [*policy.policy.mlp_extractor.value_net.parameters(), *policy.policy.value_net.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=3e-4)
    started, updates, losses = policy.num_timesteps, 0, []
    while policy.num_timesteps-started < steps:
        if not policy.collect_rollouts(policy.get_env(), callback, policy.rollout_buffer, policy.n_steps):
            raise RuntimeError("Critic warmup rollout interrupted")
        policy.policy.set_training_mode(True)
        for _ in range(8):
            for batch in policy.rollout_buffer.get(policy.batch_size):
                values = policy.policy.predict_values(batch.observations).flatten()
                loss = torch.nn.functional.mse_loss(values, batch.returns)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                updates += 1
                losses.append(float(loss.detach()))
        print(json.dumps({"critic_warmup_steps": policy.num_timesteps-started, "critic_loss": losses[-1]}), flush=True)
        if run is not None:
            run.log({"env_steps": policy.num_timesteps, "critic_warmup/loss": losses[-1]})
    if (any(not torch.equal(value.detach().cpu(), before[name]) for name, value in actor.state_dict().items())
            or not torch.equal(policy.policy.log_std.detach().cpu(), log_std)):
        raise RuntimeError("Critic warmup modified actor parameters")
    elapsed = policy.num_timesteps-started
    policy.critic_warmup_steps = getattr(policy, "critic_warmup_steps", 0)+elapsed
    policy.critic_warmup_updates = getattr(policy, "critic_warmup_updates", 0)+updates
    return {"environment_steps": elapsed, "critic_updates": updates, "actor_updates": 0,
            "actor_parameters_unchanged": True, "first_loss": losses[0], "last_loss": losses[-1]}


def save_bundle(policy, normalizer, output, options):
    output.mkdir(parents=True, exist_ok=True)
    policy.save(output / "policy.zip")
    normalizer.save(str(output / "vecnormalize.pkl"))
    actor = (ACTRhythmActor(policy.policy) if isinstance(policy.policy, ACTRhythmPolicy) else RhythmActor(policy.policy, normalizer)).eval()
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        torch.jit.trace(actor, (torch.zeros((1, len(actor.mean))), torch.zeros((1, 14)))).save(str(output / "actor.pt"))
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    files = ("policy.zip", "vecnormalize.pkl", "actor.pt")
    with make_rhythm_env(**options) as env:
        identity = scene_identity(env.scene["robot"], env.scene["scene"])
        c = env.controller
        limits = {"bounds_rad": c.bounds.tolist(), "velocity_rad_s": (.8*c.velocity).tolist(),
                  "acceleration_rad_s2": [120.]*14, "position_gains": c.kp.tolist(),
                  "velocity_gains": c.kd.tolist(), "control_hz": 100}
        trained_drums = sorted({goal.target for goal in env.score.goals})
    if "corpus_manifest" in options:
        corpus = read_manifest(options["corpus_manifest"])
        trained_drums = sorted({target for clip in corpus["clips"] if clip["split"] == "train" for target in clip["targets"]})
    (output / "limits.json").write_text(json.dumps(limits, indent=2))
    files = (*files, "limits.json")
    score_only = options.get("score_only", False)
    (output / "bundle.json").write_text(json.dumps({"schema": "openarm.score_actor.v3" if score_only else "openarm.rhythm_actor.v1", "action_dim": 14,
        "trained_drums": trained_drums,
        "proprio_anchor": "episode_start_encoders" if score_only else "timed_reference",
        "command_semantics": "normalized_joint_positions_with_encoder_gravity_and_command_velocity" if score_only else "normalized_joint_positions_with_encoder_gravity_and_timed_reference",
        "actor_trajectory_inputs": not score_only, "motor_reference": not score_only,
        "control_hz": 100, "actor_oracle_inputs": False, "steps": policy.num_timesteps,
        "scene_identity": identity,
        "reference_profiles_sha256": profile_hashes(Path(options["profiles"])),
        "model_sha256": hashlib.sha256((Path(options["scene_directory"])/"model.mjb").read_bytes()).hexdigest(),
        "optimization_updates": policy._n_updates,
        "update_counter_semantics": "SB3 PPO epoch counter",
        "critic_warmup_steps": getattr(policy, "critic_warmup_steps", 0),
        "critic_warmup_updates": getattr(policy, "critic_warmup_updates", 0),
        "sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in files}}, indent=2))


class RhythmCallback(BaseCallback):
    def __init__(self, output, options, interval, run):
        super().__init__()
        self.output, self.options, self.interval, self.run = output, options, interval, run
        self.next_save, self.best = interval, (-np.inf, -np.inf, -np.inf)

    def assess(self, name):
        stats = self.model.get_vec_normalize_env()
        report = evaluate(self.options, self.output / name, self.model, stats, episodes=4)
        score = (report["is_success"], report["f1"]-report["unsafe_contact"], report["episode_return"])
        if score > self.best:
            self.best = score
            save_bundle(self.model, stats, self.output / "best", self.options)
        if self.run is not None:
            self.run.log({"env_steps": self.model.num_timesteps, **{f"eval/{k}": v for k, v in report.items() if k != "results"}})
        print(json.dumps({"evaluation": name, "steps": self.model.num_timesteps, **{k: v for k, v in report.items() if k != "results"}}), flush=True)
        return report

    def _on_step(self):
        if self.num_timesteps >= self.next_save:
            save_bundle(self.model, self.model.get_vec_normalize_env(), self.output / f"checkpoint_{self.num_timesteps:09d}", self.options)
            self.assess(f"validation_{self.num_timesteps:09d}")
            self.next_save += self.interval
        for info in self.locals["infos"]:
            if "episode" in info:
                self.logger.record_mean("rhythm/f1", info["f1"])
                self.logger.record_mean("rhythm/unsafe", info["unsafe_contact"])
        return True

    def _on_rollout_start(self):
        check_resources()
        if self.run is not None:
            self.run.log({"env_steps": self.num_timesteps, **{k: float(v) for k, v in self.logger.name_to_value.items() if isinstance(v, (float, int, np.number))}})


@budgeted("openarm_rl.training.rhythmic")
def main():
    parser = argparse.ArgumentParser(description="Collect timed music demonstrations, train ACT and optimize fourteen-joint rhythmic PPO")
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--midi", type=Path)
    parser.add_argument("--heldout-midi", type=Path)
    parser.add_argument("--initialize-act", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--velocity-calibration", type=Path)
    parser.add_argument("--minimum-initial-success", type=float, default=.25)
    parser.add_argument("--initialize-data-run", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--entropy-coefficient", type=float)
    parser.add_argument("--ppo-epochs", type=int)
    parser.add_argument("--force-penalty", type=float)
    parser.add_argument("--critic-warmup-steps", type=int)
    parser.add_argument("--actor-update-scope", choices=["head", "all"])
    parser.add_argument("--use-sde", action="store_true")
    parser.add_argument("--sde-sample-freq", type=int, default=-1)
    parser.add_argument("--log-std-init", type=float)
    parser.add_argument("--act-observation-jitter", type=float, default=.01)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--act-steps", type=int, default=1000)
    parser.add_argument("--bc-steps", type=int, default=500)
    parser.add_argument("--demonstrations", type=int, default=16)
    parser.add_argument("--num-envs", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--warp-envs", type=int, default=0, help="Batched GPU worlds for PPO rollouts; evaluation stays on native MuJoCo")
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--domain-range", type=float, default=.05)
    parser.add_argument("--position-range", type=float, default=.002)
    parser.add_argument("--checkpoint-steps", type=int, default=4096)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--xvfb", type=Path)
    args = parser.parse_args()
    if args.warp_envs < 0 or (args.warp_envs and (not args.score_only or args.velocity_calibration or args.device != "cuda")):
        raise ValueError("MuJoCo Warp rollouts require score-only CUDA training without intensity calibration")
    args.rollout_steps = args.rollout_steps or (32 if args.warp_envs else 256)
    args.batch_size = args.batch_size or (8192 if args.warp_envs else 256)
    if (args.warp_envs or args.num_envs)*args.rollout_steps % args.batch_size:
        raise ValueError("Rollout size must be divisible by the PPO batch size")
    if args.use_sde and (not args.score_only or args.resume_checkpoint):
        raise ValueError("SDE initialization requires fresh score-only PPO from ACT")
    if args.sde_sample_freq == 0 or args.sde_sample_freq < -1:
        raise ValueError("SDE sampling frequency must be -1 or positive")
    if args.log_std_init is not None and args.resume_checkpoint:
        raise ValueError("Initial exploration scale cannot replace resumed optimizer state")
    if args.log_std_init is not None and not -20 <= args.log_std_init <= 0:
        raise ValueError("Initial log standard deviation must lie in [-20, 0]")
    if args.score_only and not args.resume_checkpoint and args.actor_update_scope is None:
        args.actor_update_scope = "head"
    if args.critic_warmup_steps is None:
        args.critic_warmup_steps = 4096 if args.score_only and not args.resume_checkpoint else 0
    if args.critic_warmup_steps < 0 or (args.critic_warmup_steps and not args.score_only):
        raise ValueError("Critic warmup requires score-only training and a nonnegative step count")
    if not 0 <= args.minimum_initial_success <= 1:
        raise ValueError("Initial success threshold must lie between zero and one")
    if not 0 <= args.act_observation_jitter <= .03:
        raise ValueError("ACT encoder augmentation must be between zero and 0.03 radians")
    if args.learning_rate is not None and not 0 < args.learning_rate <= .001:
        raise ValueError("Learning rate must lie in (0, 0.001]")
    if args.entropy_coefficient is not None and not 0 <= args.entropy_coefficient <= .1:
        raise ValueError("Entropy coefficient must lie in [0, 0.1]")
    if args.ppo_epochs is not None and not 1 <= args.ppo_epochs <= 10:
        raise ValueError("PPO epochs must lie between one and ten")
    if args.initialize_data_run and not args.corpus:
        raise ValueError("Demonstration reuse requires a corpus")
    if args.resume_checkpoint:
        if not args.score_only or not args.corpus:
            raise ValueError("Checkpoint continuation requires score-only corpus training")
        if args.initialize_act:
            raise ValueError("Checkpoint continuation loads its original ACT automatically")
        args.initialize_act = args.resume_checkpoint.parent/"act.pt"
    if args.corpus and (args.midi or args.heldout_midi):
        raise ValueError("Choose a corpus or individual MIDI files")
    if args.corpus and not args.score_only:
        raise ValueError("Multi-performance training requires --score-only")
    if min(args.steps, args.act_steps, args.bc_steps, args.demonstrations, args.checkpoint_steps) <= 0:
        raise ValueError("Training counts must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.scene_directory, args.output / "scene")
    shutil.copytree(args.profiles, args.output / "profiles")
    midi, heldout = args.output / "training.mid", args.output / "heldout.mid"
    if args.midi is None:
        write_example(midi, "duet")
    else:
        shutil.copyfile(args.midi, midi)
    if args.heldout_midi is None:
        write_example(heldout, "heldout")
    else:
        shutil.copyfile(args.heldout_midi, heldout)
    options = {"scene_directory": str(args.output / "scene"), "profiles": str(args.output / "profiles"),
               "midi_file": str(midi), "domain_range": args.domain_range, "position_range": args.position_range,
               "score_only": args.score_only,
               "force_penalty": args.force_penalty if args.force_penalty is not None else (.05 if args.score_only else 0.)}
    if args.velocity_calibration:
        shutil.copyfile(args.velocity_calibration, args.output / "velocity_calibration.json")
        options["velocity_calibration"] = str(args.output / "velocity_calibration.json")
    heldout_options = {**options, "midi_file": str(heldout)}
    validation_options = options
    if args.corpus:
        shutil.copytree(args.corpus.parent, args.output / "corpus")
        options.pop("midi_file")
        options.update(corpus_manifest=str(args.output / "corpus" / args.corpus.name), corpus_split="train")
        validation_options = {**options, "corpus_split": "validation"}
        heldout_options = {**options, "corpus_split": "test"}
    (args.output / "run_config.json").write_text(json.dumps({**vars(args), "environment": options,
        "validation_environment": validation_options, "test_environment": heldout_options}, default=str, indent=2))
    package = Path(__file__).resolve().parents[1]
    sources = [package / name for name in ("envs/rhythmic.py", "models/rhythmic.py", "models/act.py",
               "music/score.py", "music/reference.py", "music/corpus.py", "music/observations.py", "models/music_features.py", "envs/multiscore.py",
               "models/act_ppo.py", "control/servo.py", "training/rhythmic.py")]
    (args.output / "source_hashes.json").write_text(json.dumps({str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
                                                               for path in sources}, indent=2))
    with ExitStack() as resources:
        if args.xvfb:
            display = NativeDisplay(args.xvfb, args.output / "display")
            resources.callback(display.close)
        run = wandb.init(project="openarm-midi", name=args.output.name, dir=str(args.output),
                         config={**vars(args), "actor_oracle_inputs": False, "action_dim": 14}, mode=args.wandb_mode)
        resources.callback(run.finish)
        (args.output / "wandb_url.txt").write_text(str(run.url))
        reference = evaluate(options, args.output / "reference", episodes=8)
        data = reuse_demonstrations(args.initialize_data_run, options, args.output) if args.initialize_data_run else collect(options, args.output, args.demonstrations)
        with make_rhythm_env(**options) as environment:
            joint_bounds = environment.controller.bounds.copy()
        if args.initialize_act is None:
            act = train_act(data, args.output, args.act_steps, args.device, run, args.score_only,
                            joint_bounds, args.act_observation_jitter if args.score_only else 0.)
        else:
            act = ActionChunkTransformer.load(args.initialize_act, args.device)
            if (act.config.state_dim != data[0].shape[1]+14 or act.config.action_dim != 14
                    or act.config.action_semantics != ("normalized_episode_anchor_residual" if args.score_only else "normalized_joint_reference_residual")):
                raise ValueError("Supplied ACT does not match the rhythm observation/action schema")
            shutil.copyfile(args.initialize_act, args.output / "act.pt")
            source = {"ACT_updates_in_this_run": 0,
                      "ACT_source_sha256": hashlib.sha256(args.initialize_act.read_bytes()).hexdigest()}
            (args.output / "ACT_initialization.json").write_text(json.dumps(source, indent=2))
            run.config.update(source)
        act_report = evaluate(validation_options, args.output / "ACT_validation", act=act, episodes=8)
        act_training = evaluate(options, args.output / "ACT_training", act=act, episodes=8) if args.score_only else None
        if act_training is not None and act_training["is_success"] < args.minimum_initial_success:
            raise RuntimeError("ACT fails the native training-clip success gate; PPO was not started")
        if args.warp_envs:
            from ..envs.warp_rhythmic import WarpRhythmicVecEnv
            source = ({"corpus_manifest": options["corpus_manifest"], "corpus_split": "train"} if "corpus_manifest" in options
                      else {"midi_files": [options["midi_file"]]})
            vector = WarpRhythmicVecEnv(args.warp_envs, options["scene_directory"], options["profiles"], **source,
                                        domain_range=args.domain_range, position_range=args.position_range,
                                        segment_seconds=4., force_penalty=options["force_penalty"], seed=17)
        else:
            vector = make_vec_env(make_rhythm_env, n_envs=args.num_envs,
                                  env_kwargs={**options, "segment_seconds": 4.}, vec_env_cls=TrainingSubprocVecEnv,
                                  vec_env_kwargs={"start_method": "spawn"})
        resources.callback(vector.close)
        normalizer_class = ScoreNormalize if args.score_only else VecNormalize
        normalizer = (VecNormalize.load(str(args.resume_checkpoint/"vecnormalize.pkl"), vector) if args.resume_checkpoint else
                      normalizer_class(vector, norm_obs_keys=["critic"] if args.score_only else ["actor", "critic"]))
        policy_class, policy_options = RhythmicPolicy, {"log_std_init": -5.}
        if args.score_only:
            specification = {"config": asdict(act.config), **{name: getattr(act, name).cpu().tolist()
                for name in ("state_mean", "state_std", "action_mean", "action_std")}}
            policy_class, policy_options = ACTRhythmPolicy, {"log_std_init": -9. if args.use_sde else -7., "act_specification": specification,
                "actor_update_scope": args.actor_update_scope or "all"}
        if args.log_std_init is not None:
            policy_options["log_std_init"] = args.log_std_init
        overrides = {key: value for key, value in (("learning_rate", args.learning_rate),
            ("ent_coef", args.entropy_coefficient), ("n_epochs", args.ppo_epochs)) if value is not None}
        if args.resume_checkpoint:
            validate_source_run(args.resume_checkpoint.parent, options)
            normalizer.training = True
            policy = PPO.load(args.resume_checkpoint/"policy.zip", env=normalizer, device=args.device, **overrides)
            if args.actor_update_scope and args.actor_update_scope != policy.policy.actor_update_scope:
                raise ValueError("Actor update scope differs from checkpoint; initialize ACT for a different scope")
            (args.output/"PPO_initialization.json").write_text(json.dumps({"method": "resumed_checkpoint",
                "source": str(args.resume_checkpoint), "initial_steps": policy.num_timesteps,
                "sha256": hashlib.sha256((args.resume_checkpoint/"policy.zip").read_bytes()).hexdigest()}, indent=2))
        else:
            settings = {"learning_rate": 3e-5, "ent_coef": 1e-4, "n_epochs": 4, **overrides}
            policy = PPO(policy_class, normalizer, n_steps=args.rollout_steps, batch_size=args.batch_size, target_kl=.01, device=args.device,
                         policy_kwargs=policy_options, seed=17, use_sde=args.use_sde,
                         sde_sample_freq=args.sde_sample_freq, **settings)
            if args.score_only:
                policy.policy.mlp_extractor.policy_net.act.load_state_dict(act.state_dict())
                normalizer.obs_rms["critic"].update(data[3])
                normalizer.floor_statistics()
                (args.output/"PPO_initialization.json").write_text(json.dumps({"method": "direct_ACT_weights",
                    "BC_updates": 0, "ACT_sha256": hashlib.sha256((args.output/"act.pt").read_bytes()).hexdigest()}, indent=2))
            else:
                initialize(policy, normalizer, act, data, args.bc_steps)
        (args.output/"PPO_hyperparameters.json").write_text(json.dumps({"learning_rate": policy.lr_schedule(1.),
            "entropy_coefficient": policy.ent_coef, "epochs": policy.n_epochs, "batch_size": policy.batch_size,
            "rollout_steps_per_worker": policy.n_steps, "target_kl": policy.target_kl, "initial_steps": policy.num_timesteps,
            "use_sde": policy.use_sde, "sde_sample_freq": policy.sde_sample_freq,
            "initial_log_std": policy.policy.log_std.detach().cpu().tolist(),
            "actor_update_scope": getattr(policy.policy, "actor_update_scope", "reference_mlp")}, indent=2))
        policy.set_logger(configure(str(args.output), ["stdout", "csv"]))
        callback = RhythmCallback(args.output, validation_options, args.checkpoint_steps, run)
        callback.init_callback(policy)
        callback.next_save = policy.num_timesteps+args.checkpoint_steps
        initial = callback.assess("initial")
        if args.score_only and initial["is_success"] < args.minimum_initial_success:
            raise RuntimeError("Initialized PPO fails the native validation gate; optimization was not started")
        if args.critic_warmup_steps:
            report = warmup_critic(policy, args.critic_warmup_steps, run)
            (args.output/"critic_warmup.json").write_text(json.dumps(report, indent=2))
            save_bundle(policy, normalizer, args.output/"best", options)
            callback.next_save = policy.num_timesteps+args.checkpoint_steps
        policy.learn(args.steps, callback=callback,
                     reset_num_timesteps=not bool(args.resume_checkpoint or args.critic_warmup_steps))
        save_bundle(policy, normalizer, args.output / "final", options)
        optimized = callback.assess("optimized")
        best = PPO.load(args.output / "best/policy.zip", env=normalizer, device=args.device)
        normalizer = VecNormalize.load(str(args.output / "best/vecnormalize.pkl"), vector)
        normalizer.training = False
        vector.close()
        final = evaluate(options, args.output / "heldout_training_pattern", best, normalizer, seed=800000, video=args.video)
        unseen_reference = evaluate(heldout_options, args.output / "unseen_reference", seed=800000)
        unseen = evaluate(heldout_options, args.output / "unseen_pattern", best, normalizer, seed=800000, video=args.video)
        comparison = {"reference": reference, "ACT": act_report, "initialized_PPO": initial,
                      "optimized_PPO_validation": optimized, "selected_policy": final,
                      "unseen_reference": unseen_reference, "unseen_policy": unseen,
                      "selection": {"steps": best.num_timesteps, "optimization_updates": best._n_updates}}
        (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2))
        run.summary.update({f"comparison/{name}/{key}": value for name, report in comparison.items()
                            for key, value in report.items() if key != "results"})


if __name__ == "__main__":
    main()
