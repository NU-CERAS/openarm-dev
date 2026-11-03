import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
import torch
import wandb

from openarm_sim.compute_guard import budgeted, check_resources
from openarm_drummer.model import CONFIG
from openarm_drummer.taught import TaughtStrike, scene_signature

from ..envs.camera_free_strike import CameraFreeStrikeEnv
from ..models.act import ACTConfig, ActionChunkTransformer


class StrikeExpert:
    def __init__(self, env, gain):
        self.env, self.gain = env, gain
        self.data = mujoco.MjData(env.servo.model)
        self.inverse = np.zeros(env.servo.model.nv)

    def command(self, observation):
        env = self.env
        frame = observation["actor"][-env.history.frame_dim:]
        q, dq = frame[:14], frame[14:28]
        reference, reference_velocity = env.reference.sample(env.elapsed)
        nominal = env.reference.motor_target(env.elapsed, q, env.servo)
        self.data.qpos[env.servo.q] = q
        self.data.qvel[env.servo.v] = dq
        mujoco.mj_forward(env.servo.model, self.data)
        self.data.qacc[env.servo.v] = np.clip(env.reference.spline(env.elapsed, 2), -env.acceleration, env.acceleration)
        mujoco.mj_rne(env.servo.model, self.data, 1, self.inverse)
        wrist = env.skill.wrist
        gravity = (nominal[wrist] - reference[wrist]) * env.controller.kp[wrist]
        dynamic = (self.inverse[env.servo.v[wrist]] - gravity) / env.controller.kp[wrist]
        action = np.array([np.clip(self.gain * dynamic / .08, -1, 1)], np.float32)
        requested = float(nominal[wrist] + .08*action[0])
        if env.action_semantics == "absolute_wrist_position_velocity":
            requested = float(np.clip(requested, *env.controller.bounds[wrist]))
            action = env.encode_motor_command(requested, reference_velocity[wrist])
            return action, np.array([nominal[wrist], reference_velocity[wrist]]), np.array([requested, reference_velocity[wrist]])
        return action, np.array([nominal[wrist]]), np.array([requested])


class LiveDemoStrikeExpert(StrikeExpert):
    source = "opendrummer_live_demo_TaughtStrike"

    def reset(self):
        self.routine = None
        self.env.physics_observer = self.observe

    def command(self, observation):
        env = self.env
        if env.elapsed < env.reference.times[1]:
            return super().command(observation)
        if self.routine is None:
            selected = slice(0, 7) if env.skill.arm == "left" else slice(7, 14)
            scene = deepcopy(env.scene)
            next(surface for surface in scene["surfaces"] if surface["name"] == env.skill.target)["center"] = env.center.tolist()
            preset = {"arm": env.skill.arm, "drum": env.skill.target,
                      "ready_rad": env.reference.positions[0, selected].tolist(),
                      "strike_joint7_rad": float(env.reference.positions[3, env.skill.wrist]),
                      "scene_signature": scene_signature(scene)}
            motion = json.loads(CONFIG.with_name("demo_motion.json").read_text())
            duration = float(env.reference.times[3] - env.reference.times[1])
            distance = abs(preset["strike_joint7_rad"] - preset["ready_rad"][6])
            acceleration = 6 * distance / duration**2
            motion["strike_acceleration_by_drum_rad_s2"][env.skill.target] = acceleration
            self.routine = TaughtStrike(env.model, env.robot, scene, preset,
                                       initial=env.data.qpos[env.controller.q].copy(), transfer_motion=motion, concurrent=True)
            self.routine.transition("downstroke", env.data)
        torque = self.routine.prepare(env.data)
        measured = observation["actor"][-env.history.frame_dim:][:14]
        nominal = env.reference.motor_target(env.elapsed, measured, env.servo)
        nominal_velocity = env.reference.sample(env.elapsed)[1]
        if self.routine.failure is not None:
            env.failure = "Live demo expert: " + self.routine.failure
        if torque is None:
            values = np.array([nominal[env.skill.wrist], nominal_velocity[env.skill.wrist]]) if env.action_dim == 2 else np.array([nominal[env.skill.wrist]])
            return env.reference_action(), values, values.copy()
        kp = env.controller.kp * env.parameters[2]
        kd = env.controller.kd * np.sqrt(env.parameters[2])
        q, dq = env.data.qpos[env.controller.q], env.data.qvel[env.controller.v]
        wrist = env.skill.wrist
        if env.action_semantics == "absolute_wrist_position_velocity":
            velocity = self.routine.step_context[1]
            requested = q + (torque - kd*(velocity-dq)) / kp
            motor = float(np.clip(nominal[wrist] + self.gain*(requested[wrist]-nominal[wrist]), *env.controller.bounds[wrist]))
            speed = float(nominal_velocity[wrist] + self.gain*(velocity[wrist]-nominal_velocity[wrist]))
            action = env.encode_motor_command(motor, speed)
            return action, np.array([nominal[wrist], nominal_velocity[wrist]]), np.array([motor, speed])
        previous = env.command.previous if env.command.previous is not None else nominal
        requested = (torque + kp*q + kd*(previous/env.period + dq)) / (kp + kd/env.period)
        motor = float(nominal[wrist] + self.gain*(requested[wrist] - nominal[wrist]))
        action = np.array([np.clip((motor - nominal[wrist])/.08, -1, 1)], np.float32)
        motor = float(nominal[wrist] + .08*action[0])
        return action, np.array([nominal[wrist]]), np.array([motor])

    def observe(self, data):
        if self.routine is not None and not self.routine.finished:
            self.routine.observe(data)
            if self.routine.failure is not None:
                self.env.failure = "Live demo expert: " + self.routine.failure


class StrikeDemonstrations:
    def __init__(self, path):
        with h5py.File(path) as source:
            version = source.attrs["schema"]
            if version not in ("openarm.strike_demonstrations.v1", "openarm.strike_demonstrations.v2"):
                raise ValueError("Unsupported strike demonstration data")
            self.state = source["actor"][:]
            self.critic = source["critic"][:]
            self.command = source["command"][:] if version.endswith("v2") else source["motor_position_rad"][:]
            self.nominal = source["nominal"][:] if version.endswith("v2") else source["nominal_motor_position_rad"][:]
            self.ends = source["episode_ends"][:]
            self.action_semantics = source.attrs.get("action_semantics", "wrist_residual")
            self.motor_bounds = np.asarray(source.attrs.get("motor_bounds_rad", [-1.570796, 1.570796]))
            self.velocity_limit = float(source.attrs.get("velocity_limit_rad_s", 1.))
            self.history_frames = int(source.attrs.get("history_frames", 8))
        if self.state.shape[1] != self.history_frames*(58+self.command.shape[1]) or self.command.shape[0] != len(self.state) or self.ends[-1] != len(self.state):
            raise ValueError("Invalid demonstration dimensions or episode boundaries")

    def batch(self, indices, chunk, device, residual=False):
        episode = np.searchsorted(self.ends, indices, side="right")
        offsets = indices[:, None] + np.arange(chunk)
        padding = offsets >= self.ends[episode, None]
        offsets = np.minimum(offsets, self.ends[episode, None] - 1)
        command = self.command - self.nominal if residual else self.command
        return (torch.tensor(self.state[indices], device=device), torch.tensor(command[offsets], device=device),
                torch.tensor(padding, device=device))


def run_expert(options, gain, episodes, seed, data_file=None, expert_source="reference"):
    accepted, rejected, rows, observations, critics, labels, nominals, executed, ends = [], [], [], [], [], [], [], [], []
    with CameraFreeStrikeEnv(**options) as env:
        expert = LiveDemoStrikeExpert(env, gain) if expert_source == "live_demo" else StrikeExpert(env, gain)
        attempts = episodes if data_file is None else episodes * 8
        for index in range(attempts):
            check_resources()
            observation, _ = env.reset(seed=seed + index)
            if expert_source == "live_demo":
                expert.reset()
            episode, total = [], 0.
            while True:
                action, nominal, command = expert.command(observation)
                before = observation
                observation, reward, done, truncated, info = env.step(action)
                total += reward
                applied = env.trace[-1]
                motor = [applied["joint_command_rad"][env.skill.wrist]]
                if env.action_dim == 2:
                    motor.append(applied["joint_velocity_command_rad_s"][env.skill.wrist])
                episode.append((before["actor"].copy(), before["critic"].copy(), command, nominal, np.array(motor)))
                if done or truncated:
                    break
            rows.append({**info, "seed": seed + index, "episode_return": total})
            if info["is_success"]:
                accepted.append(seed + index)
                if data_file is not None:
                    for state, critic, command, nominal, motor in episode:
                        observations.append(state)
                        critics.append(critic)
                        labels.append(command)
                        nominals.append(nominal)
                        executed.append(motor)
                    ends.append(len(observations))
            else:
                rejected.append(seed + index)
            if data_file is not None and len(accepted) == episodes:
                break
    if data_file is not None:
        if len(accepted) != episodes:
            raise RuntimeError(f"Only collected {len(accepted)} of {episodes} successful native demonstrations")
        with h5py.File(data_file, "w") as target:
            target.attrs.update(schema="openarm.strike_demonstrations.v2", actor_oracle_inputs=False,
                                command_semantics="requested_control_before_shared_limits_and_delay",
                                gain=gain, profile_sha256=hashlib.sha256(Path(options["profile_file"]).read_bytes()).hexdigest(),
                                expert_source=expert_source, teacher_contact_information=expert_source == "live_demo",
                                action_semantics=env.action_semantics, motor_bounds_rad=env.controller.bounds[env.skill.wrist],
                                velocity_limit_rad_s=.8*env.controller.velocity[env.skill.wrist], history_frames=env.history_frames,
                                command_units_json=json.dumps(["rad", "rad/s"] if env.action_dim == 2 else ["rad"]),
                                live_demo_motion_sha256=hashlib.sha256(CONFIG.with_name("demo_motion.json").read_bytes()).hexdigest())
            target.attrs["live_controller_sha256"] = hashlib.sha256(Path(inspect.getfile(TaughtStrike)).read_bytes()).hexdigest()
            target.attrs["collector_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            for name, values in (("actor", observations), ("critic", critics), ("command", labels),
                                 ("nominal", nominals), ("executed", executed)):
                target.create_dataset(name, data=np.asarray(values, np.float32), compression="gzip")
            target.create_dataset("episode_ends", data=np.array(ends, np.int64))
            target.create_dataset("accepted_seeds", data=accepted)
            target.create_dataset("rejected_seeds", data=rejected)
    return {"episodes": len(rows), "success_rate": float(np.mean([row["is_success"] for row in rows])),
            "unsafe_rate": float(np.mean([row["unsafe_contact"] for row in rows])),
            "mean_return": float(np.mean([row["episode_return"] for row in rows])), "accepted": len(accepted), "results": rows}


def act_action(model, env, observation, device):
    with torch.no_grad():
        command = model.commands(torch.tensor(observation["actor"][None], device=device))[0, 0].cpu().numpy()
    if model.config.action_semantics in ("motor_position_residual_rad", "motor_position_velocity_residual"):
        correction = command
    elif model.config.action_semantics == "absolute_motor_position":
        measured = observation["actor"][-env.history.frame_dim:][:14]
        nominal = env.reference.motor_target(env.elapsed, measured, env.servo)[env.skill.wrist]
        correction = command - np.array([nominal])
    else:
        raise ValueError("Unsupported ACT strike action semantics")
    if env.action_semantics == "absolute_wrist_position_velocity":
        measured = observation["actor"][-env.history.frame_dim:][:14]
        nominal = env.reference.motor_target(env.elapsed, measured, env.servo)[env.skill.wrist]
        velocity = env.reference.sample(env.elapsed)[1][env.skill.wrist]
        return env.encode_motor_command(nominal + correction[0], velocity + correction[1])
    return np.clip(correction / .08, -1, 1).astype(np.float32)


def evaluate_act(model, options, episodes, seed, device):
    rows = []
    model.eval()
    with CameraFreeStrikeEnv(**options) as env:
        for index in range(episodes):
            observation, _ = env.reset(seed=seed + index)
            total = 0.
            while True:
                action = act_action(model, env, observation, device)
                observation, reward, done, truncated, info = env.step(action)
                total += reward
                if done or truncated:
                    rows.append({**info, "episode_return": total, "seed": seed + index})
                    break
    return {"episodes": episodes, "success_rate": float(np.mean([row["is_success"] for row in rows])),
            "unsafe_rate": float(np.mean([row["unsafe_contact"] for row in rows])),
            "mean_return": float(np.mean([row["episode_return"] for row in rows])), "results": rows}


def initialize_ppo(policy, normalizer, act_file, data_file, steps):
    data = StrikeDemonstrations(data_file)
    teacher = ActionChunkTransformer.load(act_file, policy.device)
    if (teacher.config.vision or teacher.config.state_dim != policy.observation_space["actor"].shape[0]
            or teacher.config.action_dim != data.command.shape[1]):
        raise ValueError("PPO initialization requires a matching camera-free strike ACT")
    for name, values in (("actor", data.state), ("critic", data.critic)):
        stats = normalizer.obs_rms[name]
        stats.mean, stats.var, stats.count = values.mean(0).astype(np.float64), values.var(0).astype(np.float64), len(values)
    states = torch.tensor(data.state, device=policy.device)
    targets = []
    with torch.no_grad():
        for start in range(0, len(states), 128):
            predicted = teacher.commands(states[start:start + 128])[:, 0]
            if teacher.config.action_semantics == "absolute_motor_position":
                predicted = predicted - torch.tensor(data.nominal[start:start + 128], device=policy.device)
            elif teacher.config.action_semantics not in ("motor_position_residual_rad", "motor_position_velocity_residual"):
                raise ValueError("Unsupported ACT initialization action semantics")
            if data.action_semantics == "absolute_wrist_position_velocity":
                desired = predicted + torch.tensor(data.nominal[start:start + 128], device=policy.device)
                lower, upper = data.motor_bounds
                target = torch.stack((2*(desired[:, 0]-lower)/(upper-lower)-1, desired[:, 1]/data.velocity_limit), dim=-1)
            else:
                target = predicted / .08
            targets.append(torch.clamp(target, -.99, .99))
    targets = torch.cat(targets)
    normalized = torch.tensor(normalizer.normalize_obs({"actor": data.state, "critic": data.critic})["actor"], device=policy.device)
    parameters = [*policy.policy.mlp_extractor.policy_net.parameters(), *policy.policy.action_net.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=.001)
    generator = torch.Generator(device=policy.device).manual_seed(0)
    for step in range(steps):
        indices = torch.randint(len(states), (256,), device=policy.device, generator=generator)
        prediction = torch.tanh(policy.policy.action_net(policy.policy.mlp_extractor.policy_net(normalized[indices])))
        loss = torch.nn.functional.mse_loss(prediction, targets[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            check_resources()
    return {"ACT_sha256": hashlib.sha256(Path(act_file).read_bytes()).hexdigest(),
            "data_sha256": hashlib.sha256(Path(data_file).read_bytes()).hexdigest(), "steps": steps,
            "samples": len(states), "final_loss": float(loss.detach()), "actor_oracle_inputs": False}


@budgeted("openarm_rl.imitation.strike")
def main():
    parser = argparse.ArgumentParser(description="Collect successful native strike demonstrations and train a camera-free ACT")
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--profile-file", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluate-every", type=int, default=500)
    parser.add_argument("--position-range", type=float, default=.005)
    parser.add_argument("--domain-range", type=float, default=.1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--expert-source", choices=["reference", "live_demo"], default="live_demo")
    parser.add_argument("--data-file", type=Path)
    args = parser.parse_args()
    if min(args.episodes, args.steps, args.batch_size, args.evaluate_every) <= 0:
        raise ValueError("Training counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    options = {"scene_directory": str(args.scene_directory.resolve()), "profile_file": str(args.profile_file.resolve()),
               "target": args.target, "position_range": args.position_range, "domain_range": args.domain_range}
    if args.data_file is None:
        candidates = []
        for gain in ((.5, 1.) if args.expert_source == "live_demo" else (0., .25, .5)):
            result = run_expert(options, gain, 12, 500000, expert_source=args.expert_source)
            candidates.append({"gain": gain, **result})
        selected = max(candidates, key=lambda row: (row["success_rate"] - row["unsafe_rate"], row["mean_return"]))
        (args.output / "expert_selection.json").write_text(json.dumps(candidates, indent=2))
        collection = run_expert(options, selected["gain"], args.episodes, 200000, args.output / "train.h5", args.expert_source)
        (args.output / "collection.json").write_text(json.dumps(collection, indent=2))
    else:
        import shutil

        with h5py.File(args.data_file) as supplied:
            if supplied.attrs["profile_sha256"] != hashlib.sha256(args.profile_file.read_bytes()).hexdigest():
                raise ValueError("Demonstrations belong to another reference profile")
        shutil.copyfile(args.data_file, args.output / "train.h5")
    data = StrikeDemonstrations(args.output / "train.h5")
    config = ACTConfig(state_dim=data.state.shape[1], action_dim=data.command.shape[1], chunk=10, hidden=128, layers=2, vision=False,
                       relative_commands=False, physical_oracle=False, observation_clip=10.,
                       action_semantics="motor_position_velocity_residual" if data.command.shape[1] == 2 else "motor_position_residual_rad")
    residual = data.command - data.nominal
    model = ActionChunkTransformer(config, data.state.mean(0), np.maximum(data.state.std(0), .01),
                                   residual.mean(0), np.maximum(residual.std(0), .001)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    rng = np.random.default_rng(0)
    best = (-1., -np.inf)
    with ExitStack() as resources:
        run = None
        if args.wandb_mode != "disabled":
            run = resources.enter_context(wandb.init(project="openarm-drum-skills", name=f"{args.target}-fast-ACT", dir=str(args.output),
                                                     config={**vars(args), "options": options, "actor_oracle_inputs": False}, mode=args.wandb_mode))
            if run.url:
                (args.output / "wandb_url.txt").write_text(run.url + "\n")
        for step in range(1, args.steps + 1):
            state, actions, padding = data.batch(rng.integers(0, len(data.state), args.batch_size), config.chunk, args.device, residual=True)
            model.train()
            loss, _, _ = model.distillation_loss(state, actions, padding, None)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if step % 100 == 0:
                check_resources()
                print(json.dumps({"target": args.target, "ACT_step": step, "loss": float(loss.detach())}), flush=True)
                if run is not None:
                    run.log({"ACT_step": step, "train/loss": float(loss.detach())})
            if step % args.evaluate_every == 0 or step == args.steps:
                report = evaluate_act(model, options, 16, 700000, args.device)
                score = (report["success_rate"] - report["unsafe_rate"], report["mean_return"])
                (args.output / f"validation_{step:06d}.json").write_text(json.dumps(report, indent=2))
                if score > best:
                    best = score
                    model.save(args.output / "act.pt", steps=step, actor_oracle_inputs=False,
                               data_sha256=hashlib.sha256((args.output / "train.h5").read_bytes()).hexdigest())
                print(json.dumps({"target": args.target, "ACT_step": step, "success_rate": report["success_rate"]}), flush=True)
                if run is not None:
                    run.log({"ACT_step": step, "eval/success_rate": report["success_rate"]})
        restored = ActionChunkTransformer.load(args.output / "act.pt", args.device)
        report = evaluate_act(restored, options, 64, 800000, args.device)
        (args.output / "heldout.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"target": args.target, "ACT_heldout_success": report["success_rate"]}), flush=True)


if __name__ == "__main__":
    main()
