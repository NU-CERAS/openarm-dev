"""Reproducible offline comparison against physically detected MIDI contacts.

Run ``python -m openarm_rl.experiments.compare --help``. Results are simulation
experiments, not evidence that any controller is safe on a physical robot.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import time

import mujoco
import numpy as np

from openarm_sim.compute_guard import budgeted

from ..envs.rhythmic import RhythmicDrummingEnv
from ..music.reference import profile_hashes
from ..music.score import ContactScore
from ..paths import PROJECT_ROOT
from ..planning.strikes import scene_identity
from .controllers import BCController, IKController, MPCController, NominalPhysics, ReferenceController
from .models import DynamicsResidual, RidgeModel, dynamics_features, policy_features


VARIANTS = ("reference", "ik", "mpc", "residual_mpc", "bc")
SCENARIOS = {
    "nominal": {},
    "lowered": {"height_offset_m": -.005, "domain_range": .03},
    "raised": {"height_offset_m": .005, "domain_range": .03},
    "soft": {"contact_stiffness_scale": .7, "contact_damping_scale": .9, "domain_range": .03},
    "gain": {"gain_scale": .85, "domain_range": .03},
}
TRAINING_SCENARIOS = (
    {"gain_scale": .9, "domain_range": .03},
    {"gain_scale": 1.1, "contact_stiffness_scale": .9, "domain_range": .03},
    {"gain_scale": .95, "height_offset_m": -.002, "domain_range": .03},
)


def musical_score_sha256(path, mapping=None):
    """Identity of requested contacts, independent of MIDI container metadata.

    Simultaneous events are sorted consistently; duplicate hits are retained.
    Nanosecond rounding removes floating summation differences between equivalent
    MIDI tempo/tick encodings without merging musically meaningful hit offsets.
    """
    score = ContactScore.read(path, mapping)
    contacts = sorted((round(goal.time_s, 9), goal.target, goal.velocity) for goal in score.goals)
    return hashlib.sha256(json.dumps(contacts, separators=(",", ":")).encode()).hexdigest()


def csv_values(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def seeds(value):
    try:
        result = [int(part) for part in csv_values(value)]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers") from error
    if not result or len(result) != len(set(result)) or min(result) < 0:
        raise argparse.ArgumentTypeError("Seeds must be distinct nonnegative integers")
    return result


def validate_split(train_seeds, test_seeds, learned=True):
    if len(set(train_seeds)) != len(train_seeds) or len(set(test_seeds)) != len(test_seeds):
        raise ValueError("Each seed list must contain distinct seeds")
    if not test_seeds or set(train_seeds) & set(test_seeds):
        raise ValueError("Evaluation seeds must be nonempty and disjoint from all training/validation seeds")
    if learned and len(train_seeds) < 3:
        raise ValueError("Learning requires at least three training seeds: two fitting seeds and one validation seed")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def measured_state(env):
    # Same idealized synchronous encoder measurement for every algorithm. It
    # deliberately excludes force, shifted drum position and true plant gains.
    return env.data.qpos[env.controller.q].copy(), env.data.qvel[env.controller.v].copy()


def state_error(prediction, measured):
    error = np.asarray(prediction)-np.asarray(measured)
    return {"position_rmse_rad": float(np.sqrt(np.mean(error[:, :14]**2))),
            "velocity_rmse_rad_s": float(np.sqrt(np.mean(error[:, 14:]**2)))}


def train_models(options, physics, output, train_seeds, max_steps=None):
    """Collect teacher transitions before evaluating; last seed is validation."""
    x_dynamics, y_dynamics, x_policy, y_policy, row_seeds = [], [], [], [], []
    predicted_states, actual_states, episodes = [], [], []
    started = time.perf_counter()
    for index, seed in enumerate(train_seeds):
        perturbation = TRAINING_SCENARIOS[index % len(TRAINING_SCENARIOS)]
        with RhythmicDrummingEnv(**options, **perturbation) as env:
            env.reset(seed=seed)
            teacher = ReferenceController(env.reference, physics)
            steps, excluded_partial_steps = 0, 0
            while env.active and (max_steps is None or steps < max_steps):
                q, dq = measured_state(env)
                state_time = env.elapsed
                features = policy_features(q, dq, teacher.previous_velocity, teacher.previous_target, teacher.anchor,
                                           env.score.future(env.elapsed-env.reference.preroll))
                command = teacher.command(q, dq, env.elapsed)
                next_q, next_dq, _, _ = physics.predict(q, dq, command)
                env.step_command(command.motor, command.velocity, target_position=command.target)
                steps += 1
                if not np.isclose(env.elapsed-state_time, physics.period):
                    # A safety stop can interrupt the ten physics substeps.
                    # Such a state is not a valid one-period dynamics label.
                    excluded_partial_steps += 1
                    continue
                actual_q, actual_dq = measured_state(env)
                predicted, actual = np.r_[next_q, next_dq], np.r_[actual_q, actual_dq]
                x_dynamics.append(dynamics_features(q, dq, command.motor, command.velocity))
                y_dynamics.append(actual-predicted)
                x_policy.append(features)
                y_policy.append(command.velocity)
                row_seeds.append(seed)
                predicted_states.append(predicted)
                actual_states.append(actual)
            episodes.append({"seed": seed, "perturbation": perturbation, "steps": steps,
                             "excluded_partial_steps": excluded_partial_steps,
                             "completed": not env.active and not env.unsafe,
                             "f1": env.info()["f1"], "failure": env.failure})
        print(json.dumps({"training_seed": seed, "steps": steps}), flush=True)
    xd, yd, xp, yp = map(np.asarray, (x_dynamics, y_dynamics, x_policy, y_policy))
    row_seeds = np.asarray(row_seeds)
    validation = row_seeds == train_seeds[-1]
    fit = ~validation
    if fit.sum() < 2 or validation.sum() < 2:
        raise ValueError("Too few safe transitions for separate fitting and validation")
    dynamics = RidgeModel(regularization=1.).fit(xd[fit], yd[fit])
    policy = RidgeModel(regularization=2., hidden=128).fit(xp[fit], yp[fit])
    dynamics.save(output/"dynamics-residual.npz")
    policy.save(output/"bc-policy.npz")
    np.savez_compressed(output/"training-transitions.npz", dynamics_features=xd,
                        next_state_error=yd, policy_features=xp, teacher_velocity=yp,
                        nominal_next_state=np.asarray(predicted_states), measured_next_state=np.asarray(actual_states),
                        seed=row_seeds, fit=fit)
    predicted, actual = np.asarray(predicted_states), np.asarray(actual_states)
    residual = DynamicsResidual(dynamics)
    corrections = np.clip(dynamics.predict(xd[validation]), -residual.limits, residual.limits)
    report = {
        "schema": "openarm.controller_training.v1", "fitting_seeds": train_seeds[:-1],
        "validation_seeds": train_seeds[-1:], "episodes": episodes,
        "fitting_transitions": int(fit.sum()), "validation_transitions": int(validation.sum()),
        "wall_time_s": time.perf_counter()-started,
        "dynamics": {"kind": "standardized linear ridge next-state residual",
                     "input": "q[14], dq[14], motor_position-q[14], motor_velocity[14]",
                     "output": "next measured q/dq minus nominal MuJoCo one-period prediction",
                     "period_s": physics.period, "randomized_parameters_available": False,
                     "correction_clip": {"position_rad": .01, "velocity_rad_s": .5},
                     "validation_nominal": state_error(predicted[validation], actual[validation]),
                     "validation_corrected": state_error(predicted[validation]+corrections, actual[validation])},
        "policy": {"kind": "behavior cloning, standardized ridge with 128 fixed tanh features",
                   "input": "q-initial[14], dq[14], previous command velocity[14], previous target-initial[14], next eight MIDI events[80]",
                   "output": "14 joint command velocities", "reference_at_inference": False,
                   "teacher": "reference", "reinforcement_learning": False,
                   "validation_command_rmse_rad_s": float(np.sqrt(np.mean((policy.predict(xp[validation])-yp[validation])**2)))},
    }
    write_json(output/"training.json", report)
    return dynamics, policy, report


def make_controller(variant, env, physics, dynamics=None, policy=None, horizon=4, candidates=5):
    if variant == "reference":
        return ReferenceController(env.reference, physics)
    if variant == "ik":
        return IKController(env.reference, physics)
    if variant == "mpc":
        return MPCController(env.reference, physics, horizon=horizon, candidates=candidates)
    if variant == "residual_mpc":
        if dynamics is None:
            raise ValueError("Residual MPC requires a fitted dynamics residual")
        return MPCController(env.reference, physics, DynamicsResidual(dynamics), horizon, candidates)
    if variant == "bc":
        if policy is None:
            raise ValueError("Behavior cloning requires a fitted policy")
        return BCController(env.reference, physics, env.score, policy)
    raise ValueError(f"Unknown controller {variant}")


def evaluate_episode(options, physics, variant, scenario, seed, output, dynamics=None, policy=None,
                     horizon=4, candidates=5, max_steps=None):
    output.mkdir()
    timings, diagnostics = [], []
    with RhythmicDrummingEnv(**options, **SCENARIOS[scenario]) as env:
        env.reset(seed=seed)
        controller = make_controller(variant, env, physics, dynamics, policy, horizon, candidates)
        physics_steps = physics.forward_evaluations
        started = time.perf_counter()
        failure = None
        total_reward = 0.
        while env.active and (max_steps is None or len(timings) < max_steps):
            q, dq = measured_state(env)
            tick = time.perf_counter()
            try:
                command = controller.command(q, dq, env.elapsed)
                duration = time.perf_counter()-tick
                timings.append(duration)
                _, reward, _, _, _ = env.step_command(command.motor, command.velocity, target_position=command.target)
                total_reward += reward
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                failure = f"{type(error).__name__}: {error}"
                break
            diagnostics.append({"time_s": env.elapsed, "controller_time_s": duration,
                                "mpc_cost": getattr(controller, "last_cost", None),
                                "mpc_candidate": getattr(controller, "last_candidate", None),
                                "mpc_unsafe_candidates": getattr(controller, "last_unsafe_candidates", None),
                                "ik_tip_error_m": getattr(controller, "last_tip_error_m", None)})
        info = env.info()
        completed = not env.active and not env.unsafe and failure is None and env.elapsed >= env.end-1e-8
        report = {
            "variant": variant, "scenario": scenario, "seed": seed,
            "description": controller.description,
            "reference_assistance": controller.reference_assistance,
            "perturbation": SCENARIOS[scenario], "completed": completed,
            "truncated_by_budget": bool(env.active and failure is None),
            "controller_failure": failure, "steps": len(timings), "total_reward": total_reward,
            "wall_time_s": time.perf_counter()-started,
            "controller_p50_ms": float(1000*np.percentile(timings, 50)) if timings else None,
            "controller_p95_ms": float(1000*np.percentile(timings, 95)) if timings else None,
            "deadline_s": physics.period,
            "deadline_miss_fraction": float(np.mean(np.asarray(timings)>physics.period)) if timings else None,
            "prediction_periods": physics.forward_evaluations-physics_steps,
            "metrics": info,
        }
        if variant == "residual_mpc":
            report["description"] += " plus learned next-state dynamics error, clipped to 0.01 rad / 0.5 rad/s"
        write_json(output/"result.json", report)
        write_json(output/"trace.json", env.trace)
        write_json(output/"controller-trace.json", diagnostics)
        return report


def aggregate(results):
    grouped = defaultdict(list)
    for result in results:
        grouped[(result["variant"], result["scenario"])].append(result)
    summaries = []
    for (variant, scenario), rows in grouped.items():
        entry = {"variant": variant, "scenario": scenario, "episodes": len(rows),
                 "completed_fraction": float(np.mean([r["completed"] for r in rows])),
                 "unsafe_fraction": float(np.mean([r["metrics"]["unsafe_contact"] for r in rows])),
                 "controller_failures": sum(r["controller_failure"] is not None for r in rows)}
        for key in ("f1", "timing_p95_s", "velocity_mae", "peak_force_N", "missed_hits", "extra_hits"):
            values = [r["metrics"].get(key) for r in rows if r["metrics"].get(key) is not None]
            entry[key+"_mean"] = float(np.mean(values)) if values else None
        for key in ("controller_p50_ms", "controller_p95_ms", "deadline_miss_fraction"):
            values = [r[key] for r in rows if r[key] is not None]
            entry[key+"_mean"] = float(np.mean(values)) if values else None
        summaries.append(entry)
    return summaries


def compare(scene, profiles, midi, output, train_midi=None, train_seeds=(10, 11, 12), test_seeds=(0, 1, 2),
            variants=VARIANTS, scenarios=("nominal", "lowered", "gain"), horizon=4, candidates=5,
            max_steps=None, max_training_steps=None, mapping=None, velocity_calibration=None):
    variants, scenarios, train_seeds, test_seeds = map(list, (variants, scenarios, train_seeds, test_seeds))
    if not variants or set(variants)-set(VARIANTS) or len(set(variants)) != len(variants):
        raise ValueError("Select distinct supported controller variants")
    if not scenarios or set(scenarios)-set(SCENARIOS) or len(set(scenarios)) != len(scenarios):
        raise ValueError("Select distinct supported scenarios")
    learned = bool(set(variants) & {"bc", "residual_mpc"})
    validate_split(train_seeds, test_seeds, learned)
    if horizon < 1 or candidates not in (3, 5):
        raise ValueError("MPC horizon must be positive and candidates must be 3 or 5")
    if any(value is not None and value < 1 for value in (max_steps, max_training_steps)):
        raise ValueError("Step budgets must be positive")
    scene, profiles, midi, output = map(Path, (scene, profiles, midi, output))
    train_midi = Path(train_midi) if train_midi else midi
    mapping = Path(mapping) if mapping else PROJECT_ROOT/"configs/music/percussion.json"
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((scene/"scene.json").read_text())
    physics = NominalPhysics(scene, metadata)
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest = {
        "schema": "openarm.controller_comparison.v1", "simulation_only": True,
        "scene_identity": scene_identity(metadata["robot"], metadata["scene"]),
        "model_sha256": sha(scene/"model.mjb"), "profiles_sha256": profile_hashes(profiles),
        "evaluation_midi_sha256": sha(midi), "training_midi_sha256": sha(train_midi),
        "evaluation_score_sha256": musical_score_sha256(midi, mapping),
        "training_score_sha256": musical_score_sha256(train_midi, mapping),
        "held_out_score": musical_score_sha256(midi, mapping) != musical_score_sha256(train_midi, mapping),
        "inputs": {"scene": str(scene.resolve()), "profiles": str(profiles.resolve()),
                   "midi": str(midi.resolve()), "train_midi": str(train_midi.resolve()),
                   "mapping": str(mapping.resolve()),
                   "velocity_calibration": str(Path(velocity_calibration).resolve()) if velocity_calibration else None},
        "mapping_sha256": sha(mapping),
        "controller_source_sha256": {name: sha(Path(__file__).parent/name)
                                     for name in ("compare.py", "controllers.py", "models.py")},
        "velocity_calibration_sha256": sha(velocity_calibration) if velocity_calibration else None,
        "fitting_seeds": train_seeds[:-1] if learned else [], "validation_seeds": train_seeds[-1:] if learned else [],
        "evaluation_seeds": test_seeds, "variants": variants, "scenarios": scenarios,
        "max_steps": max_steps, "max_training_steps": max_training_steps,
        "mpc": {"horizon_periods": horizon, "period_s": physics.period, "candidates": candidates,
                "optimizer": "finite candidate enumeration", "prediction_model": "native nominal MuJoCo",
                "safety": "Reject predicted unexpected contact >0.1N, per-contact penetration/force, summed arm/drum force, or tracking stop"},
        "feedback": "synchronous idealized joint encoders for every variant; no contact/plant parameters supplied",
        "nominal_seed_repetitions": "Nominal has no randomization; repeated seeds are deterministic repeats, not independent trials",
        "intensity": "measured calibration supplied" if velocity_calibration else "disabled: no measured speed-to-MIDI calibration",
        "runtime": {"python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": np.__version__,
                    "platform": platform.platform(), "processor": platform.processor()},
        "limitations": ["IK and both MPC variants use compiled reference guidance",
                        "BC is a small supervised baseline, not a test of pure reinforcement learning",
                        "No real sensor data, hardware latency or physical robot validation in this benchmark",
                        "Contact parameters are numerical assumptions until identified from a real drum",
                        "MPC may miss 10 ms deadlines; offline results do not establish real-time feasibility"],
    }
    write_json(output/"manifest.json", manifest)
    options = {"scene_directory": scene, "profiles": profiles, "midi_file": midi, "score_only": True,
               "mapping": mapping, "velocity_calibration": velocity_calibration}
    dynamics = policy = training = None
    if learned:
        dynamics, policy, training = train_models({**options, "midi_file": train_midi}, physics,
                                                 output, train_seeds, max_training_steps)
    results = []
    for scenario in scenarios:
        for seed in test_seeds:
            for variant in variants:
                result = evaluate_episode(options, physics, variant, scenario, seed,
                                          output/f"{scenario}-{seed}-{variant}", dynamics, policy,
                                          horizon, candidates, max_steps)
                results.append(result)
                write_json(output/"results.json", {"manifest": manifest, "training": training,
                                                   "episodes": results, "summary": aggregate(results)})
                print(json.dumps({"variant": variant, "scenario": scenario, "seed": seed,
                                  "f1": result["metrics"]["f1"], "completed": result["completed"],
                                  "unsafe": result["metrics"]["unsafe_contact"],
                                  "controller_p95_ms": result["controller_p95_ms"]}), flush=True)
    return {"manifest": manifest, "training": training, "episodes": results, "summary": aggregate(results)}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scene", "--scene-directory", dest="scene", required=True, help="Prepared scene directory containing model.mjb and scene.json")
    result.add_argument("--profiles", required=True, help="Validated per-drum reference profiles directory")
    result.add_argument("--midi", required=True, help="Evaluation target MIDI")
    result.add_argument("--train-midi", help="Separate training score; omission explicitly marks score overlap")
    result.add_argument("--output", required=True, help="New results directory; existing output is never overwritten")
    result.add_argument("--train-seeds", type=seeds, default=[10, 11, 12], help="Last seed is model validation, preceding seeds fit models")
    result.add_argument("--seeds", type=seeds, default=[0, 1, 2], help="Held-out evaluation seeds")
    result.add_argument("--variants", type=csv_values, default=list(VARIANTS), help=",".join(VARIANTS))
    result.add_argument("--scenarios", type=csv_values, default=["nominal", "lowered", "gain"], help=",".join(SCENARIOS))
    result.add_argument("--horizon", type=int, default=4, help="MPC lookahead in 10 ms periods")
    result.add_argument("--candidates", type=int, choices=(3, 5), default=5)
    result.add_argument("--max-steps", type=int, help="Optional smoke-test episode cap; capped episodes are never reported complete")
    result.add_argument("--max-training-steps", type=int, help="Optional teacher collection cap per seed")
    result.add_argument("--mapping", help="Percussion note-to-drum mapping JSON")
    result.add_argument("--velocity-calibration", help="Measured per-drum speed-to-MIDI calibration JSON")
    return result


@budgeted("openarm_rl.experiments.compare")
def main(argv=None):
    args = parser().parse_args(argv)
    try:
        compare(args.scene, args.profiles, args.midi, args.output, args.train_midi,
                args.train_seeds, args.seeds, args.variants, args.scenarios, args.horizon,
                args.candidates, args.max_steps, args.max_training_steps, args.mapping, args.velocity_calibration)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
