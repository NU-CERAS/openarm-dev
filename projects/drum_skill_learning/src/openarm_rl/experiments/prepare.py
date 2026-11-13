"""Build a small, reproducible native-physics MIDI lab from repository assets."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco

from openarm_sim.compute_guard import budgeted

from ..envs.camera_free_strike import CameraFreeStrikeEnv
from ..envs.rhythmic import RhythmicDrummingEnv
from ..music.corpus import write_segment
from ..planning.scene import prepare_scene
from ..planning.strikes import plan_center
from ..skills.registry import SKILLS


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(output: Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scene = output / "scene"
    prepare_scene(scene)
    model = mujoco.MjModel.from_binary_path(str(scene / "model.mjb"))
    metadata = json.loads((scene / "scene.json").read_text())
    profile = plan_center(model, metadata["robot"], metadata["scene"], SKILLS["snare"])
    if profile["status"] != "ready":
        raise RuntimeError(f"No validated snare geometry: {profile['reason']}")
    del model
    profile["configuration"]["retract_seconds"] = .28
    source = output / "source-snare.json"
    write_json(source, profile)
    with CameraFreeStrikeEnv(scene, source, position_range=0, domain_range=0) as env:
        env.reset(seed=1)
        while True:
            _, _, done, truncated, measured = env.step(env.reference_action())
            if done or truncated:
                break
        if (measured["hit_count"] != 1 or measured["unsafe_contact"] or not measured["recovered"]
                or measured["hits"][0]["contact_duration_s"] > .12):
            write_json(output / "failed-calibration.json", measured)
            raise RuntimeError("Nominal strike failed native contact/recovery validation")
        reference = env.reference.document()
        reference["impact_time_s"] = measured["hits"][0]["onset_time_s"]
        profile = dict(env.profile)
        profile["reference_bank"] = {
            "schema": "openarm.speed_conditioned_strike.v1",
            "speeds_m_s": [.6], "references": [reference],
            "calibration_speed_m_s": [measured["hits"][0]["downward_speed_m_s"]],
            "notes": "Single nominal command speed; simulated measurement, not MIDI velocity calibration.",
        }
    profiles = output / "profiles"
    profiles.mkdir()
    write_json(profiles / "snare.json", profile)
    write_json(output / "strike-measurement.json", measured)
    for name, times in (("train", [.5, 4.5, 8.5]), ("evaluation", [.8, 5.1])):
        write_segment(output / f"{name}.mid", [
            {"time_s": time, "note": 38, "velocity": 90} for time in times
        ], 0., 1.)
    # Verify repeated hits with the same command path used by the score workflow.
    with RhythmicDrummingEnv(scene, profiles, output / "evaluation.mid") as env:
        physics_timestep, control_period = float(env.model.opt.timestep), env.period
        env.reset(seed=0)
        while True:
            _, _, done, truncated, result = env.step(env.reference_action())
            if done or truncated:
                break
    write_json(output / "reference-validation.json", result)
    if not result["is_success"]:
        raise RuntimeError("Prepared repeated-hit task failed; inspect reference-validation.json")
    files = [scene / "model.mjb", scene / "scene.json", profiles / "snare.json",
             output / "train.mid", output / "evaluation.mid"]
    manifest = {
        "schema": "openarm.midi_lab.v1", "origin": "simulation", "targets": ["snare"],
        "physics_timestep_s": physics_timestep, "control_period_s": control_period,
        "hashes": {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "limits": ["One drum, sparse repeated hits, one nominal speed.",
                   "Drum contact parameters are provisional; no real sensor calibration.",
                   "Training and evaluation use different timing but the same strike primitive.",
                   "Generated files do not qualify physical robot execution."],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


@budgeted("openarm_rl.experiments.prepare")
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for scene, profiles and MIDI")
    args = parser.parse_args()
    prepare(args.output)
    print(json.dumps({"output": str(args.output), "validated": True, "targets": ["snare"]}))


if __name__ == "__main__":
    main()
