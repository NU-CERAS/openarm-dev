"""One entry point for source MIDI, simulation contacts, and received MIDI."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import mido

from ..paths import PROJECT_ROOT
from .capture import capture_midi, input_backend, write_timed_midi
from .feedback import compare_performance
from .score import ContactScore


DEFAULT_MAPPING = PROJECT_ROOT/"configs/music/percussion.json"


def write_json(path, document):
    Path(path).write_text(json.dumps(document, indent=2, allow_nan=False)+"\n")


def import_midi(source, output, mapping=None, *, arrange=False, scene_directory=None, profiles=None):
    """Keep the original bytes; arrangements are explicitly requested derivatives."""
    source, output = Path(source), Path(output)
    raw = source.read_bytes()
    midi = mido.MidiFile(source)
    if midi.type not in (0, 1):
        raise ValueError("Synchronous MIDI type 0 or 1 is required")
    if midi.ticks_per_beat <= 0 or any(message.type == "set_tempo" and message.tempo <= 0
                                     for track in midi.tracks for message in track):
        raise ValueError("MIDI requires positive ticks per beat and tempo; SMPTE timing is unsupported")
    if arrange and (scene_directory is None or profiles is None):
        raise ValueError("Explicit arrangement needs --scene-directory and --profiles")
    if output.exists():
        raise FileExistsError(output)
    if not arrange:
        # Validate before creating output; unsupported percussion never disappears.
        score = ContactScore.read(source, mapping)
    output.mkdir(parents=True, exist_ok=False)
    (output/"source.mid").write_bytes(raw)
    arrangement = None
    if arrange:
        # Arrangement loads physics models. Use its existing guarded process,
        # keeping ordinary import/capture/evaluation lightweight.
        command = [sys.executable, "-m", "openarm_rl.music.arrangement", "--midi", str(output/"source.mid"),
                   "--scene-directory", str(scene_directory), "--profiles", str(profiles),
                   "--output", str(output/"arrangement")]
        if mapping is not None:
            command.extend(["--mapping", str(mapping)])
        subprocess.run(command, check=True)
        arranged = output/"arrangement/arranged.mid"
        arrangement = json.loads((output/"arrangement/arrangement.json").read_text())
        shutil.copyfile(arranged, output/"target.mid")
        mapping = DEFAULT_MAPPING
        score = ContactScore.read(output/"target.mid", mapping)
    else:
        (output/"target.mid").write_bytes(raw)
    mapping = Path(mapping) if mapping is not None else DEFAULT_MAPPING
    shutil.copyfile(mapping, output/"mapping.json")
    mapping_document = json.loads(mapping.read_text())
    ignored = sum(message.type == "note_on" and message.velocity > 0
                  and message.channel != mapping_document["channel"] for message in midi)
    manifest = {
        "schema": "openarm.midi_import.v1", "source_path": str(source.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "target_sha256": score.source_sha256, "mapping_sha256": hashlib.sha256(mapping.read_bytes()).hexdigest(),
        "source_copy": "source.mid", "target": "target.mid", "mapping": "mapping.json",
        "source_bytes_preserved": True, "arranged": arrange,
        "target_events": len(score.goals), "target_duration_s": score.duration_s,
        "source_type": midi.type, "source_tracks": len(midi.tracks),
        "ignored_nonpercussion_note_ons": ignored,
        "arrangement_report": "arrangement/arrangement.json" if arrangement is not None else None,
        "executable": "not_checked; simulation preflight checks robot reachability and scheduling",
    }
    write_json(output/"import.json", manifest)
    write_json(output/"score.json", score.document())
    return manifest


def export_simulated_contacts(evaluation, output, mapping=None):
    """Export contact onsets, never score targets, as simulated performed MIDI."""
    evaluation, output = Path(evaluation), Path(output)
    if output.exists() or Path(str(output)+".json").exists():
        raise FileExistsError(output)
    report = json.loads(evaluation.read_text())
    if not isinstance(report, dict) or not isinstance(report.get("hits"), list):
        raise ValueError("Simulation evaluation must contain a measured hits list (an empty list is valid)")
    hits = report["hits"]
    mapping = DEFAULT_MAPPING if mapping is None else Path(mapping)
    mapping_document = json.loads(mapping.read_text())
    if mapping_document["schema"] != "openarm.percussion_mapping.v1":
        raise ValueError("Unsupported percussion mapping")
    reverse = {}
    for note, target in sorted(mapping_document["notes"].items(), key=lambda item: int(item[0])):
        reverse.setdefault(target, int(note))
    for hit in hits:
        if (not isinstance(hit, dict) or not isinstance(hit.get("target"), str)
                or hit["target"] not in reverse or not isinstance(hit.get("time_s"), (int, float))
                or not math.isfinite(hit["time_s"])):
            raise ValueError("Simulated contact requires a mapped drum and finite onset time")
        velocity = hit.get("midi_velocity")
        if velocity is not None and (not isinstance(velocity, (int, float))
                                     or not math.isfinite(velocity) or not 1 <= velocity <= 127):
            raise ValueError("Calibrated contact MIDI velocity must be finite and within 1..127")
    calibrated = bool(hits) and all(hit.get("midi_velocity") is not None for hit in hits)
    # MIDI cannot encode negative time. Keep all early contacts and state the
    # offset explicitly rather than clipping early contacts to score time zero.
    offset = max(0., -min((hit["time_s"] for hit in hits), default=0.))
    messages = []
    for hit in hits:
        velocity = round(hit["midi_velocity"]) if calibrated else 64
        time_s = hit["time_s"]+offset
        messages.extend([
            (time_s, mido.Message("note_on", channel=mapping_document["channel"], note=reverse[hit["target"]], velocity=velocity)),
            (time_s+.001, mido.Message("note_off", channel=mapping_document["channel"], note=reverse[hit["target"]], velocity=0)),
        ])
    output.parent.mkdir(parents=True, exist_ok=True)
    write_timed_midi(output, messages, max((row[0] for row in messages), default=0.),
                     "SIMULATED contact onsets; see paired .mid.json for intensity validity and time origin")
    metadata = {
        "schema": "openarm.performed_midi.v1", "measurement_source": "native_MuJoCo_contacts",
        "time_origin": "score_zero_shifted_to_preserve_negative_contact_times", "score_start_s": offset,
        "latency_correction_s": 0., "intensity": "calibrated_simulated_speed_estimate" if calibrated else "uncalibrated_placeholder",
        "placeholder_velocity": None if calibrated else 64,
        "intensity_note": "Uncalibrated placeholder is required for MIDI note_on and must not enter intensity reward.",
        "note_off_note": "1 ms note lengths are serialization markers; they do not measure contact duration.",
        "performed_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "evaluation_path": str(evaluation.resolve()), "evaluation_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
        "actual_hits": len(hits), "physical_validation": False,
    }
    write_json(Path(str(output)+".json"), metadata)
    return metadata


def simulate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    command = [sys.executable, "-m", "openarm_rl.music.perform"]
    for name in ("scene_directory", "profiles", "midi", "mapping", "output", "actor", "seed", "domain_range", "position_range", "camera",
                 "height_offset_m", "contact_stiffness_scale", "contact_damping_scale", "gain_scale", "velocity_calibration"):
        value = getattr(args, name)
        if value is not None:
            command.extend(["--"+name.replace("_", "-"), str(value)])
    for name in ("video", "contact_audio", "score_only"):
        if getattr(args, name):
            command.append("--"+name.replace("_", "-"))
    result = subprocess.run(command, check=False)
    evaluation = args.output/"performance/evaluation.json"
    if evaluation.exists():
        # Failed episodes are useful evidence too. Keep the performer's nonzero
        # exit status and export their actual contacts, including silence.
        metadata = export_simulated_contacts(evaluation, args.output/"performed.mid", args.mapping)
        feedback = compare_performance(args.midi, args.output/"performed.mid", args.mapping,
                                       score_start_s=metadata["score_start_s"])
        write_json(args.output/"midi-feedback.json", feedback)
        print(json.dumps({"performed_midi": str(args.output/"performed.mid"),
                          "feedback": str(args.output/"midi-feedback.json"), "f1": feedback["f1"],
                          "intensity_scored": feedback["intensity_scored"]}), flush=True)
    return result.returncode


def parser():
    root = argparse.ArgumentParser(description="Import target MIDI, simulate drum contacts, capture real MIDI input, and evaluate feedback. No robot motion is sent.")
    commands = root.add_subparsers(dest="command", required=True)
    imported = commands.add_parser("import", help="Validate MIDI and preserve a byte-identical source with provenance")
    imported.add_argument("--midi", type=Path, required=True)
    imported.add_argument("--output", type=Path, required=True, help="New session directory")
    imported.add_argument("--mapping", type=Path, help="Percussion mapping, or arrangement mapping with --arrange")
    imported.add_argument("--arrange", action="store_true", help="Explicitly allow note substitution/dropping; save an itemized report")
    imported.add_argument("--scene-directory", type=Path)
    imported.add_argument("--profiles", type=Path)
    capture = commands.add_parser("capture", help="Record timestamped input from a connected drum MIDI port")
    capture.add_argument("--list-ports", action="store_true")
    capture.add_argument("--port", help="Exact input name from --list-ports")
    capture.add_argument("--duration-seconds", type=float, help="Positive duration; omit to stop with Ctrl-C")
    capture.add_argument("--output", type=Path, help="New recording directory")
    evaluate = commands.add_parser("evaluate", help="Match measured notes to the target with explicit clock alignment")
    evaluate.add_argument("--target-midi", type=Path, required=True)
    evaluate.add_argument("--performed-midi", type=Path, required=True)
    evaluate.add_argument("--mapping", type=Path)
    evaluate.add_argument("--score-start-seconds", type=float, help="Target time zero on recording clock; 0 for already aligned files. Simulation sidecar supplies this automatically.")
    evaluate.add_argument("--latency-seconds", type=float, default=0., help="Measured input/module latency to subtract (default: no correction)")
    evaluate.add_argument("--intensity", choices=["auto", "off", "on"], default="auto")
    evaluate.add_argument("--output", type=Path, required=True, help="New JSON report")
    simulation = commands.add_parser("simulate", help="Run native MuJoCo and export measured contact onsets to performed.mid")
    for flag in ("scene-directory", "profiles", "midi", "output"):
        simulation.add_argument("--"+flag, type=Path, required=True)
    simulation.add_argument("--mapping", type=Path)
    simulation.add_argument("--actor", type=Path)
    simulation.add_argument("--score-only", action="store_true")
    simulation.add_argument("--seed", type=int, default=800000)
    simulation.add_argument("--domain-range", type=float, default=0.)
    simulation.add_argument("--position-range", type=float, default=0.)
    simulation.add_argument("--height-offset-m", type=float, default=0.)
    simulation.add_argument("--contact-stiffness-scale", type=float, default=1.)
    simulation.add_argument("--contact-damping-scale", type=float, default=1.)
    simulation.add_argument("--gain-scale", type=float, default=1.)
    simulation.add_argument("--velocity-calibration", type=Path, help="Measured speed-to-MIDI calibration; without it simulated intensity scoring is disabled")
    simulation.add_argument("--camera", choices=["side", "front"], default="side")
    simulation.add_argument("--video", action="store_true")
    simulation.add_argument("--contact-audio", action="store_true", help="Synthetic audio of simulated contacts; requires --video")
    exported = commands.add_parser("export-simulation", help="Export contacts from an existing simulation evaluation.json")
    exported.add_argument("--evaluation", type=Path, required=True)
    exported.add_argument("--mapping", type=Path)
    exported.add_argument("--output", type=Path, required=True, help="New .mid file; creates paired .mid.json metadata")
    return root


def run(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.command == "import":
        result = import_midi(args.midi, args.output, args.mapping, arrange=args.arrange,
                             scene_directory=args.scene_directory, profiles=args.profiles)
    elif args.command == "capture":
        if args.list_ports:
            result = {"input_ports": input_backend().get_input_names()}
        else:
            if args.port is None or args.output is None:
                cli.error("capture needs --port and --output, or --list-ports")
            result = capture_midi(args.output, args.port, args.duration_seconds)
    elif args.command == "evaluate":
        if args.output.exists():
            raise FileExistsError(args.output)
        result = compare_performance(args.target_midi, args.performed_midi, args.mapping,
                                     args.latency_seconds, score_start_s=args.score_start_seconds,
                                     intensity=args.intensity, require_alignment=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    elif args.command == "simulate":
        return simulate(args)
    else:
        result = export_simulated_contacts(args.evaluation, args.output, args.mapping)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0


def main(argv=None):
    try:
        return run(argv)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"openarm-midi: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
