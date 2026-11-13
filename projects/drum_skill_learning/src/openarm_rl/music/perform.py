import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

from openarm_sim.compute_guard import budgeted, check_resources

from ..envs.rhythmic import RhythmicDrummingEnv
from ..deployment.rhythm_actor import RhythmActorRuntime
from ..recording.display import NativeDisplay
from ..recording.music_audio import add_contact_audio
from .arrangement import CANONICAL_MAPPING, arrange_midi
from .examples import write_example
from .preflight import inspect_performance


def rollout(env, output, seed=800000, command=None, video=False, reset_options=None, controller_label=None, contact_audio=False,
            backing=(), caption=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    observation, _ = env.reset(seed=seed, options=reset_options)
    total = 0.
    with ExitStack() as resources:
        writer = resources.enter_context(imageio.get_writer(output / "simulation_silent.mp4", fps=25, codec="libx264", macro_block_size=1)) if video else None
        if writer is not None:
            writer.append_data(env.render())
        while True:
            action = env.reference_action() if command is None else command(observation)
            observation, reward, done, truncated, info = env.step(action)
            total += reward
            if env.steps % 100 == 0:
                check_resources()
            if writer is not None and env.steps % 4 == 0:
                frame = Image.fromarray(env.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 1280, 104), fill=(20, 28, 40))
                controller = controller_label or ("LEARNED ACT/PPO" if command is not None else "REFERENCE CONTROLLER")
                draw.text((20, 12), f"{controller} | score time {info['score_time_s']:.2f}s | actual contacts {info['actual_hits']} | correct {info['correct_hits']}/{info['target_events']}", fill="white", font_size=22)
                upcoming = [g for g in env.score.goals if g.time_s > info["score_time_s"]][:3]
                draw.text((20, 46), "Next: " + " | ".join(f"{g.time_s:.2f}s {g.target}" for g in upcoming), fill="#ffcc66", font_size=19)
                audio = ("Audio: synthetic drums on simulated contacts" if contact_audio else "Silent video | Evaluation: physical contact timing")
                if backing:
                    audio += " + MIDI kick backing (NOT played by the robot)"
                draw.text((20, 76), audio+(f" | {caption}" if caption else ""), fill="white", font_size=16)
                targets = sorted({goal.target for goal in env.score.goals})
                top = 720-30-22*len(targets)
                draw.rectangle((0, top, 1280, 720), fill=(20, 28, 40))
                draw.text((16, top+4), "MIDI target: yellow | Actual contact: green | Time window: -1.5 to +2.5 seconds", fill="white", font_size=15)
                now = info["score_time_s"]
                for index, target in enumerate(targets):
                    y = top+32+22*index
                    draw.text((16, y-8), target, fill="white", font_size=15)
                    draw.line((180, y, 1260, y), fill="#526070")
                    for goal in env.score.goals:
                        if goal.target == target and -1.5 <= goal.time_s-now <= 2.5:
                            x = 180+(goal.time_s-now+1.5)/4*1080
                            draw.ellipse((x-4, y-5, x+4, y+3), outline="#ffcc66", width=2)
                    for hit in env.matcher.actual:
                        if hit["target"] == target and -1.5 <= hit["time_s"]-now <= 2.5:
                            x = 180+(hit["time_s"]-now+1.5)/4*1080
                            color = "#45df95" if hit["matched_event_id"] is not None else "#ff6655"
                            draw.ellipse((x-3, y+2, x+3, y+8), fill=color)
                draw.line((585, top+25, 585, 720), fill="white", width=2)
                writer.append_data(np.asarray(frame))
            if done or truncated:
                break
    info.update(env.matcher.report(detailed=True))
    info.update(episode_return=total, seed=seed, control_hz=100, action_dim=14,
                completed=env.elapsed >= env.reference.duration-.011, actor_oracle_inputs=False,
                controller="native_reference_teacher" if command is None else "learned_actor",
                controller_label=controller_label,
                policy_uses_reference_trajectory=command is None or not env.score_only,
                intensity_control="fixed_nominal_reference_speed" if command is None else "learned_actor",
                intensity_feedback="measured_speed_to_midi_calibration" if env.velocity_calibration else "intensity_reward_disabled")
    if video:
        if contact_audio:
            add_contact_audio(output, info["hits"], env.reference.preroll-env.started, env.elapsed-env.started, backing)
        else:
            shutil.copyfile(output/"simulation_silent.mp4", output/"performance.mp4")
        info["audio_source"] = "procedural drum synthesizer triggered by measured simulation contacts" if contact_audio else "none"
        info["backing_audio"] = "MIDI foot notes, not robot contacts" if backing else "none"
        info["acoustic_validation"] = "not_performed"
        info["video_fps"] = 25
    (output / "evaluation.json").write_text(json.dumps(info, indent=2))
    (output / "trace.json").write_text(json.dumps(env.trace, indent=2))
    (output / "score.json").write_text(json.dumps(env.score.document(), indent=2))
    (output / "reference.json").write_text(json.dumps(env.reference.document(), indent=2))
    return info


def reference_playback_score_only(args) -> bool:
    """Command path for this performance.

    Without an actor, the reference teacher plays the score through the score-only
    command path: the same path `rhythmic_references` qualifies strokes under and
    that `ReferenceRuntime` sends to hardware. The trajectory-input path rate-limits
    absolute joint targets to 120 rad/s^2 for learned residual actors. Compact music
    strokes lift the wrist far faster than that, so the limited target lags and then
    overshoots the ready pose after an isolated hit, tripping the 0.2 rad tracking
    stop. An actor keeps the input schema it was trained with.
    """
    return args.score_only or args.actor is None


@budgeted("openarm_rl.music.perform")
def main():
    parser = argparse.ArgumentParser(description="Validate and perform MIDI percussion with continuous native robot physics")
    parser.add_argument("--scene-directory", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--midi", type=Path)
    source.add_argument("--example", choices=["duet", "heldout", "fill", "groove"])
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--tempo-scale", type=float, default=1.)
    parser.add_argument("--arrange", action="store_true", help="Fold General MIDI notes onto the kit and report every changed or dropped note")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actor", type=Path)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--seed", type=int, default=800000)
    parser.add_argument("--domain-range", type=float, default=0.)
    parser.add_argument("--position-range", type=float, default=0.)
    parser.add_argument("--height-offset-m", type=float, default=0.)
    parser.add_argument("--contact-stiffness-scale", type=float, default=1.)
    parser.add_argument("--contact-damping-scale", type=float, default=1.)
    parser.add_argument("--gain-scale", type=float, default=1.)
    parser.add_argument("--velocity-calibration", type=Path, help="Measured speed-to-MIDI calibration JSON")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--contact-audio", action="store_true")
    parser.add_argument("--backing-kick", action="store_true", help="Mix the source MIDI foot notes under the contact audio; the video labels them as not played")
    parser.add_argument("--camera", choices=["side", "front"], default="side")
    parser.add_argument("--xvfb", type=Path)
    args = parser.parse_args()
    if args.contact_audio and not args.video:
        raise ValueError("Contact audio requires --video")
    if args.backing_kick and not (args.contact_audio and args.arrange):
        raise ValueError("Kick backing requires --contact-audio and --arrange")
    args.output.mkdir(parents=True, exist_ok=False)
    midi = write_example(args.output / "input.mid", args.example) if args.example else args.midi
    arrangement, backing, caption = None, (), None
    if args.arrange:
        midi, arrangement = arrange_midi(midi, args.scene_directory, args.profiles, args.output/"arrangement",
                                         args.mapping, args.tempo_scale)
        args.mapping, args.tempo_scale = CANONICAL_MAPPING, 1.
        caption = f"Stick notes kept {arrangement['kept_events']}/{arrangement['stick_events']}"
        if args.backing_kick:
            backing = [{"target": "kick", "time_s": row["time_s"], "midi_velocity": row["velocity"]}
                       for row in arrangement["dropped"] if row["reason"] == "foot_event"]
        print(json.dumps({key: value for key, value in arrangement.items() if key not in ("kept", "dropped")}), flush=True)
    preflight = inspect_performance(midi, args.scene_directory, args.profiles, args.mapping, args.tempo_scale)
    (args.output/"preflight.json").write_text(json.dumps(preflight, indent=2))
    if not preflight["executable"]:
        raise RuntimeError(f"MIDI cannot be scheduled; inspect {args.output/'preflight.json'}")
    score_only = reference_playback_score_only(args)
    with ExitStack() as resources:
        if args.xvfb:
            display = NativeDisplay(args.xvfb, args.output / "display")
            resources.callback(display.close)
        env = resources.enter_context(RhythmicDrummingEnv(args.scene_directory, args.profiles, midi,
                                                        mapping=args.mapping, tempo_scale=args.tempo_scale,
                                                        score_only=score_only,
                                                        domain_range=args.domain_range, position_range=args.position_range,
                                                        height_offset_m=args.height_offset_m,
                                                        contact_stiffness_scale=args.contact_stiffness_scale,
                                                        contact_damping_scale=args.contact_damping_scale,
                                                        gain_scale=args.gain_scale,
                                                        velocity_calibration=args.velocity_calibration,
                                                        render_mode="rgb_array" if args.video else None, camera=args.camera))
        command = None
        label, manifest = None, None
        if args.actor:
            runtime = RhythmActorRuntime(args.actor, args.scene_directory, args.profiles, midi,
                                         mapping=args.mapping, tempo_scale=args.tempo_scale)
            if runtime.score_only != args.score_only:
                raise ValueError("Actor schema and --score-only setting differ")
            actor = runtime.actor
            manifest = json.loads((args.actor.parent/"bundle.json").read_text())
            label = "ACT + PPO" if manifest["optimization_updates"] > 0 else "ACT INITIALIZATION"
            def command(observation):
                with torch.no_grad():
                    return actor(torch.from_numpy(observation["actor"])[None], torch.from_numpy(observation["nominal"])[None]).numpy()[0]
        result = rollout(env, args.output / "performance", seed=args.seed, command=command, video=args.video,
                         controller_label=label, contact_audio=args.contact_audio, backing=backing, caption=caption)
        if arrangement is not None:
            result["arrangement"] = {key: value for key, value in arrangement.items() if key not in ("kept", "dropped")}
            (args.output/"performance/evaluation.json").write_text(json.dumps(result, indent=2))
        if manifest is not None:
            result.update(checkpoint_environment_steps=manifest["steps"], ppo_epochs=manifest["optimization_updates"],
                          critic_warmup_environment_steps=manifest.get("critic_warmup_steps", 0))
            (args.output/"performance/evaluation.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k not in ("hits", "timing_errors_s")}), flush=True)
        if not result["is_success"] or not result["completed"]:
            raise RuntimeError("MIDI performance failed its contact/completion checks; inspect performance/evaluation.json")


if __name__ == "__main__":
    main()
