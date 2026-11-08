# MIDI to drum contacts to measured MIDI

The loop is **target MIDI → controller → robot/drum physics → measured contact or drum-module MIDI → aligned feedback**. The target specifies drum, onset, and MIDI intensity; it does not specify joint motion. These commands never send robot motion.

Run from the repository root after installing the project and its local dependencies. `openarm-midi` and `python -m openarm_rl.music.workflow` are equivalent. Each output must be a new path.

## Try a complete simulated performance

```sh
# Build a reproducible scene, one snare stroke, and two small MIDI scores.
python -m openarm_rl.experiments.prepare --output runs/rl/midi-lab

python -m openarm_rl.music.workflow import \
  --midi runs/rl/midi-lab/evaluation.mid --output runs/rl/midi-import

python -m openarm_rl.music.workflow simulate \
  --scene-directory runs/rl/midi-lab/scene --profiles runs/rl/midi-lab/profiles \
  --midi runs/rl/midi-import/target.mid --mapping runs/rl/midi-import/mapping.json \
  --output runs/rl/midi-simulation

python -m openarm_rl.music.workflow evaluate \
  --target-midi runs/rl/midi-import/target.mid \
  --performed-midi runs/rl/midi-simulation/performed.mid \
  --output runs/rl/midi-simulation/rechecked-feedback.json
```

The bootstrap supports **snare only** and sparse repeated hits. For your own MIDI, substitute its path in `import`; `simulate` needs profiles for every requested drum. Unsupported notes fail explicitly. MIDI channel 10 (zero-based channel 9) maps 38 snare, 42 hi-hat, 43 floor tom, 47/48 rack toms, and 49/51 cymbals by default. Supply `--mapping` for another kit.

Import preserves all original bytes, tracks, and tempo changes in `source.mid`; `target.mid` is identical unless you explicitly request `--arrange --scene-directory … --profiles …`. Arrangement may change/drop notes and writes an itemized `arrangement/arrangement.json`. Import validates the score; simulation preflight separately checks scheduling and reachability.

Simulation writes:

| File | Meaning |
|---|---|
| `performance/evaluation.json` | Actual simulated contacts, timing, forces, safety/completion metrics |
| `performance/trace.json` | Controller/robot trace |
| `performed.mid` + `performed.mid.json` | Contact onsets and their simulation provenance/clock offset |
| `midi-feedback.json` | Target-versus-contact matching and reward components |

Without `--actor`, the reference teacher always plays through the score-only command path, the same path `rhythmic_references` qualifies strokes under and `ReferenceRuntime` sends to hardware. The trajectory-input path limits absolute joint-target acceleration to 120 rad/s² for learned residual actors; compact music strokes exceed that, and the limited target overshoots the ready pose after an isolated hit. An actor keeps the schema it was trained with.

Failed performances still export their measured contacts when an evaluation exists and return a nonzero exit status. Add `--video --contact-audio` for a rendered video with synthetic contact-triggered audio. This audio is not acoustic validation. Vary `--height-offset-m`, `--contact-stiffness-scale`, `--contact-damping-scale`, and `--gain-scale` to test sensitivity.

Without `--velocity-calibration calibration.json`, simulated note-ons use **velocity 64 solely as a MIDI placeholder**; intensity scoring is disabled. With calibration, intensity is an estimate derived from simulated contact speed, not a physical drum-module measurement. Keep each `.mid.json` beside its MIDI file. MIDI note-off lengths are serialization markers, not measured contact durations. Raw forces/impulses remain in the simulation report.

## Record a real drum module

Connect the module's USB/DIN MIDI input and install the optional port backend:

```sh
pip install -e 'projects/drum_skill_learning[midi]'
python -m openarm_rl.music.workflow capture --list-ports
python -m openarm_rl.music.workflow capture \
  --port 'Exact input port name' --duration-seconds 30 --output runs/rl/real-take
```

On macOS, run capture in a terminal with CoreMIDI access; some sandboxed sessions cannot initialize the native MIDI client. An empty port list means no input was detected.

Omit `--duration-seconds` to stop with Ctrl-C; silence is a valid recording. `performed.mid` contains received channel/SysEx events. `events.jsonl` preserves every incoming message, including clock/start/stop, with host monotonic receive timestamps. `performed.mid.json` records port, clock convention, hash, and provenance. A virtual MIDI source is still a virtual source: connect an actual drum module for physical feedback. This capture command does not play a score or start a robot.

Record **when target MIDI time zero occurred on the recording clock**, using a shared playback timestamp or synchronization signal. Do not align the first strike automatically; that would hide its timing error. Example: if score playback starts 2 seconds after capture and measured module/transport latency is 6 ms:

```sh
python -m openarm_rl.music.workflow evaluate \
  --target-midi runs/rl/midi-import/target.mid \
  --performed-midi runs/rl/real-take/performed.mid \
  --score-start-seconds 2.0 --latency-seconds 0.006 \
  --output runs/rl/real-take/feedback.json
```

Those numbers are illustrative: measure your own offset and latency. The comparison uses `aligned_time = recorded_time − score_start_seconds − latency_seconds`. Capture requires the explicit score offset; already aligned external MIDI uses `--score-start-seconds 0`. Simulation supplies its offset through the paired metadata, preserving early contacts instead of clipping them. No latency correction is assumed by default. Host receipt times include module/USB/OS delays; clock drift and jitter are not corrected automatically.

## Feedback and training

Hits match one-to-one on the same drum within ±60 ms. Reports include precision/recall/F1, matched timing error, intensity error when valid, missed hits, extra hits, and per-event rewards. Default feedback is `+1` per match, `−0.25 × |timing error| / 60 ms`, `−0.25 × |velocity error| / 126`, `−2` per miss, and `−0.5` per extra. A wrong drum or out-of-window hit incurs both an extra and a miss. Use `--intensity off` when velocity scales are not comparable. Empty captures produce misses, not a passing score.

This evaluator provides an inspectable **offline musical reward**; it does not update a policy or replace the simulator's effort/contact/safety costs. Real feedback can calibrate dynamics and velocity curves or supervise later training. Physical transfer still requires measured drum compliance/damping/rebound, actuator dynamics and delay, sensor calibration, and hardware validation. MIDI alone cannot represent force, stick rebound, contact duration, or acoustic timbre.

Test without hardware:

```sh
python -m pytest projects/drum_skill_learning/tests/imitation/test_midi_workflow.py -q
python -m openarm_rl.music.workflow --help
```

The tests preserve tempo changes through import, capture a fake input port, verify explicit clock/latency alignment, and cover silence, Ctrl-C, extra/missing hits, metadata hashes, and disabled placeholder intensity. Live port behavior requires a connected MIDI device.
