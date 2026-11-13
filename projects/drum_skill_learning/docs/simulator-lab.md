# MIDI simulator lab

The loop is **target MIDI → controller → robot/drum physics → detected hits →
performed MIDI → timing/intensity/miss/extra rewards**. Real drum-module MIDI can
replace the simulated performed MIDI in the evaluator. Joint feedback belongs
in the fast control loop; MIDI measures musical outcomes.

## Run a complete example

From the repository root, activate your Python environment and install:

```sh
python -m pip install -e . -e projects/opendrummer_playground \
  -e projects/cartesian_console -e 'projects/drum_skill_learning[midi]'
export OPENARM_RAM_RESERVE_GIB=1
python -m openarm_rl.experiments.prepare --output runs/rl/midi-lab
python -m openarm_rl.music.workflow import \
  --midi runs/rl/midi-lab/evaluation.mid --output runs/rl/my-song
python -m openarm_rl.music.workflow simulate \
  --scene-directory runs/rl/midi-lab/scene --profiles runs/rl/midi-lab/profiles \
  --midi runs/rl/my-song/target.mid --mapping runs/rl/my-song/mapping.json \
  --output runs/rl/my-performance
```

Use new output directories for each run. `prepare` builds the native scene from
repository assets, validates a snare stroke against simulated contacts, then checks
repeated hits. Its training and evaluation MIDI use different timings. This
starter has one drum and one nominal command speed; it does not represent a
complete expressive kit. Existing kit-planning tools can generate more profiles.

Replace `evaluation.mid` with your MIDI. Unsupported notes fail explicitly.
`import --arrange --scene-directory … --profiles …` permits an arrangement and
records every changed/dropped event. Imported source bytes are retained.
The selected profiles must cover every target and allow enough recovery/transfer
time. Dense or unreachable songs fail preflight.

`my-performance` contains:

| File | Meaning |
| --- | --- |
| `performed.mid` + `.mid.json` | Actual simulated contact onsets and provenance |
| `midi-feedback.json` | Per-event reward, timing errors, misses, extras and F1 |
| `performance/evaluation.json` | Contacts, forces, penetration and completion |
| `performance/trace.json` | Joint states and executed commands |

Add `--video --contact-audio` to simulation for a rendered performance with
synthetic contact-triggered audio. This needs a working MuJoCo OpenGL backend.
The headless workflow does not need ROS, a MIDI device or a GPU. The optional
`midi` extra is only needed for a physical MIDI input port.

The 8 GiB RAM reserve was a repository compute-guard setting. The environment
override above reserves 1 GiB; `0` disables that reserve. Process budgets and the
shared job lock still apply. See [compute settings](../../cartesian_console/docs/getting-started.md).

## Use real feedback

Connect a drum module/trigger interface and run:

```sh
python -m openarm_rl.music.workflow capture --list-ports
python -m openarm_rl.music.workflow capture --port 'Your exact MIDI input name' \
  --duration-seconds 30 --output runs/rl/real-take
python -m openarm_rl.music.workflow evaluate \
  --target-midi runs/rl/my-song/target.mid --performed-midi runs/rl/real-take/performed.mid \
  --score-start-seconds 2.0 --latency-seconds 0.006 \
  --output runs/rl/real-take/feedback.json
```

The offsets above are **examples to replace with measured values**. Score start
is target time zero on the recording clock; positive trigger/transport latency is
then subtracted. Align using a synchronization marker or shared clock. Aligning
the first hit automatically would hide its timing error.

An existing recorded MIDI file works with `evaluate` directly. Capture saves raw
received events as JSONL, MIDI, and clock/device metadata. It never moves a robot.
See [MIDI workflow](midi-workflow.md) for mappings, intensity calibration and
reward fields. A session's reward is available for subsequent optimization;
this command does not automatically train or update a physical controller.

## Compare controller choices

```sh
python -m openarm_rl.experiments.compare \
  --scene-directory runs/rl/midi-lab/scene --profiles runs/rl/midi-lab/profiles \
  --train-midi runs/rl/midi-lab/train.mid --midi runs/rl/midi-lab/evaluation.mid \
  --output runs/rl/controller-comparison
```

The common runner compares the stroke reference, Cartesian Jacobian IK,
native-physics MPC, MPC with learned next-state prediction errors, and a small
behavior-cloned policy. It reports contact outcomes, safety and controller
runtime on separate training/evaluation seeds. See
[controller experiments](controller-experiments.md) for budgets, reference
assistance and limitations. Behavior cloning is supervised learning; this pilot
does not establish how a fully trained RL policy would perform.

The [measured 50-episode pilot](controller-pilot-20261105.md) compares all five
controllers across drum-height, contact-response and servo-gain changes. It
includes the runtime failures and unsafe stops as well as successful hits.

## Physics and transfer limits

Native MuJoCo integrates robot motion and contacts at 1 kHz; commands run at
100 Hz. Use `simulate --height-offset-m 0.005`,
`--contact-stiffness-scale 0.7 --contact-damping-scale 1.2`, or
`--gain-scale 0.9` to probe model error. Height is an unobserved plant change:
the controller retains the registered nominal drum position.

The drum head is a compliant contact proxy. Its settings are provisional, not
a calibrated vibrating membrane or acoustic model. The stiffness/damping
controls scale the contact solver's effective parameters using the
[MuJoCo solver parameterization](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters).
They are not direct measured material constants in N/m and N·s/m.

Without measured calibration, simulated output MIDI uses a marked velocity
placeholder and intensity reward is disabled. `--velocity-calibration` enables
a speed-to-MIDI estimate; the current reference still uses a fixed nominal
stroke speed rather than changing its trajectory for each note's velocity.

Before drawing transfer conclusions, measure drum height/normal, rebound and
contact duration, stick/grip compliance, motor response, encoder delay and the
module's velocity/latency curve. MIDI alone does not identify force, rebound or
motor dynamics: pair it with timestamped joint state and suitable impact
measurements. Fit on one set of trials and evaluate on new songs and altered
conditions. No physical robot or drum has been validated by this lab example.
