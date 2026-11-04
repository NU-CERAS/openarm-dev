# MIDI performance and rhythmic learning

The music workflow parses a percussion track into absolute-time contact goals,
compiles a continuous two-arm reference, and evaluates actual MuJoCo contacts.
ACT and asymmetric PPO train a fourteen-joint position-target actor. Shared
corpus training uses encoder history, registered geometry and future score
events. Reference-assisted single-score training is also available.

## Environment and preparation

From an Ubuntu checkout with initialized submodules:

```bash
bash setup/linux.sh --console
export OPENARM_PYTHON="$PWD/.venv/bin/python"
bash projects/drum_skill_learning/scripts/setup.sh --gpu
mkdir -p runs/rl/cache
export TMPDIR="$PWD/runs/rl/cache"
python -m openarm_rl.planning.strikes \
  --targets snare floor_tom rack_tom_left rack_tom_right \
  --output runs/rl/music_preparation
python -m openarm_rl.planning.fast_references \
  --scene-directory runs/rl/music_preparation/scene \
  --profiles runs/rl/music_preparation/snare.json runs/rl/music_preparation/floor_tom.json \
    runs/rl/music_preparation/rack_tom_left.json runs/rl/music_preparation/rack_tom_right.json \
  --output runs/rl/music_preparation/calibrated_profiles
```

The complete seven-drum kit uses the hand-tuned ready poses in
`projects/opendrummer_playground/configs/demo_poses.json`. `kit_strikes` keeps joints
1-6 of each pose, finds the wrist angle of first contact, and proposes wrist strokes
at several lift heights. `rhythmic_references` then measures every stroke with an
eight-hit burst at its own period and keeps the shortest stroke that stays safe:

```bash
python -m openarm_rl.planning.kit_strikes \
  --scene-directory runs/rl/music_preparation/scene \
  --tuned-poses projects/opendrummer_playground/configs/demo_poses.json \
  --output runs/rl/music_kit_strikes
python -m openarm_rl.planning.rhythmic_references \
  --scene-directory runs/rl/music_preparation/scene \
  --profiles runs/rl/music_kit_strikes/{snare,floor_tom,rack_tom_left,rack_tom_right,hi_hat,cymbal_left,cymbal_right}.json \
  --maximum-period .4 --transfer-velocity 4 --transfer-acceleration 30 \
  --output runs/rl/music_references
```

Use `runs/rl/music_references/profiles` as `--profiles` for performance and training.
Each profile records `pose_source`, `impact_joint7_rad` and the qualified
`repeat_period_s`. Tuned strokes usually lift less than the saved ready pose because
fast repeated hits need a short stroke. Compared with automatic IK poses on the same
24 Groove files, these poses cut peak contact force from 17.9 N to 10.6 N. `kit_strikes
--assignments target:arm` still searches automatic IK poses and off-center strike
points, which can add cross-hand assignments.

A stick's tip, taper or shaft may strike the intended surface: at the tuned 6-21 degree
playing angles the shaft can touch a head or cymbal. Stick contact with any other
geometry remains an unsafe contact.

Activate the selected environment before direct `python` commands, or replace
`python` with `"$OPENARM_PYTHON"`. Physics uses native CPU MuJoCo; CUDA runs network
updates. The shared compute guard runs one compute job at a time with bounded
memory and CPU use. Generated files belong under `runs/rl/`.
This preparation uses the original complete drum-kit mesh and original drum
positions. Check `preflight.json` for physically validated targets. Unavailable
targets require their own reachability solution. Video execution checks that the
original kit mesh is visible. Training datasets and actor bundles must be built
for this scene identity.

## Shared MIDI segment training

The Groove MIDI Dataset contains 1,150 human-performed MIDI recordings and
provides official train/validation/test splits. Its MIDI archive is 3.11 MB.
The preparer verifies the expected archive SHA256, preserves source IDs and hashes,
and keeps every percussion event in each accepted window. Unsupported notes,
infeasible timing and sampling quotas are recorded in `rejections.json`.
The manifest rejects source-performance leakage across splits and held-out
drums absent from training. Sampling quotas encourage drum coverage.

```bash
python -m openarm_rl.music.corpus \
  --archive runs/rl/groove_data/groove-v1.0.0-midionly.zip --download \
  --scene-directory runs/rl/music_preparation/scene \
  --profiles runs/rl/music_preparation/calibrated_profiles \
  --output runs/rl/groove_segments \
  --seconds 0.5 --stride 0.25 --tempo-scale 0.2 --minimum-events 1 \
  --max-per-split 64 --max-per-target 8 --max-per-source 2 \
  --max-candidates-per-split 50000
python -m openarm_rl.music.audit \
  --corpus runs/rl/groove_segments/manifest.json \
  --scene-directory runs/rl/music_preparation/scene \
  --profiles runs/rl/music_preparation/calibrated_profiles \
  --output runs/rl/groove_native_audit --episodes 2
bash projects/drum_skill_learning/scripts/train_midi.sh \
  --corpus runs/rl/groove_segments/manifest.json --score-only --use-sde \
  --scene-directory runs/rl/music_preparation/scene \
  --profiles runs/rl/music_preparation/calibrated_profiles \
  --output runs/rl/shared_music_teacher \
  --device cuda --num-envs 4 --demonstrations 128 \
  --act-steps 12000 --steps 16384 --checkpoint-steps 8192 \
  --learning-rate 1e-6 --entropy-coefficient 1e-5 --ppo-epochs 2
```

The explicit `0.2` tempo scale multiplies event offsets in each accepted original
half-second window by five. Episode duration includes preparation and recovery
around the retained events. This is a slowed segment
experiment; it does not establish original-tempo song performance. Inspect
`summary.json` for accepted counts, drum coverage and rejected candidates.
Selection uses note support and reference feasibility; held-out scores do not
participate in gradient updates or checkpoint selection.

With `--score-only`, the actor receives 269 values: four encoder/action-history
frames (168), registered drum centers (21), and eight future MIDI goals (80).
The additional fourteen-value input named `nominal` contains the encoded joint
positions measured at the start of the current episode. ACT predicts changes
from that fixed encoder anchor. PPO directly initializes its actor with the ACT
network and weights, preserving the same encoder anchor while optimizing all
fourteen joint corrections. Both output absolute joint targets
to execution. The critic has 344 values and includes
privileged contact/domain information. No reference joint trajectory enters the
actor or motor target calculation. The shared compiler still supplies the ready
pose, safety feasibility checks and episode bounds.

Encoder positions, velocities and command history use separate normalization
scale floors (0.03 rad, 0.5 rad/s and 0.03 normalized action). Supervised encoder
augmentation adds position variation to non-wrist joints, velocity variation and
command-history variation. `--act-observation-jitter` controls the position
variation. The episode-start encoder anchor remains fixed during performance.
`--initialize-data-run` reuses native demonstrations after checking model,
corpus and environment identities and records the source checksum.
MIDI goals retain their time-in-seconds and normalized velocity values. The
networks derive local time-window features from upcoming event times to resolve
short downstrokes. Supervised sampling emphasizes imminent contacts and wrist
motion while retaining resting states.
PPO defaults to updating the ACT action output layer and action distribution;
`--actor-update-scope all` enables complete ACT fine-tuning. A separate critic
warmup collects 4,096 native environment steps and updates only the value network.
It checks that every actor parameter remains unchanged. `--critic-warmup-steps`
controls this stage. Its samples and update counts are recorded separately from
PPO actor updates. Training phase resets require settled, collision-free robot
states containing an upcoming target, matching the deployment start condition.
Optional `--use-sde` uses Stable-Baselines3 state-dependent exploration with one
noise matrix per rollout (`--sde-sample-freq -1`). Its default initial log standard
deviation is -9; independent Gaussian exploration defaults to -7.
`--log-std-init` controls initialization. These training distributions do not
change the deterministic exported actor. Compare their native execution before
interpreting stochastic rollout success as deployment performance:

```bash
python -m openarm_rl.music.exploration \
  --run runs/rl/shared_music_teacher --checkpoint best \
  --episodes 41 --seed 930000 --output runs/rl/music_exploration
python -m openarm_rl.music.exploration \
  --run runs/rl/shared_music_teacher --checkpoint best --sde \
  --episodes 41 --seed 930000 --output runs/rl/music_sde_exploration
```

The comparison preserves actor weights and reset seeds and samples one SDE noise
matrix per episode. Gaussian and SDE use their respective initialization scales;
this comparison changes both noise correlation and initial strength. It reports hit F1,
contact-limit failures, recovery, final joint speed and action variation.
An exploration audit does not optimize actor weights or establish PPO gains.

Each training clip receives reference-controller demonstrations. The ACT input
normalization is embedded in the actor; VecNormalize handles the privileged
critic and returns. Each actor bundle records the drums covered by training,
and execution rejects requests outside that coverage. ACT is checked
on complete training and validation clips; the initialized PPO actor must pass
the configured native success gate before optimization starts. The default
`--minimum-initial-success` is 0.25. A failed gate saves diagnostics and stops.
Checkpoint selection uses validation performances. Frozen final and selected
actors are evaluated on independently held-out performances.
The score-only reward includes a measured contact-force cost, default weight
0.05, integrated as squared force relative to the existing stop threshold.
`--force-penalty` controls this weight. The force, penetration and unexpected-
contact stops remain active. Eight consecutive failed native reference episodes
stop demonstration collection for inspection.

```bash
python -m openarm_rl.music.benchmark \
  --run runs/rl/shared_music_teacher --checkpoint final --episodes-per-clip 2 --seed 910000 \
  --output runs/rl/shared_music_benchmark
python -m openarm_rl.music.evaluate_act \
  --run runs/rl/shared_music_teacher --split train --episodes 8 \
  --output runs/rl/shared_act_training_evaluation
```

Continue an existing PPO checkpoint with its optimizer and critic normalization:

```bash
bash projects/drum_skill_learning/scripts/train_midi.sh \
  --corpus runs/rl/groove_segments/manifest.json --score-only \
  --scene-directory runs/rl/music_preparation/scene \
  --profiles runs/rl/music_preparation/calibrated_profiles \
  --initialize-data-run runs/rl/shared_music_teacher \
  --resume-checkpoint runs/rl/shared_music_teacher/final \
  --output runs/rl/shared_music_continued --device cuda --num-envs 4 \
  --steps 32768 --checkpoint-steps 8192 \
  --learning-rate 1e-6 --entropy-coefficient 1e-5 --ppo-epochs 2
```

The requested steps are additional steps. Bundle counters remain cumulative.
`PPO_initialization.json` records checkpoint identity; `PPO_hyperparameters.json`
records the effective optimization settings. Comparison reports distinguish the
original ACT, resumed initialization, selected policy and final checkpoint.

Dataset source: [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove),
Google LLC, CC BY 4.0. Attribution: Jon Gillick, Adam Roberts, Jesse Engel, Douglas
Eck and David Bamman, *Learning to Groove with Inverse Sequence Transformations*,
ICML 2019. Original MIDI content is downloaded into ignored run directories.

## Measured performance feedback

Compare the target MIDI with a synchronized MIDI recording returned by an
electronic drum kit:

```bash
python -m openarm_rl.music.feedback compare \
  --target-midi /path/to/target.mid --performed-midi /path/to/recorded.mid \
  --mapping projects/drum_skill_learning/configs/music/groove.json \
  --latency-seconds 0 --output runs/rl/physical_feedback.json
```

This reports one-to-one hit matching, timing errors, missed/extra hits and MIDI
velocity error. Any latency correction must come from measured device timing.
The two recordings require the same performance time origin.

Intensity supervision requires paired measurements in CSV columns `target`,
`downward_speed_m_s`, and `midi_velocity`. Each drum needs at least three distinct
positive measured speeds with monotonic MIDI velocity. Compile the measurements:

```bash
python -m openarm_rl.music.feedback calibrate \
  --observations /path/to/measured_strikes.csv \
  --measurement-source physical_electronic_kit \
  --output runs/rl/velocity_calibration.json
```

Pass `--velocity-calibration runs/rl/velocity_calibration.json` during training
to enable intensity error rewards. Without measured calibration, intensity
supervision is explicitly disabled. Synthesized demonstration audio represents
contact events; acoustic timbre control and microphone transcription require
their own measured data and evaluation. Physical-kit calibration and hardware
execution remain separate from native simulator verification.

## Perform a MIDI file

```bash
bash projects/drum_skill_learning/scripts/run_midi.sh \
  --scene-directory runs/rl/music_preparation/scene \
  --profiles runs/rl/music_preparation/calibrated_profiles \
  --midi /absolute/path/to/drums.mid \
  --output runs/rl/my_midi_performance
```

Use `--example duet` to generate a six-hit MIDI exercise, `--example groove`
for a thirty-second alternating pattern, or `--example fill` for a cross-drum
exercise. Supply exactly one of `--midi` and `--example`.
`--tempo-scale 0.5` explicitly doubles all event times. Without `--arrange` the
compiler rejects infeasible stroke intervals and transfers with the affected
score time, and it does not remove notes or change tempo.

## Arrange a complete drum part

Real drum parts contain kick and pedal notes, articulations and more hits than two
sticks can play. Add `--arrange` to perform any General MIDI or Groove file:

```bash
bash projects/drum_skill_learning/scripts/run_midi.sh \
  --scene-directory runs/rl/music_preparation/scene \
  --profiles runs/rl/music_references/profiles \
  --midi /absolute/path/to/song.mid --arrange \
  --output runs/rl/my_song
```

`configs/music/general_midi.json` folds each note onto a kit piece and lists
ordered substitutes. A beam search then keeps the heaviest subset that the two
hands can play under measured stroke periods and collision-checked transfers.
Snare outranks toms, cymbals and hi-hat, and louder notes outrank softer ones.
A hit that cannot be played is dropped, never moved in time. Cross-hand strokes are
a last resort because they can park one arm in the other's path. The compiled
two-arm path is checked again; a note that still fails is dropped as
`path_infeasible`, and a plan that needed such repairs is compared with a
primary-hands plan.

`arrangement/arrangement.json` lists every kept, substituted and dropped note with
its reason: `foot_event` (OpenDrummer has no kick or pedal actuator), `no_kit_piece`,
`merged_flam`, `hand_busy`, `no_calibrated_stroke` or `path_infeasible`.
`stick_coverage` is kept notes over stick notes in the source. Physical hit scores
compare against `arrangement/arranged.mid`; report them together with coverage.

Measure a repertoire against the source files themselves:

```bash
python -m openarm_rl.music.repertoire \
  --archive runs/rl/groove_data/groove-v1.0.0-midionly.zip --split test --files 24 \
  --scene-directory runs/rl/music_preparation/scene \
  --profiles runs/rl/music_references/profiles \
  --output runs/rl/music_repertoire
```

`source_f1` counts missing stick notes as misses, so it combines arrangement
coverage and physical execution. `faithful_coverage` excludes substituted notes.

Mido handles type 0/1 MIDI files, merged tracks and tempo changes. The default
mapping is `configs/music/percussion.json`; MIDI percussion channel 10 uses
zero-based channel value 9. Only positive-velocity note-on events create goals.
Other tracks remain outside the robot percussion score. Without `--arrange`,
unmapped percussion notes, including kick and unconfigured articulations, are
rejected by the strict mapping.

The current references use uniform calibrated strike speed. MIDI velocities are
preserved in the score and actor input; expressive intensity matching requires
additional calibration and training.

## Output and recording

Each performance writes the parsed score, compiled reference and per-event hand
assignments, requested/applied joint traces and actual hit results. Evaluation
reports overall and per-drum precision, recall, F1, timing error, missed/extra
hits, completion, peak contact force, penetration, contact duration and recovery.
A complete successful score requires F1 of at least 0.95, no unsafe contacts,
contact duration at most 120 ms and final joint speed below 0.08 rad/s. Shared
actors must separate and rearm both sticks; reference-assisted evaluation also
requires final reference joint error below 0.03 rad. Each physical contact can match one target
event within a 60 ms window; missed events receive a single penalty. Early unsafe
termination also accounts for remaining missed events.

Add `--video` to record silent native physics. On a headless machine with `xvfb`
and `xauth` installed, add `--xvfb /usr/bin/Xvfb`. The recording is saved as
`performance/performance.mp4`. Optional `--contact-audio` adds synthesized contact
cues and labels them in the video. Its `contacts.wav` contains generated cues.
`--camera front` shows both arms and the whole kit. With `--arrange`,
`--backing-kick` mixes the source foot notes under the audio; the video labels them
as not played by the robot and `evaluation.json` records `backing_audio`.
Evaluation records contact geometry names, contact coordinates, tip position,
registered drum center and measured forces for each confirmed hit.

MIDI supervises drum identity and onset timing through one-to-one physical hit
matching. Correct hits earn +1, extra hits cost 0.5, and missed hits cost 2.
The actor also receives future MIDI events, including note velocity. ACT learns
reference-controller joint targets. Audio waveforms do not enter ACT or PPO loss.
Physical sound, timbre and loudness accuracy require electronic-kit MIDI or
recorded acoustic measurements. Synthesized cues have no acoustic validation role.

## Additional hand assignments

Profiles normally use each drum's configured side. To validate an additional
hand-to-drum combination, run:

```bash
python -m openarm_rl.music.prepare \
  --scene-directory runs/rl/music_preparation/scene \
  --profiles runs/rl/music_preparation/calibrated_profiles \
  --targets snare --output runs/rl/music_hand_profiles
```

This runs IK/arc checks and native calibration for the other hand. Unreachable
assignments are recorded in `reachability.json`. Use the resulting `profiles/`
directory for performance or training. The score compiler searches hand
assignments using available calibrated profiles and stroke/transfer timing,
then checks the combined two-arm path. Simultaneous requests that exceed the
available hands are rejected.

## GPU rollouts with MuJoCo Warp

`openarm_rl.envs.warp_rhythmic.WarpRhythmicVecEnv` runs score-only PPO rollouts on
batched MuJoCo Warp worlds. Stepping, contact forces, hit detection and one-to-one
score matching run in Warp kernels; rewards and observations stay on the GPU.
`RhythmicDrummingEnv` remains the semantic reference, and validation, checkpoint
selection and held-out evaluation still run on native MuJoCo.

Build a corpus of complete arranged performances, then select batched rollouts:

```bash
python -m openarm_rl.music.arranged_corpus \
  --archive runs/rl/groove_data/groove-v1.0.0-midionly.zip \
  --scene-directory runs/rl/music_preparation/scene \
  --profiles runs/rl/music_references/profiles \
  --train 32 --validation 8 --test 8 --max-seconds 30 \
  --output runs/rl/arranged_corpus
bash projects/drum_skill_learning/scripts/train_midi.sh \
  --corpus runs/rl/arranged_corpus/manifest.json --score-only \
  --scene-directory runs/rl/music_preparation/scene \
  --profiles runs/rl/music_references/profiles \
  --output runs/rl/warp_music_teacher --device cuda \
  --warp-envs 1024 --rollout-steps 32 --batch-size 8192 \
  --demonstrations 64 --act-steps 12000 --steps 50000000 --checkpoint-steps 5000000
```

Install the `gpu-sim` extra first. MuJoCo Warp computes world-attached geom poses once,
so kit randomization writes `geom_xpos` directly at reset. Intensity calibration is not
implemented on this path. Mass randomization scales the inverse-weight constants by
the mass factor; armature makes that approximate, and `tests/envs/test_warp_rhythmic.py`
checks the batched episode against native contact physics.

## ACT and PPO

```bash
"$OPENARM_PYTHON" -m wandb login
bash projects/drum_skill_learning/scripts/train_midi.sh \
  --scene-directory runs/rl/music_preparation/scene \
  --profiles runs/rl/music_preparation/calibrated_profiles \
  --output runs/rl/music_teacher \
  --device cuda --num-envs 4 --steps 8192 \
  --demonstrations 16 --act-steps 1000 --bc-steps 500
```

Supply `--midi` and `--heldout-midi` for training and held-out songs. Omitting
them generates separately specified training and held-out exercises. The command
collects successful native reference-controller episodes, trains a ten-step ACT,
initializes the reference-assisted MLP actor from ACT outputs, trains randomly initialized four-second
segments, and evaluates continuous complete scores. Networks have two 128-unit
actor/critic hidden layers; ACT uses a transformer. PPO uses 256 steps per worker,
batch size 256, four epochs, learning rate `3e-5`, entropy coefficient `1e-4`,
initial log standard deviation `-5`, target KL `0.01`, and observation/reward
normalization. W&B project: `openarm-midi`.

The actor receives four encoder/action-history frames (168 values), reference
joint position/velocity (28), registered nominal drum centers (21), and eight
future contact goals (80). A separate fourteen-value nominal-target input
provides a reference prior for the joint-position output. This is a reference-
assisted adaptation of the paper's score-conditioned PPO. The critic additionally
receives actual joint state, contact, tip position and randomized-domain
information, giving 372 critic inputs. ACT receives the 297 actor values and
fourteen nominal values and predicts ten steps of normalized joint-target
corrections relative to the reference. Actual
randomized drum positions are excluded from actor observations.

Use `--initialize-act /path/to/act.pt` to reuse a compatible collected-data ACT
checkpoint. Its checksum is recorded with the run. `final/` contains the actor
after the requested PPO updates; `best/` can select the initialized actor when
subsequent updates do not improve validation. Check `selection.steps` and
`selection.optimization_updates` in `comparison.json` when reporting results.

Evaluate the final updated actor against its reference on identical reset seeds:

```bash
python -m openarm_sim.compute_guard python -m openarm_rl.music.benchmark \
  --run runs/rl/music_teacher --checkpoint final --episodes 8 \
  --output runs/rl/music_teacher_benchmark
```

Default randomization is 2 mm XY kit translation and 5% mass/friction/gain
variation, with small initial encoder-position perturbations, 50/100 Hz encoder
sampling and 0–20 ms delivery delay. Training and runtime share bounded linear
encoder prediction; control samples older than 30 ms are rejected. Existing v1.0
pickup and single-strike workflows retain their own configuration.

Compare `reference/`, `ACT_validation/`, `initial/`, `validation_*/`, and
`comparison.json`. `best/actor.pt` exports only actor inference and its matching
normalization. `best/` and `final/` retain PPO and VecNormalize checkpoints.
Learning completion and contact F1 alone do not establish safe or improved
performance. Report reference, ACT, initialized PPO and selected PPO separately,
including unsafe-contact rates and unseen-pattern results.

## ROS execution

The `openarm_rl.deployment.rhythm_actor` runtime constructs its inputs from
encoder feedback, the supplied score, registered geometry and reference. The
`openarm_rl.deployment.ros_music` entry point previews by default and requires
`--execute` to send commands. It uses the shared ROS trajectory actions for
preparation and paired position/velocity streams for timed performance.
Physical execution requires the reviewed mapping and controller settings
described in [camera-free drumming](camera-free-drumming.md). Physical hit
verification requires electronic-kit MIDI feedback or operator observation.
Native simulator contact metrics do not establish physical hit accuracy.

Export a checkpoint and its deployment identities before ROS use:

```bash
python -m openarm_rl.music.export --run runs/rl/music_teacher \
  --output runs/rl/music_teacher/deployment
python -m openarm_rl.music.ros_demo \
  --actor runs/rl/music_teacher/deployment/actor.pt \
  --scene-directory runs/rl/music_teacher/scene \
  --profiles runs/rl/music_teacher/profiles \
  --midi runs/rl/music_teacher/heldout.mid \
  --output runs/rl/music_ros_demo --repeats 2 --execute
```

The demo owns a simulated ROS backend and shuts down its services afterward.
Subsequent performances prepare from measured feedback. The deployment runtime
checks actor checksums and nominal model/layout identities. ROS timing/encoder
checks are distinct from simulator contact evaluation.
For Groove-derived MIDI, pass `--percussion-mapping
projects/drum_skill_learning/configs/music/groove.json` to the ROS command.
For native playback of a shared actor, pass `--score-only` and the same mapping
through `--mapping`.

Method reference: [Robot Drummer: Learning Rhythmic Skills for Humanoid Drumming](https://arxiv.org/abs/2507.11498).
Sim-to-real reference for later hardware work: [Learning to Play Piano in the Real World](https://arxiv.org/abs/2503.15481).
