# Compare controllers

This experiment compares physically detected drum hits under the same target MIDI, robot limits, contact model and safety stops. It runs offline in native MuJoCo and never sends robot commands. Start with the validated, sparse snare task; a score requesting other drums needs validated profiles for those drums.

See the [measured 50-episode pilot](controller-pilot-20261105.md) for the first
results, controller failures and practical implications.

From the repository root, using the Python environment installed for this project:

```bash
# The default guard reserves 8 GiB. This explicit override reserves 1 GiB.
export OPENARM_RAM_RESERVE_GIB=1
python -m openarm_rl.experiments.prepare --output runs/rl/midi-lab
python -m openarm_rl.experiments.compare \
  --scene runs/rl/midi-lab/scene \
  --profiles runs/rl/midi-lab/profiles \
  --train-midi runs/rl/midi-lab/train.mid \
  --midi runs/rl/midi-lab/evaluation.mid \
  --output runs/rl/controller-comparison \
  --train-seeds 10,11,12 --seeds 0,1,2 \
  --scenarios nominal,lowered,gain
```

Both commands acquire the existing compute guard. Output directories must be new. `--scene-directory` is an alias for `--scene`. `--help` lists all options. A quick plumbing check can add `--max-steps 8 --max-training-steps 8 --seeds 0 --scenarios nominal`; it ends before the hits and **is not a performance experiment**. Capped episodes are marked incomplete.

| Variant | What it actually computes | Reference assistance |
| --- | --- | --- |
| `reference` | Existing stroke trajectory with gravity compensation and bounded joint velocity/acceleration | Complete compiled trajectory |
| `ik` | Damped Jacobian inverse kinematics for stick-tip position/orientation, then bounded joint commands | Desired Cartesian curve and posture come from the reference |
| `mpc` | Receding-horizon finite-candidate search; each candidate runs the nominal MuJoCo dynamics with PD torques, effort limits and contacts | Reference supplies candidate commands and tracking cost |
| `residual_mpc` | Same search, adding a learned prediction of the nominal model's next joint-state error | Same reference guidance as MPC |
| `bc` | Supervised behavior clone with 128 fixed nonlinear features and a fitted ridge output layer | Reference teacher during training; ready pose only at inference |

The BC policy sees current joint positions/velocities, previous commanded position/velocity, and upcoming MIDI events. It has no timed reference trajectory at inference. This is a deliberately small learning baseline, **not pure reinforcement learning**. Its failure would not establish that all learning methods fail.

MPC defaults to five candidates and four 10 ms prediction periods. It simulates actual MuJoCo dynamics; it is a small discrete search, not a global optimizer. Predicted unexpected contacts, excessive penetration, summed stick/drum contact forces, or tracking errors reject a candidate using the plant's stop thresholds. `--horizon` and `--candidates 3|5` expose the compute tradeoff. Every episode records controller runtime and the fraction of decisions exceeding the 10 ms deadline. Offline playback can complete while missing every real-time deadline.

## Training and held-out evaluation

Teacher rollouts use training seeds 10 and 11 with mild gain/contact randomization. Seed 12 is reserved for model validation and is never fitted. Evaluation seeds must be disjoint from all three. The prepared evaluation MIDI uses different hit times from the training MIDI. Held-out score identity compares parsed hit times, drums and velocities; metadata or note-off changes do not count as a new task. Omitting `--train-midi` explicitly marks score overlap in the manifest; changing only the seed is not a new musical task.

The dynamics residual is a 56-input, 28-output linear ridge model. Inputs are measured `q`, `dq`, commanded motor position minus `q`, and commanded motor velocity. Its target is the next measured `q,dq` minus a one-period nominal MuJoCo prediction. Predicted corrections are bounded to 0.01 rad and 0.5 rad/s. The nominal predictor loads a separate, unmodified scene and receives no randomized masses, gains, contacts or drum offsets. Validation reports prediction error before and after correction; a learned correction is not assumed to improve it.

All controllers currently receive the same idealized synchronous joint feedback. This comparison does **not** test feedback delay, encoder noise, missed communication deadlines, or real drum sensors. Nominal trials are deterministic repeated measurements; varied seeds in perturbed scenarios change the plant. Available scenarios are:

- `nominal`: unchanged scene.
- `lowered` / `raised`: drum kit moved vertically by −5 / +5 mm, plus 3% domain randomization.
- `gain`: 0.85 actuator gain scale, plus 3% domain randomization.
- `soft`: 0.7 contact stiffness and 0.9 damping scales, plus 3% domain randomization.

These are perturbations of an assumed contact model, not fitted material properties of a real drum membrane.

## Read the results

`results.json` contains every episode and per-controller/scenario averages. Inspect completion, unsafe stops and missed/extra hits alongside F1 and timing; a run that stopped early must not be counted as a successful performance. Intensity error stays `null` unless `--velocity-calibration` supplies measured speed-to-MIDI calibration. MIDI velocities alone do not identify real contact force or sound amplitude.

Each episode directory contains `result.json`, `trace.json` (joint state and exact applied commands), and `controller-trace.json` (decision time, MPC candidate/cost or IK error). The top-level files include:

- `manifest.json`: scene/model/profile/MIDI hashes, seeds, perturbations, software versions and limits.
- `training.json`: fitting/validation counts, teacher failures, dynamics error and BC validation error.
- `training-transitions.npz`: observable features, teacher labels, nominal and measured next states, and seed split.
- `dynamics-residual.npz`, `bc-policy.npz`: fitted model weights and normalization; load with `RidgeModel.load()`.

Use these results to decide which controller deserves a larger experiment. A useful next stage is more drums/tempos, measured latency and drum-contact identification, followed by held-out physical MIDI recordings. The [MIDI workflow](midi-workflow.md) documents capture/evaluation. No result here establishes sim-to-real transfer.
