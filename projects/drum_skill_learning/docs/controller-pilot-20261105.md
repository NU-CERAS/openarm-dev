# Controller pilot — 2026-11-05

**Reference and IK are the useful baselines in this pilot. Neither MPC variant nor the small learned policy improved contact outcomes.** These are controlled simulation results, not a comparison of fully trained RL methods or a hardware qualification.

## Task and results

50 episodes: five controllers × five conditions × two held-out seeds (0, 1). Each episode requests two snare hits at 0.8 and 5.1 seconds. Conditions are nominal, kit height −5 mm, kit height +5 mm, contact stiffness/damping ×0.7/0.9, and servo gains ×0.85. Non-nominal conditions additionally randomize the domain by 3%. Nominal seed repeats are deterministic.

| Controller | Mean F1 | Completed | Unsafe stops | Predicted-safety rejections | Decision p95, ms¹ | >10 ms decisions¹ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stroke reference | 1.00 | 10/10 | 0 | 0 | 0.21 | 0.0% |
| Jacobian IK | 1.00 | 10/10 | 0 | 0 | 1.10 | 0.0% |
| Physics MPC | 0.80 | 8/10 | 0 | 2 | 22.56 | 100.0% |
| MPC + learned dynamics | 0.80 | 8/10 | 0 | 2 | 23.94 | 100.0% |
| Behavior clone | 0.00 | 2/10 | 8 | 0 | 0.08 | 0.0% |

¹ Means of per-episode measurements on the local Mac under the compute guard. Decision timing excludes the evaluated plant step and real I/O. The control deadline is 10 ms. Completion alone does not imply correct hits or real-time operation.

| Controller | Nominal F1 | Lowered F1 | Raised F1 | Softer F1 | Gain-error F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stroke reference | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Jacobian IK | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Physics MPC | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| MPC + learned dynamics | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| Behavior clone | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## What this tells us

- The unchanged reference and online IK hit both targets in every tested condition. The task is sparse and uses a validated stroke; this is a low bar for general musical drumming.
- Both MPC variants reject every candidate in the lowered-drum trials. Their nominal model predicts unsafe contact while the evaluated plant permits the reference/IK stroke. A joint-state residual alone did not repair this geometry mismatch.
- MPC uses five candidate proposals and a 40 ms horizon. Its decision time exceeds the 100 Hz budget. This implementation needs faster planning and better model/state estimation before considering deployment.
- The behavior clone misses the requested hits; eight of ten trials stop on excessive summed contact force. It learned from only two reference rollouts. This result measures this small clone, not the potential of RL or imitation learning with adequate data.
- All controller variants still use the validated ready pose. IK/MPC additionally use the reference trajectory; BC sees MIDI, joint feedback and its own prior command state at inference. The experiment compares control/adaptation, not arbitrary movement synthesis.

## Learning evidence

Fit: 2250 transitions from seeds 10 and 11. Validation: 1125 transitions from seed 12, excluded from fitting. The training score has three hits at 0.5, 4.5 and 8.5 seconds. Evaluation score identity is checked from parsed musical events as well as file provenance.

The dynamics residual changes validation joint-position RMSE from 0.00016310 to 0.00016223 rad, and velocity RMSE from 0.022283 to 0.022141 rad/s. This small prediction improvement did not improve hit F1.

## Next priorities

1. Measure drum position/normal, rebound, contact duration, motor response and timing latency. Use MIDI for musical outcomes and synchronized encoders/impact measurements for dynamics identification.
2. Keep the reference/IK baseline. Evaluate learned dynamics corrections against measured data and harder held-out scores before adding more control complexity.
3. Add expressive per-note intensity control using measured velocity curves and genuinely speed-conditioned stroke banks. The reference currently selects a fixed nominal speed.
4. Extend evaluation to multiple drums, dense rhythms, varying tempos, noisy/delayed encoders and real recordings. Fit uncertain contact models; the current compliant contact proxy has no calibrated membrane modes or acoustics.

## Reproduce and inspect

Follow the [simulator lab](simulator-lab.md) and [controller experiment guide](controller-experiments.md), then use `--seeds 0,1 --scenarios nominal,lowered,raised,soft,gain --horizon 4 --candidates 5`. Outputs must use a new directory.

The [compact measured report](controller-pilot-20261105.json) retains input/code hashes, seed split, metrics, failures and software versions. Full traces and fitted models are in `runs/rl/controller-comparison-final-20261105/` (Git-ignored). An earlier exploratory run in `controller-comparison-20261105/` predates review fixes and is superseded by this report.

No physical robot or drum module was tested. CoreMIDI input discovery succeeded with no connected ports. MIDI import → simulated contacts → performed MIDI → evaluation independently achieved F1 1.0 and about 3 ms timing MAE; uncalibrated intensity was excluded.

## Implementation validation

The affected simulator, controller, MIDI, compute-guard and deployment regression
suite finished with **153 passed, 1 skipped** in 69.76 seconds. The skip is an
existing transfer test requiring a separate, untracked prepared-kit run. Results
are saved in `runs/rl/simulator-validation-20261105.xml`. Existing TorchScript
deprecation/tracing warnings remain.

Rehearsal now executes exact motor commands and requires successful nominal and
deployment conditions. Normalized simulation actions and learned runtime share
motor-position bounds. Saved-pose fingerprints tolerate insignificant numerical
differences; all 21 shipped presets were revalidated with two successful native
strike cycles each before changing signatures, preserving every joint angle.
The evidence is in `projects/opendrummer_playground/configs/pose_signature_migration.json`.
