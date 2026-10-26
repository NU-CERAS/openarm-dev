# ACT Teacher Implementation Plan

> Implementation follows the user-approved reference-controller, ACT, residual PPO, and student-distillation design. Execute within the existing checkout and preserve user changes.

**Goal:** Train and evaluate an oracle ACT pickup teacher, refine its commands with residual PPO, and distill combined-teacher actions into a camera/proprioception ACT student.

**Architecture:** Native MuJoCo collects physically successful reference rollouts into HDF5, with disjoint train/evaluation seeds. A state-conditioned ACT CVAE predicts command chunks. A frozen ACT supplies nominal commands for bounded residual PPO on MuJoCo Warp; student data records combined commands with camera observations. Saved checkpoints include action conventions, normalization, model settings, and provenance.

**Technologies:** PyTorch, MuJoCo, MuJoCo Warp, Stable-Baselines3, HDF5, W&B.

## Constraints

- Preserve actual grasping contacts and success checks; use no attachment constraint or fabricated data.
- Use existing seven-joint radians and gripper-meter command conventions.
- Keep intermediate artifacts under repository `runs/cartesian_console/`; use no system configuration changes.
- Use English identifiers, documentation, GUI text, and Chinese comments.
- Keep prior unsuccessful PPO training stopped. New experiments have separate output paths and W&B runs.

## Review focus

- Reject mixed train/evaluation seeds, incomplete episodes, invalid command units, and incompatible normalization.
- Mask padded action chunks and prevent chunks crossing episode boundaries.
- Verify commands rather than measured positions are imitation targets.
- Verify zero residual reproduces ACT command execution and preserve frozen teacher weights.
- Exclude privileged object/reward state from student inputs; labels come from final combined commands.

## Tasks

- [ ] `act_data.py`: Collect reference episodes with measured state, position targets, physics state, success metadata, and optional camera frames; validate HDF5 and split manifests.
- [ ] `act_model.py`: Implement state/image-conditioned ACT with posterior action encoder, Gaussian latent, transformer chunk decoder, masked reconstruction/KL losses, and saved normalization.
- [ ] `act_train.py`: Train on oracle data, evaluate on independent physical resets, record videos, and track W&B metrics.
- [ ] `act_residual.py`: Add frozen-ACT nominal commands and bounded PPO residual execution with deterministic evaluation and complete checkpoint state.
- [ ] `act_distill.py`: Collect combined-teacher rollouts and student-state queries; train and evaluate image/proprioception ACT without oracle inputs.
- [ ] Targeted validation: Real reference data round trip, chunk padding/masks, physical ACT rollout, zero-residual equivalence, teacher freeze, student input boundary, saved-model reload.
- [ ] Run an actual teacher learning experiment and publish measured results, videos, and limitations.
