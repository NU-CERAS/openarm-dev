# Refactor integration validation

Validated on 2026-10-25 in `feat/jixin-refactor`, based on upstream `refactor`
commit `74c1df4` and the simulation feature branch at `acee8e0`.

The root distribution remains `openarm-tools`. Task code is installed as the
independent `openarm-sim` distribution from `projects/cartesian_console`.
Both packages were installed editable into ignored
`runs/cartesian_console/validation-env`, using the existing conda environment's
real dependencies. Import locations were checked against this checkout.

## Automated checks

| Checks | Results |
| --- | --- |
| Reach, camera tools, execution and pickup | 19 passed |
| ACT, GPU pickup, drumming and residual strike | 27 passed |
| Exported kinematics, cuRobo collision planning and imported kit | 9 passed |
| Shared model/configuration/trajectory and CLI simulation | 6 passed |
| ACT physical-episode aggregation and supervision | 1 passed |

Planning assets were generated from the canonical `robots/v1.0` model. The
downloaded electronic kit was converted into `runs/cartesian_console/drumset`.
The shared package, canonical robot files, ROS package and shared tests retain
their upstream contents. Selected shared tests executed actual MuJoCo and CLI
operations. Project tests use actual physics, rendering, GPU actions and models.

## Native GUI execution

The installed console command generated GUI events and checked measured state:

| Demo | Measured result |
| --- | --- |
| `--planner mink --smoke-test` | Reach succeeded, TCP error 0.689 mm |
| `--scene pickup --smoke-test` | Both fingers contacted the free stick; lift 136 mm |
| `--scene strike --smoke-test` | One measured impact, retained grasp, task success in this episode |
| `--scene kit --planner curobo --smoke-test` | Infeasible selection rejected; another reach succeeded, error 0.890 mm |

## Existing checkpoint execution

The selected ACT checkpoint from the original checkout was loaded without
format conversion and executed in eight distinct held-out starting states.
All eight succeeded. This is a migration check; the earlier 64-state experiment
remains the source of the reported 58/64 success result.

Files under repository `runs/cartesian_console/migration_teacher_check/` include
`evaluation.json`, `execution.json` and a rendered `execution.mp4` from the actual
GPU state. The checkpoint and original datasets remain in the original checkout.
No new training was started during integration. Physical robot execution and
the ROS runtime remain unvalidated.
