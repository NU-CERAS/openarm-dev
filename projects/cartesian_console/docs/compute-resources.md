# Local compute resource policy

Training, dataset collection and evaluation commands use one shared compute
slot. The pipeline executes these jobs serially. Its orchestration processes
perform no GPU computation; each GPU job runs in a separate process and releases
its CUDA context when that job completes.

## Limits

The Linux launcher creates a transient user systemd scope for each job:

| Resource | Setting |
| --- | --- |
| Total job memory, including worker processes | At most 14 GiB, reduced if necessary to reserve 8 GiB of currently available system memory |
| Memory pressure threshold | At most 12 GiB |
| Job swap | Disabled |
| CPU quota | Half the available logical CPUs, capped at eight CPU equivalents |
| Scheduling priority | Nice 10 |
| BLAS / Torch CPU threads | One per worker |
| Torch CUDA allocator | At most 50% of total device memory |
| Runtime reserve checks | Stop the job below 8 GiB system memory or 4 GiB free GPU memory |

The Torch allocator limit does not include Warp or graphics allocations. The
runtime GPU reserve check measures total free device memory. CUDA jobs use
private streams scoped to simulation operations, preserving the caller's stream.

The launcher checks system memory once per second. Training and collection
also check available resources between work units. The scope applies the memory
and CPU limits to child workers. A resource failure terminates the job and is
reported in its log. The pipeline stops on execution errors.

## Commands and inspection

The training commands enter the launcher automatically. For an explicit command:

```bash
openarm-train-act --student --data PATH --output PATH
```

Use the same launcher for GPU tests. Run tests while the training slot is free;
the shared file lock queues a second compute command until the current one ends.
`runs/cartesian_console/active_compute.json` records the active unit, command,
process ID and configured limits. Resource information for that unit is visible
through `systemctl --user show UNIT.scope`.

## Collection and recovery

Oracle teacher collection runs 32 physics worlds in one process. It records
actual observations, commands, joint/free-object positions and velocities, then
renders the recorded states into RGB images. Episodes become available to the
student dataset after their complete image and state data have been written.
Pending episode data is retained separately if collection is interrupted.

`--resume` verifies dataset seed, teacher checkpoint, residual checkpoint,
supervision mode and image resolution. It continues after saved attempts.
Student-query collection uses one world because the student's next action
depends on the current rendered image. Visual training uses minibatches of 32.

ACT saves checkpoints every 200 optimization updates. Strike saves a recovery
checkpoint each rollout, and separately saves validation-selected checkpoints.
The pipeline reuses completed pickup assessment and resumes stored datasets.
Completion of optimization and qualification of a skill are recorded separately.

The resource checks and scope limit admission were verified locally. The
post-recovery interoperability check passed five actual ACT/cuRobo/Warp tests.
A batched collection check saved four successful physical episodes and their
rendered images. A strike checkpoint reload check executed two successful
physical episodes without optimization. These checks ran one at a time.
