# OpenDrummer playground

The playground imports the shared OpenDrummer URDF into MuJoCo and combines it
with the drum kit. Step one provides the checked robot model and holding
controller; step two adds configured kit placement, eight surface markers, and
static approach-pose previews. Step three adds primitive collision geometry
and a slow stick–snare contact check. Rhythmic strikes and motion planning are
later steps.

## Run on macOS

From the repository root, activate the existing project environment:

```sh
~/.pixi/bin/pixi shell --manifest-path dependencies/mac/pixi.toml
```

Run the automated checks:

```sh
python projects/opendrummer_playground/scripts/validate.py
```

Expect `"status": "PASS"`. The checks cover all 14 hinge joints and their
limits, fixed-assembly mass aggregation, positive physical inertia tensors,
39 poses compared against independent URDF forward kinematics (including zero
and all 28 joint-limit endpoints), both stick tips, five seconds of gravity
holding after a 0.03-radian perturbation, and a small excursion of each joint.
Motion checks enforce effort and velocity limits and reject solver warnings.
These checks establish import consistency and numerical stability, not physical
calibration or collision clearance.

Open the combined scene:

```sh
mjpython projects/opendrummer_playground/scripts/run.py
```

Both arms should hold still beside the drum kit, with attached shells and sticks.
Cyan disks identify fitted playing surfaces; yellow markers show centers and
normals. RGB axes mark the world origin (+X forward, +Y left, +Z up).
The left stick tip
is marked green, and the right tip orange. Drag to orbit and scroll to zoom.
Close the window to stop. The base is fixed; there are no free-base degrees of
freedom. The initial holding pose bends both fourth joints by 0.35 radians;
the import's zero pose is checked separately by validation.

To inspect each joint moving individually:

```sh
mjpython projects/opendrummer_playground/scripts/run.py --robot-only --exercise
```

This cycles through `lj1`–`lj7`, then `rj1`–`rj7`, four seconds per joint.
The terminal names the current joint. Each moves through a small positive
0.15-radian excursion and returns. Watch that the corresponding arm and its
attached geometry move together. This is a visual joint check, not drumming.

A GUI-free smoke run is also available:

```sh
python projects/opendrummer_playground/scripts/run.py --headless --seconds 5
```

## Check scene alignment (step two)

```sh
python projects/opendrummer_playground/scripts/validate_scene.py
```

This step-two validator intentionally builds a visual-only model to isolate
alignment checks. The normal viewer and the step-three test enable contacts.
Expect `"status": "PASS"`. This checks the eight fitted surfaces, floor height,
unchanged 14-DOF robot, exact robot elevation, support floor/top alignment,
joint-limited initial approach poses,
two seconds of holding those poses, and stick-centerline intersections against
the kit triangles in both the ready and approach configurations. It does not
certify full-body clearance, finite stick thickness, or a path between poses.

Preview each initial target:

```sh
mjpython projects/opendrummer_playground/scripts/run.py --arm left --preview-target snare
mjpython projects/opendrummer_playground/scripts/run.py --arm right --preview-target rack_tom_right
```

These commands **start in a static approach pose** with the selected tip 8 cm
above the fitted surface. They do not move through a planned trajectory or strike
a drum. The other arm stays in its holding pose. The terminal reports positional
and directional IK error. Other target/arm combinations may be unreachable.

Use `--labels` to display site names. Surface/axis markers are in MuJoCo site
group 3, which the viewer enables; turn this group off in the viewer to hide them.
Use `--robot-only` for the original step-one view. `--exercise` requires
`--robot-only`, since motion through the kit is not planned yet.

To save a preview without opening the viewer:

```sh
python projects/opendrummer_playground/scripts/run.py --snapshot runs/opendrummer_playground/scene-preview.png
```

### Placement and scale

Edit `configs/scene.json` to change kit XY position, yaw, uniform scale, approach
height, and initial target assignments. `robot_elevation_m` raises the whole
robot above the world floor; it is currently **0.3048 m (1 foot)**. A fixed visual
support block, **0.4 × 0.4 × 0.3048 m**, sits beneath it. Change its footprint
with `support_size_xy_m`. The robot remains fixed at XY `(0, 0)`; the
`--robot-only` view retains the original floor-level model. The scene is rebuilt on every launch; markers follow the kit
transform automatically. Re-run scene validation after changing placement.

The OBJ has no physical unit metadata. The initial **0.6 scale is an explicit
visual normalization**, giving a bass-head diameter of about 0.57 m, rather than
an assertion of measured kit dimensions. Kit center XY is initially `(0.55, 0)` m.
The source's +Y up-axis is mapped to world +Z, +Z to world +X, and +X to world +Y.
The lowest non-platform kit vertex is placed on the world floor.

The export includes a display slab; only its 12 triangles are omitted in the
generated scene. Material groups are split into generated visual OBJ files to
preserve diffuse colors, with opaque rendering; source OBJ/MTL files are not
edited. An extra tilted cymbal near the floor remains part of the source visual
but is not annotated as an active playing surface.

### Surface definitions

| Target | Definition |
|---|---|
| `snare` | Left-side near drumhead, selected for initial left-arm reach |
| `rack_tom_right`, `rack_tom_left` | Two inclined upper drumheads |
| `floor_tom` | Right-side lower tom |
| `cymbal_right`, `cymbal_left` | Upper cymbals, approximate fitted planes |
| `hi_hat` | Upper of the paired hi-hat surfaces |
| `kick` | Rear/batter bass-drum head, annotation only; excluded from arm IK |

Names are scene labels based on geometry, not manufacturer metadata. The builder
locates material-connected components near configured source-coordinate seeds,
fits their planes, and records centers, normals, radii, and planarity errors.
Cymbal markers are approximate planes through curved meshes; their bells and
strike zones are approximated by planar contact disks for now; no bell geometry
or flexible membrane is modeled.

`runs/opendrummer_playground/scene-report.json` records the kit transform, bounds,
surface fits, and source/config hashes. `scene-validation.json` records acceptance
checks and the two initial approach configurations. Per-target preview commands
also write `approach-<arm>-<target>.json`.

## Collisions and slow contact check (step three)

The default combined scene now enables collision physics. Rendering meshes still
have collisions disabled; separate primitive proxies handle contact without
turning the entire drum kit into a convex solid.

```sh
# One slow left-stick/snare touch and withdrawal (6.5 seconds), then freeze:
mjpython projects/opendrummer_playground/scripts/run.py --contact-test

# Inspect collision shapes with visual meshes hidden:
mjpython projects/opendrummer_playground/scripts/run.py --contact-test --collision-only

# Or overlay the proxies on the render meshes:
mjpython projects/opendrummer_playground/scripts/run.py --show-collisions
```

The test starts in an IK pose 4 cm above the snare. Torque-controlled joints
follow a continuous descent, briefly load the head, then retract. The desired
endpoint is 3 mm below the head plane, but contact should prevent the actual tip
from following it through the surface. There are no runtime qpos teleports.
The viewer freezes when the check completes or stops; close it to exit. The
terminal prints measured force, penetration, contact duration in steps, and
success/failure. A headless run uses the same dynamics:

```sh
python projects/opendrummer_playground/scripts/run.py --contact-test --headless
python projects/opendrummer_playground/scripts/contact_check.py
```

The second command is the acceptance test. It checks all eight head proxies with
an independent small sphere, verifies the ready pose is contact-free, then runs
the slow contact at 1 ms and 0.5 ms timesteps. It requires actual rounded-tip/head
contact force, bounded penetration, no unexpected contacts or solver warnings,
actuator limits, and successful withdrawal. A negative control disables contact:
the same trajectory must pass through the head plane with zero contact force.
The timestep comparison allows at most a 25% change in peak force.

### Collision approximations

- Sticks: shaft and tapered-section capsules plus a 4 mm rounded tip sphere;
  the front of the tip sphere aligns with the existing stick-tip site.
- Arms: bulk capsules fitted to each moving link's visual bounds; custom tool
  mounts use bounding boxes. These simplify the real outlines and recesses.
- Robot base: pedestal, shoulder housing, and base-plate proxies, plus the
  existing 1-foot support block and floor.
- Drumheads: 8 mm thick cylinders, with their upper surfaces at the fitted head
  planes. Contact compliance is numerical, not a calibrated membrane model.
- Rims: capsule rings. Shells: hollow segmented walls fitted to the source shell
  components, rather than solid cylinders filling the drum interiors.
- Stands: 68 elongated metal components approximated as capsules in this export.
- Cymbals/hi-hat: planar disks; curvature, bell contacts, wobble, and sound are
  not modeled. Decorative/non-playing cymbal geometry is not fully covered.

Collision proxies use geom group 4. `--collision-only` hides render groups 1/2
and the surface-marker group 3; `--show-collisions` overlays the shapes. Exactly
coplanar heads can show rendering interference in the overlay, so the isolated
view is clearer. Contact points and force arrows are enabled in debug views.

Environment–environment collisions are filtered out. Robot–environment,
arm–arm, stick–stick, and nonadjacent same-arm contacts are eligible. Neighboring
joint bodies are excluded, as are J5/tool contacts within each compact wrist and
base-housing/J1–J2 contacts: simplified bulk shapes fill their real assembly
recesses. These exclusions are recorded in `collision-report.json`; this is not
an exact self-collision model of every mechanical detail.

Settings live in `configs/contact.json`. Defaults use head `solref=[0.012, 1]`,
rigid-proxy `solref=[0.006, 1]`, and friction `[0.5, 0.005, 0.0001]`. The slow
check stops for force above 20 N, penetration above 3 mm, unintended loaded
contacts, or joint speed/effort violations. These are simulation test thresholds,
not verified hardware limits or calibrated drum-impact parameters. See the
[MuJoCo contact documentation](https://mujoco.readthedocs.io/en/3.6.0/modeling.html#contact-parameters).

Results are written to `runs/opendrummer_playground/contact-test.json`,
`contact-trace.json`, and `contact-validation.json`; proxy definitions and
exclusions are summarized in `collision-report.json`. This demonstrates one
slow contact and retraction, not rhythmic drumming or arbitrary safe paths.

## Source and generated files

- `robots/opendrummer/urdf/opendrummer.urdf` and `meshes/` remain canonical.
- `scripts/model.py` converts the model reproducibly and adds tip markers and
  torque motors. All commands rebuild automatically from source.
- `scripts/scene.py` composes the kit, extracts surface targets, and solves static IK.
- `scripts/collisions.py` builds contact proxies; `contact_check.py` runs and validates the slow touch.
- `configs/contact.json` contains proxy sizes and contact-test settings.
- `configs/scene.json` contains kit placement and surface-selection settings.
- `configs/import.json` contains explicit simulation assumptions, holding gains,
  joint armature, timestep, and the holding pose. Arrays use joints 1–7 per arm.
- `runs/opendrummer_playground/` holds generated URDF/MJCF, `build-report.json`,
  and `validation-report.json`. This directory is ignored by Git. Generated
  files contain checkout-specific mesh paths; rebuild after moving the repo.

The importer copies no meshes and does not edit the URDF; no new environment is needed.
The existing v0.3/v1.0 CLI and Cartesian Console are unaffected.

## Official inertial properties and custom tools

The canonical URDF now contains official OpenArm v1.0 properties pinned to
`enactic/openarm_description` commit `10ca9bf2f49ef93a2f0f83f97ed2c0fb42dd9a90`.
The importer uses those values directly, with no global mass/inertia multiplier.
See [inertial provenance](../../robots/opendrummer/provenance/README.md).

Official body/base and arm links 0–6 provide mass, center of mass, and the full
inertia tensor. Those tensors are rotated into the CAD link frames. Distributed
CAD inertias are removed from replaced assemblies to avoid double-counting.
The visual meshes, joints, axes, limits, and tip locations are unchanged.

The standard link-7 end-effector assembly is absent from this robot. Its custom
Base/ScrewOn mounts, screws, and sticks retain separate CAD estimates, totaling
**0.101493484 kg per tool**. Only the four Base/ScrewOn parts still use the
provisional 1000x unit correction, baked into the canonical URDF and documented
in `robots/opendrummer/inertial-provenance.json`. These parts need weighing or
CAD material/unit confirmation; they are not official OpenArm mass data.

Total source assembly mass is now **14.666956 kg**, including the fixed body and
both arm bases; moving mass is approximately **4.320250 kg per arm**, including
its custom tool. Fixed base mass does not participate in fixed-base dynamics.
The build report records masses, source/config hashes, and import settings.

Validation additionally checks all 15 official inertial assignments in world
coordinates and the compiled moving-link COMs/tensors across 11 poses against
the upstream model, accounting for reversed joint axes. This verifies frame
mapping and import consistency, not measured physical calibration.

The controller still uses the initial simulation PD gains, bounded motor torque,
bias compensation, rate-limited targets, and provisional 0.01 kg·m² joint
armature. This inertial update does not substitute official hardware gains or
claim a calibrated actuator model.
