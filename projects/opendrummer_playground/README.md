# OpenDrummer playground

The playground imports the shared OpenDrummer URDF into MuJoCo and combines it
with the drum kit. Step one provides the checked robot model and holding
controller; step two adds configured kit placement, eight surface markers, and
static approach-pose previews. Step three adds primitive collision geometry
and a slow stick–snare contact check. Step four adds a contact-triggered
single-arm snare strike and recovery, optionally repeated. Two-arm rhythms and
general collision-aware motion planning are later steps.

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

## Controlled single strikes (step four)

With the project environment active, run one left-arm snare strike:

```sh
mjpython projects/opendrummer_playground/scripts/run.py --strike
```

Repeat three isolated cycles, with full recovery between them:

```sh
mjpython projects/opendrummer_playground/scripts/run.py --strike --strikes 3
```

Use `--collision-only` to inspect proxies and contact points, or `--headless` to
run without a window. These routines start in a raised ready pose above the
snare; they do not plan a path from the hanging-arm pose. They freeze on completion
or failure. The other arm remains held still. This stage supports left/snare
only; `--arm right` is rejected with `--strike`.

The state sequence is:

```text
ready → approach → downstroke → contact → retract → recovery → ready
```

- Ready height: 10 cm above the head, with a 0.4 s initial settle.
- Approach: 0.8 s to position joints 1–6 at the drum and cock J7, leaving the
  strike tip 5 cm above the head within the strike area.
- Downstroke: only J7 changes its target over 0.09 s (5× the original strike speed), rotating the stick toward
  a virtual endpoint 8 mm below the plane. Joints 1–6 hold their approach targets.
  That endpoint is a virtual command, not intended actual penetration.
- Contact: measured rounded-tip/head force and impulse trigger withdrawal;
  the routine never continues down to the endpoint after a registered hit.
- Retraction: only J7 returns the tip to 10 cm clearance over 0.65 s, followed
  by 0.5 s of recovery. Repeated hits on the same drum retain the arm position.
- Another strike is allowed only after one recorded hit, stable recovery,
  sufficient clearance, and detector rearming.

The planner solves a centered impact pose with a specified J7 angle, then finds
J7 angles for the raised and lowered positions while fixing the other joints.
The tip follows a wrist arc, so its raised position need not be directly above
the exact impact point. The sampled arc must remain within the drum strike area
and clear unintended geometry. The below-head endpoint is a virtual command;
measured contact starts withdrawal before reaching it.

Joint PD tracks position and desired velocity with bias compensation and torque
clamped to the URDF limits. Only J7's position stiffness reduces to 45% during
contact and the first 80 ms of withdrawal. The other joint targets remain fixed
through downstroke, contact, retraction, and recovery. Those joints are still
dynamic and can deflect slightly under reaction forces; they are not welded.
Stick-direction tracking is checked against the changing wrist arc orientation.
Joint state is set only at initial placement, never teleported during a strike.

### Hit detection and logs

A hit requires actual contact between the intended rounded stick tip and snare
head, force of at least 0.5 N, and accumulated normal impulse of at least
0.0002 N·s. The detector counts once per episode. Rearming requires force below
0.05 N and at least 5 mm tip clearance for 40 ms. Brief force chatter therefore
does not count as extra strikes. A second qualified hit in the same cycle,
unintended loaded contact, or missing hit is a failure.

The terminal prints state transitions and final hit summaries. Full results:

- `runs/opendrummer_playground/strike-report.json`: hits, onset/confirmation
  times, impact speed, force peak, impulse, contact duration, nominal timing
  error, impact offset, held-joint deviations, wrist arc, recovered cycles, limits,
  and state transitions.
- `runs/opendrummer_playground/strike-trace.json`: per-step state, tip height,
  force, penetration, speed, tracking error, and commanded/actual joint positions.
- `runs/opendrummer_playground/strike-validation.json`: repeated-cycle and
  timestep checks plus the disabled-contact negative control.

Timing error is relative to the commanded tip-plane crossing, not to a musical
beat clock. Impact strengths are numerical model outputs, not calibrated acoustic
loudness or verified hardware force predictions.

### Validate strikes

```sh
python -m unittest discover -s projects/opendrummer_playground/tests -v
python projects/opendrummer_playground/scripts/strike.py
```

Validation requires ten consecutive strikes at 1 ms and five at 0.5 ms, with one
hit and stable recovery per cycle, brief contact (under 100 ms), repeatable timing
and force, no unintended contacts or solver warnings, and bounded joint motion,
tracking error, and contact penetration. It also requires less than 25% change
in peak force between timesteps. With contact disabled, the same command must
produce zero hits and a missed-strike failure. Unit tests independently check
force/impulse thresholds, sustained contact, chatter, and clearance-based rearming.

Tune simulation settings in `configs/strike.json`. Current stop thresholds include
30 N contact force, 3 mm penetration, 0.20 rad joint tracking error, 2 m/s tip speed,
and 15° stick-direction error, plus URDF joint limits. These are simulation test
settings, not a hardware commissioning configuration. The initial 0.30 s stroke
exceeded the force threshold; the current 0.45 s stroke passed without raising it.

## Source and generated files

- `robots/opendrummer/urdf/opendrummer.urdf` and `meshes/` remain canonical.
- `scripts/model.py` converts the model reproducibly and adds tip markers and
  torque motors. All commands rebuild automatically from source.
- `scripts/scene.py` composes the kit, extracts surface targets, and solves static IK.
- `scripts/collisions.py` builds contact proxies; `contact_check.py` runs and validates the slow touch.
- `configs/contact.json` contains proxy sizes and contact-test settings.
- `scripts/strike.py` and `configs/strike.json` define the strike controller, event detector, and validation.
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

## Hit readout and drum flashes

The combined viewer now includes a right sidebar showing the latest hit's peak
normal contact force (N), impulse (N·s), and each surface's hit count and most
recent measurements. Run from the repository root in the shared Mac environment:

```sh
mjpython projects/opendrummer_playground/scripts/run.py --strike --strikes 3
```

Expect three snare hits, roughly **25.6 N** peak per hit with the
current configuration. The struck head flashes for 0.8 seconds: blue below 5 N,
amber from 5 to below 20 N, and red at 20 N or above. Readings persist after the
flash fades and after the routine finishes. These are simulated contact estimates,
not measured hardware forces or calibrated sound levels. The color bands and
flash duration are adjustable in `configs/telemetry.json`.

Measurements sample every physics step, sum stick/head normal contact forces,
and use the strike detector's force/impulse thresholds and separation debounce.
Rim, shell, stand, and arm contacts are excluded. Simultaneous contacts on the
same head form one combined event. Both sticks and all eight target surfaces are
monitored, although the current automated routine still targets only the snare.

On viewer exit (or headless completion), measurements are written to
`runs/opendrummer_playground/hit-telemetry.json`. Use `--no-hit-panel` to restore
the standard MuJoCo panels while keeping head flashes and logging. `--robot-only`
does not enable drum telemetry. Flashes are visual-only sites and do not alter
contact physics. The sidebar uses Pillow, already available in the shared Mac
Pixi environment; other environments can install `requirements-ui.txt`.

Verify event counting and measured force/impulse consistency with:

```sh
python -m unittest discover -s projects/opendrummer_playground/tests -v
```

## Native pose teaching mode

Teaching runs in the local MuJoCo window, separate from demo playback. From the
repository root with the shared Mac environment active:

```sh
mjpython projects/opendrummer_playground/scripts/run.py --teach
```

Drag the **J1–J7 sliders on the right**, or use their ± buttons. All displayed
angles and step sizes are in **degrees**. Select left/right arm with the buttons,
and choose a drum with `<` / `>`. Save, load, edit, test, and stop also have
clickable buttons; keyboard shortcuts are optional. The scene supports left-drag
orbit, right-drag pan, and scroll zoom. Close the window to exit. No browser or
local web server is used.

The built-in passive MuJoCo viewer disables its joint sliders, so teaching uses
a native GLFW/MuJoCo window with a directly controlled mouse panel. On macOS the
launcher creates an app wrapper under the ignored run directory so window creation
runs on the OS main thread. Both `python ... --teach` and `mjpython ... --teach`
work; the ordinary demo continues to use the existing MuJoCo viewer.

Focus the scene before using teaching shortcuts:

| Key | Action |
|---|---|
| F8 | Switch left/right arm |
| F9 | Select the next drum |
| Up / Down | Select J1–J7 for keyboard jogging |
| Left / Right | Jog selected joint |
| `-` / `=` | Decrease/increase jog step: 0.1°, 0.5°, 1°, 5° |
| F10 | Save ready pose and lock J1–J6 |
| F11 | Save J7 strike angle |
| F12 | Test one physical hit |
| Home | Load the selected preset's ready pose |
| End | Unlock the selected arm to edit ready |
| Space | Stop the physical test |

On a Mac, hold **Fn/Globe** to send F8–F12 when the function row controls system
features. **Fn+Left/Right** sends Home/End on compact keyboards.

1. Select an arm and drum using the panel buttons or F8/F9. Each arm/drum has its own preset.
2. Drag the sliders or use arrow-key jogging to position the
   stick above the head. Jogging is a kinematic preview with physics paused.
   Only the selected arm's seven sliders are shown.
3. Press **F10** to save ready, including the raised J7 starting angle. The tip
   must be over the head with at least 10 mm clearance, and the pose must clear
   other geometry. J1–J6 then lock and their sliders disable.
4. Lower **only J7** until the tip touches or slightly crosses the head. Press
   **F11** to save the strike endpoint.
5. Press **F12** to test one hit from the saved ready pose using contact physics.
   Watch peak force, impulse, head flashes, and the pass/failure message. Slider
   and pose edits cannot alter a running test. **Space** stops it and returns to
   preview.
6. **Home** reloads ready; **End** unlocks J1–J6 for editing. Re-saving ready
   clears its old strike angle so it must be taught again. Use F8/F9 to teach
   other arm/drum combinations.

Presets persist in `configs/taught_poses.json` as radians, keyed by `arm/drum`.
They include starting/strike J7 angles and stroke/rebound durations (default
90 ms / 650 ms). Robot/scene/contact configuration changes invalidate old
presets for playback; re-teach them against the new geometry. Existing presets
are never replaced by sample poses during setup or tests.

Run a saved preset in the **separate demo mode**:

```sh
mjpython projects/opendrummer_playground/scripts/run.py --strike --preset snare --arm left --strikes 3
mjpython projects/opendrummer_playground/scripts/run.py --strike --preset rack_tom_right --arm right
```

Demo playback begins at the selected preset's ready pose. It holds J1–J6 while
J7 strikes and rebounds. Playback validates the wrist arc, joint/velocity limits,
head reach, and unwanted geometry intersections; runtime contact, force,
penetration, speed, and recovery checks can stop a bad preset. Use **F12** before
demo playback to check the dynamic hit. Physical test results are written to
`runs/opendrummer_playground/taught-strike-report.json` and hit telemetry to
`hit-telemetry.json` in that output directory.

The automatic snare demo (`--strike` without `--preset`) continues to use its
existing generated pose. Standalone preset playback starts at ready. For live transfers between the
provisional drum poses, use the drum-button demo below.

## Live drum-button demo

Run a separate native demo window with clickable drum buttons:

```sh
python projects/opendrummer_playground/scripts/run_demo.py
```

`mjpython` also works in the shared Mac environment. Click a head to plan a clear
path from the current pose, move the arm to its provisional ready position, hit
with J7 only, and rebound. Click while busy to queue up to eight more requests.
**Stop / clear queue** (or Space) cancels queued work and holds the current arm
positions. Keys 1–8 correspond to the displayed drum buttons. The hit readout and
head color flashes continue across successive button presses.

Provisional poses in `configs/demo_poses.json` cover seven stick targets:

| Left arm | Right arm |
|---|---|
| Snare | Rack tom R |
| Rack tom L | Floor tom |
| Hi-hat | Cymbal R |
| Cymbal L | |

The ready tip clearance is approximately 4 cm (5 cm on the snare). Each stored
preset includes its J7 strike angle, 90 ms downstroke, rebound duration, and a
physical simulation check. Some impact points are shifted toward the robot while
remaining inside the selected head. These are provisional simulation poses.
Your manually taught poses in `configs/taught_poses.json` are preserved; this
live demo currently loads the generated demo file, while `run.py --strike
--preset ...` loads your taught file.

The kick button is disabled: the current demo has two stick arms and no pedal
controller. Between targets, the unused arm returns to its parked pose. A
joint-space planner samples collision geometry along the transfer, then the
controller follows the path through physics without resetting joints or time.
Planning runs in the background so the window remains interactive. A planning
failure or a force/contact/limit failure clears pending requests and displays
why the action stopped. The impact stops remain 30 N and 2 m/s. Transfer travel has its own 4 m/s
tip-speed stop and an 8 mm planning clearance (existing closer endpoint gaps
are preserved).

A seven-head continuous simulation run is available with:

```sh
python projects/opendrummer_playground/scripts/validate_demo.py
```

It checks one hit per requested drum, continuous simulation time, and constant
non-J7 commands during strikes. Results are saved in
`runs/opendrummer_playground/live-demo-validation.json`. Interactive runs log to
`live-demo-report.json` in the same directory. To regenerate the provisional
poses after changing robot/kit geometry:

```sh
python projects/opendrummer_playground/scripts/generate_demo_poses.py
```

Generation validates wrist arcs and physical impacts and does not edit taught
poses. Rebuilding the preset file may take several minutes.

### Arm travel speed

`configs/demo_motion.json` sets `transfer_speed_fraction` to **1.0**. Planned
travel between drum poses uses each joint's URDF motor-speed limit separately:
J1–J2: **16.7547 rad/s**, J3–J4: **5.4454 rad/s**, J5–J7: **20.9439 rad/s**.
These are ceilings. Cubic ramps respect provisional acceleration limits and
synchronize the joints, so short moves and slower-axis-constrained moves can
peak below 100%. A 2% command headroom helps the physical servo stay inside the
speed ceiling; measured transfer speeds are checked during execution.

The same config holds acceleration limits, transfer tip-speed stop, and planner
clearance. Clearance is evaluated on a private collision model and does not
modify physical geometry or impact forces. Stroke timing comes from each drum
preset. The report records commanded and measured transfer speeds as fractions
of the motor limits, along with transfer duration. Restart the demo after
changing motion settings.
