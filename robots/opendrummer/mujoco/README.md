# MuJoCo import

The reproducible importer and initial holding viewer are in
[`projects/opendrummer_playground`](../../../projects/opendrummer_playground/README.md).
It reads this robot's original URDF and meshes and writes generated models under
`runs/opendrummer_playground/`. The importer does not modify the canonical assets.

The current build contains all visual shapes, 14 actuated hinge joints, a fixed
base, and named `left_stick_tip` / `right_stick_tip` sites. The standalone import
is visual-only; the playground adds primitive robot/kit collision geometry and
a slow stick–snare contact test. See its documentation for collision debug views,
official inertial mapping, remaining custom-tool estimates, and validation commands.
