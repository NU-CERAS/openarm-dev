# OpenDrummer robot assets

Imported from the Desktop `OpenDrummer` export. The joint names, mesh references, and kinematic/visual geometry are preserved.
Inertial properties have been updated from pinned official OpenArm data; see
[provenance and custom-tool exceptions](provenance/README.md).
The export directory has been flattened so URDF, meshes, and launch files live
directly under this robot variant.

- Robot description: `urdf/opendrummer.urdf`
- Geometry: `meshes/`
- Joint-limit values and provenance: `joint-limits.json`
- Source export's repair validation report: `validation.json`
- Original export notes: `README.txt` (paths updated for this layout)
- OpenArm license: `OPENARM-LICENSE`
- Inertial mapping and custom estimates: `inertial-provenance.json`
- Original unmodified URDF: `provenance/original-opendrummer.urdf`

The URDF uses `package://opendrummer/meshes/...` references. When loading it in
a URDF viewer or importer, map the `opendrummer` package to this
`robots/opendrummer/` directory. The export does not include a ROS
package manifest. Its `launch/opendrummer.launch` is a legacy ROS 1
launch file, not a ROS 2 Jazzy launch file.

The [OpenDrummer playground](../../projects/opendrummer_playground/README.md)
provides a MuJoCo importer, validation checks, and holding viewer. The shared
CLI simulator still uses the existing v1.0 scene. The export's movable joints are `lj1` through `lj7` and `rj1`
through `rj7`, so integration with the shared controllers requires an explicit
joint-name mapping. See the original export notes for the joint-limit source
and the scope of the collision checks.
