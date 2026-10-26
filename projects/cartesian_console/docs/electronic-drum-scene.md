# Electronic drum scene

The scene imports the user's downloaded **Electronic Drum Set** by **SINNIK**, licensed under **CC BY 4.0**. The original `license.txt` is preserved with the converted files. Full attribution is in [ATTRIBUTION.md](../assets/electronic_drum_set/ATTRIBUTION.md).

The downloaded glTF contains 45 mesh instances and 416,246 triangles. The converter loads the complete glTF scene graph through trimesh, applies its node transforms, and exports MuJoCo-compatible OBJ geometry. Source base colors, metallic factors, roughness, and mesh normals are used. The source stool normal map remains available in the extracted archive.

## Import and launch

Run from the repository root in the `mujoco` environment after the [project installation](../README.md). The optional `assets` dependency is included by the planning setup.

```bash
openarm-import-drum /home/jixinyan/Downloads/electronic_drum_set.zip
openarm-reach --scene kit --drum-asset runs/cartesian_console/drumset/converted/manifest.json
```

The converter supports `--position X Y Z` and `--scale`. The default rigid transform maps source Y-up to MuJoCo Z-up and places the source origin at `(1.15, 0, 0)` m. Scale is 1.0. This is a simulation placement; the actual drum dimensions, robot mounting, and relative transform still require physical measurements.

The kit scene defaults to a robot observation camera at `(0.038, 0, 1.55)` m, looking toward `(0.70, -0.05, 0.65)` m. Its vertical field of view is 60 degrees, giving approximately 91.5 degrees horizontally at 1280x720. The camera uses perspective projection and covers the six fitted pad disks. **Robot head camera** restores this preset, and **Kit overview** selects the presentation view. Position, look-at point, and `fovy` remain editable. This is a provisional camera configuration; matching physical optics requires the actual camera intrinsics and mounting calibration. All GUI controls are in English.

Select an arm and click a visible pad. The calibrated pixel ray intersects the fitted pad disk; the target is 0.14 m along its outward normal. cuRobo plans the empty-gripper reach. The tested target on `Object_28` is reachable by the left arm. Targets remain subject to workspace, IK, and collision feasibility checks.

When IK or cuRobo cannot find a feasible plan, the GUI displays a rejection with the arm, target, and reason. It retains the current joint commands and remains available for another selection.

## Physical contact surfaces

Six disk-shaped source meshes are mapped to independently named collision surfaces:

- `kit_Object_26`
- `kit_Object_27`
- `kit_Object_28`
- `kit_Object_29`
- `kit_Object_31`
- `kit_Object_33`

Identifiers retain the source node names. Each surface uses a PCA normal, a geometry-derived center and radius, and a thin cylindrical collider. The disk is an approximation of the source surface, including a vertical striking surface. Collision response uses explicit friction and solver parameters. Each surface also exposes a named target site. Positions, normals, radii, source identifiers, and conversion provenance are recorded in `manifest.json`.

The six surfaces are physical colliders. Rack, cables, stool, and the remaining imported parts are visual geometry. Their collision geometry and mechanical compliance are not represented. Reach planning in this scene accounts for the configured pad colliders. The existing pickup and residual-strike training tasks continue to use their dedicated single-drum scene; the imported kit is exposed through `--scene kit`.

## Verified behavior

- The complete imported scene compiles and renders in MuJoCo.
- The original geometry contains 45 meshes and 12 material groups.
- The robot remains numerically stable during physical stepping.
- Every fitted pad produces a measured contact force when struck by a free capsule representing a drumstick.
- Pixel projection and fitted-pad intersection agree within 5 mm.
- The real GUI mouse-event check executes cuRobo reach above the imported pad, with TCP error below 1 mm in the checked rollout.

The contact checks establish simulator functionality. Pad dimensions, friction, impact response, compliance, mounting, and camera transforms require calibration for physical deployment. No RL learning or physical motor execution was performed.

## Videos

```bash
openarm-record-demo --drum-asset runs/cartesian_console/drumset/converted/manifest.json \
  --output runs/cartesian_console/demo/openarm_electronic_drum.mp4
openarm-record-demo --include-kit runs/cartesian_console/drumset/converted/manifest.json \
  --output runs/cartesian_console/demo/openarm_boss_demo.mp4
```

The standalone preview contains the imported kit and a measured cuRobo reach. The complete demonstration includes the camera-guided reach, obstacle-aware reach, physical reference pickup, nominal reference strike, and the electronic kit scene. Both use English captions. Model attribution and license links appear in the electronic-kit section. Videos have matching JSON metadata and recorded state traces.
