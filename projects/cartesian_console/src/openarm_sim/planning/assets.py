import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from yourdfpy import Collision, Geometry, Joint, Limit, Link, Mesh, Robot, URDF, Visual

from ..simulation import ReachSimulation
from ..paths import MODEL, ROOT, RUNS


def transform(position, rotation):
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(rotation).reshape(3, 3)
    matrix[:3, 3] = position
    return matrix


def export_robot(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    with ReachSimulation() as sim:
        model = sim.model
        data = mujoco.MjData(model)
        data.qpos[:] = 0
        mujoco.mj_forward(model, data)
        robot = Robot(name="openarm_v1_planning", links=[Link(name="world")])
        for body_id in sorted(sim.robot_bodies):
            body = model.body(body_id)
            name = body.name
            link = Link(name=name)
            body_world = transform(data.xpos[body_id], data.xmat[body_id])
            meshes = []
            for geom_id in range(int(model.body_geomadr[body_id]), int(model.body_geomadr[body_id] + model.body_geomnum[body_id])):
                if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
                    continue
                if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                    raise ValueError(f"Unsupported robot collision geometry: {model.geom(geom_id).name}")
                mesh_id = int(model.geom_dataid[geom_id])
                va, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
                fa, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
                mesh = trimesh.Trimesh(model.mesh_vert[va:va + vn], model.mesh_face[fa:fa + fn], process=False).convex_hull
                geom_world = transform(data.geom_xpos[geom_id], data.geom_xmat[geom_id])
                mesh.apply_transform(np.linalg.inv(body_world) @ geom_world)
                meshes.append(mesh)
            if meshes:
                mesh_path = output / f"{name}.stl"
                trimesh.util.concatenate(meshes).export(mesh_path)
                geometry = Geometry(mesh=Mesh(filename=str(mesh_path.resolve())))
                link.collisions.append(Collision(name=f"{name}_collision", geometry=geometry, origin=np.eye(4)))
                link.visuals.append(Visual(name=f"{name}_visual", geometry=geometry, origin=np.eye(4)))
            robot.links.append(link)
            parent_id = int(model.body_parentid[body_id])
            parent_name = model.body(parent_id).name if parent_id else "world"
            parent_world = transform(data.xpos[parent_id], data.xmat[parent_id])
            origin = np.linalg.inv(parent_world) @ body_world
            if int(model.body_jntnum[body_id]) > 1:
                raise ValueError(f"Multiple joints on {name} require explicit intermediate links")
            if model.body_jntnum[body_id]:
                joint_id = int(model.body_jntadr[body_id])
                if np.linalg.norm(model.jnt_pos[joint_id]) > 1e-9:
                    raise ValueError(f"Nonzero joint origin on {name}")
                kind = "revolute" if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE else "prismatic"
                lower, upper = model.jnt_range[joint_id]
                joint = Joint(
                    name=model.joint(joint_id).name, type=kind, parent=parent_name, child=name,
                    origin=origin, axis=model.jnt_axis[joint_id].copy(),
                    limit=Limit(lower=float(lower), upper=float(upper), effort=40.0, velocity=0.6 if kind == "revolute" else 0.1),
                )
            else:
                joint = Joint(name=f"{name}_fixed", type="fixed", parent=parent_name, child=name, origin=origin)
            robot.joints.append(joint)
        path = output / "openarm.urdf"
        URDF(robot, build_scene_graph=False, load_meshes=False).write_xml_file(str(path))
        return path


def main():
    parser = argparse.ArgumentParser(description="Export the current MJCF robot and fit cuRobo collision spheres")
    parser.add_argument("--output", type=Path, default=RUNS / "planning")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    path = export_robot(args.output.resolve())
    if args.export_only:
        print(path)
        return
    from curobo.robot_builder import RobotBuilder

    builder = RobotBuilder(str(path), asset_path=str(path.parent), tool_frames=["openarm_right_hand_tcp"])
    builder.fit_collision_spheres(sphere_density=1.5, clip_links={"openarm_body_link0": ("z", 0.0)})
    builder.compute_collision_matrix(prune_collisions=False)
    specification = mujoco.MjSpec.from_file(str(MODEL))
    for exclusion in specification.excludes:
        builder.add_collision_ignore(exclusion.bodyname1, [exclusion.bodyname2])
    builder.save(builder.build(), str(path.parent / "openarm.yml"))
    (path.parent / "provenance.json").write_text(json.dumps({
        "source": str(MODEL.relative_to(ROOT)),
        "geometry": "MuJoCo compiled convex collision meshes",
        "sphere_density": 1.5, "collision_pruning": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
