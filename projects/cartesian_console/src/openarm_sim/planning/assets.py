import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from yourdfpy import Collision, Geometry, Joint, Limit, Link, Mesh, Robot, URDF, Visual

from ..simulation import reach_simulation
from ..paths import RUNS


def transform(position, rotation):
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(rotation).reshape(3, 3)
    matrix[:3, 3] = position
    return matrix


def export_robot(output: Path, robot_model="v1.0") -> Path:
    output.mkdir(parents=True, exist_ok=True)
    with reach_simulation(robot_model) as sim:
        model = sim.model
        data = mujoco.MjData(model)
        data.qpos[:] = 0
        mujoco.mj_forward(model, data)
        robot = Robot(name=f"{robot_model}_planning", links=[Link(name="world")])
        for body_id in sorted(sim.robot_bodies):
            body = model.body(body_id)
            name = body.name
            link = Link(name=name)
            body_world = transform(data.xpos[body_id], data.xmat[body_id])
            meshes = []
            for geom_id in range(int(model.body_geomadr[body_id]), int(model.body_geomadr[body_id] + model.body_geomnum[body_id])):
                if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
                    continue
                kind, size = model.geom_type[geom_id], model.geom_size[geom_id]
                if kind == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh_id = int(model.geom_dataid[geom_id])
                    va, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
                    fa, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
                    mesh = trimesh.Trimesh(model.mesh_vert[va:va + vn], model.mesh_face[fa:fa + fn], process=False).convex_hull
                elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                    mesh = trimesh.creation.box(extents=2 * size)
                elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                    mesh = trimesh.creation.icosphere(radius=size[0])
                elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    mesh = trimesh.creation.capsule(height=2 * size[1], radius=size[0])
                elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1])
                else:
                    raise ValueError(f"Unsupported robot collision geometry: {model.geom(geom_id).name}")
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
        for arm in ("left", "right"):
            sid = sim.site_ids[arm]
            parent = int(model.site_bodyid[sid])
            name = f"{arm}_planning_tcp"
            robot.links.append(Link(name=name))
            robot.joints.append(Joint(name=f"{name}_fixed", type="fixed",
                                      parent=model.body(parent).name if parent else "world", child=name,
                                      origin=np.linalg.inv(transform(data.xpos[parent], data.xmat[parent])) @
                                             transform(data.site_xpos[sid], data.site_xmat[sid])))
        path = output / "openarm.urdf"
        URDF(robot, build_scene_graph=False, load_meshes=False).write_xml_file(str(path))
        return path


def main():
    parser = argparse.ArgumentParser(description="Export the current MJCF robot and fit cuRobo collision spheres")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", choices=["v1.0", "opendrummer"], default="v1.0")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    output = args.output or (RUNS / "planning" / "opendrummer" if args.model == "opendrummer" else RUNS / "planning")
    path = export_robot(output.resolve(), args.model)
    if args.export_only:
        print(path)
        return
    from curobo.robot_builder import RobotBuilder

    frame = "openarm_right_hand_tcp" if args.model == "v1.0" else "right_planning_tcp"
    builder = RobotBuilder(str(path), asset_path=str(path.parent), tool_frames=[frame])
    base = "openarm_body_link0" if args.model == "v1.0" else "robot_base_collision"
    builder.fit_collision_spheres(sphere_density=1.5, clip_links={base: ("z", 0.0)})
    builder.compute_collision_matrix(prune_collisions=False)
    with reach_simulation(args.model) as sim:
        for body1, body2 in zip(sim.model.exclude_signature >> 16, sim.model.exclude_signature & 0xFFFF):
            builder.add_collision_ignore(sim.model.body(int(body1)).name, [sim.model.body(int(body2)).name])
    builder.save(builder.build(), str(path.parent / "openarm.yml"))
    (path.parent / "provenance.json").write_text(json.dumps({
        "robot_model": args.model,
        "geometry": "MuJoCo compiled convex collision meshes",
        "sphere_density": 1.5, "collision_pruning": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
