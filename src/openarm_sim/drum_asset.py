import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import mujoco
import numpy as np
import trimesh


MODEL_URL = "https://sketchfab.com/3d-models/electronic-drum-set-51b95e62da844b95b6ca871c23b6e858"
PAD_NODES = ("Object_26", "Object_27", "Object_28", "Object_29", "Object_31", "Object_33")
SOURCE_TO_WORLD = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)


def surface_from_mesh(mesh: trimesh.Trimesh, name: str) -> dict:
    _, _, vectors = np.linalg.svd(mesh.vertices - mesh.vertices.mean(axis=0), full_matrices=False)
    normal = vectors[-1]
    if normal[2] < 0 or (abs(normal[2]) < 0.5 and normal[0] > 0):
        normal = -normal
    center = np.average(mesh.triangles_center, weights=mesh.area_faces, axis=0)
    normal_projection = (mesh.vertices - center) @ normal
    center = center + normal * float(np.quantile(normal_projection, 0.95))
    displacement = mesh.vertices - center
    planar = displacement - (displacement @ normal)[:, None] * normal
    radius = float(np.max(np.linalg.norm(planar, axis=1))) * 0.90
    quaternion = np.empty(4)
    mujoco.mju_quatZ2Vec(quaternion, normal)
    return {"name": f"kit_{name}", "source_node": name, "center": center.tolist(),
            "normal": normal.tolist(), "quaternion_wxyz": quaternion.tolist(),
            "radius": radius, "half_thickness": 0.003,
            "mesh_thickness_m": float(np.ptp(normal_projection)),
            "surface_model": "PCA disk approximation of source geometry"}


def convert(source: Path, output: Path, position=(1.15, 0, 0), scale: float = 1.0) -> Path:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Asset scale must be positive")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".zip":
        extracted = output / "source"
        extracted.mkdir(exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                destination = (extracted / member.filename).resolve()
                if not destination.is_relative_to(extracted):
                    raise ValueError("Archive contains a path outside the extraction directory")
            archive.extractall(extracted)
        models = list(extracted.rglob("*.gltf")) + list(extracted.rglob("*.glb"))
        if len(models) != 1:
            raise ValueError("Archive must contain exactly one glTF or GLB model")
        model_path = models[0]
    else:
        model_path = source.resolve()
    scene = trimesh.load_scene(model_path, process=False)
    world = np.eye(4)
    world[:3, :3] = SOURCE_TO_WORLD * scale
    world[:3, 3] = position
    rows, pads, all_vertices = [], [], []
    for index, node in enumerate(scene.graph.nodes_geometry):
        node_transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry].copy()
        mesh.apply_transform(world @ node_transform)
        if len(mesh.faces) == 0:
            raise ValueError(f"Source node has no triangles: {node}")
        material = mesh.visual.material
        color = np.asarray(material.baseColorFactor, dtype=float) / 255
        color[:3] = trimesh.visual.color.linear_to_srgb(color[:3])
        filename = f"part_{index:03d}.obj"
        obj = trimesh.exchange.obj.export_obj(mesh, include_normals=True, include_color=False,
                                              include_texture=False, write_texture=False)
        (output / filename).write_text(obj)
        rows.append({"name": f"kit_visual_{index:03d}", "node": node, "file": filename,
                     "material": material.name, "rgba": color.tolist(),
                     "metallic": 1.0 if material.metallicFactor is None else float(material.metallicFactor),
                     "roughness": 1.0 if material.roughnessFactor is None else float(material.roughnessFactor),
                     "faces": len(mesh.faces), "bounds": mesh.bounds.tolist()})
        all_vertices.append(mesh.vertices)
        if node in PAD_NODES:
            pads.append(surface_from_mesh(mesh, node))
    if len(pads) != len(PAD_NODES):
        raise ValueError("The selected electronic drum model is missing expected pad nodes")
    vertices = np.concatenate(all_vertices)
    payload = {
        "schema": "openarm.electronic_drum_asset.v1", "model": "Electronic Drum Set", "author": "SINNIK",
        "source_url": MODEL_URL, "license": "CC-BY-4.0", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scale": scale, "position": list(position), "rotation": SOURCE_TO_WORLD.tolist(),
        "bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        "parts": rows, "pads": pads,
        "calibration": "Source geometry in meters; scene placement is configurable and not hardware-calibrated",
        "appearance": "Source base colors, metallic factors, and roughness; stool normal texture remains in source archive",
    }
    license_path = model_path.parent / "license.txt"
    if not license_path.is_file():
        raise ValueError("The electronic drum download must include its license.txt")
    shutil.copyfile(license_path, output / "license.txt")
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    return manifest


def load_asset(spec: mujoco.MjSpec, path: Path) -> dict:
    data = json.loads(path.read_text())
    if data["schema"] != "openarm.electronic_drum_asset.v1":
        raise ValueError("Unsupported electronic drum asset schema")
    for part in data["parts"]:
        name = part["name"]
        spec.add_mesh(name=name, file=str((path.parent / part["file"]).resolve()), maxhullvert=64,
                      inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        spec.add_material(name=f"{name}_material", rgba=part["rgba"], metallic=part["metallic"],
                          roughness=part["roughness"], specular=0.3, shininess=0.2)
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                               material=f"{name}_material", contype=0, conaffinity=0, group=2)
    for pad in data["pads"]:
        center = np.array(pad["center"]) - np.array(pad["normal"]) * pad["half_thickness"]
        spec.worldbody.add_geom(name=pad["name"], type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                               pos=center, quat=pad["quaternion_wxyz"], size=[pad["radius"], pad["half_thickness"], 0],
                               rgba=[0.2, 0.7, 0.3, 0], group=3, friction=[0.7, 0.01, 0.001],
                               solref=[0.008, 0.5], solimp=[0.9, 0.95, 0.001, 0.5, 2])
        spec.worldbody.add_site(name=f"{pad['name']}_target", pos=pad["center"],
                               quat=pad["quaternion_wxyz"], size=[0.006, 0, 0], rgba=[1, 0.6, 0, 0], group=4)
    return data


def pick_pad(u: int, v: int, frame, width: int, height: int, pads: list[dict]):
    if not 0 <= u < width or not 0 <= v < height:
        raise ValueError("Pixel coordinates are outside the image")
    local = np.array([(u - (width - 1) / 2) / frame.focal,
                      -(v - (height - 1) / 2) / frame.focal, -1.0])
    ray = frame.rotation @ local
    intersections = []
    for pad in pads:
        normal = np.array(pad["normal"])
        denominator = float(ray @ normal)
        if abs(denominator) < 1e-9:
            continue
        distance = float((np.array(pad["center"]) - frame.position) @ normal) / denominator
        if distance <= 0:
            continue
        position = frame.position + distance * ray
        if np.linalg.norm(position - pad["center"]) <= pad["radius"]:
            intersections.append((distance, pad, position))
    if not intersections:
        return None
    _, pad, position = min(intersections, key=lambda item: item[0])
    return pad, position


def main():
    parser = argparse.ArgumentParser(description="Convert the SINNIK electronic drum glTF asset for MuJoCo")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path(".work/drumset/converted"))
    parser.add_argument("--position", type=float, nargs=3, default=[1.15, 0, 0])
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    manifest = convert(args.source, args.output, args.position, args.scale)
    print(manifest)


if __name__ == "__main__":
    main()
