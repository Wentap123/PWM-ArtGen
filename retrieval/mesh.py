"""Mesh export adapted from Singapo; see singapo/LICENSE-SINGAPO."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import trimesh

from .singapo.obj_retrieval import find_obj_candidates, pick_and_rescale_parts


def retrieve_meshes(info, database_root, output_dir, hashbook):
    info = deepcopy(info)
    candidates = find_obj_candidates(info, str(database_root), str(hashbook),
                                     gt_file_name="object_pwm.json", num_states=5,
                                     metric_num_samples=4096, keep_top=5)
    if not candidates:
        raise RuntimeError("No candidate objects were found in the mesh database.")
    specs = pick_and_rescale_parts(info, candidates, str(database_root),
                                  gt_file_name="object_pwm.json", save_path=str(output_dir),
                                  hashbook_path=hashbook)
    if len(specs) != len(info["diffuse_tree"]) or any(not item for item in specs):
        raise RuntimeError("Mesh retrieval did not produce every required part.")
    mesh_dir = Path(output_dir) / "plys"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    for index, (part, spec) in enumerate(zip(info["diffuse_tree"], specs)):
        meshes = [trimesh.load(Path(spec["dir"]) / name, force="mesh") for name in spec["files"]]
        mesh = trimesh.util.concatenate(meshes)
        mesh.vertices -= mesh.bounding_box.centroid
        mesh.apply_transform(trimesh.transformations.compose_matrix(
            scale=spec["scale_factor"], angles=[0, 0, np.radians(90) if spec["z_rotate_90"] else 0],
            translate=part["aabb"]["center"]))
        mesh.export(mesh_dir / f"part_{index}.ply")
        part["plys"] = [f"plys/part_{index}.ply"]
        scene.add_geometry(mesh)
    scene.export(Path(output_dir) / "object.ply")
    return info
