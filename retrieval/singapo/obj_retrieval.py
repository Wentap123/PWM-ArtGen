# Adapted from https://github.com/3dlg-hcvc/singapo
# Copyright (c) 2024 3dlg-hcvc. See LICENSE-SINGAPO.
import os
import sys
import random
import json
import numpy as np
from copy import deepcopy

from retrieval.singapo.metrics.iou_cdist import IoU_cDist
from retrieval.hashing import get_hash
from retrieval.database import object_json_path, load_hashbook, candidate_ids, resolve_hashbook

all_categories = [
    "Table",
    "StorageFurniture",
    "WashingMachine",
    "Microwave",
    "Dishwasher",
    "Refrigerator",
    "Oven",
]


def _verify_mesh_exists(dir, ply_files, verbose=False):
    """
    Verify that the mesh files exist\n

    - dir: the directory of the object\n
    - ply_files: the list of mesh files\n
    - verbose (optional): whether to print the progress\n

    return:\n
    - True if the mesh files exist, False otherwise
    """

    for ply_file in ply_files:
        if not os.path.exists(os.path.join(dir, ply_file)):
            if verbose:
                print(f" - {os.path.join(dir, ply_file)} does not exist!!!")
            return False
    return True


def _generate_output_part_dicts(
    candidate_dict,
    part_idx,
    candidate_dir,
    requirement_part_bbox_sizes,
    bbox_size_eps=1e-3,
    verbose=False,
):
    """
    Generate the output part dictionary for all parts that are fulfilled by the candidate part and computing the scale factor of the parts\n

    - candidate_dict: the candidate object dictionary\n
    - part_idx: the index of the part in the candidate object\n
    - candidate_dir: the directory of the candidate object\n
    - requirement_part_bbox_sizes: the bounding box sizes of the requirement part in the form: [[lx1, ly1, lz1], [lx2, ly2, lz2], ...]\n
    - bbox_size_eps (optional): the epsilon to avoid zero volume parts\n
    - verbose (optional): whether to print the progress\n

    Return:\n
    - part_dicts: the output part dictionaries in the form:
        - [{name, dir, files, scale_factor=[sx, sy, sz]}, z_rotate_90]
            - z_rotate_90 is True if the part needs to be rotated by 90 degrees around the z-axis
    - [{}, ...] if any of the mesh files do not exist
    """

    part_dicts = [{} for _ in range(len(requirement_part_bbox_sizes))]
    fixed_portion = {
        "name": candidate_dict["diffuse_tree"][part_idx]["name"],
        "dir": candidate_dir,
        "files": candidate_dict["diffuse_tree"][part_idx]["plys"],
        "z_rotate_90": False,
    }

    # Verify that the mesh files exist
    if not _verify_mesh_exists(fixed_portion["dir"], fixed_portion["files"], verbose):
        if verbose:
            print(
                f" - ! Found invalid mesh files in {fixed_portion['dir']}, skipping..."
            )
        return part_dicts  # List of empty dicts

    candidate_bbox_size = np.array(
        candidate_dict["diffuse_tree"][part_idx]["aabb"]["size"]
    )
    candidate_bbox_size = np.maximum(
        candidate_bbox_size, bbox_size_eps
    )  # Avoid zero volume parts

    for i, requirement_part_bbox_size in enumerate(requirement_part_bbox_sizes):
        part_dicts[i] = deepcopy(fixed_portion)

        # Handles are already part of their parent mesh in the PWM database.
        part_dicts[i]["scale_factor"] = list(
            np.array(requirement_part_bbox_size) / candidate_bbox_size
        )

    return part_dicts


def find_obj_candidates(
    requirement_dict,
    dataset_dir,
    hashbook_path,
    num_states=5,
    metric_compare_handles=False,
    metric_iou_include_base=True,
    metric_num_samples=10000,
    keep_top=5,
    gt_file_name="object_pwm.json",
    verbose=False,
):
    """
    Find the best object candidates for selecting the base part using AID\n

    - requirement_dict: the object dictionary of the requirement\n
    - dataset_dir: the directory of the dataset to search in\n
    - hashbook_path: the path to the hashbook for filtering candidates\n
    - num_states: the number of states to average the metric over\n
    - metric_transform_plucker (optional): whether to use Plucker coordinates to move parts when computing the metric\n
    - metric_compare_handles (optional): whether to compare handles when computing the metric\n
    - metric_iou_include_base (optional): whether to include the base when computing the IoU\n
    - metric_scale_factor (optional): the scale factor to scale the object before computing the metric\n
        - Scaling up the object makes the sampling more well distributed\n
    - metric_num_samples (optional): the number of samples to use when computing the metric\n
    - keep_top (optional): the number of top candidates to keep\n
    - gt_file_name (optional): the name of the ground truth json file, which describes a candidate object\n
    - verbose (optional): whether to print the progress\n

    return:\n
    - a list of best object candidates of the form:
        - {"category", "dir", "score"}
    """
    dataset_dir = os.path.abspath(dataset_dir)

    # Load the hashbook
    hashbook = load_hashbook(hashbook_path)

    # Resolve paths to directories
    category_specified = False
    requirement_category = ""

    # if the category is specified, only search in that category, otherwise search in all categories
    if "obj_cat" in requirement_dict["meta"]:
        requirement_category = requirement_dict["meta"]["obj_cat"]
        category_specified = True

    category_dirs = (
        [os.path.join(dataset_dir, requirement_category)]
        if category_specified
        else [os.path.join(dataset_dir, category) for category in all_categories]
    )

    # Extract requirement data
    requirement_part_names = []
    requirement_part_bboxes = []
    for part in requirement_dict["diffuse_tree"]:
        requirement_part_names.append(part["name"])
        requirement_part_bboxes.append(
            np.concatenate([part["aabb"]["center"], part["aabb"]["size"]])
        )

    # Compute hash of the requirement graph
    requirement_graph_hash = get_hash(requirement_dict)

    # Category/object pairs prevent IDs in different categories from colliding.
    same_hash_objects = {
        (os.path.basename(directory), oid)
        for directory in category_dirs
        for oid in candidate_ids(hashbook, os.path.basename(directory), requirement_graph_hash)
        if object_json_path(os.path.join(directory, oid), gt_file_name).is_file()
    }

    # Iterate through all candidate objects and keep the top k candidates
    best_obj_candidates = []
    for category_dir in category_dirs:
        category = os.path.basename(category_dir)
        obj_ids = candidate_ids(hashbook, category)
        for i, obj_id in enumerate(obj_ids):
            if verbose:
                print(
                    f"\r - Finding candidates from {category_dir.split('/')[-1]}: {i+1}/{len(obj_ids)}",
                    end="",
                )

            # Load the candidate object
            obj_dir = os.path.join(category_dir, obj_id)
            object_path = object_json_path(obj_dir, gt_file_name)
            if not object_path.is_file():
                continue
            if same_hash_objects and (category, obj_id) not in same_hash_objects:
                continue
            with open(object_path, "r") as f:
                obj_dict = json.load(f)
                if "diffuse_tree" not in obj_dict:  # Rename for compatibility
                    obj_dict["diffuse_tree"] = obj_dict.pop("arti_tree")

            # Compute metric for selecting the base if the hash matches or if there are no objects with the same hash
            if (category, obj_id) in same_hash_objects or not same_hash_objects:
                scores = IoU_cDist(
                    requirement_dict,
                    obj_dict,
                    num_states=num_states,
                    compare_handles=metric_compare_handles,
                    iou_include_base=metric_iou_include_base,
                    num_samples=metric_num_samples,
                )
                base_score = scores["AS-cDist"] + scores["AS-IoU"] + scores["RS-IoU"] + scores["RS-cDist"]
                # scores["AS-cDist"] + scores["AS-IoU"]
                # scores["AS-cDist"]
                # + scores["AS-IoU"] + scores["RS-IoU"] + scores["RS-cDist"]

                # Add the candidate to the list of best candidates and keep the top k candidates
                best_obj_candidates.append(
                    {
                        "category": category_dir.split("/")[-1],
                        "dir": obj_dir,
                        "score": base_score,
                    }
                )
                best_obj_candidates = sorted(
                    best_obj_candidates, key=lambda x: x["score"]
                )[:keep_top]

        # if verbose:
        print("best_obj_candidates", best_obj_candidates)

    return best_obj_candidates


def pick_and_rescale_parts(
    requirement_dict,
    obj_candidates,
    dataset_dir,
    gt_file_name="object_pwm.json",
    verbose=True,
    max_retry=3,
    save_path=None,
    hashbook_path=None,
):
    """
    Pick and rescale parts from the object candidates safely (with anti-dead-loop logic)
    """

    hashbook = load_hashbook(resolve_hashbook(dataset_dir, hashbook_path))
    allowed = {cat: set(candidate_ids(hashbook, cat)) for cat in all_categories}

    # Extract requirement data
    requirement_part_names = []
    requirement_part_bbox_sizes = []
    for part in requirement_dict["diffuse_tree"]:
        requirement_part_names.append(part["name"])
        requirement_part_bbox_sizes.append(part["aabb"]["size"])

    # Group same-name parts (e.g. 4 legs)
    unique_requirement_part_names = {}
    for i, part_name in enumerate(requirement_part_names):
        unique_requirement_part_names.setdefault(part_name, []).append(i)

    # Initialize empty outputs
    parts_to_render = [{} for _ in requirement_part_names]

    # ---------- Step 1: use given candidates ----------
    for candidate in obj_candidates:
        if all(len(p) > 0 for p in parts_to_render):
            break
        if os.path.basename(candidate['dir']) not in allowed.get(candidate['category'], set()):
            continue
        with open(object_json_path(candidate["dir"], gt_file_name), "r") as f:
            candidate_dict = json.load(f)

        for candidate_part_idx, part in enumerate(candidate_dict["diffuse_tree"]):
            if part["name"] not in unique_requirement_part_names:
                continue

            # Find which parts still missing for this name
            target_idxs = [
                i for i in unique_requirement_part_names[part["name"]]
                if len(parts_to_render[i]) == 0
            ]
            if not target_idxs:
                continue

            bbox_sizes = [requirement_part_bbox_sizes[i] for i in target_idxs]
            part_dicts = _generate_output_part_dicts(
                candidate_dict,
                candidate_part_idx,
                candidate["dir"],
                bbox_sizes,
                verbose=verbose,
            )
            for j, idx in enumerate(target_idxs):
                parts_to_render[idx].update(part_dicts[j])

    # ---------- Step 2: search in dataset if still missing ----------
    if any(len(p) == 0 for p in parts_to_render):
        remaining_names = list({
            requirement_part_names[i]
            for i, p in enumerate(parts_to_render)
            if len(p) == 0
        })

        if verbose:
            print(f" -{save_path} Missing parts: {remaining_names}, searching in dataset...")

        requirement_category = requirement_dict["meta"].get("obj_cat", "")
        category_specified = requirement_category != ""
        if category_specified:
            category_dirs = [os.path.join(dataset_dir, requirement_category)]
        else:
            category_dirs = [
                os.path.join(dataset_dir, cat) for cat in (all_categories or [])
            ]

        retry_count = 0
        prev_missing = len(remaining_names)

        while retry_count < max_retry:
            retry_count += 1
            if verbose:
                print(f"\n[Search Attempt {retry_count}/{max_retry}] "
                      f"Categories: {[os.path.basename(d) for d in category_dirs]}")

            for category_dir in category_dirs:
                if not os.path.exists(category_dir):
                    continue

                obj_ids = candidate_ids(hashbook, os.path.basename(category_dir))
                random.shuffle(obj_ids)

                for i, obj_id in enumerate(obj_ids):
                    if verbose and i % 20 == 0:
                        print(f"\r - Searching {category_dir.split('/')[-1]}: {i+1}/{len(obj_ids)}", end="")

                    obj_dir = os.path.join(category_dir, obj_id)
                    gt_path = object_json_path(obj_dir, gt_file_name)
                    if not os.path.exists(gt_path):
                        continue

                    with open(gt_path, "r") as f:
                        candidate_dict = json.load(f)

                    for candidate_part_idx, part in enumerate(candidate_dict["diffuse_tree"]):
                        if part["name"] not in remaining_names:
                            continue

                        target_idxs = [
                            i for i in unique_requirement_part_names[part["name"]]
                            if len(parts_to_render[i]) == 0
                        ]
                        if not target_idxs:
                            continue

                        bbox_sizes = [requirement_part_bbox_sizes[i] for i in target_idxs]
                        part_dicts = _generate_output_part_dicts(
                            candidate_dict,
                            candidate_part_idx,
                            obj_dir,
                            bbox_sizes,
                            verbose=verbose,
                        )
                        for j, idx in enumerate(target_idxs):
                            parts_to_render[idx].update(part_dicts[j])

                    if all(len(p) > 0 for p in parts_to_render):
                        if verbose:
                            print(" -> All missing parts found.")
                        break

                if all(len(p) > 0 for p in parts_to_render):
                    break

            # progress check
            cur_missing = len([p for p in parts_to_render if len(p) == 0])
            if cur_missing == 0:
                break
            elif cur_missing >= prev_missing and not category_specified:
                if verbose:
                    print(f"\nNo further progress after attempt {retry_count}, stopping search.")
                break
            else:
                prev_missing = cur_missing

            # fallback: broaden search to all categories
            if category_specified and cur_missing > 0:
                category_specified = False
                category_dirs = [
                    os.path.join(dataset_dir, cat)
                    for cat in (all_categories or [])
                    if cat != requirement_category
                ]
                if verbose:
                    print(f"\n -> Still missing {cur_missing} parts, expanding search to all categories.")

    # ---------- Step 3: final check ----------
    missing_parts = [
        requirement_part_names[i]
        for i, p in enumerate(parts_to_render)
        if len(p) == 0
    ]
    if missing_parts:
        raise RuntimeError(f"No mesh found for parts: {missing_parts}")

    return parts_to_render
