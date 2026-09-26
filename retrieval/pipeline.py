"""One-pass prediction loading, alignment, assembly, retrieval, and evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import random

from .assemble import assemble_object
from .database import object_json_path, resolve_hashbook, load_hashbook


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parser():
    ap = argparse.ArgumentParser(description="Postprocess PWM predictions and retrieve part meshes.")
    for name in ("pred_root", "data_root", "database_root", "out_dir"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--view_id", help="View number or directory name, required when a source is ambiguous.")
    ap.add_argument("--no-postprocess", action="store_true", help="Disable the default frontal-view alignment.")
    ap.add_argument("--evaluate", action="store_true", help="Compute metrics with the optional PyTorch3D dependencies.")
    ap.add_argument("--gt_root", type=Path)
    ap.add_argument("--hashbook", type=Path, help="Override the database's generated retrieval index.")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, help="Optional reproducible per-object random seed.")
    ap.add_argument("--dry-run", action="store_true", help="Validate inputs and list jobs without writing files.")
    return ap


def validate_prediction(prediction):
    for key, length in (("axis_dir", 3), ("axis_ori", 3), ("range", 1),
                        ("aabb_min", 3), ("aabb_max", 3),
                        ("aabb_base_min", 3), ("aabb_base_max", 3)):
        values = prediction.get(key)
        if not isinstance(values, list) or len(values) != length or not all(math.isfinite(x) for x in values):
            raise ValueError(f"Invalid prediction field: {key}")
    if prediction.get("type") not in (0, 1):
        raise ValueError("The final PWM configuration expects joint type 0 or 1.")


def prepare_job(object_dir, args):
    matches = [path for path in (args.data_root / args.split).glob(f"*/{object_dir.name}") if path.is_dir()]
    if len(matches) != 1:
        raise ValueError(f"Expected one dataset object for {object_dir.name}, found {len(matches)}.")
    source = matches[0]
    category = source.parent.name
    view = args.view_id
    if view is not None and view.isdecimal():
        view = f"view_id_{int(view):02d}"
    views = sorted(source.glob(view or "view_id_*"))
    views = [path for path in views if (path / "process/graph_renum.json").is_file()]
    if len(views) != 1:
        raise ValueError(f"Expected one graph view for {object_dir.name}, found {len(views)}; specify --view_id.")
    graph = read_json(views[0] / "process/graph_renum.json")
    graph.setdefault("meta", {})["obj_cat"] = category
    joints = sorted((path for path in object_dir.glob("joint_*") if path.is_dir()),
                    key=lambda path: int(path.name.removeprefix("joint_")))
    if not joints:
        raise ValueError("No joint predictions found.")
    predictions, boxes, keys = [], [], []
    for joint in joints:
        pred = read_json(joint / "geom_pred.json")["pred"]
        validate_prediction(pred)
        predictions.append(pred)
        if not args.no_postprocess:
            box = read_json(views[0] / joint.name / "mask_renum_bbox.json")
            for field in ("joint_bbox_cxcywh_norm", "base_bbox_cxcywh_norm"):
                if len(box.get(field, [])) != 4 or not all(math.isfinite(x) for x in box[field]):
                    raise ValueError(f"Invalid {field} for {joint.name}.")
            boxes.append(box)
            # Keep the reference alignment's stable IDs and tie-breaking keys.
            key = (f"id={box['id']}" if box.get("id") is not None else
                   "part_" + hashlib.md5(str(joint / "meta.json").encode()).hexdigest()[:8])
            keys.append(key)
    assemble_object(graph, predictions)  # Validate the graph before creating outputs.
    if not (args.database_root / category).is_dir():
        raise FileNotFoundError(f"No retrieval category: {args.database_root / category}")
    if args.evaluate and not object_json_path(args.gt_root / category / object_dir.name).is_file():
        raise FileNotFoundError(f"Missing GT object for {category}/{object_dir.name}")
    output = args.out_dir / f"0@{category}@{object_dir.name}" / "0"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output already contains results: {output}; choose a fresh --out_dir.")
    return dict(object_id=object_dir.name, category=category, graph=graph, predictions=predictions,
                boxes=boxes, keys=keys, output=str(output), postprocess=not args.no_postprocess,
                database_root=str(args.database_root), evaluate=args.evaluate,
                gt_dir=str(args.gt_root / category / object_dir.name) if args.evaluate else None,
                seed=args.seed, hashbook=str(resolve_hashbook(args.database_root, args.hashbook)))


def process_job(job):
    import numpy as np
    from .postprocess import align_predictions
    from .mesh import retrieve_meshes

    if job["seed"] is not None:
        seed = (job["seed"] + int(hashlib.sha256(job["object_id"].encode()).hexdigest()[:8], 16)) % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        if job["evaluate"]:
            import torch
            torch.manual_seed(seed)
    predictions = job["predictions"]
    if job["postprocess"]:
        predictions = align_predictions(predictions, job["boxes"], job["keys"])
    info = assemble_object(job["graph"], predictions)
    output = Path(job["output"])
    info = retrieve_meshes(info, job["database_root"], output, job["hashbook"])
    write_json(output / "object_pwm.json", info)
    result = {"object_id": job["object_id"], "output": str(output), "status": "ok"}
    if job["evaluate"]:
        from .evaluate import evaluate_object
        metrics = evaluate_object(info, output, job["gt_dir"])
        write_json(output / "metrics.json", metrics)
        result["metrics"] = metrics
    return result


def run(args):
    for name in ("pred_root", "data_root", "database_root", "out_dir", "gt_root", "hashbook"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.expanduser().resolve())
    if args.workers < 1:
        raise ValueError("--workers must be positive.")
    if args.evaluate and args.gt_root is None:
        raise ValueError("--evaluate requires --gt_root.")
    args.hashbook = resolve_hashbook(args.database_root, args.hashbook)
    load_hashbook(args.hashbook)
    for path in (args.pred_root, args.data_root / args.split, args.database_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for source in (args.pred_root, args.data_root, args.database_root, args.gt_root):
        if source is not None and (args.out_dir == source or args.out_dir.is_relative_to(source) or source.is_relative_to(args.out_dir)):
            raise ValueError("Output must be separate from prediction, data, database, and GT directories.")
    objects = sorted(path for path in args.pred_root.iterdir()
                     if path.is_dir() and any(path.glob("joint_*")))
    if not objects:
        raise ValueError("No objects containing joint predictions were found.")
    jobs, results = [], []
    for object_dir in objects:
        try:
            jobs.append(prepare_job(object_dir, args))
        except Exception as error:
            results.append(dict(object_id=object_dir.name, status="failed", error=str(error)))
            print(f"[FAIL] {object_dir.name}: {error}", flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"[DRY] {job['object_id']}: postprocess={job['postprocess']} -> {job['output']}; evaluate={job['evaluate']}")
        return 1 if results else 0

    def collect(job, operation):
        try:
            result = operation()
            results.append(result)
            print(f"[OK] {job['object_id']}: {job['output']}", flush=True)
        except Exception as error:
            results.append(dict(object_id=job["object_id"], status="failed", error=str(error)))
            print(f"[FAIL] {job['object_id']}: {error}", flush=True)

    if args.workers == 1:
        for job in jobs:
            collect(job, lambda job=job: process_job(job))
    elif jobs:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(process_job, job): job for job in jobs}
            for future in as_completed(futures):
                collect(futures[future], future.result)
    results.sort(key=lambda result: result["object_id"])
    write_json(args.out_dir / "retrieval_summary.json", results)
    metrics = [result["metrics"] for result in results if "metrics" in result]
    if metrics:
        from .evaluate import aggregate_metrics
        write_json(args.out_dir / "metrics.json", aggregate_metrics(metrics))
    return int(any(result["status"] != "ok" for result in results))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        print(f"[FAIL] {error}", flush=True)
        return 1
