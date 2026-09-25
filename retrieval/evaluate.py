"""Optional Singapo metrics for the merged-handle PWM representation."""
import json
from pathlib import Path

from .database import object_json_path


def evaluate_object(prediction, prediction_dir, gt_dir):
    # PyTorch3D/CUDA is required only when evaluation is explicitly requested.
    from .singapo.metrics.cd import CD
    from .singapo.metrics.aor import AOR
    from .singapo.metrics.iou_cdist import IoU_cDist

    with object_json_path(gt_dir).open() as stream:
        target = json.load(stream)
    if "diffuse_tree" not in target:
        target["diffuse_tree"] = target.pop("arti_tree")
    distances = CD(prediction, str(prediction_dir), target, str(gt_dir), include_base=True)
    overlap = AOR(prediction)
    scores = IoU_cDist(prediction, target, compare_handles=False, iou_include_base=True)
    return {"AS-IoU": float(scores["AS-IoU"]), "RS-IoU": float(scores["RS-IoU"]),
            "AS-cDist": float(scores["AS-cDist"]), "RS-cDist": float(scores["RS-cDist"]),
            "AS-CD": float(distances["AS-CD"]), "RS-CD": float(distances["RS-CD"]),
            "AOR": None if overlap == -1 else float(overlap)}


def aggregate_metrics(metrics):
    return {key: (sum(values)/len(values) if values else None)
            for key in metrics[0]
            for values in [[item[key] for item in metrics if item[key] is not None]]}
