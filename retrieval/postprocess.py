"""In-memory version of the final frontal-view bounding-box postprocessing."""
from copy import deepcopy
import numpy as np

from . import geometry as g


def align_predictions(predictions, boxes, keys, threshold=0.05):
    """Return adjusted copies; preserve the original single-part/overlap gates."""
    result = deepcopy(predictions)
    if len(result) < 2:
        return result
    if not (len(result) == len(boxes) == len(keys)) or len(set(keys)) != len(keys):
        raise ValueError("Each joint requires distinct metadata and a unique alignment key.")
    width = height = 512
    axes = ("x", "y")
    parts = []
    for pred, box, key in zip(result, boxes, keys):
        center, size = g.aabb_center_size_from_minmax(pred["aabb_min"], pred["aabb_max"])
        base_center, base_size = g.aabb_center_size_from_minmax(pred["aabb_base_min"], pred["aabb_base_max"])
        parts.append(dict(key=key, disp=key, aabb_center=center, aabb_size=size,
                          base_center=base_center, base_size=base_size,
                          joint_norm=box["joint_bbox_cxcywh_norm"]))
    base_center = np.median([part["base_center"] for part in parts], axis=0).tolist()
    base_size = np.median([part["base_size"] for part in parts], axis=0).tolist()
    base_norm = np.median([box["base_bbox_cxcywh_norm"] for box in boxes], axis=0)
    bx0, by0, bx1, by1 = g.norm_cxcywh_to_xyxy(base_norm, width, height)
    bw, bh = max(1e-6, bx1 - bx0), max(1e-6, by1 - by0)
    percentages = {}
    for part in parts:
        cx, cy, w, h = g.xyxy_to_cxcywh(*g.norm_cxcywh_to_xyxy(part["joint_norm"], width, height))
        percentages[part["key"]] = dict(cx_pct=(cx-bx0)/bw, cy_pct=(cy-by0)/bh,
                                       w_pct=w/bw, h_pct=h/bh)
    projected = {"base": g.world_xyxy_to_pixel_xyxy(
        *g.aabb_world_xyxy_on_plane(base_center, base_size, axes), width, height, axes)}
    for part in parts:
        projected[part["key"]] = g.world_xyxy_to_pixel_xyxy(
            *g.aabb_world_xyxy_on_plane(part["aabb_center"], part["aabb_size"], axes),
            width, height, axes)
    if not g.need_align_due_to_overlap(parts, axes, threshold=threshold, metric="iomin"):
        return result
    aligned = g.align_percent(projected, percentages, canvas_wh=(width, height),
                              size_mode="rel_ratio", size_clip=(0.5, 2.0),
                              base_pad_px=3.0, base_pad_ratio=0.02,
                              collision_axis="auto", gap=2.0)

    def reproject(pixel_box, center, size):
        x0, y0, x1, y1 = g.pixel_xyxy_to_world_xyxy(*pixel_box, width, height, axes)
        center, size = list(center), list(size)
        center[0], center[1] = float(g.clamp((x0+x1)*0.5, -1, 1)), float(g.clamp((y0+y1)*0.5, -1, 1))
        size[0], size[1] = float(g.clamp(x1-x0, 1e-6, 2)), float(g.clamp(y1-y0, 1e-6, 2))
        lo, hi = g.aabb_minmax_from_center_size(center, size)
        return center, lo, hi

    _, base_min, base_max = reproject(projected["base"], base_center, base_size)
    for pred, part in zip(result, parts):
        center, low, high = reproject(aligned[part["key"]], part["aabb_center"], part["aabb_size"])
        for axis in (0, 1):
            pred["aabb_min"][axis], pred["aabb_max"][axis] = low[axis], high[axis]
            pred["aabb_base_min"][axis], pred["aabb_base_max"][axis] = base_min[axis], base_max[axis]
        origin = list(pred.get("axis_ori", [0.0, 0.0, 0.0]))
        for axis in (0, 1):
            origin[axis] += center[axis] - part["aabb_center"][axis]
        pred["axis_ori"] = origin
    return result
