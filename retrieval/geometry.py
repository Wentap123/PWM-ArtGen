"""Geometric alignment helpers retained from the final PWM postprocessing."""
from pathlib import Path
from typing import Dict, Tuple
import hashlib
import numpy as np

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def xyxy_to_cxcywh(x0, y0, x1, y1):
    w = max(0.0, x1 - x0); h = max(0.0, y1 - y0)
    cx = x0 + w * 0.5; cy = y0 + h * 0.5
    return cx, cy, w, h

def cxcywh_to_xyxy(cx, cy, w, h):
    return cx - w*0.5, cy - h*0.5, cx + w*0.5, cy + h*0.5

def world_to_px(val: float, length: int) -> float:
    u = (val + 1.0) * 0.5
    return u * (length - 1)

def px_to_world(px: float, length: int) -> float:
    if length <= 1: return 0.0
    u = px / (length - 1)
    return u * 2.0 - 1.0

def world_to_px_axis(val: float, length: int, axis: str) -> float:
    if axis.lower() == "y":
        u = (val + 1.0) * 0.5
        return (1.0 - u) * (length - 1)
    else:
        return world_to_px(val, length)

def px_to_world_axis(px: float, length: int, axis: str) -> float:
    if length <= 1: return 0.0
    if axis.lower() == "y":
        v = 1.0 - px / (length - 1)
        return v * 2.0 - 1.0
    else:
        return px_to_world(px, length)

def aabb_center_size_from_minmax(aabb_min, aabb_max):
    c = [(a+b)*0.5 for a,b in zip(aabb_min, aabb_max)]
    s = [max(1e-9, b-a) for a,b in zip(aabb_min, aabb_max)]
    return c, s

def aabb_minmax_from_center_size(center, size):
    half = [v*0.5 for v in size]
    aabb_min = [c-h for c,h in zip(center, half)]
    aabb_max = [c+h for c,h in zip(center, half)]
    return aabb_min, aabb_max

def aabb_world_xyxy_on_plane(center, size, plane_axes: Tuple[str, str]):
    cx, cy, cz = center; sx, sy, sz = size
    half = {"x": sx * 0.5, "y": sy * 0.5, "z": sz * 0.5}
    c = {"x": cx, "y": cy, "z": cz}
    ax, ay = plane_axes
    a0 = c[ax] - half[ax]; a1 = c[ax] + half[ax]
    b0 = c[ay] - half[ay]; b1 = c[ay] + half[ay]
    return a0, b0, a1, b1

def world_xyxy_to_pixel_xyxy(a0, b0, a1, b1, W, H, plane_axes: Tuple[str, str]):
    ax, ay = plane_axes
    x0 = world_to_px_axis(a0, W, ax)
    x1 = world_to_px_axis(a1, W, ax)
    y0 = world_to_px_axis(b1, H, ay)  # flip for image-y
    y1 = world_to_px_axis(b0, H, ay)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    return x0, y0, x1, y1

def pixel_xyxy_to_world_xyxy(x0, y0, x1, y1, W, H, plane_axes: Tuple[str, str]):
    ax, ay = plane_axes
    a0 = px_to_world_axis(x0, W, ax)
    a1 = px_to_world_axis(x1, W, ax)
    b1 = px_to_world_axis(y0, H, ay)
    b0 = px_to_world_axis(y1, H, ay)
    a0, a1 = min(a0, a1), max(a0, a1)
    b0, b1 = min(b0, b1), max(b0, b1)
    return a0, b0, a1, b1

def norm_cxcywh_to_xyxy(norm, W, H):
    cx, cy, w, h = norm
    cx *= W; cy *= H; w *= W; h *= H
    return cxcywh_to_xyxy(cx, cy, w, h)

def choose_axes(plane: str) -> Tuple[str,str]:
    plane = plane.upper()
    if plane == "XY": return ("x","y")
    if plane == "XZ": return ("x","z")
    if plane == "YZ": return ("y","z")
    raise ValueError("plane must be XY/XZ/YZ")

def choose_base_name(boxes: Dict[str, Tuple[float,float,float,float]]):
    return "base" if "base" in boxes else max(boxes.items(), key=lambda kv: (kv[1][2]-kv[1][0])*(kv[1][3]-kv[1][1]))[0]

def get_inner_base(base_xyxy, pad_px: float = 0.0, pad_ratio: float = 0.0):
    x0, y0, x1, y1 = base_xyxy
    w = max(0.0, x1 - x0); h = max(0.0, y1 - y0)
    ix0 = x0 + pad_px + w*pad_ratio
    iy0 = y0 + pad_px + h*pad_ratio
    ix1 = x1 - pad_px - w*pad_ratio
    iy1 = y1 - pad_px - h*pad_ratio
    if ix1 < ix0:
        m = (x0+x1)/2; ix0 = ix1 = m
    if iy1 < iy0:
        m = (y0+y1)/2; iy0 = iy1 = m
    return (ix0, iy0, ix1, iy1)

def uniq_key_from_path(path: Path) -> str:
    h = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8]
    return f"part_{h}"

def rect_intersection_area(r1, r2):
    """r = (x0,y0,x1,y1)"""
    x0 = max(r1[0], r2[0]); y0 = max(r1[1], r2[1])
    x1 = min(r1[2], r2[2]); y1 = min(r1[3], r2[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)

def need_align_due_to_overlap(parts, axes, threshold=0.10, metric="iomin"):

    proj_rects = []  # (name, (x0,y0,x1,y1), area)
    for it in parts:
        a0,b0,a1,b1 = aabb_world_xyxy_on_plane(it["aabb_center"], it["aabb_size"], axes)
        x0,y0,x1,y1 = min(a0,a1), min(b0,b1), max(a0,a1), max(b0,b1)
        area = max(0.0, (x1-x0)) * max(0.0, (y1-y0))
        proj_rects.append( (it["disp"], (x0,y0,x1,y1), area) )

    n = len(proj_rects)
    for i in range(n):
        for j in range(i+1, n):
            _, r1, a1 = proj_rects[i]
            _, r2, a2 = proj_rects[j]
            inter = rect_intersection_area(r1, r2)
            if inter <= 0.0 or a1 <= 0.0 or a2 <= 0.0:
                continue
            if metric == "iou":
                union = a1 + a2 - inter
                score = inter / max(1e-12, union)
            elif metric in ("iof_max",):
                score = inter / max(a1, a2)
            else:  # "iomin" or alias
                score = inter / min(a1, a2)
            if score >= threshold:
                return True
    return False

def align_percent(pred_boxes: Dict[str, Tuple[float, float, float, float]],
                  gt_percent: Dict[str, Dict[str, float]],
                  canvas_wh, size_mode="rel_ratio", size_clip=(0.5, 2.0),
                  base_pad_px=3.0, base_pad_ratio=0.02,
                  collision_axis="auto", gap=2.0):

    W, H = canvas_wh
    base_name = choose_base_name(pred_boxes)
    base_xyxy = pred_boxes[base_name]
    inner = get_inner_base(base_xyxy, base_pad_px, base_pad_ratio)

    out = dict(pred_boxes)

    for n, p in gt_percent.items():
        if n == base_name or n not in out: continue
        cx_t = inner[0] + p["cx_pct"] * max(1e-6, inner[2] - inner[0])
        cy_t = inner[1] + p["cy_pct"] * max(1e-6, inner[3] - inner[1])
        x0, y0, x1, y1 = out[n]
        w0 = max(1e-6, x1 - x0)
        h0 = max(1e-6, y1 - y0)

        if size_mode == "rel_ratio":
            w_t = p["w_pct"] * max(1e-6, inner[2] - inner[0])
            h_t = p["h_pct"] * max(1e-6, inner[3] - inner[1])
            fx = max(size_clip[0], min(size_clip[1], w_t / w0))
            fy = max(size_clip[0], min(size_clip[1], h_t / h0))
            w_new = w0 * fx
            h_new = h0 * fy
        elif size_mode == "keep":
            w_new, h_new = w0, h0
        else:  # absolute
            w_new = w_t
            h_new = h_t

        nx0, ny0, nx1, ny1 = cxcywh_to_xyxy(cx_t, cy_t, w_new, h_new)
        ix0, iy0, ix1, iy1 = inner
        nx0 = max(ix0, min(ix1, nx0))
        nx1 = max(ix0, min(ix1, nx1))
        ny0 = max(iy0, min(iy1, ny0))
        ny1 = max(iy0, min(iy1, ny1))
        out[n] = (nx0, ny0, nx1, ny1)

    if collision_axis == "auto":
        xs = [p["cx_pct"] for p in gt_percent.values()]
        ys = [p["cy_pct"] for p in gt_percent.values()]

        axis = "x" if (np.var(xs) if xs else 0.0) >= (np.var(ys) if ys else 0.0) else "y"
    else:
        axis = collision_axis

    aligned = out

    return aligned
