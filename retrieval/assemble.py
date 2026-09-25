"""Assemble PWM object graphs directly from per-joint predictions."""
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

TYPE_ID_TO_NAME = {0: "revolute", 1: "prismatic", 2: "continuous", 3: "screw"}
ALLOWED_CORE_KEYS = ("id", "parent", "name", "children")


def assemble_object(graph, predictions):
    """Fill graph nodes in their original order from numerically sorted joints."""
    data = simplify_payload(graph)
    tree = data["diffuse_tree"]
    if not tree or [node["id"] for node in tree] != list(range(len(tree))):
        raise ValueError("Graph node IDs must be contiguous and match node order.")
    if any("handle" in str(node["name"]).lower() for node in tree):
        raise ValueError("Handles must already be merged into their parent parts.")
    bases = [node for node in tree if str(node["name"]).lower() == "base"]
    if len(bases) != 1:
        raise ValueError("Expected exactly one base node.")
    base = bases[0]
    if base["parent"] != -1:
        raise ValueError("The base must be the root node.")
    visited, pending = set(), [base["id"]]
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            raise ValueError("The graph must be a tree without repeated children or cycles.")
        visited.add(node_id)
        node = tree[node_id]
        for child in node["children"]:
            if not isinstance(child, int) or not 0 <= child < len(tree) or tree[child]["parent"] != node_id:
                raise ValueError("Inconsistent graph parent/child references.")
            pending.append(child)
    if len(visited) != len(tree):
        raise ValueError("All parts must be connected to the base.")
    parts = [node for node in tree if node is not base]
    if len(parts) != len(predictions):
        raise ValueError(f"Joint/node count mismatch: {len(predictions)} predictions, {len(parts)} nodes.")
    for node, prediction in zip(parts, predictions):
        joint, box, base_box = build_joint_aabbs_from_block(prediction, node["name"])
        node.update(joint=joint, aabb=box, aabb_base=base_box)
    box = _avg_base_aabb_from_nodes(parts)
    base.update(joint={"type": "fixed", "range": [0.0, 0.0],
                       "axis": {"direction": [0.0, 0.0, 0.0], "origin": [0.0, 0.0, 0.0]}},
                aabb=box, aabb_base={"center": list(box["center"]), "size": list(box["size"])})
    return data

def simplify_node(node: Dict[str, Any]) -> Dict[str, Any]:
    
    out: Dict[str, Any] = {
        "id": node.get("id"),
        "parent": node.get("parent"),
        "name": node.get("name"),
        "children": node.get("children", []) if isinstance(node.get("children", []), list) else [],
    }

    if "joint" in node:
        out["joint"] = {"type": None, "range": None}

    for key in node.keys():
        if key in ALLOWED_CORE_KEYS or key == "joint":
            continue
        if key not in out:
            out[key] = None

    return out


def simplify_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    meta = data.get("meta", {})
    tree_in = data.get("diffuse_tree", [])
    if not isinstance(tree_in, list):
        tree_in = []
    tree_out = [simplify_node(n) for n in tree_in]
    return {"meta": meta, "diffuse_tree": tree_out}


def _norm_vec3(v: Iterable[float]) -> List[float]:
    x, y, z = (float(vv) for vv in (list(v) + [0, 0, 1])[:3])
    n = math.sqrt(x*x + y*y + z*z)
    if n < 1e-8:
        return [0.0, 0.0, 1.0]
    return [x/n, y/n, z/n]

def _minmax_to_center_size(vmin: Iterable[float], vmax: Iterable[float]) -> Dict[str, List[float]]:
    a = [(list(vmin) + [0,0,0])[i] for i in range(3)]
    b = [(list(vmax) + [0,0,0])[i] for i in range(3)]
    center = [(ai+bi)*0.5 for ai,bi in zip(a,b)]
    size   = [ (bi-ai)     for ai,bi in zip(a,b)]
    return {"center": center, "size": size}

def _avg_vectors(vs: List[Iterable[float]]) -> List[float]:
    if not vs:
        return [0.0, 0.0, 0.0]
    acc = [0.0, 0.0, 0.0]; cnt = 0
    for v in vs:
        v = list(v)
        if len(v) >= 3:
            acc = [acc[i] + float(v[i]) for i in range(3)]
            cnt += 1
    if cnt == 0:
        return [0.0, 0.0, 0.0]
    return [acc[i]/cnt for i in range(3)]

def _ensure_aabb_from_block(block: Dict[str, Any]) -> Dict[str, List[float]]:
    mn, mx = block.get("aabb_min"), block.get("aabb_max")
    if isinstance(mn, (list, tuple)) and isinstance(mx, (list, tuple)):
        return _minmax_to_center_size(mn, mx)
    aabb = block.get("aabb")
    if isinstance(aabb, dict) and "center" in aabb and "size" in aabb:
        c, s = list(aabb["center"])[:3], list(aabb["size"])[:3]
        return {"center": c, "size": s}
    return {"center": [0.0, 0.0, 0.0], "size": [0.0, 0.0, 0.0]}

def _ensure_aabb_base_from_block(block: Dict[str, Any]) -> Dict[str, List[float]]:
    mn, mx = block.get("aabb_base_min"), block.get("aabb_base_max")
    if isinstance(mn, (list, tuple)) and isinstance(mx, (list, tuple)):
        return _minmax_to_center_size(mn, mx)
    return _ensure_aabb_from_block(block)

def _parse_joint_type(block: Dict[str, Any]) -> str:
    jt = block.get("type", 0)
    if isinstance(jt, int):
        return TYPE_ID_TO_NAME.get(jt, "revolute")
    if isinstance(jt, str):
        name = jt.strip().lower()
        return name if name in {"revolute","prismatic","continuous","screw","fixed"} else "revolute"
    return "revolute"

def _uninorm_range(jtype: str, span_val: float) -> List[float]:
    if jtype in ("revolute", "continuous"):
        hi = float(span_val) * 360.0
    else:
        hi = float(span_val)
    return [0.0, hi]

def _extract_pred_block(d: Dict[str, Any]) -> Dict[str, Any]:
    if "pred" in d and isinstance(d["pred"], dict):
        return d["pred"]
    if "gt" in d and isinstance(d["gt"], dict):
        return d["gt"]
    return d


def build_joint_aabbs_from_block(block: Dict[str, Any], node_name: Optional[str]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    jtype = _parse_joint_type(block)
    axis_dir = _norm_vec3(block.get("axis_dir", [0.0, 0.0, 1.0]))
    axis_ori = list(block.get("axis_ori", [0.0, 0.0, 0.0]))[:3]

    rng = block.get("range", [0.0])
    span = float(rng[0]) if isinstance(rng, (list, tuple)) and len(rng) > 0 else 0.0
    jrange = _uninorm_range(jtype, span)

    aabb = _ensure_aabb_from_block(block)
    aabb_base = _ensure_aabb_base_from_block(block)

    if jtype in ("prismatic", "screw") and (node_name is not None) and (node_name != "door"):
        axis_ori = list(aabb["center"])[:3]

    joint = {
        "type": jtype,
        "range": jrange,
        "axis": {"direction": axis_dir, "origin": axis_ori}
    }
    return joint, aabb, aabb_base

def _avg_base_aabb_from_nodes(nodes: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    centers, sizes = [], []
    for n in nodes:
        bb = n.get("aabb_base")
        if isinstance(bb, dict) and "center" in bb and "size" in bb:
            centers.append(bb["center"]); sizes.append(bb["size"])
        else:
            bb2 = n.get("aabb")
            if isinstance(bb2, dict) and "center" in bb2 and "size" in bb2:
                centers.append(bb2["center"]); sizes.append(bb2["size"])
    return {"center": _avg_vectors(centers), "size": _avg_vectors(sizes)}
