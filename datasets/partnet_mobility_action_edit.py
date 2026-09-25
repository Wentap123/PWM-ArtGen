import os
import re
import json
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

_FRAME_RE = re.compile(r"frame_(\d{3})_view_(\d{2})\.png")

def _natural_sort_key(s: str):

    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', s)]

def _read_img_uint8(path: str, size: Tuple[int, int]) -> np.ndarray:

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im = im.resize(size, Image.Resampling.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)
            return arr
    except Exception:
        H, W = size[1], size[0]
        return np.zeros((H, W, 3), dtype=np.uint8)

def _read_mask_label_bin(path: str, label_id: int, size: Tuple[int, int]) -> np.ndarray:

    try:
        with Image.open(path) as im:
            if im.mode == "P" or im.mode in ("I", "L"):
                im = im.resize(size, Image.Resampling.NEAREST)
                arr = np.asarray(im)
            else:
                im = im.convert("L").resize(size, Image.Resampling.NEAREST)
                arr = np.asarray(im)
            return (arr == int(label_id)).astype(np.uint8)
    except Exception:
        H, W = size[1], size[0]
        return np.zeros((H, W), dtype=np.uint8)

def build_type_codes(K: int, device=None) -> torch.Tensor:

    I = torch.eye(K, device=device)
    J = torch.ones(K, K, device=device) / K
    M = I - J
    U, S, Vh = torch.linalg.svd(M, full_matrices=True)
    C = U[:, :K-1]
    C = C / (C.norm(dim=1, keepdim=True) + 1e-8)
    return C  # [K, K-1]

def _ensure_odd(k: int) -> int:
    k = int(max(1, k))
    return k if k % 2 == 1 else k + 1

def make_soft_mask(
    hw3,
    kernel: int = 5,
    strength: float = 0.7,
    out_dtype=None,
    use_gauss: bool = True,
    gauss_sigma: float = 1.0,
    do_close: bool = False,
    close_kernel: int = 3
):
    is_numpy = isinstance(hw3, np.ndarray)
    if is_numpy:
        x = torch.from_numpy(hw3)
        in_dtype = hw3.dtype
    else:
        x = hw3
        in_dtype = hw3.dtype

    assert x.ndim == 3, f"expect (H,W,C), got {tuple(x.shape)}"
    H, W, C = x.shape

    x = x.to(torch.float32)
    if x.max() > 1.0:
        x = x / 255.0
    x = x.clamp_(0, 1)

    m = x.permute(2, 0, 1).unsqueeze(0)

    if do_close and close_kernel and close_kernel > 1:
        ck = _ensure_odd(close_kernel)
        pad = ck // 2
        m = F.max_pool2d(m, kernel_size=ck, stride=1, padding=pad)
        m = 1.0 - F.max_pool2d(1.0 - m, kernel_size=ck, stride=1, padding=pad)

    if kernel and kernel > 1:
        k = _ensure_odd(kernel)
        pad = k // 2
        if use_gauss:
            r = torch.arange(-pad, pad + 1, device=m.device, dtype=m.dtype)
            g1 = torch.exp(-0.5 * (r / (gauss_sigma + 1e-6))**2)
            g1 = g1 / g1.sum()
            kx = g1.view(1, 1, 1, k).repeat(C, 1, 1, 1)  # (C,1,1,k)
            ky = g1.view(1, 1, k, 1).repeat(C, 1, 1, 1)  # (C,1,k,1)
            y = F.pad(m, (pad, pad, 0, 0), mode="reflect")
            y = F.conv2d(y, kx, groups=C)
            y = F.pad(y, (0, 0, pad, pad), mode="reflect")
            m_soft = F.conv2d(y, ky, groups=C)
        else:
            w = torch.ones((C, 1, k, k), device=m.device, dtype=m.dtype) / (k * k)
            y = F.pad(m, (pad, pad, pad, pad), mode="reflect")
            m_soft = F.conv2d(y, w, groups=C)
        m_soft = m_soft.clamp(0, 1)
    else:
        m_soft = m

    m_hard = (m >= 0.5).to(m.dtype)
    s = float(max(0.0, min(1.0, strength)))
    out = ((1.0 - s) * m_hard + s * m_soft).clamp(0, 1)

    out = out.squeeze(0).permute(1, 2, 0)

    if out_dtype is not None:
        if is_numpy:
            out = out.to(torch.float32) if out_dtype == np.float32 else out.to(torch.float32)
        else:
            out = out.to(out_dtype)
    else:
        if is_numpy:
            pass
        else:
            out = out.to(in_dtype if out.dtype.is_floating_point else torch.float32)

    if is_numpy:
        return out.cpu().numpy()
    return out
class PartNetMobilityActionsDataset(Dataset):

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        ho: int = 1,
        ha: int = 1,
        use_views: Optional[int] = 1,
        num_types: int = 2,
        expand_single_view: bool = True,
        seed: int = 0,
        eval_fixed_view_idx: Optional[List[int]] = None, # "view_idx_00"
        eval_fixed_frame_list: Optional[List[int]] = None,#[0,13]
    ):
        super().__init__()
        assert ho >= 1 and ha >= 1
        self.root = os.path.join(root, split)
        self.W, self.H = image_size
        self.ho, self.ha = ho, ha
        self.use_views = use_views
        self.num_types = num_types
        self.expand_single_view = expand_single_view
        self.rng = np.random.RandomState(seed)

        self.cam_pose_dim = 0    # q(4)+t(3)+f(2)
        self.base_bbox_dim = 4
        self.joint_bbox_dim = 4
        self._per_view_low_dim = self.cam_pose_dim + self.base_bbox_dim + self.joint_bbox_dim

        if (self.use_views is not None) and (self.use_views > 0):
            self.low_dim_dim = self._per_view_low_dim * int(self.use_views)
        else:
            self.low_dim_dim = self._per_view_low_dim

        self.type_codes = build_type_codes(num_types)
        self.D_t = int(self.type_codes.shape[1])

        self.action_dim = 7 + self.D_t + 12 #= 3+3+1+1+D_t+12

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Split root not found: {self.root}")

        self.tr: List[Dict[str, Any]] = []
        for cat_name in sorted(os.listdir(self.root), key=_natural_sort_key):
            cat_dir = os.path.join(self.root, cat_name) # cat_dir
            if not os.path.isdir(cat_dir):
                continue
            for obj_name in sorted(os.listdir(cat_dir), key=_natural_sort_key): #obj_dir
                obj_dir = os.path.join(cat_dir, obj_name)
                if not os.path.isdir(obj_dir):
                    continue
                for jname in sorted(os.listdir(obj_dir), key=_natural_sort_key):
                    if not jname.startswith("joint_"):
                        continue
                    jdir = os.path.join(obj_dir, jname)
                    rgb_root = os.path.join(jdir, "RGB")
                    mask_root = None
                    for cand in ("Mask", "mask"):
                        p = os.path.join(jdir, cand)
                        if os.path.isdir(p):
                            mask_root = p
                            break
                    if not os.path.isdir(rgb_root):
                        continue

                    if eval_fixed_view_idx is not None:
                        idx = eval_fixed_view_idx
                        prefixes = idx if isinstance(idx, (list, tuple, set)) else [idx]
                        prefixes = [f"view_idx_{p:02d}" if isinstance(p, int) else str(p) for p in prefixes]
                        view_dirs = [
                            d for d in sorted(os.listdir(rgb_root), key=_natural_sort_key)
                            if any(d.startswith(p) for p in prefixes)
                        ]
                    else:
                        view_dirs = [d for d in sorted(os.listdir(rgb_root), key=_natural_sort_key) if d.startswith("view_idx_")]
                    if len(view_dirs) == 0:
                        continue

                    per_view_ids = []
                    for v in view_dirs:
                        vdir = os.path.join(rgb_root, v)
                        ids = set()
                        for fn in os.listdir(vdir):
                            m = _FRAME_RE.match(fn)
                            if m:
                                ids.add(int(m.group(1)))
                        per_view_ids.append(ids)
                    if len(per_view_ids) == 0:
                        continue
                    if len(per_view_ids) > 1:
                        common_ids = sorted(list(set.intersection(*per_view_ids)))
                    else:
                        common_ids = sorted(list(per_view_ids[0]))
                    if len(common_ids) == 0:
                        continue

                    if eval_fixed_frame_list is not None:
                        filtered_ids = [fid for fid in common_ids if fid in eval_fixed_frame_list]
                        if len(filtered_ids) == 0:
                            print(f"Warning: No frames from {eval_fixed_frame_list} found for joint {jname}. Skipping.")
                            continue
                        common_ids = filtered_ids
                    T = len(common_ids)
                    if not ((self.use_views == 1) and (self.ho == 1)):
                        if T < (self.ho + self.ha):
                            continue

                    meta_path = os.path.join(rgb_root, view_dirs[0], "info", "meta.json")
                    meta_sample_raw = self._read_meta(meta_path)
                    if meta_sample_raw is None:
                        continue
                    if meta_sample_raw.get("aabb") is None or meta_sample_raw.get("aabb_base") is None:
                        continue
                    if meta_sample_raw.get("base_bbox_cxcywh_norm") is None or meta_sample_raw.get("joint_bbox_cxcywh_norm") is None:
                        continue
                    parsed_geom = self._parse_meta_geom(meta_sample_raw)
                    if parsed_geom is None:
                        continue

                    try:
                        jid = int(jname.split("_")[1])
                    except Exception:
                        jid = 0

                    expected_label = self._get_expected_label(obj_dir, jname)
                    if expected_label is None:
                        print(f"[WARN] missing expected label for {obj_name}/{jname} under {obj_dir}, skip this joint.")
                        continue

                    self.tr.append({
                        "oid": obj_name,
                        "jid": jid,
                        "jdir": jdir,
                        "rgb_root": rgb_root,
                        "mask_root": mask_root,
                        "views": view_dirs,
                        "frame_ids": common_ids,
                        "T": T,
                        "meta_sample": parsed_geom,
                        "expected_label": int(expected_label),
                    })
        self.index: List[Tuple[int, int, Optional[int]]] = []
        for tidx, tr in enumerate(self.tr):
            T = tr["T"]
            V = len(tr["views"])
            if (self.use_views == 1) and (self.ho == 1):
                time_iters = range(0, max(T - 1, 0))
            else:
                limit = T - (self.ho + self.ha)
                time_iters = range(0, max(limit + 1, 0))
            if (self.use_views == 1) and self.expand_single_view:
                for t0 in time_iters:
                    for vidx in range(V):
                        self.index.append((tidx, t0, vidx))
            else:
                for t0 in time_iters:
                    self.index.append((tidx, t0, None))

        if len(self.index) == 0:
            raise RuntimeError(f"No samples found under {self.root} for ho={ho}, ha={ha}")

        unique_oids = []
        for tr in self.tr:
            oid = tr.get("oid")
            if oid not in unique_oids:
                unique_oids.append(oid)
        self.oid_to_idx = {oid: i for i, oid in enumerate(unique_oids)}

    def _read_meta(self, path: str) -> Optional[Dict[str, Any]]:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _safe_array_from_meta_field(self, val, expected_len: int) -> np.ndarray:
        if expected_len <= 0:
            return np.zeros((0,), dtype=np.float32)
        if val is None:
            return np.zeros((expected_len,), dtype=np.float32)
        try:
            arr = np.array(val, dtype=np.float32)
        except Exception:
            return np.zeros((expected_len,), dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(-1)
        arr = arr.ravel()
        out = np.zeros((expected_len,), dtype=np.float32)
        copy_len = min(arr.size, expected_len)
        if copy_len > 0:
            out[:copy_len] = arr[:copy_len].astype(np.float32)
        return out

    def _parse_meta_geom(self, meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:

        if not isinstance(meta, dict):
            return None
        required_top_keys = ["joint", "aabb"] #["joint", "aabb", "qtf_list"]
        for k in required_top_keys:
            if k not in meta:
                return None

        joint = meta.get("joint", {})
        name = meta.get("name", "unknown")
        if not isinstance(joint, dict):
            return None
        jtype = joint.get("type", "revolute")
        type_map = {"revolute": 0, "prismatic": 1, "continuous": 2, "screw": 3}
        type_id = type_map.get(jtype, 0)

        axis = joint.get("axis", {})
        axis_dir = np.array(axis.get("direction", [0.0, 0.0, 1.0]), dtype=np.float32)
        nrm = np.linalg.norm(axis_dir) + 1e-8
        axis_dir = axis_dir / nrm if nrm > 0 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

        flip = (np.sum(axis_dir > 0) < np.sum(-axis_dir > 0))
        if flip:
            axis_dir = -axis_dir

        axis_ori = np.array(axis.get("origin", [0.0, 0.0, 0.0]), dtype=np.float32)

        jr = joint.get("range", [0.0, 0.0])
        if isinstance(jr, (list, tuple)) and len(jr) >= 2:
            lo, hi = jr[0], jr[1]
        else:
            try:
                lo = float(jr[0]); hi = float(jr[0])
            except Exception:
                lo, hi = 0.0, 0.0
        span_norm = np.array([hi / 360.0], dtype=np.float32) if jtype in ("revolute", "continuous") else np.array([hi], dtype=np.float32)

        if flip:
            span_norm = -span_norm

        aabb = meta.get("aabb", {})
        if not isinstance(aabb, dict):
            return None
        aabb_center = self._safe_array_from_meta_field(aabb.get("center", None), 3)
        aabb_size = self._safe_array_from_meta_field(aabb.get("size", None), 3)

        aabb_base = meta.get("aabb_base", {})
        if not isinstance(aabb_base, dict):
            aabb_base_center = np.zeros((3,), dtype=np.float32)
            aabb_base_size = np.zeros((3,), dtype=np.float32)
        else:
            aabb_base_center = self._safe_array_from_meta_field(aabb_base.get("center", None), 3)
            aabb_base_size = self._safe_array_from_meta_field(aabb_base.get("size", None), 3)

        if (jtype == 'prismatic' or jtype == 'screw') and name != 'door':
            axis_ori = aabb_center

        base_bb = self._safe_array_from_meta_field(meta.get("base_bbox_cxcywh_norm", None), self.base_bbox_dim)
        joint_bb = self._safe_array_from_meta_field(meta.get("joint_bbox_cxcywh_norm", None), self.joint_bbox_dim)

        aabb_max = aabb_center + aabb_size / 2
        aabb_min = aabb_center - aabb_size / 2

        aabb_base_max = aabb_base_center + aabb_base_size / 2
        aabb_base_min = aabb_base_center - aabb_base_size / 2
        return {
            "type": type_id,
            "axis_dir": axis_dir,
            "axis_ori": axis_ori,
            "range": span_norm,
            "aabb_max": aabb_max,
            "aabb_min": aabb_min,
            "aabb_base_max": aabb_base_max,
            "aabb_base_min": aabb_base_min,
            "qtf_list": [],
            "base_bbox_cxcywh_norm": base_bb,
            "joint_bbox_cxcywh_norm": joint_bb,
        }

    def _read_view_rgb_stack(self, rgb_root: str, view: str, fid_list: List[int]) -> np.ndarray:
        vdir = os.path.join(rgb_root, view)
        m = re.match(r"view_idx_(\d+)", view)
        v_id = int(m.group(1)) if m else 0
        imgs = []
        for fid in fid_list:
            fn = f"edited_frame_{fid:03d}_view_{v_id:02d}.png"
            p = os.path.join(vdir, fn)
            imgs.append(_read_img_uint8(p, (self.W, self.H)))
        return np.stack(imgs, axis=0)

    def _read_view_masks_and_masked_rgb(self, mask_root: Optional[str], rgb_root: str, view: str, fid_list: List[int], label_id: int) -> Tuple[np.ndarray, np.ndarray]:
        v_rgb_dir = os.path.join(rgb_root, view)
        v_mask_dir = os.path.join(mask_root, view) if (mask_root is not None) else None
        m = re.match(r"view_idx_(\d+)", view)
        v_id = int(m.group(1)) if m else 0
        rgbs = []
        mrgbs = []
        for fid in fid_list:
            fn_edited = f"edited_frame_{fid:03d}_view_{v_id:02d}.png"
            fn = f"frame_{fid:03d}_view_{v_id:02d}.png"
            rgb_p = os.path.join(v_rgb_dir, fn_edited)
            rgb = _read_img_uint8(rgb_p, (self.W, self.H))
            if v_mask_dir is not None:
                mask_p = os.path.join(v_mask_dir, fn)
                mask = _read_mask_label_bin(mask_p, label_id, (self.W, self.H))
            else:
                mask = np.zeros((self.H, self.W), dtype=np.uint8)
            m3 = np.repeat(mask[..., None], 3, axis=-1)
            alpha = make_soft_mask(m3, do_close=True, close_kernel=3, kernel=5, gauss_sigma=1.2)
            mrgb = (rgb * alpha).astype(np.uint8)
            rgbs.append(rgb)
            mrgbs.append(mrgb)
        return np.stack(rgbs, axis=0), np.stack(mrgbs, axis=0)

    def _load_expected_map_for_object(self, object_dir: str):

        if not hasattr(self, "_expected_map_cache"):
            self._expected_map_cache = {}  # {object_dir: (mapping, json_path)}

        if object_dir in self._expected_map_cache:
            return self._expected_map_cache[object_dir]

        json_files = sorted(
            [f for f in os.listdir(object_dir) if f.endswith("_joints_info.json")],
            key=_natural_sort_key
        )
        if not json_files:
            mapping, jp = {}, None
            self._expected_map_cache[object_dir] = (mapping, jp)
            return mapping, jp

        jp = os.path.join(object_dir, json_files[0])
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            mapping, jp_warn = {}, jp
            self._expected_map_cache[object_dir] = (mapping, jp_warn)
            return mapping, jp_warn

        names = data.get("new_joint_name_list")
        gids  = data.get("new_joint_global_id_list")
        if not isinstance(names, list) or not isinstance(gids, list) or len(names) != len(gids):
            names = data.get("joint_name_list")
            gids  = data.get("joint_global_id_list")

        mapping = {}
        if isinstance(names, list) and isinstance(gids, list) and len(names) == len(gids):
            for n, g in zip(names, gids):
                if isinstance(n, str) and isinstance(g, int):
                    mapping[n] = int(g)

        self._expected_map_cache[object_dir] = (mapping, jp)
        return mapping, jp

    def _get_expected_label(self, object_dir: str, joint_name: str):
        mapping, _ = self._load_expected_map_for_object(object_dir)
        return mapping.get(joint_name, None)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tr_idx, t0, view_fixed = self.index[idx]
        tr = self.tr[tr_idx]
        views_all = tr["views"]
        frame_ids = tr["frame_ids"]
        T = tr["T"]
        jid = tr["jid"]

        if view_fixed is not None:
            selected_views = [views_all[view_fixed]]
        elif (self.use_views is None) or (self.use_views >= len(views_all)):
            selected_views = views_all
        else:
            sel = self.rng.choice(len(views_all), self.use_views, replace=False)
            selected_views = [views_all[i] for i in sel]
        K = len(selected_views)

        fids_obs = frame_ids[t0: t0 + self.ho]
        if (self.use_views == 1) and (self.ho == 1):
            fids_next = [frame_ids[-1]]
        else:
            start_next = t0 + self.ha
            end_next = start_next + self.ho
            fids_next = frame_ids[start_next: end_next] if start_next < len(frame_ids) else [frame_ids[-1]]

        obs_dict: Dict[str, Any] = {}
        next_dict: Dict[str, Any] = {}
        rgb_stacks = []
        mrgb_stacks = []
        next_stacks = []

        for vi, v in enumerate(selected_views):
            rgb_stack = self._read_view_rgb_stack(tr["rgb_root"], v, fids_obs)   # [T,H,W,3]
            rgb_stacks.append(rgb_stack)
            rgb_n = self._read_view_rgb_stack(tr["rgb_root"], v, fids_next)     # [Tn,H,W,3]
            next_stacks.append(rgb_n)
            rgb_v, mrgb_v = self._read_view_masks_and_masked_rgb(tr["mask_root"], tr["rgb_root"], v, fids_obs, tr["expected_label"])
            mrgb_stacks.append(mrgb_v)

        rgb_all = np.stack(rgb_stacks, axis=0)
        mrgb_all = np.stack(mrgb_stacks, axis=0)
        next_all = np.stack(next_stacks, axis=0)

        per_view_vecs = []
        for v in selected_views:
            meta_v = self._read_meta(os.path.join(tr["rgb_root"], v, "info", "meta.json"))
            parsed = self._parse_meta_geom(meta_v) if meta_v is not None else tr["meta_sample"]
            if parsed is None:
                parsed = tr["meta_sample"]

            if self.cam_pose_dim > 0:
                qtf_list = parsed.get("qtf_list", []) if isinstance(parsed, dict) else []
                if len(qtf_list) > 0 and qtf_list[0].get("q") is not None:
                    ent = qtf_list[0]
                    parts = []
                    if ent.get("q") is not None:
                        parts += list(ent["q"].reshape(-1).tolist())
                    if ent.get("t") is not None:
                        parts += list(ent["t"].reshape(-1).tolist())
                    if ent.get("f") is not None:
                        parts += list(ent["f"].reshape(-1).tolist())
                    cam = np.array(parts, dtype=np.float32)
                else:
                    cam = np.zeros((self.cam_pose_dim,), dtype=np.float32)
                if cam.size != self.cam_pose_dim:
                    tmp = np.zeros((self.cam_pose_dim,), dtype=np.float32)
                    tmp[:min(cam.size, self.cam_pose_dim)] = cam.ravel()[:self.cam_pose_dim]
                    cam = tmp

            base_bb = parsed.get("base_bbox_cxcywh_norm", None)
            joint_bb = parsed.get("joint_bbox_cxcywh_norm", None)
            base_bb = self._safe_array_from_meta_field(base_bb, self.base_bbox_dim)
            joint_bb = self._safe_array_from_meta_field(joint_bb, self.joint_bbox_dim)

            if self.cam_pose_dim == 0:
                per_view_vec = np.concatenate([base_bb, joint_bb], axis=0).astype(np.float32)  # per-view low-dim
            else:
                per_view_vec = np.concatenate([cam, base_bb, joint_bb], axis=0).astype(np.float32)  # per-view low-dim
            per_view_vecs.append(per_view_vec)

        if len(per_view_vecs) > 0:
            low_vec = np.concatenate(per_view_vecs, axis=0).astype(np.float32)
        else:
            low_vec = np.zeros((self._per_view_low_dim * K,), dtype=np.float32)

        if low_vec.size < self.low_dim_dim:
            tmp = np.zeros((self.low_dim_dim,), dtype=np.float32)
            tmp[:low_vec.size] = low_vec
            low_vec = tmp
        elif low_vec.size > self.low_dim_dim:
            low_vec = low_vec[:self.low_dim_dim]

        low_seq = np.tile(low_vec[None, :], (self.ho, 1))  # [T, Dlow]

        for v in range(K):
            obs_dict[f"rgb_{v:02d}"] = torch.from_numpy(rgb_all[v])
            obs_dict[f"rgb_{v+1:02d}"] = torch.from_numpy(mrgb_all[v])
            next_dict[f"rgb_{v:02d}"] = torch.from_numpy(rgb_all[v])
            next_dict[f"rgb_{v+1:02d}"] = torch.from_numpy(next_all[v])

        obs_dict["low_dim"] = torch.from_numpy(low_seq)
        next_dict["low_dim"] = torch.from_numpy(low_seq)

        g = tr["meta_sample"]
        t_idx = int(g.get("type", 0))
        type_code = self.type_codes[t_idx].numpy().astype(np.float32)
        axis_dir = g.get("axis_dir", np.zeros((3,), dtype=np.float32))
        axis_ori = g.get("axis_ori", np.zeros((3,), dtype=np.float32))
        span = g.get("range", np.zeros((1,), dtype=np.float32))

        fid0 = fids_obs[0]
        idx0 = frame_ids.index(fid0)
        alpha = idx0 / max(T - 1, 1)
        pos = np.array([alpha * float(span[0])], dtype=np.float32)

        aabb_max = g.get("aabb_max", np.zeros((3,), dtype=np.float32))
        aabb_min = g.get("aabb_min", np.zeros((3,), dtype=np.float32))
        aabb_base_max = g.get("aabb_base_max", np.zeros((3,), dtype=np.float32))
        aabb_base_min = g.get("aabb_base_min", np.zeros((3,), dtype=np.float32))
        aabb_max = np.array(aabb_max, dtype=np.float32).reshape(3,)
        aabb_min = np.array(aabb_min, dtype=np.float32).reshape(3,)
        aabb_base_max = np.array(aabb_base_max, dtype=np.float32).reshape(3,)
        aabb_base_min = np.array(aabb_base_min, dtype=np.float32).reshape(3,)

        action_vec = np.concatenate([
            axis_dir.reshape(3,),
            axis_ori.reshape(3,),
            span.reshape(1,),
            type_code.reshape(self.D_t,),
            aabb_max,
            aabb_min,
            aabb_base_max,
            aabb_base_min
        ], axis=0)
        if action_vec.shape[0] != self.action_dim:
            raise RuntimeError(f"action vector dim {action_vec.shape[0]} != expected action_dim {self.action_dim}")

        actions = np.tile(action_vec[None, :], (self.ha, 1)).astype(np.float32)

        geom = {
            "type": torch.tensor(t_idx, dtype=torch.long),
            "axis_dir": torch.tensor(axis_dir, dtype=torch.float32),
            "axis_ori": torch.tensor(axis_ori, dtype=torch.float32),
            "range": torch.tensor(span, dtype=torch.float32),
            "aabb_max": torch.tensor(aabb_max, dtype=torch.float32),
            "aabb_min": torch.tensor(aabb_min, dtype=torch.float32),
            "aabb_base_max": torch.tensor(aabb_base_max, dtype=torch.float32),
            "aabb_base_min": torch.tensor(aabb_base_min, dtype=torch.float32),
            "type_code": torch.tensor(type_code, dtype=torch.float32),
            "oid_idx": torch.tensor(int(self.oid_to_idx.get(tr["oid"], -1)), dtype=torch.long),
        }

        sample = {
            "obs": obs_dict,
            "next": next_dict,
            "actions": torch.from_numpy(actions),
            "action_mask": torch.tensor(0, dtype=torch.bool),
            "geom": geom,
            "meta": {
                "oid": tr["oid"],
                "jid": tr["jid"],
                "num_views": K,
                "frames": self.ho,
                "low_dim": self.low_dim_dim,
                "action_dim": self.action_dim,
                "tr_idx": tr_idx,
                "t0": t0,
                "views": selected_views,
            }
        }
        return sample
