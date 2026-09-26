"""
Inference-only PartNet dataset — object/view/joint layout, no 'next' returned.

Path pattern assumed:
  masked images:
    <root>/<split>/<cat>/<obj>/view_id_XX/joint_YY/frame_000_view_ZZ_masked.png
  rgb images:
    <root>/<split>/<cat>/<obj>/view_id_XX/joint_YY/frame_000_view_ZZ.png
  bbox json (optional):
    <root>/<split>/<cat>/<obj>/view_id_XX/joint_YY/mask_renum_bbox.json

Notes:
- This dataset NEVER reads or returns "next".
- It returns samples with keys:
    sample["obs"][f"rgb_{v:02d}"]  -> raw obs rgb (tensor)
    sample["obs"][f"rgb_{v+1:02d}"]-> masked obs rgb (tensor) (or same as raw if masked missing)
    sample["obs"]["low_dim"]      -> low-dim tiled sequence (tensor)
  and sample["meta"] with metadata.
- No eval_fixed_view/frame options (removed).
"""

import os
import re
import json
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

_FRAME_RE = re.compile(r"frame_(\d{3})_view_(\d{1,2})(?:_masked)?\.png", re.IGNORECASE)
_VIEW_DIR_RE = re.compile(r"view[_\-]?id[_\-]?(\d+)", re.IGNORECASE)
_JOINT_DIR_RE = re.compile(r"joint[_\-]?(\d+)", re.IGNORECASE)
def build_type_codes(K: int, device=None) -> torch.Tensor:

    I = torch.eye(K, device=device)
    J = torch.ones(K, K, device=device) / K
    M = I - J
    U, S, Vh = torch.linalg.svd(M, full_matrices=True)
    C = U[:, :K-1]
    C = C / (C.norm(dim=1, keepdim=True) + 1e-8)
    return C  # [K, K-1]

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

class PartNetMobilityInferenceDataset(Dataset):
    """
    Inference-only dataset for object/view/joint layout, never returns 'next'.
    """
    def __init__(
        self,
        root: str,
        split: str = "test",
        image_size: Tuple[int, int] = (224, 224),
        ho: int = 1,
        ha: int = 0,                      # kept for compatibility but ignored for 'next'
        use_views: Optional[int] = 1,
        expand_single_view: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        assert ho >= 1
        self.root = os.path.join(root, split)
        self.W, self.H = image_size
        self.ho = int(ho)
        self.ha = int(ha)
        self.use_views = use_views
        self.expand_single_view = expand_single_view
        self.rng = np.random.RandomState(int(seed))
        self.action_dim = 8  + 12
        self.base_bbox_dim = 4
        self.joint_bbox_dim = 4
        self._per_view_low_dim = self.base_bbox_dim + self.joint_bbox_dim
        if (self.use_views is not None) and (self.use_views > 0):
            self.low_dim_dim = self._per_view_low_dim * int(self.use_views)
        else:
            self.low_dim_dim = self._per_view_low_dim

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Split root not found: {self.root}")

        self.tr: List[Dict[str, Any]] = []
        for cat_name in sorted(os.listdir(self.root), key=_natural_sort_key):
            cat_dir = os.path.join(self.root, cat_name)
            if not os.path.isdir(cat_dir):
                continue
            for obj_name in sorted(os.listdir(cat_dir), key=_natural_sort_key):
                obj_dir = os.path.join(cat_dir, obj_name)
                if not os.path.isdir(obj_dir):
                    continue
                joints_map: Dict[str, Dict[str, Any]] = {}

                for view_d in sorted(os.listdir(obj_dir), key=_natural_sort_key):
                    view_d_full = os.path.join(obj_dir, view_d)
                    if not os.path.isdir(view_d_full):
                        continue
                    mview = _VIEW_DIR_RE.match(view_d)
                    if mview:
                        view_id = int(mview.group(1))
                    else:
                        digits = re.findall(r"(\d+)", view_d)
                        view_id = int(digits[-1]) if digits else None
                        if view_id is None:
                            continue

                    for jname in sorted(os.listdir(view_d_full), key=_natural_sort_key):
                        jdir_full = os.path.join(view_d_full, jname)
                        if not os.path.isdir(jdir_full):
                            continue
                        if not _JOINT_DIR_RE.match(jname):
                            continue

                        files = sorted(os.listdir(jdir_full), key=_natural_sort_key)
                        fids = set()
                        for fn in files:
                            m = _FRAME_RE.match(fn)
                            if not m:
                                continue
                            fid = int(m.group(1))
                            fids.add(fid)

                        if len(fids) == 0:
                            continue

                        jm = joints_map.get(jname)
                        if jm is None:
                            jm = {"view_jdir": {}, "per_view_ids": {}}
                            joints_map[jname] = jm
                        jm["view_jdir"][view_id] = jdir_full
                        jm["per_view_ids"][view_id] = fids

                for jname, jm in joints_map.items():
                    view_ids_all = sorted(jm["view_jdir"].keys())
                    if len(view_ids_all) == 0:
                        continue
                    if len(view_ids_all) > 1:
                        common_ids = sorted(list(set.intersection(*[jm["per_view_ids"][v] for v in view_ids_all])))
                    else:
                        common_ids = sorted(list(jm["per_view_ids"][view_ids_all[0]]))

                    if len(common_ids) == 0:
                        continue

                    T = len(common_ids)
                    if T < self.ho:
                        continue

                    first_vid = view_ids_all[0]
                    first_jdir = jm["view_jdir"][first_vid]
                    parsed_geom = self._try_load_mask_renum_bbox(first_jdir)

                    jid = int(re.sub(r"[^\d]", "", jname)) if re.search(r"\d+", jname) else 0

                    self.tr.append({
                        "oid": obj_name,
                        "jid": jid,
                        "view_jdir_map": jm["view_jdir"],   # {view_id: jdir_full}
                        "views": view_ids_all,              # list of ints
                        "frame_ids": common_ids,            # list of ints
                        "T": T,
                        "meta_sample": parsed_geom,
                        "jname": jname,
                    })

        if len(self.tr) == 0:
            raise RuntimeError(f"No trajectories found under {self.root} (ho={self.ho})")

        self.index: List[Tuple[int, int, Optional[int]]] = []
        for tidx, tr in enumerate(self.tr):
            T = tr["T"]
            V = len(tr["views"])
            time_iters = range(0, max(T - self.ho + 1, 0))
            if (self.use_views == 1) and self.expand_single_view:
                for t0 in time_iters:
                    for vidx in range(V):
                        self.index.append((tidx, t0, vidx))
            else:
                for t0 in time_iters:
                    self.index.append((tidx, t0, None))

        if len(self.index) == 0:
            raise RuntimeError(f"No samples (index) constructed under {self.root} for ho={self.ho}")

    def _try_load_mask_renum_bbox(self, jdir: str) -> Dict[str, Any]:
        candidates = [
            os.path.join(jdir, "mask_renum_bbox.json"),
            os.path.join(jdir, "mask_renum_bbox.JSON"),
            os.path.join(os.path.dirname(jdir), "mask_renum_bbox.json"),
        ]
        meta = None
        for p in candidates:
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    break
                except Exception:
                    meta = None
        return self._parse_meta_geom(meta)

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

    def _parse_meta_geom(self, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        base_bb = np.zeros((self.base_bbox_dim,), dtype=np.float32)
        joint_bb = np.zeros((self.joint_bbox_dim,), dtype=np.float32)
        if isinstance(meta, dict):
            base_bb = self._safe_array_from_meta_field(meta.get("base_bbox_cxcywh_norm", None), self.base_bbox_dim)
            joint_bb = self._safe_array_from_meta_field(meta.get("joint_bbox_cxcywh_norm", None), self.joint_bbox_dim)
        return {
            "base_bbox_cxcywh_norm": base_bb.astype(np.float32),
            "joint_bbox_cxcywh_norm": joint_bb.astype(np.float32),
        }

    def _frame_filename(self, fid: int, vid: int, masked: bool = False) -> str:
        if masked:
            return f"frame_{fid:03d}_view_{vid:02d}_masked.png"
        else:
            return f"frame_{fid:03d}_view_{vid:02d}.png"

    def _read_view_rgb_stack(self, jdir: str, vid: int, fid_list: List[int]) -> np.ndarray:
        imgs = []
        for fid in fid_list:
            p = os.path.join(jdir, self._frame_filename(fid, vid, masked=False))
            imgs.append(_read_img_uint8(p, (self.W, self.H)))
        return np.stack(imgs, axis=0)

    def _read_view_masked_stack(self, jdir: str, vid: int, fid_list: List[int]) -> Optional[np.ndarray]:
        imgs = []
        for fid in fid_list:
            p = os.path.join(jdir, self._frame_filename(fid, vid, masked=True))
            if not os.path.isfile(p):
                return None
            imgs.append(_read_img_uint8(p, (self.W, self.H)))
        if len(imgs) == 0:
            return None
        return np.stack(imgs, axis=0)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tr_idx, t0, view_fixed = self.index[idx]
        tr = self.tr[tr_idx]
        view_list = tr["views"]               # list of ints
        frame_ids = tr["frame_ids"]
        T = tr["T"]

        if view_fixed is not None:
            selected_views = [view_list[view_fixed]]
        elif (self.use_views is None) or (self.use_views >= len(view_list)):
            selected_views = view_list.copy()
        else:
            sel = self.rng.choice(len(view_list), int(self.use_views), replace=False)
            selected_views = [view_list[i] for i in sel]
        K = len(selected_views)

        fids_obs = frame_ids[t0: t0 + self.ho]
        if len(fids_obs) < self.ho:
            last = frame_ids[-1]
            while len(fids_obs) < self.ho:
                fids_obs.append(last)

        obs_dict: Dict[str, Any] = {}
        rgb_stacks = []
        mrgb_stacks = []

        for vi, vid in enumerate(selected_views):
            view_jdir_map = tr["view_jdir_map"]
            if vid not in view_jdir_map:
                raise RuntimeError(f"View id {vid} not found for tr_idx {tr_idx}")
            jdir = view_jdir_map[vid]

            rgb_stack = self._read_view_rgb_stack(jdir, vid, fids_obs)   # [ho, H, W, 3]
            rgb_stacks.append(rgb_stack)

            mrgb_stack = self._read_view_masked_stack(jdir, vid, fids_obs)
            if mrgb_stack is None:
                mrgb_stack = rgb_stack.copy()
            mrgb_stacks.append(mrgb_stack)

        rgb_all = np.stack(rgb_stacks, axis=0)    # [K, ho, H, W, 3]
        mrgb_all = np.stack(mrgb_stacks, axis=0)  # [K, ho, H, W, 3]

        per_view_vecs = []
        for vid in selected_views:
            parsed = tr.get("meta_sample", {})
            base_bb = parsed.get("base_bbox_cxcywh_norm", np.zeros((self.base_bbox_dim,), dtype=np.float32))
            joint_bb = parsed.get("joint_bbox_cxcywh_norm", np.zeros((self.joint_bbox_dim,), dtype=np.float32))
            base_bb = self._safe_array_from_meta_field(base_bb, self.base_bbox_dim)
            joint_bb = self._safe_array_from_meta_field(joint_bb, self.joint_bbox_dim)
            per_view_vec = np.concatenate([base_bb, joint_bb], axis=0).astype(np.float32)
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

        low_seq = np.tile(low_vec[None, :], (self.ho, 1))  # [ho, Dlow]

        for v in range(K):
            obs_dict[f"rgb_{v:02d}"] = torch.from_numpy(rgb_all[v])
            obs_dict[f"rgb_{v+1:02d}"] = torch.from_numpy(mrgb_all[v])

        obs_dict["low_dim"] = torch.from_numpy(low_seq)

        sample: Dict[str, Any] = {
            "obs": obs_dict,
            "meta": {
                "oid": tr["oid"],
                "jid": tr["jid"],
                "num_views": K,
                "frames": self.ho,
                "low_dim": self.low_dim_dim,
                "tr_idx": tr_idx,
                "t0": t0,
                "views": selected_views,
                "frame_ids": tr["frame_ids"],
                "jdir_sample": tr["view_jdir_map"],
                "jname": tr.get("jname", ""),
            }
        }

        return sample
