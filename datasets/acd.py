"""ACD prepared view-10 observations (GT masks, not the SAM photo pipeline).

Layout: <root>/<source>/<category>/<id>/{object_pwm.json,info/10_bbox.json,
rgbs/10.png,masks/10.png}. Legacy object_uwm.json is also accepted.
Adapted from the final ACD reference loader; preserves its soft-mask recipe.
"""
import os
import json
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from preprocess.soft_mask import make_soft_mask

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

class ACDInferenceDataset(Dataset):
    """
    Prepared ACD RGB/mask/bbox observations; never returns 'next'.
    """
    def __init__(
        self,
        root: str,
        split: str = "test",
        image_size: Tuple[int, int] = (224, 224),
        ho: int = 1,
        ha: int = 0,
        use_views: Optional[int] = 1,
        expand_single_view: bool = True,
        seed: int = 0,
        object_ids=None,
    ):
        super().__init__()
        assert ho >= 1
        self.root = os.path.join(root)
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
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.tr = []

        dataset_dirs = sorted(
            [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        )

        for data_name in dataset_dirs:
            dataset_dir = os.path.join(self.root, data_name)
            if not os.path.isdir(dataset_dir):
                continue

            for cat_name in sorted(os.listdir(dataset_dir)):
                cat_dir = os.path.join(dataset_dir, cat_name)
                if not os.path.isdir(cat_dir):
                    continue

                for obj_name in sorted(os.listdir(cat_dir)):
                    if object_ids is not None and f"{data_name}/{cat_name}/{obj_name}" not in object_ids:
                        continue
                    obj_dir = os.path.join(cat_dir, obj_name)
                    if not os.path.isdir(obj_dir):
                        continue

                    object_path = os.path.join(obj_dir, "object_pwm.json")
                    if not os.path.isfile(object_path):
                        object_path = os.path.join(obj_dir, "object_uwm.json")
                    bbox_path = os.path.join(obj_dir, "info", "10_bbox.json")

                    if not os.path.exists(object_path) or not os.path.exists(bbox_path):
                        continue

                    try:
                        with open(object_path, "r", encoding="utf-8") as f:
                            obj_json = json.load(f)
                        with open(bbox_path, "r", encoding="utf-8") as f:
                            bbox_json = json.load(f)
                            mask_path = bbox_json.get("mask_path", "")
                            rgb_path = bbox_json.get("rgb_path", "")
                            local_mask = os.path.join(obj_dir, "masks", "10.png")
                            local_rgb = os.path.join(obj_dir, "rgbs", "10.png")
                            mask_path = local_mask if os.path.isfile(local_mask) else mask_path
                            rgb_path = local_rgb if os.path.isfile(local_rgb) else rgb_path
                            if not os.path.isfile(mask_path) or not os.path.isfile(rgb_path):
                                raise FileNotFoundError(f"Missing ACD RGB/mask for {obj_dir}")
                    except Exception as e:
                        print(f"[WARN] failed reading {obj_dir}: {e}")
                        continue

                    diffuse_tree = obj_json.get("diffuse_tree", [])
                    bboxes = bbox_json.get("bboxes", {})

                    if not isinstance(diffuse_tree, list) or not isinstance(bboxes, dict):
                        print(f"[WARN] invalid format in {obj_dir}, skipping.")
                        continue

                    valid_joints = []
                    skip_object = False
                    for node in diffuse_tree:
                        name = node.get("name")
                        if name == "base":
                            continue

                        seg_id = node.get("seg_id")
                        jid = node.get("id")
                        if seg_id is None or jid is None:
                            continue

                        seg_key = str(seg_id)
                        if seg_key not in bboxes:
                            skip_object = True
                            break

                        bbox_entry = bboxes[seg_key]
                        base_bbox = bbox_entry.get("base_bbox_cxcywh_norm")
                        joint_bbox = bbox_entry.get("joint_bbox_cxcywh_norm")

                        if base_bbox is None or joint_bbox is None:
                            print(f"⚠️ skip {obj_name} ({cat_name}) in {data_name}: null bbox for seg_id={seg_id}")
                            skip_object = True
                            break

                        valid_joints.append({
                            "data_name": data_name,         # ✅ 数据集名
                            "cid": cat_name,                # 类别
                            "oid": obj_name,                # 对象 ID
                            "jid": jid,                     # joint ID
                            "seg_id": seg_id,               # mask segment ID
                            "name": name,                   # joint 名称
                            "base_bbox": base_bbox,
                            "joint_bbox": joint_bbox,
                            "mask_path": mask_path,
                            "rgb_path": rgb_path
                        })

                    if skip_object or len(valid_joints) == 0:
                        continue

                    for jinfo in valid_joints:
                        self.tr.append(jinfo)

        if len(self.tr) == 0:
            raise RuntimeError(f"No valid joints found under {self.root}")

        self.index = list(range(len(self.tr)))

        print(f"✅ Loaded {len(self.tr)} joints from {self.root} ({len(dataset_dirs)} datasets)")

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

    def _read_mask_label_bin(self, path: str, label_id: int, size: Tuple[int, int]) -> np.ndarray:
        """读取 mask，失败时返回 zeros mask。"""
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

    def _read_view_masks_and_masked_rgb(self, rgb_root: Optional[str], mask_root: str, label_id: int) -> Tuple[np.ndarray, np.ndarray]:
        v_rgb_dir = os.path.join(rgb_root)
        v_mask_dir = os.path.join(mask_root) if (mask_root is not None) else None
        
        rgbs = []
        mrgbs = []
       
        rgb = _read_img_uint8(v_rgb_dir, (self.W, self.H))
        mask = self._read_mask_label_bin(v_mask_dir, label_id, (self.W, self.H))
         
        m3 = np.repeat(mask[..., None], 3, axis=-1)
        alpha = make_soft_mask(m3, do_close=True, close_kernel=3, kernel=5, gauss_sigma=1.2)
        mrgb = (rgb * alpha).astype(np.uint8)
        rgbs.append(rgb)
        mrgbs.append(mrgb)
        return np.stack(rgbs, axis=0), np.stack(mrgbs, axis=0)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
  
 
        tr = self.tr[idx]

        seg_id=tr["seg_id"]
        rgb_path = tr["rgb_path"]
        mask_path = tr["mask_path"]
        base_bbox = np.array(tr["base_bbox"], dtype=np.float32)
        joint_bbox = np.array(tr["joint_bbox"], dtype=np.float32)
        K = 1
        obs_dict: Dict[str, Any] = {}
        rgb_stacks = []
        mrgb_stacks = []
        rgb_stack,mrgb_stack = self._read_view_masks_and_masked_rgb(rgb_path,mask_path, seg_id)

        rgb_stacks.append(rgb_stack)
        if mrgb_stack is None:
            mrgb_stack = rgb_stack.copy()
        mrgb_stacks.append(mrgb_stack)

        rgb_all = np.stack(rgb_stacks, axis=0)    # [K, ho, H, W, 3]
        mrgb_all = np.stack(mrgb_stacks, axis=0)  # [K, ho, H, W, 3]

        per_view_vecs = []
        base_bb = base_bbox
        joint_bb = joint_bbox
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
                "data_name": tr["data_name"],
                "cid": tr["cid"],
                "oid": tr["oid"],
                "jid": tr["jid"],
                "num_views": K,
                "frames": self.ho,
                "low_dim": self.low_dim_dim,
                "seg_id": seg_id,
                "jname": tr["name"],

            }
        }

        return sample
