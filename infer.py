"""Dataset inference: export geometry predictions and generated RGB images."""

import os, json, argparse, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, default_collate
import numpy as np
from PIL import Image

from datasets.inference import (
    PartNetMobilityInferenceDataset,
    build_type_codes,
)
from models.pwm.obs_encoder import PWMObservationEncoder
from models.pwm.pwm import PartWorldModel

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)

def build_shape_meta(num_views: int, H: int, W: int, low_dim: int):

    obs = {}
    for i in range(num_views):
        obs[f"rgb_{i:02d}"] = {"type": "rgb", "shape": (H, W, 3)}
        obs[f"rgb_{i+1:02d}"] = {"type": "rgb", "shape": (H, W, 3)}
    obs["low_dim"] = {"type": "low_dim", "shape": (low_dim,)}
    return {"obs": obs}

def to_device_nested(x, device):
    if isinstance(x, dict):
        return {k: to_device_nested(v, device) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    else:
        return x


def collate_inference_batch(samples):
    """Stack observations while keeping variable-length metadata per sample."""
    return {"obs": default_collate([sample["obs"] for sample in samples]),
            "meta": [sample["meta"] for sample in samples]}

def main():
    ap = argparse.ArgumentParser("Dump predicted actions + next-frame visuals")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--dataset", choices=["prepared", "pm", "acd"], default="prepared")
    ap.add_argument("--test_ids", type=Path, help="Fixed object ID manifest for PM/ACD.")
    ap.add_argument("--view_ids", type=int, nargs="+", default=[0, 1], help="Benchmark view indices (default: 0 1).")
    ap.add_argument("--vae_path", type=str, required=True, help="Path to the SDXL VAE used for training.")
    ap.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    ap.add_argument("--views", type=int, default=1)
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--img_size", type=int, nargs=2, default=[256,256])
    ap.add_argument("--num_types", type=int, default=2)

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sample_steps", type=int, default=10)
    ap.add_argument("--num_samples", type=int, default=0, help="Number of samples; 0 processes the full dataset.")

    ap.add_argument("--ckpt", type=str, default=str(Path(__file__).resolve().parent / "checkpoints/pwm_final.pt"))
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--vision_backbone", type=str, default="dinov2")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    H, W = args.img_size
    if args.dataset != "prepared":
        if (args.views, args.frames, args.num_types) != (1, 1, 2):
            ap.error("PM/ACD benchmarks require --views 1 --frames 1 --num_types 2")
        from evaluation.inputs import build_dataset
        ds = build_dataset(args)
    else:
        ds = PartNetMobilityInferenceDataset(
            root=args.data_root, split=args.split,
            image_size=(H, W), ho=args.frames, ha=1,
            use_views=args.views, expand_single_view=True,
        )
    inference_data = Subset(ds, range(min(args.num_samples, len(ds)))) if args.num_samples > 0 else ds
    dl = DataLoader(inference_data, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=(device.type=="cuda"),
                    drop_last=False, collate_fn=collate_inference_batch)
    print(f"Vision backbone: {args.vision_backbone}")
    shape_meta = build_shape_meta(args.views, H, W, low_dim=ds.low_dim_dim)
    obs_encoder = PWMObservationEncoder(
        shape_meta=shape_meta,
        num_frames=args.frames,
        embed_dim=768,
        resize_shape=(H, W),
        crop_shape=None,
        random_crop=False,
        color_jitter=None,
        imagenet_norm=False,
        vision_backbone=args.vision_backbone,
        use_low_dim=True,
        use_language=False,
        vae_path=args.vae_path,
    ).to(device)

    model = PartWorldModel(
        action_len=ds.ha,
        action_dim=ds.action_dim,
        obs_encoder=obs_encoder,
        embed_dim=768, timestep_embed_dim=512,
        latent_patch_shape=(2, 2, 2),
        depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, num_registers=8,
        num_train_steps=100, num_inference_steps=args.sample_steps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
    ).to(device)
    model.register_type_codes(build_type_codes(args.num_types, device=device))

    assert os.path.isfile(args.ckpt), f"Checkpoint not found: {args.ckpt}"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    saved = 0
    t0 = time.time()

    for batch in dl:

        bsz = len(batch["meta"])
        batch = to_device_nested(batch, device)

        with torch.no_grad():
            geom_out = model.sample_geom(batch["obs"], steps=args.sample_steps)


        with torch.no_grad():
            next_lat_pred = model.sample_marginal_next_obs(batch["obs"])
            next_rgb_pred = model.obs_encoder.apply_vae(next_lat_pred, inverse=True)

        for i in range(bsz):
            joint_id = int(batch["meta"][i]["jid"])
            from evaluation.inputs import sample_path
            sub_path = sample_path(batch["meta"][i], args.dataset)
            sample_dir = os.path.join(out_root,sub_path)
            sample_dir = Path(sample_dir)
            sample_dir.mkdir(parents=True, exist_ok=True)

            view_keys = sorted([k for k in batch["obs"].keys() if k.startswith("rgb_")])
            for vk in view_keys:
                for t_sel in [0]:
                    obs_img  = batch["obs"][vk][i, t_sel]   # [H,W,3] uint8
                    Image.fromarray(obs_img.detach().cpu().numpy()).save(sample_dir / f"obs_{vk}_t{t_sel:02d}.png")

            lat_i = next_lat_pred[i]  # [V,C,T,H,W]

            rgb_i = next_rgb_pred[i]  # [V,3,T,H,W]

            V, C, T, Hh, Ww = lat_i.shape
            for v in range(V):
                for t_sel in [0, 1]:
                    rgb = (rgb_i[v, :, t_sel].clamp(0,1) * 255.0).byte().permute(1,2,0).cpu().numpy()  # [H,W,3] uint8
                    Image.fromarray(rgb).save(sample_dir / f"next_pred_rgb_view_{v:02d}_t{t_sel:02d}.png")

            out = {}

            def to_list(t):
                return None if t is None else t[i].detach().cpu().reshape(-1).tolist()

            out["pred"] = {
                "axis_dir": to_list(geom_out.get("axis_dir")),
                "axis_ori": to_list(geom_out.get("axis_ori")),
                "range":    to_list(geom_out.get("range")),
                "aabb_max": to_list(geom_out.get("aabb_max")),
                "aabb_min": to_list(geom_out.get("aabb_min")),
                "aabb_base_max": to_list(geom_out.get("aabb_base_max")),
                "aabb_base_min": to_list(geom_out.get("aabb_base_min")),
                "type":     None if geom_out.get("type") is None else int(geom_out["type"][i].item()),
                "type_sim": None if geom_out.get("type_sim") is None else geom_out["type_sim"][i].detach().cpu().tolist(),
            }

            if "action_raw" in geom_out:
                out["pred"]["action_raw"] = geom_out["action_raw"][i].detach().cpu().tolist()

            with open(sample_dir / "geom_pred.json", "w") as f:
                json.dump(out, f, indent=2)

            saved += 1

        if saved % 10 == 0:
            print(f"[Saved {saved}]  elapsed: {time.time() - t0:.1f}s")

    print(f"Done. Saved {saved} samples to: {args.out_dir}")

if __name__ == "__main__":
    main()
