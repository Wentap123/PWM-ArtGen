"""Train PWM with Accelerate, mixed observation sources, and REPA alignment."""

import argparse
import os
import time
from typing import Dict, Any
import re
import shutil
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

wandb = None


def load_tracker():
    """Load the original preferred backend only when tracking is requested."""
    try:
        import swanlab
        return swanlab
    except Exception:
        try:
            import wandb as backend
            return backend
        except Exception:
            return None

from diffusers.optimization import get_scheduler
from contextlib import nullcontext

from accelerate import Accelerator

from datasets.partnet_mobility_action import (
    PartNetMobilityActionsDataset,
    build_type_codes,
)

from datasets.partnet_mobility_action_edit import PartNetMobilityActionsDataset as PartNetMobilityActionsDataset_edit
from datasets.mixture_sampler import InterleavedConcatSampler
from models.pwm.obs_encoder import PWMObservationEncoder
from models.pwm.pwm import PartWorldModel

def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def build_shape_meta(num_views: int, H: int, W: int, low_dim: int, ho: int):
    """
    Observation shapes:
      - rgb/mrgb use channel-last description (H, W, 3); encoder will handle conversion
      - low_dim: describe as (ho, low_dim) so encoder can read last dim as D
    """
    obs = {}
    for i in range(num_views):
        obs[f"rgb_{i:02d}"] = {"type": "rgb", "shape": (H, W, 3)}
        obs[f"rgb_{i+1:02d}"] = {"type": "rgb", "shape": (H, W, 3)}
    obs["low_dim"] = {"type": "low_dim", "shape": (ho, low_dim)}
    return {"obs": obs}

def to_device_nested(x, device):
    if isinstance(x, dict):
        return {k: to_device_nested(v, device) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    elif isinstance(x, (list, tuple)):
        return type(x)(to_device_nested(v, device) for v in x)
    else:
        return x

def train_one_step(accelerator: Accelerator, model, optimizer, batch, use_amp: bool):

    model.train()
    optimizer.zero_grad(set_to_none=True)

    ac = accelerator.autocast() if use_amp else nullcontext()

    with ac:
        actions = batch["actions"]
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        loss, info = model(batch["obs"], batch["next"], actions, action_mask=batch["action_mask"], geom_gt= None)

    accelerator.backward(loss)
    optimizer.step()

    return loss, info

@torch.no_grad()
def maybe_evaluate(
    step: int,
    every: int,
    model,
    val_source,
    accelerator: Accelerator,
    use_amp: bool,
    amp_dtype: torch.dtype = torch.bfloat16,
    eval_batches: int | None = None,
    use_wandb: bool = False,
):
    if every is None or every <= 0 or (step % every != 0):
        return

    was_training = model.training
    model.eval()

    ac = accelerator.autocast() if use_amp else nullcontext()

    if isinstance(val_source, DataLoader):
        local_loss_sum = torch.tensor(0.0, device=accelerator.device)
        local_count_sum = torch.tensor(0.0, device=accelerator.device)

        info_sums_local: Dict[str, torch.Tensor] = {}
        count_sum_local = torch.tensor(0.0, device=accelerator.device)

        for i, batch in enumerate(val_source, start=1):
            with ac:
                actions = batch["actions"]
                if actions.ndim == 2:
                    actions = actions.unsqueeze(0)
                loss, info = model(batch["obs"], batch["next"], actions, geom_gt=batch.get("geom", None))

            bs = batch["actions"].shape[0]
            loss_val = loss.detach().float()
            local_loss_sum += loss_val * bs
            local_count_sum += bs
            count_sum_local += bs

            for k, v in info.items():
                v = v.detach().float() if torch.is_tensor(v) else torch.tensor(float(v), device=accelerator.device)
                if k not in info_sums_local:
                    info_sums_local[k] = torch.zeros(1, device=accelerator.device, dtype=torch.float32)
                info_sums_local[k] += v * bs

            if eval_batches is not None and eval_batches > 0 and i >= eval_batches:
                break

        gathered_loss = accelerator.gather_for_metrics(local_loss_sum)
        gathered_cnt  = accelerator.gather_for_metrics(local_count_sum)

        gathered_info = {}
        for k, v_sum in info_sums_local.items():
            gathered_info[k] = accelerator.gather_for_metrics(v_sum)

        metrics = {}
        if accelerator.is_main_process:
            total_loss_sum = gathered_loss.sum().item()
            total_cnt = gathered_cnt.sum().item()
            mean_loss = total_loss_sum / max(total_cnt, 1)

            metrics["val/loss"] = mean_loss
            for k, arr in gathered_info.items():
                total_sum = arr.sum().item()
                metrics[f"val/{k}"] = total_sum / max(total_cnt, 1)

            if use_wandb and (wandb is not None) and hasattr(wandb, "log"):
                try:
                    wandb.log(metrics, step=step)
                except Exception:
                    print("[W&B] log failed.")
            else:
                msg = " | ".join([f"{k}:{v:.4f}" for k, v in metrics.items()])
                print(f"[Eval@{step}] {msg}")

    else:
        batch = val_source
        with ac:
            actions = batch["actions"]
            if actions.ndim == 2:
                actions = actions.unsqueeze(0)
            loss, info = model(batch["obs"], batch["next"], actions, geom_gt=batch.get("geom", None))

        loss_t = accelerator.gather_for_metrics(loss.detach().float())
        info_g = {k: accelerator.gather_for_metrics(v.detach().float() if torch.is_tensor(v) else torch.tensor(float(v), device=accelerator.device))
                  for k, v in info.items()}

        if accelerator.is_main_process:
            metrics = {"val/loss": loss_t.mean().item()}
            for k, arr in info_g.items():
                metrics[f"val/{k}"] = arr.mean().item()
            if use_wandb and (wandb is not None) and hasattr(wandb, "log"):
                try:
                    wandb.log(metrics, step=step)
                except Exception:
                    print("[W&B] log failed.")
            else:
                msg = " | ".join([f"{k}:{v:.4f}" for k, v in metrics.items()])
                print(f"[Eval@{step}] {msg}")

    if was_training:
        model.train()

def maybe_save_checkpoint(step: int, every: int, final_step: int, ckpt_path: str,
                          model, optimizer, scheduler, accelerator: Accelerator,
                          max_ckpts: int = 3):

    if every <= 0:
        return
    need_save = (step % every == 0) or (step == final_step - 1)
    if not need_save:
        return

    if not accelerator.is_main_process:
        return

    if os.path.isdir(ckpt_path) or ckpt_path.endswith(os.sep):
        ckpt_dir = ckpt_path if os.path.isdir(ckpt_path) else ckpt_path.rstrip(os.sep)
        try:
            script_basename = os.path.splitext(os.path.basename(__file__))[0]
        except Exception:
            script_basename = "checkpoint"
        base = script_basename
        ext = ".pt"
    else:
        ckpt_dir = os.path.dirname(ckpt_path) or "."
        base = os.path.splitext(os.path.basename(ckpt_path))[0]
        ext = os.path.splitext(os.path.basename(ckpt_path))[1] or ".pt"

    os.makedirs(ckpt_dir, exist_ok=True)
    final_name = f"{base}.step{step}{ext}"
    final_path = os.path.join(ckpt_dir, final_name)
    tmp_path = final_path + ".tmp"

    model_to_save = accelerator.unwrap_model(model)

    torch.save({
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
    }, tmp_path)
    os.replace(tmp_path, final_path)
    print(f"[Checkpoint] saved at step {step}: {final_path}")

    latest_name = f"{base}.latest{ext}"
    latest_path = os.path.join(ckpt_dir, latest_name)
    tmp_latest = latest_path + ".tmp"
    try:
        target = os.path.basename(final_path)
        try:
            if os.path.lexists(tmp_latest):
                os.remove(tmp_latest)
            os.symlink(target, tmp_latest)
            os.replace(tmp_latest, latest_path)
        except (AttributeError, OSError):
            shutil.copy2(final_path, tmp_latest)
            os.replace(tmp_latest, latest_path)
    except Exception as e:
        print(f"[Checkpoint] warning: failed to update latest pointer: {e}")

    if max_ckpts is None:
        max_ckpts = 3
    if max_ckpts > 0:
        pattern = re.compile(rf"^{re.escape(base)}\.step(\d+){re.escape(ext)}$")
        candidates = []
        for fn in os.listdir(ckpt_dir):
            m = pattern.match(fn)
            if m:
                try:
                    s = int(m.group(1))
                    candidates.append((s, os.path.join(ckpt_dir, fn)))
                except ValueError:
                    continue
        candidates.sort(key=lambda x: x[0], reverse=True)
        if len(candidates) > max_ckpts:
            to_remove = candidates[max_ckpts:]
            for _, path in to_remove:
                try:
                    os.remove(path)
                    print(f"[Checkpoint] removed old checkpoint: {path}")
                except Exception as e:
                    print(f"[Checkpoint] failed to remove {path}: {e}")

def maybe_resume_checkpoint(ckpt_path: str, model, optimizer, scheduler, accelerator: Accelerator, map_location="cpu"):

    if (ckpt_path is None) or (not os.path.exists(ckpt_path)):
        return 0

    unwrapped = accelerator.unwrap_model(model)

    try:
        dist_available = torch.distributed.is_available() and torch.distributed.is_initialized()
    except Exception:
        dist_available = False

    ckpt = None
    if dist_available:
        if accelerator.is_main_process:
            try:
                ckpt = torch.load(ckpt_path, map_location=map_location)
            except Exception as e:
                print(f"[Resume] failed to load checkpoint {ckpt_path} on main process: {e}")
                ckpt = None
        else:
            ckpt = None
        obj = [ckpt]
        try:
            torch.distributed.broadcast_object_list(obj, src=0)
            ckpt = obj[0]
        except Exception:
            try:
                ckpt = torch.load(ckpt_path, map_location=map_location)
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"[Resume] per-process load also failed: {e}")
                ckpt = None
    else:
        try:
            ckpt = torch.load(ckpt_path, map_location=map_location)
        except Exception as e:
            if accelerator.is_main_process:
                print(f"[Resume] failed to load checkpoint {ckpt_path}: {e}")
            ckpt = None

    if ckpt is None:
        accelerator.wait_for_everyone()
        return 0

    if "model" in ckpt and ckpt["model"] is not None:
        try:
            unwrapped.load_state_dict(ckpt["model"])
        except Exception as e:
            if accelerator.is_main_process:
                print(f"[Resume] failed to load model.state_dict(): {e}")

    if optimizer is not None and ckpt.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            if accelerator.is_main_process:
                print(f"[Resume] warning: optimizer.state_dict() load failed: {e}")

    if scheduler is not None and ckpt.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            if accelerator.is_main_process:
                print(f"[Resume] warning: scheduler.state_dict() load failed: {e}")

    step = int(ckpt.get("step", 0))
    if accelerator.is_main_process:
        print(f"[Resume] from {ckpt_path}, step={step}")

    accelerator.wait_for_everyone()
    return step

def main():
    global args, wandb
    parser = argparse.ArgumentParser("Train PWM with Accelerate")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True, help="Path to the SDXL VAE used for training.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--views", type=int, default=1)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--img_size", type=int, nargs=2, default=[256, 256])

    parser.add_argument("--num_types", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=10)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--scheduler", type=str, default="constant_with_warmup",
                        choices=["constant", "linear", "cosine", "cosine_with_restarts",
                                 "polynomial", "constant_with_warmup"])
    parser.add_argument("--warmup_steps", type=int, default=50000)

    parser.add_argument("--num_steps", type=int, default=200000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--eval_batches", type=int, default=0, help="Validation batches; 0 evaluates the full split.")
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--ckpt", type=str, default="./outputs/")
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run", type=str, default="pwm")
    parser.add_argument("--pretrained_blocks_path", type=str, default=None)
    parser.add_argument("--vision_backbone", type=str, default="dinov2")
    parser.add_argument("--repa", action="store_true", help="Enable REPA")
    parser.add_argument("--repa-coeff", type=float, default=0.0, help="REPA loss weight λ (0 = off)")
    parser.add_argument("--repa-depth", type=int, default=0, help="Align first K transformer blocks")
    parser.add_argument("--repa-teacher", type=str, default="clip",
                        choices=["clip", "vit_b_32", "dinov2_vitb14_reg"], help="Teacher backbone")
    parser.add_argument("--repa-stop-steps", type=int, default=None,
                        help="Early stop REPA after N forward calls (None = no stop)")
    args = parser.parse_args()

    mixed_precision = "bf16" if args.use_amp else "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    device = accelerator.device

    set_seed(args.seed)

    if args.use_wandb and accelerator.is_main_process:
        wandb = load_tracker()
        if wandb is not None and hasattr(wandb, "init"):
            try:
                wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))
                print("[W&B] initialized")
            except Exception as e:
                print("[W&B] init failed:", e)
        else:
            print("[W&B] not available in this environment; continuing without W&B.")

    H, W = args.img_size
    train_set_org = PartNetMobilityActionsDataset(
        root=args.data_root, split=args.split,
        image_size=(H, W), ho=args.frames, ha=1,
        use_views=args.views, num_types=args.num_types,
        expand_single_view=True,
        eval_fixed_frame_list = [0,2],
    )

    train_set_edit = PartNetMobilityActionsDataset_edit(
        root=args.data_root, split=args.split,
        image_size=(H, W), ho=args.frames, ha=1,
        use_views=args.views, num_types=args.num_types,
        expand_single_view=True,
        eval_fixed_view_idx = [
        "view_idx_00","view_idx_01","view_idx_02","view_idx_03","view_idx_04",
        "view_idx_05","view_idx_06","view_idx_07","view_idx_08","view_idx_09",
        "view_idx_10","view_idx_11","view_idx_12","view_idx_13","view_idx_14",
        "view_idx_15","view_idx_16","view_idx_17","view_idx_18","view_idx_19"],
        eval_fixed_frame_list = [0,2],
    )

    train_set = torch.utils.data.ConcatDataset([train_set_org, train_set_edit])

    assert train_set_org.low_dim_dim == train_set_edit.low_dim_dim
    assert train_set_org.action_dim  == train_set_edit.action_dim
    setattr(train_set, "low_dim_dim", train_set_org.low_dim_dim)
    setattr(train_set, "action_dim",  train_set_org.action_dim)

    mixture_sampler = InterleavedConcatSampler(
        len_a=len(train_set_org), len_b=len(train_set_edit),
        seed=args.seed, shuffle=True, drop_last=True
    )
    if accelerator.is_main_process:
        print("train_set_org:", len(train_set_org))
        print("train_set_edit:", len(train_set_edit))
        print("train_set:", len(train_set))
        print("train_set low_dim_dim:", getattr(train_set, "low_dim_dim", None), "action_dim:", getattr(train_set, "action_dim", None))
    val_set = PartNetMobilityActionsDataset(
        root=args.data_root, split="test",
        image_size=(H, W), ho=args.frames, ha=1,
        use_views=args.views, num_types=args.num_types,
        expand_single_view=True,
        eval_fixed_view_idx = ["view_idx_00","view_idx_01"], # "view_idx_00"
        eval_fixed_frame_list = [0,2],#[0,13]
    )
    if accelerator.is_main_process:
        print("val_set:", len(val_set))
        print("val_set low_dim_dim:", getattr(val_set, "low_dim_dim", None), "action_dim:", getattr(val_set, "action_dim", None))

    val_loader = DataLoader(
        val_set, batch_size=max(1, args.batch_size // 4),
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        drop_last=False, persistent_workers=(args.workers > 0)
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
        sampler=mixture_sampler,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        drop_last=True, persistent_workers=(args.workers > 0)
    )

    shape_meta = build_shape_meta(num_views=args.views, H=H, W=W, low_dim=getattr(train_set, "low_dim_dim", 0), ho=args.frames)
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
    )

    model = PartWorldModel(
        action_len=getattr(train_set, "ha", 1),
        action_dim=getattr(train_set, "action_dim", 8 + 12),
        obs_encoder=obs_encoder,
        embed_dim=768, timestep_embed_dim=512,
        latent_patch_shape=(2, 2, 2),
        depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, num_registers=8,
        num_train_steps=100, num_inference_steps=10,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        debug=False,
        pretrained_blocks_path=args.pretrained_blocks_path,
        repa=args.repa, repa_coeff=args.repa_coeff, repa_depth=args.repa_depth,
        repa_teacher=args.repa_teacher, repa_stop_steps=args.repa_stop_steps,
    )

    try:
        codes = build_type_codes(args.num_types)
        if codes is not None:
            if not torch.is_tensor(codes):
                codes = torch.tensor(codes, dtype=torch.float32)
            model.register_type_codes(codes.to(device))
            if accelerator.is_main_process:
                print("Registered type codes.")
    except Exception as e:
        if accelerator.is_main_process:
            print("build_type_codes failed or not available:", e)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2)
    )
    lr_scheduler = get_scheduler(
        name=args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_steps,
    )

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    start_step = maybe_resume_checkpoint(args.resume, model, optimizer, lr_scheduler, accelerator, map_location="cpu")

    total_steps = args.num_steps
    step = int(start_step)
    steps_per_epoch = len(train_loader)
    assert steps_per_epoch > 0, "Empty training loader: check batch_size, dataset size, and drop_last."
    epoch = step // steps_per_epoch
    offset = step % steps_per_epoch
    first_round = True
    pbar = None
    if accelerator.is_main_process:
        pbar = tqdm(total=total_steps, initial=step, desc="Training", dynamic_ncols=True)
    model.train()
    t0 = time.time()

    while step < total_steps:
        mixture_sampler.set_epoch(epoch)
        epoch += 1

        epoch_loss_sum_local = torch.tensor(0.0, device=accelerator.device)
        epoch_count_local    = torch.tensor(0.0, device=accelerator.device)

        if first_round and offset > 0:
            it = iter(train_loader)
            for _ in range(offset):
                try:
                    next(it)
                except StopIteration:
                    break
            iterable = it
            first_round = False
        else:
            iterable = iter(train_loader)

        for batch in iterable:
            step += 1
            batch = to_device_nested(batch, accelerator.device)

            loss, info = train_one_step(accelerator, model, optimizer, batch, use_amp=args.use_amp)

            if lr_scheduler is not None:
                lr_scheduler.step()

            bs = batch["actions"].shape[0]
            loss_for_epoch = loss.detach().float()
            epoch_loss_sum_local += loss_for_epoch * bs
            epoch_count_local    += bs

            if accelerator.is_main_process and args.use_wandb and (wandb is not None) and hasattr(wandb, "log"):
                try:
                    wandb.log({f"train/{k}": (v.item() if torch.is_tensor(v) else float(v)) for k, v in info.items()}, step=step)
                    wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=step)
                except Exception:
                    pass

            if step % args.log_every == 0 and accelerator.is_main_process:
                dt = time.time() - t0
                msg = (
                    f"step: {step}, loss: {info.get('loss', 0.0):.6f}, "
                    f"L_act: {info.get('action_loss', 0.0):.6f}, "
                    f"L_dyn: {info.get('dynamics_loss', 0.0):.6f}, dt: {dt:.2f}s,"
                    f"L_repa: {info.get('L_repa', 0.0):.6f}"
                )
                pbar.set_description(msg)
                t0 = time.time()

            maybe_evaluate(step, args.eval_every, model, val_loader, accelerator,
                           use_amp=args.use_amp, amp_dtype=(torch.bfloat16 if args.use_amp else torch.float32),
                           eval_batches=args.eval_batches, use_wandb=args.use_wandb)
            maybe_save_checkpoint(step, args.save_every, total_steps, args.ckpt, model, optimizer, lr_scheduler, accelerator, max_ckpts=2)

            if accelerator.is_main_process:
                pbar.update(1)
            if step >= total_steps:
                break

        if epoch_count_local.item() > 0:
            gathered_loss_sum = accelerator.gather_for_metrics(epoch_loss_sum_local).sum()
            gathered_cnt      = accelerator.gather_for_metrics(epoch_count_local).sum()
            epoch_loss_mean   = (gathered_loss_sum / torch.clamp_min(gathered_cnt, 1)).item()

            if accelerator.is_main_process:
                if (wandb is not None) and hasattr(wandb, "log") and args.use_wandb:
                    try:
                        wandb.log({"train_epoch/loss": epoch_loss_mean}, step=step)
                    except Exception:
                        print(f"[Train@epoch] loss={epoch_loss_mean:.6f}")
                else:
                    print(f"[Train@epoch] loss={epoch_loss_mean:.6f}")

    if accelerator.is_main_process:
        pbar.close()
        print("Training finished.")

if __name__ == "__main__":
    main()
