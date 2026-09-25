from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from einops import rearrange
import numpy as np
import os

from models.common.adaln_attention import AdaLNAttentionBlock, AdaLNFinalLayer
from models.common.utils import SinusoidalPosEmb, init_weights
from .obs_encoder import PWMObservationEncoder
from models.common.vision import GlobalImageTeacher

class MultiViewVideoPatchifier(nn.Module):
    def __init__(
        self,
        num_views: int,
        input_shape: tuple[int, ...] = (8, 224, 224),
        patch_shape: tuple[int, ...] = (2, 8, 8),
        num_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.num_views = num_views
        iT, iH, iW = input_shape
        pT, pH, pW = patch_shape
        assert iT % pT == 0 and iH % pH == 0 and iW % pW == 0, \
                f"latent (T,H,W)=({iT},{iH},{iW}) must be divisible by patch (pT,pH,pW)=({pT},{pH},{pW})"
        self.T, self.H, self.W = iT // pT, iH // pH, iW // pW
        self.pT, self.pH, self.pW = pT, pH, pW

        self.patch_encoder = nn.Conv3d(
            in_channels=num_chans,
            out_channels=embed_dim,
            kernel_size=patch_shape,
            stride=patch_shape,
        )
        self.patch_decoder = nn.Linear(embed_dim, num_chans * pT * pH * pW)

    def forward(self, imgs):
        return self.patchify(imgs)

    def patchify(self, imgs):
        imgs = rearrange(imgs, "b v c t h w -> (b v) c t h w")
        feats = self.patch_encoder(imgs)
        feats = rearrange(feats, "(b v) c t h w -> b (v t h w) c", v=self.num_views)
        return feats

    def unpatchify(self, feats):
        imgs = self.patch_decoder(feats)
        imgs = rearrange(
            imgs,
            "b (v t h w) (c pt ph pw) -> b v c (t pt) (h ph) (w pw)",
            v=self.num_views, t=self.T, h=self.H, w=self.W, pt=self.pT, ph=self.pH, pw=self.pW
        )
        return imgs

    @property
    def num_patches(self):
        return self.num_views * self.T * self.H * self.W

class DualTimestepEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, mlp_ratio: float = 4.0):
        super().__init__()
        self.sinusoidal_pos_emb = SinusoidalPosEmb(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, t1, t2):
        temb1 = self.sinusoidal_pos_emb(t1)
        temb2 = self.sinusoidal_pos_emb(t2)
        temb = torch.cat([temb1, temb2], dim=-1)
        return self.proj(temb)

class DualNoisePredictionNet(nn.Module):
    def __init__(
        self,
        global_cond_dim: int,
        image_shape: tuple[int, ...],
        patch_shape: tuple[int, ...],
        num_chans: int,
        num_views: int,
        action_len: int,
        action_dim: int,
        embed_dim: int = 768,
        timestep_embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        num_registers: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.obs_patchifier = MultiViewVideoPatchifier(
            num_views=num_views,
            input_shape=image_shape,
            patch_shape=patch_shape,
            num_chans=num_chans,
            embed_dim=embed_dim,
        )
        obs_len = self.obs_patchifier.num_patches

        hidden_dim = int(max(action_dim, embed_dim) * mlp_ratio)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.timestep_embedding = DualTimestepEncoder(timestep_embed_dim)

        self.registers = nn.Parameter(
            torch.empty(1, num_registers, embed_dim).normal_(std=0.02)
        )

        total_len = action_len + obs_len + num_registers
        self.pos_embed = nn.Parameter(
            torch.empty(1, total_len, embed_dim).normal_(std=0.02)
        )

        self.cond_norm = nn.LayerNorm(global_cond_dim)
        self.temb_norm = nn.LayerNorm(timestep_embed_dim)

        cond_dim = global_cond_dim + timestep_embed_dim
        self.blocks = nn.ModuleList(
            [
                AdaLNAttentionBlock(
                    dim=embed_dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )
        self.head = AdaLNFinalLayer(dim=embed_dim, cond_dim=cond_dim)
        self.action_inds = (0, action_len)
        self.next_obs_inds = (action_len, action_len + obs_len)

        self.initialize_weights()

        self.capture_hidden: bool = False
        self.capture_layers: int = 0
        self._cached_hiddens: list[torch.Tensor] = []

    def pop_cached_hiddens(self, return_meta: bool = False):
        hs = self._cached_hiddens
        self._cached_hiddens = []
        if return_meta:
            meta = getattr(self, "_cached_meta", None)
            setattr(self, "_cached_meta", None)
            return hs, meta
        return hs

    def initialize_weights(self):
        self.apply(init_weights)

        w = self.obs_patchifier.patch_encoder.weight.data
        nn.init.normal_(w.view([w.shape[0], -1]), mean=0.0, std=0.02)
        nn.init.constant_(self.obs_patchifier.patch_encoder.bias, 0)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def forward(self, global_cond, action, action_t, next_obs, next_obs_t):
        action_embed = self.action_encoder(action)
        next_obs_embed = self.obs_patchifier(next_obs)

        if len(action_t.shape) == 0:
            action_t = action_t.expand(action.shape[0]).to(
                dtype=torch.long, device=action.device
            )
        if len(next_obs_t.shape) == 0:
            next_obs_t = next_obs_t.expand(next_obs.shape[0]).to(
                dtype=torch.long, device=next_obs.device
            )
        temb = self.timestep_embedding(action_t, next_obs_t)

        global_cond = self.cond_norm(global_cond)
        temb = self.temb_norm(temb)

        registers = self.registers.expand(next_obs.shape[0], -1, -1)
        x = torch.cat((action_embed, next_obs_embed, registers), dim=1)
        x = x + self.pos_embed
        cond = torch.cat((global_cond, temb), dim=-1)

        repa_hiddens = [] if (self.capture_hidden and self.capture_layers > 0) else None
        for li, block in enumerate(self.blocks):
            x = block(x, cond)
            if repa_hiddens is not None and li < self.capture_layers:
                pt, ph, pw = self.obs_patchifier.pT, self.obs_patchifier.pH, self.obs_patchifier.pW
                T_lat, H_lat, W_lat = self.obs_patchifier.T, self.obs_patchifier.H, self.obs_patchifier.W
                V_lat = self.obs_patchifier.num_views
                self._cached_meta = {"grid": (T_lat, H_lat, W_lat), "views": V_lat}
                n0, n1 = self.next_obs_inds
                nxt = x[:, n0:n1]                 # next-observation tokens
                repa_hiddens.append(nxt)

        x = self.head(x, cond)

        if repa_hiddens is not None:
            self._cached_hiddens = repa_hiddens

        action_noise_pred = x[:, self.action_inds[0] : self.action_inds[1]]
        next_obs_noise_pred = x[:, self.next_obs_inds[0] : self.next_obs_inds[1]]

        action_noise_pred = self.action_decoder(action_noise_pred)
        next_obs_noise_pred = self.obs_patchifier.unpatchify(next_obs_noise_pred)
        return action_noise_pred, next_obs_noise_pred

class PartWorldModel(nn.Module):
    def __init__(
        self,
        action_len: int,
        action_dim: int,
        obs_encoder: PWMObservationEncoder,
        embed_dim: int = 768,
        timestep_embed_dim: int = 512,
        latent_patch_shape: tuple[int, ...] = (1, 4, 4),
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        num_registers: int = 8,
        num_train_steps: int = 100,
        num_inference_steps: int = 10,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        debug: bool = False,
        pretrained_blocks_path: str | None = None,
        repa: bool = False,
        repa_coeff: float = 0.0,
        repa_depth: int = 0,
        repa_teacher: str = "clip",
        repa_stop_steps: int | None = None,
    ):

        super().__init__()
        self.action_len = action_len
        self.action_dim = action_dim
        self.action_shape = (action_len, action_dim)

        self.obs_encoder = obs_encoder
        self.latent_img_shape = self.obs_encoder.latent_img_shape()

        global_cond_dim = self.obs_encoder.feat_dim()
        image_shape = self.latent_img_shape[2:]
        num_views, num_chans = self.latent_img_shape[:2]
        self.noise_pred_net = DualNoisePredictionNet(
            global_cond_dim=global_cond_dim,
            image_shape=image_shape,
            patch_shape=latent_patch_shape,
            num_chans=num_chans,
            num_views=num_views,
            action_len=action_len,
            action_dim=action_dim,
            embed_dim=embed_dim,
            timestep_embed_dim=timestep_embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            num_registers=num_registers,
        )

        if pretrained_blocks_path is not None and os.path.exists(pretrained_blocks_path):
            print(f"Loading pretrained blocks from {pretrained_blocks_path}")
            checkpoint = torch.load(pretrained_blocks_path, map_location="cpu")
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint  # Assume it's just the state_dict

            pretrained_block_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("noise_pred_net.blocks."):
                    new_key = k.replace("noise_pred_net.", "")
                    pretrained_block_state_dict[new_key] = v

            try:
                self.noise_pred_net.load_state_dict(pretrained_block_state_dict, strict=False)
                print(f"Successfully loaded {len(pretrained_block_state_dict)} pretrained weights for noise_pred_net.blocks.")
            except Exception as e:
                print(f"Error loading pretrained blocks: {e}")
        elif pretrained_blocks_path is not None:
            print(f"Warning: Pretrained checkpoint not found at {pretrained_blocks_path}. Skipping block loading.")

        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_steps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
        )

        self.register_buffer("type_codes", None, persistent=True)
        self.debug = bool(debug)

        self.repa_enabled = bool(repa and repa_coeff > 0.0 and repa_depth > 0)
        self.repa_coeff = float(repa_coeff)
        self.repa_depth = int(repa_depth)
        self.repa_stop_steps = None if repa_stop_steps is None else int(repa_stop_steps)
        self._fwd_calls = 0
        if self.repa_enabled:
            self.teacher = GlobalImageTeacher(embed_dim=self.noise_pred_net.embed_dim,
                                             backbone=repa_teacher)
        else:
            self.teacher = None

    def register_type_codes(self, type_codes: torch.Tensor):
        self.register_buffer("type_codes", type_codes, persistent=True)

    def _safe(self, name, x):
        if not torch.isfinite(x).all():
            m = x.detach().float()
            print(f"[NaN@{name}] mean={m.mean().item():.4f} std={m.std().item():.4f} "
                  f"min={m.min().item():.4f} max={m.max().item():.4f} shape={tuple(x.shape)}")
            raise FloatingPointError(name)

    def _add_noise_safe(self, x: torch.Tensor, noise: torch.Tensor, t: torch.LongTensor):
        with torch.autocast(device_type="cuda", enabled=False):
            ac = self.noise_scheduler.alphas_cumprod.to(x.device)[t].clamp(1e-5, 1 - 1e-5)
            s1 = ac.sqrt().view(-1, *([1] * (x.ndim - 1))).to(torch.float32)
            s2 = (1.0 - ac).sqrt().clamp_min(1e-5).view(-1, *([1] * (x.ndim - 1))).to(torch.float32)
            y = s1 * x.to(torch.float32) + s2 * noise.to(torch.float32)
        return y.to(x.dtype)

    def forward(self, obs_dict, next_obs_dict, action, action_mask=None, geom_gt=None):

        B, device = action.shape[0], action.device
        self._fwd_calls += 1

        obs = self.obs_encoder.encode_curr_obs(obs_dict)
        next_imgs = self.obs_encoder.apply_transform(next_obs_dict)   # (B,V,C,T,H,W)
        next_obs  = self.obs_encoder.apply_vae(next_imgs)             # latent (B,V,C',T',H',W')

        action_noise = torch.randn_like(action)
        action_t = torch.randint(0, self.num_train_steps, (B,), device=device).long()
        if action_mask is not None:
            action_t[~action_mask] = self.num_train_steps - 1
        if self.debug:
            noisy_action = self._add_noise_safe(action, action_noise, action_t)
            self._safe("noisy_action", noisy_action)
        else:
            noisy_action = self.noise_scheduler.add_noise(action, action_noise, action_t)

        next_noise = torch.randn_like(next_obs)
        next_t = torch.randint(0, self.num_train_steps, (B,), device=device).long()
        if self.debug:
            noisy_next = self._add_noise_safe(next_obs, next_noise, next_t)
            self._safe("noisy_next_obs", noisy_next)
        else:
            noisy_next = self.noise_scheduler.add_noise(next_obs, next_noise, next_t)

        repa_on = self.repa_enabled and self.training and \
                  (self.repa_stop_steps is None or self._fwd_calls <= self.repa_stop_steps)
        if repa_on:
            self.noise_pred_net.capture_hidden = True
            self.noise_pred_net.capture_layers = self.repa_depth
        else:
            self.noise_pred_net.capture_hidden = False
            self.noise_pred_net.capture_layers = 0

        eps_a, eps_next = self.noise_pred_net(obs, noisy_action, action_t, noisy_next, next_t)

        L_act = torch.mean((eps_a - action_noise)**2)
        L_dyn = torch.mean((eps_next - next_noise)**2)

        L_repa = torch.zeros((), device=device)
        if repa_on:
            with torch.no_grad():
                t_out = self.teacher(next_imgs)    # dict or tensor
            if isinstance(t_out, dict):
                t_g = F.normalize(t_out["global"], dim=-1)   # (B,V,T,D)
                t_p = t_out["patch"]                         # (B,V,T,Ht,Wt,D)
                Ht, Wt = t_out["grid"]
            else:
                t_g = F.normalize(t_out, dim=-1)             # (B, D)
                t_p = None

            hs, meta = self.noise_pred_net.pop_cached_hiddens(return_meta=True)
            L_repa = torch.zeros((), device=device)
            if len(hs) > 0:
                sims = []
                for h in hs:
                    if t_p is None:
                        h_g = F.normalize(h.mean(dim=1), dim=-1)   # (B,D)
                        sims.append((h_g * t_g).sum(dim=-1).mean())
                    else:
                        B, V, C, T, H_img, W_img = next_imgs.shape
                        T_lat, H_lat, W_lat = meta["grid"]
                        h = h.view(B, V, T_lat, H_lat, W_lat, -1)
                        h = F.normalize(h, dim=-1)
                        t = t_p[:, :, 1:2, ...]
                        t = F.normalize(t, dim=-1)
                        sim = (h * t).sum(dim=-1).mean()
                        sims.append(sim)

                L_repa = 1.0 - torch.stack(sims).mean()

        geom_loss = torch.zeros((), device=device)
        info_geom = {}

        D_t = max(self.action_dim - (7 + 12), 0)

        if geom_gt is not None and self.action_len >= 1 and self.action_dim >= 8:
            abar = self.noise_scheduler.alphas_cumprod.to(device)[action_t].clamp(1e-5, 1-1e-5)
            abar = abar.view(-1, 1, 1)
            x0_hat = (noisy_action - (1.0 - abar).sqrt() * eps_a) / abar.sqrt()  # [B, ha, action_dim]

            dir_hat = F.normalize(x0_hat[..., 0:3], dim=-1)
            ori_hat = x0_hat[..., 3:6]
            range_hat = x0_hat[..., 6:7]
            type_hat = x0_hat[..., 7:7 + D_t] if D_t > 0 else None

            aabb_start = 7 + D_t #8 + D_t -> 7 + D_t
            if self.action_dim >= aabb_start + 12:
                aabb_hat = x0_hat[..., aabb_start:aabb_start + 12]  # [B, ha, 12]
                aabb_max_hat = aabb_hat[..., 0:3]
                aabb_min_hat = aabb_hat[..., 3:6]
                aabb_base_max_hat = aabb_hat[..., 6:9]
                aabb_base_min_hat = aabb_hat[..., 9:12]
            else:
                aabb_max_hat = aabb_min_hat = aabb_base_max_hat = aabb_base_min_hat = None

            d = F.normalize(geom_gt["axis_dir"], dim=-1).unsqueeze(1)
            o = geom_gt["axis_ori"].unsqueeze(1)
            r = geom_gt["range"].unsqueeze(1)

            if "type_code" in geom_gt:
                t_code = geom_gt["type_code"].unsqueeze(1)
            elif (self.type_codes is not None) and ("type" in geom_gt):
                idx = geom_gt["type"].long()
                t_code = self.type_codes[idx].unsqueeze(1)
            else:
                t_code = None

            Ld = torch.minimum((dir_hat - d).pow(2).sum(-1), (dir_hat + d).pow(2).sum(-1)).mean()
            Lo = F.mse_loss(ori_hat, o)
            Lr = F.mse_loss(range_hat, r)

            if (t_code is not None) and (type_hat is not None):
                if t_code.shape[-1] == 1:
                    target = ((t_code + 1) / 2).float()
                    Lt = F.binary_cross_entropy_with_logits(type_hat, target)
                else:
                    Lt = 1.0 - F.cosine_similarity(
                        F.normalize(type_hat, dim=-1),
                        F.normalize(t_code, dim=-1),
                        dim=-1
                    ).mean()
            else:
                Lt = torch.zeros((), device=device)

            if (aabb_max_hat is not None) and ("aabb_max" in geom_gt):
                gt_c = geom_gt["aabb_max"].unsqueeze(1)
                gt_s = geom_gt["aabb_min"].unsqueeze(1)
                gt_bc = geom_gt["aabb_base_max"].unsqueeze(1)
                gt_bs = geom_gt["aabb_base_min"].unsqueeze(1)

                La_c = F.mse_loss(aabb_max_hat, gt_c)
                La_s = F.mse_loss(aabb_min_hat, gt_s)
                La_bc = F.mse_loss(aabb_base_max_hat, gt_bc)
                La_bs = F.mse_loss(aabb_base_min_hat, gt_bs)
                La = La_c + La_s + La_bc + La_bs
            else:
                La = torch.zeros((), device=device)

            geom_loss = Ld + Lo + Lr + Lt + La
            info_geom = {
                "Ld": float(Ld.item()), "Lo": float(Lo.item()), "Lr": float(Lr.item()),
                "Lt": float(Lt.item()), "La": float(La.item() if isinstance(La, torch.Tensor) else La)
            }

        loss = L_act + L_dyn + self.repa_coeff * L_repa + geom_loss
        info = {"loss": float(loss.item()), "action_loss": float(L_act.item()), "dynamics_loss": float(L_dyn.item()), **info_geom}
        if repa_on:
            info["L_repa"] = L_repa.detach()
        return loss, info

    @torch.no_grad()
    def sample_geom(self, obs_dict, steps: int | None = None):

        if steps is None:
            steps = self.num_inference_steps

        obs_feat = self.obs_encoder.encode_curr_obs(obs_dict)  # global cond
        a = torch.randn((obs_feat.shape[0],) + self.action_shape, device=obs_feat.device)
        x_next = torch.randn((obs_feat.shape[0],) + self.latent_img_shape, device=obs_feat.device)

        self.noise_scheduler.set_timesteps(steps)
        for t in self.noise_scheduler.timesteps:
            eps_a, _ = self.noise_pred_net(obs_feat, a, t, x_next, self.noise_scheduler.timesteps[0])
            a = self.noise_scheduler.step(eps_a, t, a).prev_sample

        x = a[:, 0, :]
        D_t = max(self.action_dim - (7 + 12), 0)

        dir_hat = F.normalize(x[:, 0:3], dim=-1) if self.action_dim >= 3 else None
        ori_hat = x[:, 3:6] if self.action_dim >= 6 else None
        range_hat = x[:, 6:7] if self.action_dim >= 7 else None
        type_hat = x[:, 7:7 + D_t] if D_t > 0 else None

        aabb_start = 7 + D_t
        if x.shape[-1] >= aabb_start + 12:
            aabb = x[:, aabb_start:aabb_start + 12]
            aabb_max = aabb[:, 0:3]
            aabb_min = aabb[:, 3:6]
            aabb_base_max = aabb[:, 6:9]
            aabb_base_min = aabb[:, 9:12]
        else:
            aabb_max = aabb_min = aabb_base_max = aabb_base_min = None

        if (type_hat is not None) and (self.type_codes is not None):
            codes = self.type_codes.to(x.device, x.dtype)
            if codes.shape[1] != type_hat.shape[-1]:
                raise RuntimeError("Type-code dim mismatch")
            if type_hat.shape[-1] == 1:
                logits = type_hat.view(-1)
                type_prob = torch.sigmoid(logits)
                type_idx = (type_prob > 0.5).long()
                sim = type_prob.unsqueeze(-1)
            else:
                codes_n = F.normalize(codes, dim=-1)
                type_hat_n = F.normalize(type_hat, dim=-1)
                sim = type_hat_n @ codes_n.T
                type_idx = sim.argmax(dim=-1)
        else:
            sim = None
            type_idx = None

        return {
            "axis_dir": dir_hat,
            "axis_ori": ori_hat,
            "range": range_hat,
            "type": type_idx,
            "type_sim": sim,
            "aabb_max": aabb_max,
            "aabb_min": aabb_min,
            "aabb_base_max": aabb_base_max,
            "aabb_base_min": aabb_base_min,
            "action_raw": a,
        }

    @torch.no_grad()
    def sample_geom_inverse(self, obs_dict, next_obs_dict, steps: int | None = None):

        if steps is None:
            steps = self.num_inference_steps

        obs_feat, next_obs = self.obs_encoder.encode_curr_and_next_obs(obs_dict, next_obs_dict)
        a = torch.randn((obs_feat.shape[0],) + self.action_shape, device=obs_feat.device)

        self.noise_scheduler.set_timesteps(steps)
        next_t = self.noise_scheduler.timesteps[-1]
        for t in self.noise_scheduler.timesteps:
            eps_a, _ = self.noise_pred_net(obs_feat, a, t, next_obs, next_t)
            a = self.noise_scheduler.step(eps_a, t, a).prev_sample

        x = a[:, 0, :]
        D_t = max(self.action_dim - (7 + 12), 0)

        dir_hat = F.normalize(x[:, 0:3], dim=-1) if self.action_dim >= 3 else None
        ori_hat = x[:, 3:6] if self.action_dim >= 6 else None
        range_hat = x[:, 6:7] if self.action_dim >= 7 else None
        type_hat = x[:, 7:7 + D_t] if D_t > 0 else None

        aabb_start = 7 + D_t
        if x.shape[-1] >= aabb_start + 12:
            aabb = x[:, aabb_start:aabb_start + 12]
            aabb_max = aabb[:, 0:3]
            aabb_min = aabb[:, 3:6]
            aabb_base_max = aabb[:, 6:9]
            aabb_base_min = aabb[:, 9:12]
        else:
            aabb_max = aabb_min = aabb_base_max = aabb_base_min = None

        if (type_hat is not None) and (self.type_codes is not None):
            codes = self.type_codes.to(x.device, x.dtype)
            if codes.shape[1] != type_hat.shape[-1]:
                raise RuntimeError("Type-code dim mismatch")
            if type_hat.shape[-1] == 1:
                logits = type_hat.view(-1)
                type_prob = torch.sigmoid(logits)
                type_idx = (type_prob > 0.5).long()
                sim = type_prob.unsqueeze(-1)
            else:
                codes_n = F.normalize(codes, dim=-1)
                type_hat_n = F.normalize(type_hat, dim=-1)
                sim = type_hat_n @ codes_n.T
                type_idx = sim.argmax(dim=-1)
        else:
            sim = None
            type_idx = None

        return {
            "axis_dir": dir_hat,
            "axis_ori": ori_hat,
            "range": range_hat,
            "type": type_idx,
            "type_sim": sim,
            "aabb_max": aabb_max,
            "aabb_min": aabb_min,
            "aabb_base_max": aabb_base_max,
            "aabb_base_min": aabb_base_min,
            "action_raw": a,
        }
    @torch.no_grad()
    def sample(self, obs_dict):
        return self.sample_marginal_action(obs_dict)

    @torch.no_grad()
    def sample_marginal_action(self, obs_dict):
        obs = self.obs_encoder.encode_curr_obs(obs_dict)
        action = torch.randn((obs.shape[0],) + self.action_shape, device=obs.device)
        next_obs = torch.randn((obs.shape[0],) + self.latent_img_shape, device=obs.device)

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        next_t = self.noise_scheduler.timesteps[0]
        for t in self.noise_scheduler.timesteps:
            eps_a, _ = self.noise_pred_net(obs, action, t, next_obs, next_t)
            action = self.noise_scheduler.step(eps_a, t, action).prev_sample
        return action

    @torch.no_grad()
    def sample_marginal_next_obs(self, obs_dict):
        obs = self.obs_encoder.encode_curr_obs(obs_dict)
        action = torch.randn((obs.shape[0],) + self.action_shape, device=obs.device)
        next_obs = torch.randn((obs.shape[0],) + self.latent_img_shape, device=obs.device)

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        action_t = self.noise_scheduler.timesteps[0]
        for t in self.noise_scheduler.timesteps:
            _, eps_next = self.noise_pred_net(obs, action, action_t, next_obs, t)
            next_obs = self.noise_scheduler.step(eps_next, t, next_obs).prev_sample
        return next_obs

    @torch.no_grad()
    def sample_joint(self, obs_dict):
        obs = self.obs_encoder.encode_curr_obs(obs_dict)
        action = torch.randn((obs.shape[0],) + self.action_shape, device=obs.device)
        next_obs = torch.randn((obs.shape[0],) + self.latent_img_shape, device=obs.device)

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            eps_a, eps_next = self.noise_pred_net(obs, action, t, next_obs, t)
            next_obs = self.noise_scheduler.step(eps_next, t, next_obs).prev_sample
            action   = self.noise_scheduler.step(eps_a,   t, action).prev_sample
        return next_obs, action

    @torch.no_grad()
    def sample_forward_dynamics(self, obs_dict, action):
        obs = self.obs_encoder.encode_curr_obs(obs_dict)

        next_obs = torch.randn((obs.shape[0],) + self.latent_img_shape, device=obs.device)

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        action_t = self.noise_scheduler.timesteps[-1]
        for next_t in self.noise_scheduler.timesteps:
            _, eps_next = self.noise_pred_net(obs, action, action_t, next_obs, next_t)
            next_obs = self.noise_scheduler.step(eps_next, next_t, next_obs).prev_sample
        return next_obs

    @torch.no_grad()
    def sample_inverse_dynamics(self, obs_dict, next_obs_dict):
        obs, next_obs = self.obs_encoder.encode_curr_and_next_obs(obs_dict, next_obs_dict)

        action = torch.randn((obs.shape[0],) + self.action_shape, device=obs.device)

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        next_t = self.noise_scheduler.timesteps[-1]
        for action_t in self.noise_scheduler.timesteps:
            eps_a, _ = self.noise_pred_net(obs, action, action_t, next_obs, next_t)
            action = self.noise_scheduler.step(eps_a, action_t, action).prev_sample
        return action
