from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from models.common.language import CLIPTextEncoder
from models.common.transforms import VideoTransform, VAEDownsample
from models.common.vision import ResNetImageEncoder, ViTImageEncoder, DinoV2ImageEncoder
from models.common.vision import GlobalImageTeacher

class PWMObservationEncoder(nn.Module):
    def __init__(
        self,
        shape_meta: dict,
        num_frames: int,
        embed_dim: int,
        resize_shape: Tuple[int, int] = None,
        crop_shape: Tuple[int, int] = None,
        random_crop: bool = True,
        color_jitter: Optional[Dict] = None,
        imagenet_norm: bool = False,
        vision_backbone: str = "resnet",
        use_low_dim: bool = True,
        use_language: bool = False,
        vae_path: str = None,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        self.num_frames = num_frames*2  # curr + next
        self.embed_dim = embed_dim

        self.rgb_keys = sorted([k for k,v in shape_meta["obs"].items() if v["type"] == "rgb"])
        self.low_dim_keys = sorted([k for k,v in shape_meta["obs"].items() if v["type"] == "low_dim"])

        self.num_views = len(self.rgb_keys) // 2

        self.obs_transform = VideoTransform(
            resize_shape=resize_shape,
            crop_shape=crop_shape,
            random_crop=random_crop,
            color_jitter=color_jitter,
            imagenet_norm=imagenet_norm,
        )

        def _make_encoder(nv: int):
            if vision_backbone == "vit":
                return ViTImageEncoder(num_views=nv, embed_dim=embed_dim)
            elif vision_backbone == "resnet":
                return ResNetImageEncoder(num_views=nv, embed_dim=embed_dim)
            elif vision_backbone == "dinov2":
                return DinoV2ImageEncoder(num_views=nv, embed_dim=embed_dim)
            else:
                raise NotImplementedError(f"Unsupported backbone: {vision_backbone}")

        self.img_encoder = _make_encoder(self.num_views)

        self.use_low_dim = use_low_dim
        self.use_language = use_language
        self.text_encoder = CLIPTextEncoder(embed_dim=embed_dim) if use_language else None

        self.vae = VAEDownsample(vae_path=vae_path)

    def _apply_transform_for_keys(self, obs_dicts: Union[dict, list[dict]], keys: list[str]):
        if isinstance(obs_dicts, dict):
            obs_dicts = [obs_dicts]
            singleton = True
        else:
            singleton = False
        assert isinstance(obs_dicts, list)
        n = len(obs_dicts)
        out = [[] for _ in range(n)]
        for k in keys:
            if k not in obs_dicts[0]:
                continue
            combo = torch.cat([od[k] for od in obs_dicts], dim=0)
            combo = self.obs_transform(combo)
            chunks = combo.chunk(n, dim=0)
            for i, ch in enumerate(chunks):
                out[i].append(ch)
        out = [
            v.permute(0, 3, 2, 1, 4, 5) if v.numel() > 0 else torch.empty(0)
            for v in [torch.stack(v, dim=1) if len(v) > 0 else torch.empty(0) for v in out]
        ]
        return out[0] if singleton else out

    def apply_transform(self, obs_dicts: Union[dict, list[dict]]):
        return self._apply_transform_for_keys(obs_dicts, self.rgb_keys)

    def apply_vae(self, imgs_list: Union[torch.Tensor, list[torch.Tensor]], inverse: bool=False, microbatch_size: int=32):
        if isinstance(imgs_list, torch.Tensor):
            imgs_list = [imgs_list]; singleton=True
        else:
            singleton=False
        imgs = torch.cat(imgs_list, dim=0)
        B, V = imgs.shape[:2]
        imgs = rearrange(imgs, "b v c t h w -> (b v t) c h w")

        device = next(self.vae.parameters()).device
        dtype = next(self.vae.parameters()).dtype
        out = []
        for i in range(0, imgs.shape[0], microbatch_size):
            batch = imgs[i:i+microbatch_size].to(device, dtype=dtype)
            y = self.vae.inverse(batch) if inverse else self.vae(batch)
            out.append(y)
        out = torch.cat(out, dim=0)
        out = rearrange(out, "(b v t) c h w -> b v c t h w", b=B, v=V)
        if not singleton:
            sizes = [x.shape[0] for x in imgs_list]
            out = list(out.split(sizes, dim=0))
        return out

    def encode_curr_obs(self, curr_obs_dict: dict):

        curr_imgs = self.apply_transform(curr_obs_dict)      # (B,V,C,T,H,W)
        feats = self.img_encoder(curr_imgs)                  # (B, V*T*D)

        if self.use_low_dim and len(self.low_dim_keys) > 0:
            lows = [curr_obs_dict[k] for k in self.low_dim_keys]
            lows = torch.cat(lows, dim=-1).flatten(1)        # (B, T*D_low)
            feats = torch.cat([feats, lows], dim=-1)

        if self.use_language:
            lang = self.text_encoder(
                input_ids=curr_obs_dict["input_ids"],
                attention_mask=curr_obs_dict["attention_mask"],
            )
            feats = torch.cat([feats, lang], dim=-1)
        return feats

    def encode_next_obs(self, next_obs_dict: dict):
        next_imgs = self.apply_transform(next_obs_dict)
        next_latents = self.apply_vae(next_imgs)
        return next_latents

    def encode_curr_and_next_obs(self, curr_obs_dict: dict, next_obs_dict: dict):
        curr_imgs, next_imgs = self.apply_transform([curr_obs_dict, next_obs_dict])
        feats = self.img_encoder(curr_imgs)

        if self.use_low_dim and len(self.low_dim_keys) > 0:
            lows = [curr_obs_dict[k] for k in self.low_dim_keys]
            lows = torch.cat(lows, dim=-1).flatten(1)
            feats = torch.cat([feats, lows], dim=-1)

        if self.use_language:
            lang = self.text_encoder(
                input_ids=curr_obs_dict["input_ids"],
                attention_mask=curr_obs_dict["attention_mask"],
            )
            feats = torch.cat([feats, lang], dim=-1)

        next_latents = self.apply_vae(next_imgs)
        return feats, next_latents

    def feat_dim(self):
        low_dim_size = sum(self.shape_meta["obs"][k]["shape"][-1] for k in self.low_dim_keys) if len(self.low_dim_keys)>0 else 0
        return (
            self.num_views * self.num_frames * self.embed_dim
            + (low_dim_size if self.use_low_dim and low_dim_size>0 else 0)
            + (self.embed_dim if self.use_language else 0)
        )


    def latent_img_shape(self):
        dummy_obs = {}
        for k in self.rgb_keys:
            img_shape = self.shape_meta["obs"][k]["shape"]
            dummy_obs[k] = torch.zeros(
                1, self.num_views, *img_shape, dtype=torch.uint8
            )
        with torch.no_grad():
            latent = self.encode_next_obs(dummy_obs)
        return tuple(latent.shape[1:])
