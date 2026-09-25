from typing import Callable

import timm
import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from torchvision.transforms import Normalize
import torch.nn.functional as F
import contextlib
from torchvision.transforms import v2 as Tv2
def get_imagenet_norm(inplace=True):
    """
    Construct an ImageNet normalization transform.
    """
    return Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=inplace
    )

transform = Tv2.Compose([
    Tv2.Resize(256, interpolation=Tv2.InterpolationMode.BICUBIC, antialias=True),
    Tv2.CenterCrop(224),
    Tv2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Recursively replace submodules that satisfy a given predicate.

    Args:
        root_module (nn.Module): The root module to process.
        predicate (Callable[[nn.Module], bool]): A function that takes a module as input and
            returns True if the module should be replaced.
        func (Callable[[nn.Module], nn.Module]): A function that takes a module as input and
            returns a new module to replace it.
        **kwargs: Additional keyword arguments to be passed to the ResNet model constructor.
    """
    if predicate(root_module):
        return func(root_module)

    module_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in module_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)

    module_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(module_list) == 0
    return root_module

def get_resnet(name, embed_dim, weights=None, replace_batch_norm=True, **kwargs):
    """
    Construct a ResNet model with a custom output embedding dimension and optional batch norm replacement.

    Args:
        name (str): The name of the ResNet architecture to use (e.g., "resnet18", "resnet34", "resnet50").
        embed_dim (int): The dimension of the output embedding.
        weights (Optional[str]): Pre-trained weights to load (e.g., "IMAGENET1K_V1"). If None, no pre-trained weights are used.
        replace_batch_norm (bool, optional): If True, replaces `nn.BatchNorm2d` layers with `nn.GroupNorm` layers. Default is True.
    """
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = nn.Linear(resnet.fc.in_features, embed_dim)
    if replace_batch_norm:
        resnet = replace_submodules(
            root_module=resnet,
            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            func=lambda x: nn.GroupNorm(
                num_groups=x.num_features // 16,
                num_channels=x.num_features,
            ),
        )
    return resnet

def get_vit(name, embed_dim, weights=None, **kwargs):
    """
    Construct a Vision Transformer (ViT) model with a custom output embedding dimension.

    Args:
        name (str): The name of the ViT architecture to use (e.g., "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14").
        embed_dim (int): The dimension of the output embedding.
        weights (Optional[str]): Pre-trained weights to load (e.g., "IMAGENET1K_V1"). If None, no pre-trained weights are used.
        **kwargs: Additional keyword arguments to be passed to the ViT model constructor.
    """
    func = getattr(torchvision.models, name)
    vit = func(weights=weights, **kwargs)
    vit.heads = nn.Linear(768, embed_dim)
    return vit

def get_clip(embed_dim, **kwargs):
    """
    Construct a pretrained CLIP encoder with a custom output embedding dimension.

    Args:
        embed_dim (int): The dimension of the output embedding.
        **kwargs: Additional keyword arguments to be passed to the timm model creation function.
    """
    clip = timm.create_model(
        "hf_hub:timm/vit_base_patch32_clip_224.openai", pretrained=True, **kwargs
    )
    clip.head = nn.Linear(768, embed_dim)
    return clip

def get_dinov2(name: str, embed_dim: int, **kwargs) -> nn.Module:

    backbone = torch.hub.load('facebookresearch/dinov2', name, **kwargs)
    return backbone

class DinoV2ImageEncoder(nn.Module):

    def __init__(self, num_views: int, embed_dim: int =768, model_name: str = 'dinov2_vitb14_reg'):
        super().__init__()
        self.num_views = num_views
        self.model = get_dinov2(model_name, embed_dim)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def forward(self, imgs: torch.Tensor):
        B, V = imgs.shape[:2]
        x = rearrange(imgs, "b v c t h w -> (b v t) c h w")
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        feats = self.model.forward_features(transform(x))["x_norm_clstoken"]                # (b*v*t, embed_dim)
        feats = rearrange(feats, "(b v t) c -> b (v t c)", b=B, v=V)
        return feats

class ResNetImageEncoder(nn.Module):
    """
    Multi-view image encoder using a ResNet backbone.

    The input is expected to be a tensor with shape (B, V, C, T, H, W), where:
      - B is the batch size,
      - V is the number of views,
      - C is the number of channels,
      - T is the number of frames,
      - H and W are the image height and width, respectively.
    The encoder reshapes the input to treat each view and frame as an individual image,
    extracts features using the ResNet model, and then concatenates the features across
    all views and frames.

    Args:
        num_views (int): Number of camera views in the input.
        embed_dim (int): Dimension of the output embedding features.
    """

    def __init__(self, num_views: int, embed_dim: int):
        super().__init__()
        self.num_views = num_views
        self.norm = get_imagenet_norm()
        self.model = get_resnet("resnet18", embed_dim, weights="IMAGENET1K_V1")

    def forward(self, imgs: torch.Tensor):
        B, V = imgs.shape[:2]
        imgs = rearrange(imgs, "b v c t h w -> (b v t) c h w")
        feats = self.model(self.norm(imgs))
        feats = rearrange(feats, "(b v t) c -> b (v t c)", b=B, v=V)
        return feats

class ViTImageEncoder(nn.Module):
    """
    Multi-view image encoder using a Vision Transformer (ViT) backbone.

    Args:
        num_views (int): Number of camera views in the input.
        embed_dim (int): Dimension of the output embedding features.
    """

    def __init__(self, num_views: int, embed_dim: int):
        super().__init__()
        self.num_views = num_views
        self.norm = get_imagenet_norm()
        self.model = get_vit("vit_b_32", embed_dim, weights="IMAGENET1K_V1")

    def forward(self, imgs: torch.Tensor):
        B, V = imgs.shape[:2]
        imgs = rearrange(imgs, "b v c t h w -> (b v t) c h w")
        imgs = self.norm(imgs)

        x = self.model._process_input(imgs)

        batch_cls_token = self.model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_cls_token, x], dim=1)

        x = self.model.encoder(x)
        x = self.model.heads(x[:, 0])
        feats = rearrange(x, "(b v t) c -> b (v t c)", b=B, v=V)
        return feats  # (b, v*t, c)

class ViTImagePatchEncoder(nn.Module):
    """
    Multi-view image patch encoder using a Vision Transformer (ViT) backbone with learnable positional embeddings.

    Args:
        num_views (int): Number of camera views in the input.
        num_frames (int): Number of frames per view.
        embed_dim (int): Dimension of the output embedding features.
    """

    def __init__(self, num_views: int, num_frames: int, embed_dim: int):
        super().__init__()
        self.num_views = num_views
        self.norm = get_imagenet_norm()
        self.model = get_vit("vit_b_32", embed_dim, weights="IMAGENET1K_V1")

        self.pos_shift = nn.Parameter(
            torch.zeros(1, num_views * num_frames, 1, embed_dim),
            requires_grad=True,
        )
        self.pos_scale = nn.Parameter(
            torch.zeros(1, num_views * num_frames, 1, embed_dim),
            requires_grad=True,
        )

    def forward(self, imgs: torch.Tensor):
        B, V = imgs.shape[:2]
        imgs = rearrange(imgs, "b v c t h w -> (b v t) c h w")
        imgs = self.norm(imgs)

        x = self.model._process_input(imgs)

        batch_cls_token = self.model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_cls_token, x], dim=1)

        x = self.model.encoder(x)
        x = self.model.heads(x)

        feats = rearrange(x, "(b v t) n c -> b (v t) n c", b=B, v=V)
        feats = feats * (1 + self.pos_scale) + self.pos_shift
        return feats.flatten(1, 2)  # (b, v*t*n, c)

class DinoV2ImagePatchEncoder(nn.Module):

    def __init__(
        self,
        num_views: int,
        freeze_backbone: bool = True,
        model_name: str = "dinov2_vitb14_reg",
        **hub_kwargs,
    ):
        super().__init__()
        self.num_views = num_views
        self.norm = get_imagenet_norm()
        self.backbone = get_dinov2_backbone(model_name, **hub_kwargs)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def forward(self, imgs: torch.Tensor):
        B, V = imgs.shape[:2]
        x = rearrange(imgs, "b v c t h w -> (b v t) c h w")
        x = self.norm(x)

        ctx = (torch.no_grad()
               if not any(p.requires_grad for p in self.backbone.parameters())
               else contextlib.nullcontext())
        with ctx:
            feats_dict = self.backbone.forward_features(x)
            patch_tokens = feats_dict["x_norm_patchtokens"]      # (BVT, N, 768)

        feats = rearrange(patch_tokens, "(b v t) n c -> b (v t n) c", b=B, v=V)
        return feats  # (B, V*T*N, 768)

class GlobalImageTeacher(nn.Module):

    def __init__(self, embed_dim: int, backbone: str = "dinov2_vitb14_reg"):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim

        if "dinov2" in backbone:
            self.model = torch.hub.load("facebookresearch/dinov2", backbone, pretrained=True)
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
            self.in_norm = Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], inplace=False)
            self.in_size = 224
            d_in = getattr(self.model, "embed_dim", 768)
            self.proj = nn.Linear(d_in, embed_dim)
            for p in self.proj.parameters():
                p.requires_grad = False
            self._mode = "dinov2"
        elif backbone == "vit_b_32":
            self.model = timm.create_model("hf_hub:timm/vit_base_patch32_clip_224.openai", pretrained=True)
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
            self.in_norm = Normalize(mean=[0.48145466,0.4578275,0.40821073],
                                     std=[0.26862954,0.26130258,0.27577711], inplace=False)
            self.in_size = 224
            d_in = 768
            self.proj = nn.Linear(d_in, embed_dim)
            self._mode = "clip32"
        else:
            raise NotImplementedError(f"Unknown REPA teacher backbone: {backbone}")

    @torch.no_grad()
    def forward(self, imgs):
        B, V, C, T, H, W = imgs.shape
        if imgs.min() < 0:
            imgs = (imgs + 1.0) * 0.5
        x = imgs.flatten(0,1)          # (B*V, C, T, H, W)
        x = x.permute(0,2,1,3,4)       # (B*V, T, C, H, W)
        x = x.reshape(B*V*T, C, H, W)  # (B*V*T, C, H, W)

        x = F.interpolate(x, size=(self.in_size, self.in_size), mode="bilinear", align_corners=False)
        x = self.in_norm(x)

        if self._mode == "dinov2":
            feats = self.model.forward_features(x)
            patch = feats["x_norm_patchtokens"]     # (B*V*T, L, d_in), L = 16*16
            cls   = feats["x_norm_clstoken"]        # (B*V*T, d_in)
        else:
            tokens = self.model.forward_features(x)
            cls = tokens[:, 0]
            patch = tokens[:, 1:]

        patch = self.proj(patch)     # (B*V*T, L, D)
        cls   = self.proj(cls)       # (B*V*T, D)

        L = patch.shape[1]
        Ht = Wt = int(L ** 0.5)      # DINOv2-B/14 @224 -> 16x16
        patch = patch.view(B, V, T, Ht, Wt, -1)        # (B,V,T,Ht,Wt,D)
        cls   = cls.view(B, V, T, -1)                  # (B,V,T,D)

        return {"global": cls, "patch": patch, "grid": (Ht, Wt)}
