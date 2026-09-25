import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class GaussianFourierEmb(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, dim, scale=16, trainable=True):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.trainable = trainable

        W_init = torch.randn(self.dim // 2) * self.scale
        self.W = nn.Parameter(W_init, requires_grad=self.trainable)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([x_proj.sin(), x_proj.cos()], dim=-1)

class PatchEmbed2D(nn.Module):
    """Convert image to patch embedding"""

    def __init__(
        self,
        patch_size=16,
        num_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=num_chans,
            out_channels=embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class PatchEmbed3D(nn.Module):
    """Convert video to patch embedding"""

    def __init__(
        self,
        patch_shape=(2, 16, 16),
        num_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels=num_chans,
            out_channels=embed_dim,
            kernel_size=patch_shape,
            stride=patch_shape,
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class MultiLayerPatchEmbed3D(nn.Module):
    """Convert video to patch embedding"""

    def __init__(
        self,
        num_chans=3,
        embed_dims=[384, 768],
        patch_shapes=[(2, 8, 8), (2, 4, 4)],
    ):
        super().__init__()
        layers = list()
        in_chans = num_chans
        for i, (patch_shape, out_chans) in enumerate(zip(patch_shapes, embed_dims)):
            layers.append(
                nn.Conv3d(
                    in_channels=in_chans,
                    out_channels=out_chans,
                    kernel_size=patch_shape,
                    stride=patch_shape,
                )
            )
            if i < len(patch_shapes) - 1:
                layers.append(nn.GELU())
            in_chans = out_chans
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class MultiviewPatchEmbed3D(nn.Module):
    """Convert video with multiple views to patch embedding"""

    def __init__(
        self,
        patch_shape=(2, 16, 16),
        num_views=1,
        num_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels=num_chans * num_views,
            out_channels=embed_dim * num_views,
            kernel_size=patch_shape,
            stride=patch_shape,
            groups=num_views,
        )
        self.num_views = num_views

    def forward(self, x):
        x = rearrange(x, "B V C T H W -> B (V C) T H W")
        x = self.proj(x)  # (B, V*D, t, h, w)
        x = rearrange(x, "B (V D) t h w -> B (V t h w) D", V=self.num_views)
        return x

def get_nd_rotary_embed(dim, grid_shape, cls_token=False, base=10000):
    """Create n-dimensional rotary positional embeddings.

    Args:
        dim: an int of the embedding dimension.
        grid_shape: a sequence of int of the length along each axis.
        base: the base from which to calculate the rotation angles.

    Returns:
        pos_embed: a tensor of shape (grid_shape[0]*...*grid_shape[N-1], dim) of positional embeddings.

    """
    num_axis = len(grid_shape)
    assert dim % num_axis == 0
    axis_dim = dim // num_axis
    assert axis_dim % 2 == 0

    axis_ticks = [torch.arange(length).float() for length in grid_shape]
    axis_grids = torch.meshgrid(*axis_ticks, indexing="ij")

    axis_thetas = [
        get_1d_rotary_embed(axis_dim, axis_grid.flatten(), base)
        for axis_grid in axis_grids
    ]
    thetas = torch.cat(axis_thetas, dim=-1)
    return thetas

def get_1d_rotary_embed(dim, pos, base=10000):
    """Create 1D rotary positional embeddings from a grid of positions.

    Args:
        dim: the output dimension for each position.
        pos: a tensor of size (seq_len,) of positions to be encoded.

    Returns:
        thetas: a tensor of size (seq_len, dim) of rotary positional embeddings.
    """
    assert dim % 2 == 0
    thetas = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    thetas = torch.outer(pos, thetas)  # (N, D/2)
    thetas = thetas.repeat(1, 2)  # (N, D)
    return thetas

def apply_rotary_embed(x, thetas):
    """Rotates the input tensors by the positional embeddings.

    Args:
        x: a tensor of shape (..., seq_len, dim).
        thetas: a tensor of shape (..., seq_len, dim) of positional embeddings.

    Returns:
        x: a tensor of shape (..., seq_len, dim) of the rotated input tensors.
    """
    assert x.shape[-2:] == thetas.shape[-2:]
    x1, x2 = x.chunk(2, dim=-1)
    x_rotate_half = torch.cat([-x2, x1], dim=-1)
    return x * thetas.cos() + x_rotate_half * thetas.sin()

def get_nd_sinusoidal_embed(dim, grid_shape, cls_token=False):
    """Create n-dimensional sinusoidal positional embeddings.

    Args:
        dim: an int of the embedding dimension.
        grid_shape: a sequence of int of the length along each axis.

    Returns:
        pos_embed: an array of shape (grid_shape[0]*...*grid_shape[N-1], dim) of positional embeddings.

    """
    num_axis = len(grid_shape)
    axis_dim = int(np.ceil(dim / (2 * num_axis))) * 2

    axis_ticks = [np.arange(length, dtype=float) for length in grid_shape]
    axis_grids = np.meshgrid(*axis_ticks, indexing="ij")
    axis_embeds = [
        get_1d_sinusoidal_embed(axis_dim, axis.reshape(-1)) for axis in axis_grids
    ]

    pos_embed = np.concatenate(axis_embeds, axis=1)
    pos_embed = pos_embed[:, :dim]

    if cls_token:
        pos_embed = np.concatenate([np.zeros((1, dim)), pos_embed], axis=0)
    return pos_embed

def get_1d_sinusoidal_embed(dim, pos):
    """Create 1D sinusoidal positional embeddings from a grid of positions.

    Args:
        dim: the output dimension for each position.
        pos: a list of size (seq_len,) of positions to be encoded.

    Returns:
        emb: an array of size (seq_len, dim) of positional embeddings.
    """
    assert dim % 2 == 0
    omega = np.arange(dim // 2, dtype=float)
    omega /= dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    out = np.einsum("m,d->md", pos, omega)  # (N, D/2), outer product

    emb_sin = np.sin(out)  # (N, D/2)
    emb_cos = np.cos(out)  # (N, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (N, D)
    return emb

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
        nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
