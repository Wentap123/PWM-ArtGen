import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import apply_rotary_embed

class MLP(nn.Module):
    """Multilayer perceptron with two hidden layers."""

    def __init__(self, in_dim, hidden_dim, out_dim, act=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """Multiheaded self-attention."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        is_causal=False,
        causal_block=1,
        use_sdpa=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal
        self.causal_block = causal_block

        if is_causal and causal_block > 1:
            print("Disabling torch spda kernel for block causal attention")
            self.use_sdpa = False
        else:
            self.use_sdpa = use_sdpa

        if not self.use_sdpa:
            self.causal_block_mat = nn.Parameter(
                torch.ones((causal_block, causal_block)).bool(),
                requires_grad=False,
            )

    def forward(self, x, pos_embed=None, attn_mask=None):
        B, N, D = x.shape

        if attn_mask is not None:
            assert len(attn_mask.shape) == 3
            attn_mask = attn_mask.unsqueeze(1)

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, D // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        if pos_embed is not None:
            q = apply_rotary_embed(q, pos_embed)
            k = apply_rotary_embed(k, pos_embed)

        if self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask, dropout_p=self.attn_drop.p, is_causal=self.is_causal
            )
        else:
            attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
            if self.is_causal:
                assert attn_mask is None
                assert N % self.causal_block == 0
                num_blocks = N // self.causal_block
                block_diag_mat = torch.block_diag(
                    *[self.causal_block_mat for _ in range(num_blocks)]
                )
                triu_mat = torch.triu(
                    torch.ones(N, N, device=x.device), diagonal=1
                ).bool()
                mask = torch.logical_and(~block_diag_mat, triu_mat)
                attn = attn.masked_fill(mask.view(1, 1, N, N), float("-inf"))
            if attn_mask is not None:
                attn = attn.masked_fill(~attn_mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttentionBlock(nn.Module):
    """Multiheaded self-attention block.

    Combines an attention layer and an MLP with residual connections.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        act=nn.GELU,
        norm=nn.LayerNorm,
        is_causal=False,
        causal_block=1,
    ):
        super().__init__()
        self.norm1 = norm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            is_causal=is_causal,
            causal_block=causal_block,
        )
        self.norm2 = norm(dim)
        self.mlp = MLP(
            in_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            out_dim=dim,
            act=act,
            drop=drop,
        )

    def forward(self, x, pos_embed=None, attn_mask=None):
        x = x + self.attn(self.norm1(x), pos_embed, attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class CrossAttention(nn.Module):
    """Multiheaded cross-attention."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        use_spda=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_spda = use_spda

    def forward(self, x, c, x_pos_embed=None, c_pos_embed=None):
        B, Nx, D = x.shape
        q = (
            self.q(x)
            .reshape(B, Nx, self.num_heads, D // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        if x_pos_embed is not None:
            q = apply_rotary_embed(q, x_pos_embed)

        B, Nc, D = c.shape
        kv = (
            self.kv(c)
            .reshape(B, Nc, 2, self.num_heads, D // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        if c_pos_embed is not None:
            k = apply_rotary_embed(k, c_pos_embed)

        if self.use_spda:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        else:
            xattn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
            xattn = xattn.softmax(dim=-1)
            xattn = self.attn_drop(xattn)
            x = xattn @ v

        x = x.transpose(1, 2).reshape(B, Nx, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionBlock(nn.Module):
    """Multiheaded cross-attention block.

    Combines a cross-attention layer and an MLP with residual connections.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        act=nn.GELU,
        norm=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm(dim)
        self.xattn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = norm(dim)
        self.mlp = MLP(
            in_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            out_dim=dim,
            act=act,
            drop=drop,
        )

    def forward(self, x, c, x_pos_embed=None, c_pos_embed=None):
        x = x + self.xattn(x, self.norm1(c), x_pos_embed, c_pos_embed)
        x = x + self.mlp(self.norm2(x))
        return x

class MixedAttentionBlock(nn.Module):
    """Multiheaded mixed-attention block.

    Combines a self-attention, a cross-attention, and an MLP with residual connections.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        act=nn.GELU,
        norm=nn.LayerNorm,
        is_causal=False,
        causal_block=1,
    ):
        super().__init__()
        self.norm1 = norm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            is_causal=is_causal,
            causal_block=causal_block,
        )
        self.norm2 = norm(dim)
        self.xattn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm3 = norm(dim)
        self.mlp = MLP(
            in_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            out_dim=dim,
            act=act,
            drop=drop,
        )

    def forward(self, x, c, x_pos_embed=None, c_pos_embed=None):
        x = x + self.attn(self.norm1(x), x_pos_embed)
        x = x + self.xattn(self.norm2(x), c, x_pos_embed, c_pos_embed)
        x = x + self.mlp(self.norm3(x))
        return x
