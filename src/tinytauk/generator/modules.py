"""Low-level AuK Flux2 transformer modules.

Adapted from Tencent-Hunyuan/AuK under the MIT license. TinyTAuK uses the
PyTorch SDPA path.
"""
# mypy: ignore-errors

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import apply_rotary_pos_emb


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -exponent)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )
        self.layer_need_mask_idx = [
            index for index, layer in enumerate(self.conv1d) if isinstance(layer, nn.Conv1d)
        ]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)
        x = x.permute(0, 2, 1)
        if mask is not None:
            x = x.masked_fill(~mask, 0.0)
        for index, block in enumerate(self.conv1d):
            x = block(x)
            if mask is not None and index in self.layer_need_mask_idx:
                x = x.masked_fill(~mask, 0.0)
        return x.permute(0, 2, 1)


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        mod = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(mod, 6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormFinal(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        mod = self.linear(self.silu(emb))
        scale, shift = torch.chunk(mod, 2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class SwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int | None = None, mult: float = 3.0) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=False)
        self.act_fn = SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_out(self.act_fn(self.linear_in(x)))


def _apply_rope(value: torch.Tensor, rope: Any) -> torch.Tensor:
    if rope is None:
        return value
    freqs, xpos_scale = rope
    scale = xpos_scale if xpos_scale is not None else 1.0
    return apply_rotary_pos_emb(value, freqs, scale)


class AttnProcessor:
    def __init__(self, *, attn_mask_enabled: bool = True) -> None:
        self.attn_mask_enabled = attn_mask_enabled

    def __call__(
        self,
        attn: Any,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        rope: Any = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        query, key, value = attn.to_qkv(x).chunk(3, dim=-1)
        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        query = attn.q_norm(query)
        key = attn.k_norm(key)
        if rope is not None:
            freqs, xpos_scale = rope
            q_scale, k_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_scale)
            key = apply_rotary_pos_emb(key, freqs, k_scale)

        attn_mask = None
        if self.attn_mask_enabled and mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(1)
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        x = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        x = attn.to_out[1](attn.to_out[0](x))
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return x


class JointAttnProcessor:
    def __init__(self, *, attn_mask_enabled: bool = True) -> None:
        self.attn_mask_enabled = attn_mask_enabled

    def __call__(
        self,
        attn: Any,
        x: torch.Tensor,
        *,
        c: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: Any = None,
        c_rope: Any = None,
        c_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        audio_mask = mask
        batch_size = c.shape[0]

        query, key, value = attn.to_qkv(x).chunk(3, dim=-1)
        c_query, c_key, c_value = attn.to_qkv_c(c).chunk(3, dim=-1)
        head_dim = key.shape[-1] // attn.heads

        def heads(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        query, key, value = heads(query), heads(key), heads(value)
        c_query, c_key, c_value = heads(c_query), heads(c_key), heads(c_value)
        query, key = attn.q_norm(query), attn.k_norm(key)
        c_query, c_key = attn.c_q_norm(c_query), attn.c_k_norm(c_key)

        if rope is not None:
            freqs, xpos_scale = rope
            q_scale, k_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_scale)
            key = apply_rotary_pos_emb(key, freqs, k_scale)
        if c_rope is not None:
            freqs, xpos_scale = c_rope
            q_scale, k_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            c_query = apply_rotary_pos_emb(c_query, freqs, q_scale)
            c_key = apply_rotary_pos_emb(c_key, freqs, k_scale)

        query = torch.cat([query, c_query], dim=2)
        key = torch.cat([key, c_key], dim=2)
        value = torch.cat([value, c_value], dim=2)

        combined_mask = mask
        if self.attn_mask_enabled and combined_mask is not None:
            if c_mask is not None:
                combined_mask = torch.cat([combined_mask, c_mask], dim=1)
            else:
                combined_mask = F.pad(combined_mask, (0, c.shape[1]), value=True)

        attn_mask = None
        if self.attn_mask_enabled and combined_mask is not None:
            attn_mask = combined_mask.unsqueeze(1).unsqueeze(1)
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        x, c = output[:, : residual.shape[1]], output[:, residual.shape[1] :]
        x = attn.to_out[1](attn.to_out[0](x))
        c = attn.to_out_c(c)
        if audio_mask is not None:
            x = x.masked_fill(~audio_mask.unsqueeze(-1), 0.0)
        if c_mask is not None:
            c = c.masked_fill(~c_mask.unsqueeze(-1), 0.0)
        return x, c


class Attention(nn.Module):
    def __init__(
        self,
        processor: AttnProcessor | JointAttnProcessor,
        dim: int,
        *,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout
        self.context_dim = context_dim
        self.to_qkv = nn.Linear(dim, 3 * self.inner_dim)
        self.q_norm = nn.RMSNorm(dim_head, elementwise_affine=True)
        self.k_norm = nn.RMSNorm(dim_head, elementwise_affine=True)
        if context_dim is not None:
            self.to_qkv_c = nn.Linear(context_dim, 3 * self.inner_dim)
            self.c_q_norm = nn.RMSNorm(dim_head, elementwise_affine=True)
            self.c_k_norm = nn.RMSNorm(dim_head, elementwise_affine=True)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim), nn.Dropout(dropout)])
        if context_dim is not None:
            self.to_out_c = nn.Linear(self.inner_dim, context_dim)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        *,
        mask: torch.Tensor | None = None,
        rope: Any = None,
        c_rope: Any = None,
        c_mask: torch.Tensor | None = None,
    ) -> Any:
        if c is not None:
            return self.processor(
                self,
                x,
                c=c,
                mask=mask,
                rope=rope,
                c_rope=c_rope,
                c_mask=c_mask,
            )
        return self.processor(self, x, mask=mask, rope=rope)


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        *,
        ff_mult: float = 4,
        dropout: float = 0.1,
        attn_mask_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            AttnProcessor(attn_mask_enabled=attn_mask_enabled),
            dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = SwiGLUFeedForward(dim=dim, mult=ff_mult)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        rope: Any = None,
    ) -> torch.Tensor:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp.unsqueeze(1) * self.ff(norm)
        return x


class MMDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        *,
        ff_mult: float = 4,
        dropout: float = 0.1,
        context_dim: int | None = None,
        attn_mask_enabled: bool = False,
    ) -> None:
        super().__init__()
        context_dim = context_dim or dim
        self.attn_norm_c = AdaLayerNorm(context_dim)
        self.attn_norm_x = AdaLayerNorm(dim)
        self.attn = Attention(
            JointAttnProcessor(attn_mask_enabled=attn_mask_enabled),
            dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            context_dim=context_dim,
        )
        self.ff_norm_c = nn.LayerNorm(context_dim, elementwise_affine=False, eps=1e-6)
        self.ff_c = SwiGLUFeedForward(dim=context_dim, mult=ff_mult)
        self.ff_norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_x = SwiGLUFeedForward(dim=dim, mult=ff_mult)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        rope: Any = None,
        c_rope: Any = None,
        c_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(c, emb=t)
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(x, emb=t)
        x_attn, c_attn = self.attn(
            x=norm_x,
            c=norm_c,
            mask=mask,
            rope=rope,
            c_rope=c_rope,
            c_mask=c_mask,
        )
        c = c + c_gate_msa.unsqueeze(1) * c_attn
        norm_c = self.ff_norm_c(c) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        c = c + c_gate_mlp.unsqueeze(1) * self.ff_c(norm_c)
        x = x + x_gate_msa.unsqueeze(1) * x_attn
        norm_x = self.ff_norm_x(x) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]
        x = x + x_gate_mlp.unsqueeze(1) * self.ff_x(norm_x)
        return c, x


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256) -> None:
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        hidden = self.time_embed(timestep).to(timestep.dtype)
        return self.time_mlp(hidden)
