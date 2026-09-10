"""AuK Flux2Edit backbone for standalone TinyTAuK inference.

Adapted from Tencent-Hunyuan/AuK under the MIT license. The CPU baseline keeps
only the execution path needed by AuK-Flash inference.
"""
# mypy: ignore-errors

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from .modules import AdaLayerNormFinal, ConvPositionEmbedding, DiTBlock, MMDiTBlock, TimestepEmbedding


class AudioPromptEmbedding(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(out_dim)

    def _embed(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.linear(x)
        return self.conv_pos_embed(x, mask=mask) + x

    def forward(
        self,
        x: torch.Tensor,
        ref: torch.Tensor | None = None,
        *,
        drop_audio_cond: bool = False,
        mask: torch.Tensor | None = None,
        ref_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x_emb = self._embed(x, mask=mask)
        if ref is None:
            return x_emb
        if drop_audio_cond:
            ref = torch.zeros_like(ref)
        return x_emb, self._embed(ref, mask=ref_mask)


class Flux2Edit(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 8,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        ff_mult: float = 4,
        latent_dim: int = 100,
        text_hidden_dim: int = 2048,
        checkpoint_activations: bool = False,
        checkpoint_every_n_layers: int = 1,
        attn_backend: str = "torch",
        attn_mask_enabled: bool = False,
        num_layers: int = 8,
        num_single_layers: int = 24,
        **_ignored: Any,
    ) -> None:
        super().__init__()
        if attn_backend != "torch":
            raise ValueError("TinyTAuK CPU baseline supports only torch attention")
        if checkpoint_activations:
            raise ValueError("TinyTAuK CPU inference does not use activation checkpointing")

        self.dim = dim
        self.depth = depth
        self.time_embed = TimestepEmbedding(dim)
        self.txt_norm = nn.RMSNorm(dim, elementwise_affine=True)
        self.txt_proj = nn.Linear(text_hidden_dim, dim)
        self.audio_embed = AudioPromptEmbedding(latent_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.transformer_blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    ff_mult=ff_mult,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for _ in range(num_layers)
            ]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for _ in range(num_single_layers)
            ]
        )
        self.norm_out = AdaLayerNormFinal(dim)
        self.proj_out = nn.Linear(dim, latent_dim)
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_every_n_layers = max(1, checkpoint_every_n_layers)
        self.text_cond: torch.Tensor | None = None
        self.text_uncond: torch.Tensor | None = None
        self.initialize_weights()

    def initialize_weights(self) -> None:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm_x.linear.weight, 0)
            nn.init.constant_(block.attn_norm_x.linear.bias, 0)
            nn.init.constant_(block.attn_norm_c.linear.weight, 0)
            nn.init.constant_(block.attn_norm_c.linear.bias, 0)
        for block in self.single_transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def project_text(self, text: torch.Tensor, *, drop_text: bool = False) -> torch.Tensor:
        context = self.txt_norm(self.txt_proj(text))
        if drop_text:
            context = torch.zeros_like(context)
        return context

    def _project_text_cached(
        self,
        text: torch.Tensor,
        *,
        drop_text: bool,
        cache: bool,
    ) -> torch.Tensor:
        cached = self.text_uncond if drop_text else self.text_cond
        if cache and cached is not None:
            return cached

        context = self.project_text(text, drop_text=drop_text)
        if cache:
            if drop_text:
                self.text_uncond = context
            else:
                self.text_cond = context
        return context

    def clear_cache(self) -> None:
        self.text_cond = None
        self.text_uncond = None

    def _embed_audio(
        self,
        x: torch.Tensor,
        ref: torch.Tensor | None,
        *,
        drop_audio_cond: bool,
        mask: torch.Tensor | None,
        ref_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        if ref is not None and ref.shape[1] == 0:
            ref = None
        embedded = self.audio_embed(
            x,
            ref=ref,
            drop_audio_cond=drop_audio_cond,
            mask=mask,
            ref_mask=ref_mask,
        )
        if not isinstance(embedded, tuple):
            return embedded, mask, 0

        x_emb, ref_emb = embedded
        prompt_len = ref_emb.shape[1]
        audio = torch.cat([ref_emb, x_emb], dim=1)
        if mask is None and ref_mask is None:
            return audio, None, prompt_len
        batch, target_len = x_emb.shape[:2]
        if mask is None:
            mask = torch.ones(batch, target_len, dtype=torch.bool, device=x_emb.device)
        if ref_mask is None:
            ref_mask = torch.ones(batch, prompt_len, dtype=torch.bool, device=ref_emb.device)
        return audio, torch.cat([ref_mask, mask], dim=1), prompt_len

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        c_mask: torch.Tensor | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
        ref: torch.Tensor | None = None,
        ref_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)
        timestep = self.time_embed(time)
        if c_mask is None:
            c_mask = text.abs().sum(-1) > 0

        if cfg_infer:
            c_cond = self._project_text_cached(text, drop_text=False, cache=cache)
            x_cond, mask_cond, prompt_len = self._embed_audio(
                x,
                ref,
                drop_audio_cond=False,
                mask=mask,
                ref_mask=ref_mask,
            )
            c_uncond = self._project_text_cached(text, drop_text=True, cache=cache)
            x_uncond, mask_uncond, _ = self._embed_audio(
                x,
                ref,
                drop_audio_cond=True,
                mask=mask,
                ref_mask=ref_mask,
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            context = torch.cat((c_cond, c_uncond), dim=0)
            timestep = torch.cat((timestep, timestep), dim=0)
            if mask_cond is not None and mask_uncond is not None:
                audio_mask = torch.cat((mask_cond, mask_uncond), dim=0)
            else:
                audio_mask = None
            c_mask = torch.cat((c_mask, c_mask), dim=0)
        else:
            context = self._project_text_cached(text, drop_text=drop_text, cache=cache)
            x, audio_mask, prompt_len = self._embed_audio(
                x,
                ref,
                drop_audio_cond=drop_audio_cond,
                mask=mask,
                ref_mask=ref_mask,
            )

        seq_len = x.shape[1]
        text_len = context.shape[1]
        rope_audio = self.rotary_embed.forward_from_seq_len(seq_len)
        rope_text = self.rotary_embed.forward_from_seq_len(text_len)

        for block in self.transformer_blocks:
            context, x = block(
                x,
                context,
                timestep,
                mask=audio_mask,
                rope=rope_audio,
                c_rope=rope_text,
                c_mask=c_mask,
            )

        x = torch.cat([context, x], dim=1)
        rope = self.rotary_embed.forward_from_seq_len(text_len + seq_len)
        single_mask = torch.cat([c_mask, audio_mask], dim=1) if audio_mask is not None else None
        for block in self.single_transformer_blocks:
            x = block(x, timestep, mask=single_mask, rope=rope)

        x = x[:, text_len + prompt_len :]
        return self.proj_out(self.norm_out(x, timestep))
