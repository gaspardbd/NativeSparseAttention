# -*- coding: utf-8 -*-

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope(dim: int, max_len: int = 4096, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class ReferenceNativeSparseAttention(nn.Module):
    """Pure PyTorch NSA: compression + block selection + sliding window."""

    def __init__(self, cfg) -> None:
        super().__init__()
        if cfg.block_counts < 1 or cfg.block_size < 1 or cfg.window_size < 1:
            raise ValueError("block_counts, block_size and window_size must be >= 1")

        dim = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5
        self.block_size = cfg.block_size
        self.block_counts = cfg.block_counts
        self.window_size = cfg.window_size
        self.mode = getattr(cfg, "nsa_mode", "all")
        valid_modes = {"all", "compression", "selection", "sliding"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid nsa_mode={self.mode!r}. Expected one of {sorted(valid_modes)}")

        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.g_proj = nn.Linear(dim, self.num_heads * 3, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        num_heads = self.num_heads
        head_dim = self.head_dim
        block_size = self.block_size

        q = self.q_proj(x).reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        gates = self.g_proj(x).reshape(batch_size, seq_len, num_heads, 3).permute(0, 2, 1, 3).sigmoid()
        g_cmp, g_slc, g_swa = gates.unbind(dim=-1)

        token_idx = torch.arange(seq_len, device=x.device)
        num_blocks = math.ceil(seq_len / block_size)
        pad_len = num_blocks * block_size - seq_len

        if pad_len > 0:
            k_padded = F.pad(k, (0, 0, 0, pad_len))
            v_padded = F.pad(v, (0, 0, 0, pad_len))
        else:
            k_padded = k
            v_padded = v

        use_cmp = self.mode in {"all", "compression", "selection"}
        use_slc = self.mode in {"all", "selection"}
        use_swa = self.mode in {"all", "sliding"}

        if use_cmp:
            k_cmp = k_padded.reshape(batch_size, num_heads, num_blocks, block_size, head_dim).mean(dim=3)
            v_cmp = v_padded.reshape(batch_size, num_heads, num_blocks, block_size, head_dim).mean(dim=3)

            att_cmp = torch.matmul(q, k_cmp.transpose(-1, -2)) * self.scale
            block_idx = torch.arange(num_blocks, device=x.device)
            block_end = ((block_idx + 1) * block_size - 1).clamp(max=seq_len - 1)
            cmp_ok = token_idx.unsqueeze(1) >= block_end.unsqueeze(0)
            att_cmp = att_cmp.masked_fill(~cmp_ok.unsqueeze(0).unsqueeze(0), float("-inf"))
            w_cmp = F.softmax(att_cmp, dim=-1).nan_to_num(0.0)
            o_cmp = torch.matmul(self.attn_drop(w_cmp), v_cmp)
        else:
            block_idx = torch.arange(num_blocks, device=x.device)
            w_cmp = None
            o_cmp = torch.zeros_like(v)

        att_raw = None
        if use_slc or use_swa:
            att_raw = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if use_slc:
            if w_cmp is None:
                raise RuntimeError("Selection branch requires compression scores")
            block_scores = w_cmp.detach().clone()
            local_block = token_idx // block_size
            is_local_block = block_idx.unsqueeze(0) == local_block.unsqueeze(1)
            block_scores = block_scores.masked_fill(is_local_block.unsqueeze(0).unsqueeze(0), 1.0)

            topk = min(self.block_counts, num_blocks)
            top_blocks = block_scores.topk(topk, dim=-1).indices
            selected_blocks = torch.zeros(
                batch_size,
                num_heads,
                seq_len,
                num_blocks,
                dtype=torch.bool,
                device=x.device,
            )
            selected_blocks.scatter_(3, top_blocks, True)
            token_block = token_idx // block_size
            selection_mask = selected_blocks[..., token_block]
            causal_mask = token_idx.unsqueeze(0) <= token_idx.unsqueeze(1)
            selection_mask = selection_mask & causal_mask.unsqueeze(0).unsqueeze(0)

            att_slc = att_raw.masked_fill(~selection_mask, float("-inf"))
            w_slc = F.softmax(att_slc, dim=-1).nan_to_num(0.0)
            o_slc = torch.matmul(self.attn_drop(w_slc), v)
        else:
            o_slc = torch.zeros_like(v)

        if use_swa:
            token_distance = token_idx.unsqueeze(1) - token_idx.unsqueeze(0)
            sliding_mask = (token_distance >= 0) & (token_distance < self.window_size)
            att_swa = att_raw.masked_fill(~sliding_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            w_swa = F.softmax(att_swa, dim=-1).nan_to_num(0.0)
            o_swa = torch.matmul(self.attn_drop(w_swa), v)
        else:
            o_swa = torch.zeros_like(v)

        if self.mode == "compression":
            out = g_cmp.unsqueeze(-1) * o_cmp
        elif self.mode == "selection":
            out = g_slc.unsqueeze(-1) * o_slc
        elif self.mode == "sliding":
            out = g_swa.unsqueeze(-1) * o_swa
        else:
            out = (
                g_cmp.unsqueeze(-1) * o_cmp
                + g_slc.unsqueeze(-1) * o_slc
                + g_swa.unsqueeze(-1) * o_swa
            )
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(out)