"""
NSA vs Full Attention — Training Comparison on WikiText-2
=========================================================
Self-contained script for Google Colab Free (T4 GPU).
No Triton / flash-attn / fla dependency — pure PyTorch.

Implements the three NSA branches from the paper:
  1. Compression  (mean-pooled block keys/values)
  2. Selection    (top-k block selection based on compression scores)
  3. Sliding Window (local causal window)
Combined via learned sigmoid gates, compared against standard causal attention.

Usage on Colab:
  1.  !pip install -q datasets transformers matplotlib tqdm
  2.  Copy-paste this file or upload & run:  !python train_nsa_vs_full.py
"""

import math
import time
import copy
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# Configuration
# ============================================================
class Cfg:
    # --- model ---
    vocab_size:    int = 50257        # GPT-2 tokenizer
    hidden_size:   int = 256
    num_heads:     int = 8
    head_dim:      int = 32           # hidden_size // num_heads
    num_layers:    int = 6
    mlp_hidden:    int = 512
    max_seq_len:   int = 512
    dropout:       float = 0.1

    # --- NSA ---
    block_size:    int = 64           # compression / selection block size
    block_counts:  int = 4            # top-k blocks per query
    window_size:   int = 128          # sliding-window width

    # --- training ---
    batch_size:    int = 8
    lr:            float = 5e-4
    weight_decay:  float = 0.1
    epochs:        int = 5
    warmup_steps:  int = 200
    eval_every:    int = 100          # evaluate every N steps
    grad_clip:     float = 1.0
    seed:          int = 42

CFG = Cfg()

# ============================================================
# Building blocks
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).type_as(x) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def precompute_rope(dim, max_len=4096, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """x: [B, H, T, D]  —  cos/sin: [max_len, D//2]"""
    T = x.shape[2]
    c = cos[:T].unsqueeze(0).unsqueeze(0)          # [1, 1, T, D//2]
    s = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


# ============================================================
# Full Causal Attention
# ============================================================
class FullAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.hidden_size
        self.H = cfg.num_heads
        self.D = cfg.head_dim
        self.scale = self.D ** -0.5

        self.q_proj = nn.Linear(dim, self.H * self.D, bias=False)
        self.k_proj = nn.Linear(dim, self.H * self.D, bias=False)
        self.v_proj = nn.Linear(dim, self.H * self.D, bias=False)
        self.o_proj = nn.Linear(self.H * self.D, dim, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope_cos, rope_sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.H, self.D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.H, self.D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.H, self.D).transpose(1, 2)

        q, k = apply_rope(q, rope_cos, rope_sin), apply_rope(k, rope_cos, rope_sin)

        att = (q @ k.transpose(-1, -2)) * self.scale
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        att = att.masked_fill(causal, float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))

        out = (att @ v).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


# ============================================================
# Native Sparse Attention (pure-PyTorch, mask-based)
# ============================================================
class NSAAttention(nn.Module):
    """
    Faithful implementation of NSA's three branches:
      • Compression:     attend to mean-pooled block representations
      • Selection:       top-k block selection guided by compression scores
      • Sliding window:  local causal window
    Combined through learned per-head sigmoid gates.
    """

    def __init__(self, cfg):
        super().__init__()
        dim = cfg.hidden_size
        self.H = cfg.num_heads
        self.D = cfg.head_dim
        self.scale = self.D ** -0.5
        self.BS = cfg.block_size
        self.N_SEL = cfg.block_counts
        self.W = cfg.window_size

        self.q_proj = nn.Linear(dim, self.H * self.D, bias=False)
        self.k_proj = nn.Linear(dim, self.H * self.D, bias=False)
        self.v_proj = nn.Linear(dim, self.H * self.D, bias=False)
        self.o_proj = nn.Linear(self.H * self.D, dim, bias=False)
        self.g_proj = nn.Linear(dim, self.H * 3, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope_cos, rope_sin):
        B, T, _ = x.shape
        BS, H, D = self.BS, self.H, self.D
        C = T // BS                          # number of compression blocks

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)   # [B,H,T,D]
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
        q, k = apply_rope(q, rope_cos, rope_sin), apply_rope(k, rope_cos, rope_sin)

        gates = self.g_proj(x).view(B, T, H, 3).permute(0, 2, 1, 3).sigmoid()
        g_cmp, g_slc, g_swa = gates.unbind(-1)   # each [B,H,T]

        t_idx = torch.arange(T, device=x.device)
        c_idx = torch.arange(C, device=x.device)

        # ---- 1. Compression branch ----
        k_cmp = k.view(B, H, C, BS, D).mean(3)   # [B,H,C,D]
        v_cmp = v.view(B, H, C, BS, D).mean(3)

        att_cmp = (q @ k_cmp.transpose(-1, -2)) * self.scale       # [B,H,T,C]
        # causal: query t may use block c only when all of c's tokens have been seen
        cmp_ok = t_idx.unsqueeze(1) >= (c_idx.unsqueeze(0) + 1) * BS  # [T,C]
        att_cmp = att_cmp.masked_fill(~cmp_ok.unsqueeze(0).unsqueeze(0), float("-inf"))
        w_cmp = F.softmax(att_cmp, dim=-1).nan_to_num(0)              # [B,H,T,C]
        o_cmp = self.attn_drop(w_cmp) @ v_cmp                         # [B,H,T,D]

        # ---- 2. Selection branch ----
        # block importance = compression attention weight  (detached — no grad through selection)
        blk_imp = w_cmp.detach().clone()
        # the block containing the query always gets high importance (cf. paper §3.3.2)
        local_blk = t_idx // BS                                       # [T]
        local_eq  = c_idx.unsqueeze(0) == local_blk.unsqueeze(1)      # [T,C]
        blk_imp   = blk_imp.masked_fill(local_eq.unsqueeze(0).unsqueeze(0), 1.0)

        n = min(self.N_SEL, C)
        _, top_c = blk_imp.topk(n, dim=-1)                            # [B,H,T,n]

        # scatter into [B,H,T,C] bool → expand to token-level [B,H,T,T]
        sel = torch.zeros(B, H, T, C, dtype=torch.bool, device=x.device)
        sel.scatter_(3, top_c, True)
        pos_blk = t_idx // BS                                         # [T]
        slc_mask = sel[..., pos_blk]                                   # [B,H,T,T]
        causal_2d = t_idx.unsqueeze(0) <= t_idx.unsqueeze(1)          # [T,T]
        slc_mask  = slc_mask & causal_2d.unsqueeze(0).unsqueeze(0)

        att_raw = (q @ k.transpose(-1, -2)) * self.scale              # [B,H,T,T]
        att_slc = att_raw.masked_fill(~slc_mask, float("-inf"))
        w_slc   = F.softmax(att_slc, dim=-1).nan_to_num(0)
        o_slc   = self.attn_drop(w_slc) @ v                           # [B,H,T,D]

        # ---- 3. Sliding-window branch ----
        diff = t_idx.unsqueeze(1) - t_idx.unsqueeze(0)                # [T,T]
        swa_mask = (diff >= 0) & (diff < self.W)
        att_swa = att_raw.masked_fill(~swa_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        w_swa   = F.softmax(att_swa, dim=-1).nan_to_num(0)
        o_swa   = self.attn_drop(w_swa) @ v                           # [B,H,T,D]

        # ---- gated combination ----
        out = (g_cmp.unsqueeze(-1) * o_cmp
             + g_slc.unsqueeze(-1) * o_slc
             + g_swa.unsqueeze(-1) * o_swa)

        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


# ============================================================
# Transformer block + Language model
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(self, cfg, attn_cls):
        super().__init__()
        self.ln1  = RMSNorm(cfg.hidden_size)
        self.attn = attn_cls(cfg)
        self.ln2  = RMSNorm(cfg.hidden_size)
        self.mlp  = SwiGLU(cfg.hidden_size, cfg.mlp_hidden)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.ln1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.ln2(x))
        return x


class SmallLM(nn.Module):
    def __init__(self, cfg, attn_cls):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList(
            [TransformerBlock(cfg, attn_cls) for _ in range(cfg.num_layers)]
        )
        self.ln_f    = RMSNorm(cfg.hidden_size)
        self.head    = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight         # weight tying

        rope_cos, rope_sin = precompute_rope(cfg.head_dim, cfg.max_seq_len + 64)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        self.apply(self._init_weights)
        # scaled residual init (GPT-2 style)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.o_proj.weight, std=0.02 / math.sqrt(2 * cfg.num_layers))
            nn.init.normal_(blk.mlp.w2.weight,      std=0.02 / math.sqrt(2 * cfg.num_layers))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin)
        return self.head(self.ln_f(x))

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Dataset: WikiText-2 (tokenised with GPT-2 tokenizer)
# ============================================================
def load_data(cfg):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading WikiText-2 …")
    ds  = load_dataset("wikitext", "wikitext-2-raw-v1")
    tok = AutoTokenizer.from_pretrained("gpt2")

    def encode_split(split):
        ids = []
        for line in split["text"]:
            if line.strip():
                ids.extend(tok.encode(line))
        length = cfg.max_seq_len + 1        # +1 for next-token target
        n = len(ids) // length
        return torch.tensor(ids[: n * length], dtype=torch.long).view(n, length)

    train_t = encode_split(ds["train"])
    val_t   = encode_split(ds["validation"])
    print(f"  train chunks: {len(train_t):,}  |  val chunks: {len(val_t):,}")
    return train_t, val_t


# ============================================================
# Training utilities
# ============================================================
def cosine_lr(step, warmup, total, lr):
    if step < warmup:
        return lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, total_tok = 0.0, 0
    for (batch,) in loader:
        batch = batch.to(DEVICE)
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tok  += y.numel()
    model.train()
    return total_loss / total_tok


def train_model(model, train_loader, val_loader, cfg, label="model"):
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    total_steps = cfg.epochs * len(train_loader)
    step = 0
    train_losses, val_losses, val_steps = [], [], []

    val_loss = evaluate(model, val_loader)
    val_losses.append(val_loss)
    val_steps.append(0)
    print(f"[{label}] step 0 — val loss {val_loss:.4f}  ppl {math.exp(val_loss):.1f}")

    for epoch in range(1, cfg.epochs + 1):
        pbar = tqdm(train_loader, desc=f"[{label}] epoch {epoch}/{cfg.epochs}")
        for (batch,) in pbar:
            batch = batch.to(DEVICE)
            x, y = batch[:, :-1], batch[:, 1:]

            lr = cosine_lr(step, cfg.warmup_steps, total_steps, cfg.lr)
            for pg in opt.param_groups:
                pg["lr"] = lr

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            train_losses.append(loss.item())
            step += 1

            if step % cfg.eval_every == 0:
                val_loss = evaluate(model, val_loader)
                val_losses.append(val_loss)
                val_steps.append(step)
                pbar.set_postfix(loss=f"{loss.item():.3f}", val=f"{val_loss:.3f}", lr=f"{lr:.1e}")

    val_loss = evaluate(model, val_loader)
    val_losses.append(val_loss)
    val_steps.append(step)
    print(f"[{label}] done — final val loss {val_loss:.4f}  ppl {math.exp(val_loss):.1f}")
    return {"train": train_losses, "val": val_losses, "val_steps": val_steps}


# ============================================================
# Main
# ============================================================
def main():
    cfg = CFG
    torch.manual_seed(cfg.seed)

    # ---- data ----
    train_t, val_t = load_data(cfg)
    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(TensorDataset(train_t), batch_size=cfg.batch_size, shuffle=True,  generator=g)
    val_loader   = DataLoader(TensorDataset(val_t),   batch_size=cfg.batch_size, shuffle=False)

    # ---- Full Attention ----
    print("\n========== Full Attention ==========")
    torch.manual_seed(cfg.seed)
    model_full = SmallLM(cfg, FullAttention).to(DEVICE)
    print(f"  params: {model_full.param_count():,}")
    res_full = train_model(model_full, train_loader, val_loader, cfg, label="FullAttn")

    # ---- NSA ----
    print("\n========== NSA ==========")
    torch.manual_seed(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(TensorDataset(train_t), batch_size=cfg.batch_size, shuffle=True, generator=g)
    model_nsa = SmallLM(cfg, NSAAttention).to(DEVICE)
    print(f"  params: {model_nsa.param_count():,}")
    res_nsa = train_model(model_nsa, train_loader, val_loader, cfg, label="NSA")

    # ---- plots ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # smoothed training loss
    def smooth(vals, w=50):
        out = []
        for i in range(len(vals)):
            lo = max(0, i - w)
            out.append(sum(vals[lo:i+1]) / (i + 1 - lo))
        return out

    ax1.plot(smooth(res_full["train"]), label="Full Attention", alpha=0.85)
    ax1.plot(smooth(res_nsa["train"]),  label="NSA",            alpha=0.85)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Loss (cross-entropy)")
    ax1.set_title("Training Loss (smoothed)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(res_full["val_steps"], res_full["val"], "o-", label="Full Attention")
    ax2.plot(res_nsa["val_steps"],  res_nsa["val"],  "s-", label="NSA")
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("nsa_vs_full.png", dpi=150)
    plt.show()
    print("Plot saved → nsa_vs_full.png")

    # ---- save numbers ----
    with open("results.json", "w") as f:
        json.dump({
            "full_attention": {
                "final_val_loss": res_full["val"][-1],
                "final_val_ppl":  math.exp(res_full["val"][-1]),
            },
            "nsa": {
                "final_val_loss": res_nsa["val"][-1],
                "final_val_ppl":  math.exp(res_nsa["val"][-1]),
            },
        }, f, indent=2)
    print("Numbers saved → results.json")


if __name__ == "__main__":
    main()
