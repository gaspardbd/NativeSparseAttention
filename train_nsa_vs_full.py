"""Training script: Full Attention vs NSA reference on WikiText-2."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from native_sparse_attention.pytorch_reference import (
    ReferenceNativeSparseAttention,
    apply_rope,
    precompute_rope,
)


@dataclass
class ExperimentConfig:
    tokenizer_name: str = "gpt2"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    text_field: str = "text"
    vocab_size: int = 50257

    hidden_size: int = 256
    num_heads: int = 8
    num_layers: int = 4
    mlp_hidden: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.0

    block_size: int = 32
    block_counts: int = 4
    window_size: int = 64

    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.1
    epochs: int = 2
    warmup_steps: int = 20
    eval_every: int = 20
    grad_clip: float = 1.0
    seed: int = 42

    max_train_chunks: int = 512
    max_val_chunks: int = 128
    smoothing_window: int = 20
    zero_init_nsa_gates: bool = True
    show_elapsed_time_plot: bool = False
    compare_nsa_branches: bool = False
    nsa_mode: str = "all"
    device: Optional[str] = None

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        return self.hidden_size // self.num_heads


EXPERIMENT_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "hidden_size": 256,
        "num_heads": 8,
        "num_layers": 4,
        "mlp_hidden": 1024,
        "max_seq_len": 256,
        "batch_size": 8,
        "epochs": 2,
        "eval_every": 20,
        "max_train_chunks": 512,
        "max_val_chunks": 128,
        "block_size": 32,
        "block_counts": 4,
        "window_size": 64,
    },
    "standard": {
        "hidden_size": 256,
        "num_heads": 8,
        "num_layers": 4,
        "mlp_hidden": 1024,
        "max_seq_len": 256,
        "batch_size": 8,
        "epochs": 3,
        "eval_every": 40,
        "max_train_chunks": 1536,
        "max_val_chunks": 256,
        "block_size": 32,
        "block_counts": 4,
        "window_size": 64,
    },
    "representative": {
        "hidden_size": 384,
        "num_heads": 8,
        "num_layers": 6,
        "mlp_hidden": 1536,
        "max_seq_len": 256,
        "batch_size": 8,
        "epochs": 3,
        "eval_every": 50,
        "max_train_chunks": 2048,
        "max_val_chunks": 256,
        "block_size": 32,
        "block_counts": 4,
        "window_size": 64,
    },
}


def make_preset_config(preset: str = "standard", **overrides: Any) -> ExperimentConfig:
    if preset not in EXPERIMENT_PRESETS:
        valid = ", ".join(sorted(EXPERIMENT_PRESETS))
        raise ValueError(f"Unknown preset {preset!r}. Valid presets: {valid}")
    values = dict(EXPERIMENT_PRESETS[preset])
    values.update(overrides)
    return ExperimentConfig(**values)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).type_as(x) * self.weight.type_as(x)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class FullAttention(nn.Module):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        dim = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        att = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        causal = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal, float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))

        out = torch.matmul(att, v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ExperimentConfig, attn_cls: type[nn.Module]) -> None:
        super().__init__()
        self.ln1 = RMSNorm(cfg.hidden_size)
        self.attn = attn_cls(cfg)
        self.ln2 = RMSNorm(cfg.hidden_size)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.mlp_hidden)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.ln2(x))
        return x


class SmallLM(nn.Module):
    def __init__(self, cfg: ExperimentConfig, attn_cls: type[nn.Module]) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(cfg, attn_cls) for _ in range(cfg.num_layers))
        self.ln_f = RMSNorm(cfg.hidden_size)
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        rope_cos, rope_sin = precompute_rope(cfg.head_dim, cfg.max_seq_len + 64)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        self.apply(self._init_weights)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.o_proj.weight, std=0.02 / math.sqrt(2 * cfg.num_layers))
            nn.init.normal_(blk.mlp.w2.weight, std=0.02 / math.sqrt(2 * cfg.num_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin)
        return self.head(self.ln_f(x))

    def param_count(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: Optional[str] = None) -> torch.device:
    if name is not None:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["gpu_name"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / (1024 ** 3), 1)
    return info


def get_model_specs(cfg: ExperimentConfig) -> list[dict[str, str]]:
    specs = [
        {
            "key": "full_attention",
            "label": "Full Attention",
            "progress_label": "FullAttn",
            "kind": "full",
        },
        {
            "key": "nsa",
            "label": "NSA (All)",
            "progress_label": "NSA",
            "kind": "nsa",
            "nsa_mode": "all",
        },
    ]
    if cfg.compare_nsa_branches:
        specs.extend(
            [
                {
                    "key": "nsa_compression",
                    "label": "NSA Compression",
                    "progress_label": "NSA-Cmp",
                    "kind": "nsa",
                    "nsa_mode": "compression",
                },
                {
                    "key": "nsa_selection",
                    "label": "NSA Selection",
                    "progress_label": "NSA-Slc",
                    "kind": "nsa",
                    "nsa_mode": "selection",
                },
                {
                    "key": "nsa_sliding",
                    "label": "NSA Sliding",
                    "progress_label": "NSA-Swa",
                    "kind": "nsa",
                    "nsa_mode": "sliding",
                },
            ]
        )
    return specs


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_runtime_stats(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {}
    return {
        "mem_allocated_gb": round(torch.cuda.memory_allocated(device) / (1024 ** 3), 3),
        "mem_reserved_gb": round(torch.cuda.memory_reserved(device) / (1024 ** 3), 3),
    }


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_data(cfg: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Loading {cfg.dataset_name}/{cfg.dataset_config} with tokenizer {cfg.tokenizer_name}")
    dataset = load_dataset(cfg.dataset_name, cfg.dataset_config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

    def encode_split(split_name: str, max_chunks: int) -> torch.Tensor:
        token_ids: list[int] = []
        for text in dataset[split_name][cfg.text_field]:
            stripped = text.strip()
            if not stripped:
                continue
            token_ids.extend(tokenizer.encode(stripped + "\n", add_special_tokens=False))

        chunk_len = cfg.max_seq_len + 1
        available_chunks = len(token_ids) // chunk_len
        used_chunks = min(available_chunks, max_chunks)
        if used_chunks == 0:
            raise RuntimeError(f"No chunks available for split {split_name}")

        usable_tokens = used_chunks * chunk_len
        return torch.tensor(token_ids[:usable_tokens], dtype=torch.long).reshape(used_chunks, chunk_len)

    train_tokens = encode_split("train", cfg.max_train_chunks)
    val_tokens = encode_split("validation", cfg.max_val_chunks)
    print(
        f"Prepared {len(train_tokens):,} train chunks and {len(val_tokens):,} validation chunks "
        f"with sequence length {cfg.max_seq_len}"
    )
    return train_tokens, val_tokens


def make_loader(
    tensor: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def cosine_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def smooth_curve(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    smoothed: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def validation_plot_kwargs(num_points: int) -> dict[str, Any]:
    markevery = max(1, num_points // 20)
    return {
        "linestyle": "-",
        "marker": "o",
        "markersize": 2.5,
        "markevery": markevery,
        "linewidth": 1.5,
    }


def maybe_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    sync_device(device)
    start = time.perf_counter()
    for (batch,) in loader:
        batch = batch.to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]
        with maybe_autocast(device):
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    sync_device(device)
    elapsed = time.perf_counter() - start

    model.train()
    return total_loss / total_tokens, elapsed


def copy_matching_state(src: nn.Module, dst: nn.Module) -> int:
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    copied = 0
    for name, tensor in dst_state.items():
        if name in src_state and src_state[name].shape == tensor.shape:
            tensor.copy_(src_state[name])
            copied += 1
    dst.load_state_dict(dst_state)
    return copied


def init_nsa_gates(model: SmallLM) -> None:
    for block in model.blocks:
        if hasattr(block.attn, "g_proj"):
            nn.init.zeros_(block.attn.g_proj.weight)


def train_model(
    model: SmallLM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
    label: str,
    progress_callback: Optional[ProgressCallback] = None,
    show_progress_bar: bool = True,
) -> dict[str, Any]:
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    total_steps = cfg.epochs * len(train_loader)
    if total_steps == 0:
        raise RuntimeError("Empty train loader")

    history: dict[str, Any] = {
        "label": label,
        "train_steps": [],
        "train_loss": [],
        "val_steps": [],
        "val_loss": [],
        "val_elapsed_sec": [],
        "step_time_sec": [],
        "tokens_per_sec": [],
        "runtime_stats": [],
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_start = time.perf_counter()
    step = 0
    interval_tokens = 0
    interval_steps = 0
    interval_train_time = 0.0
    total_tokens_seen = 0

    val_loss, eval_time_sec = evaluate(model, val_loader, device)
    history["val_steps"].append(0)
    history["val_loss"].append(val_loss)
    history["val_elapsed_sec"].append(0.0)
    history["runtime_stats"].append(get_runtime_stats(device))
    if progress_callback is not None:
        progress_callback({
            "label": label, "step": 0, "total_steps": total_steps,
            "val_loss": val_loss, "history": history,
        })
    print(f"[{label}] step 0/{total_steps} | val {val_loss:.4f} | ppl {math.exp(val_loss):.2f}")

    for epoch in range(1, cfg.epochs + 1):
        iterator = tqdm(
            train_loader,
            desc=f"[{label}] epoch {epoch}/{cfg.epochs}",
            leave=False,
            disable=not show_progress_bar,
        )
        for (batch,) in iterator:
            sync_device(device)
            batch_start = time.perf_counter()

            batch = batch.to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]

            lr = cosine_lr(step, cfg.warmup_steps, total_steps, cfg.lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            with maybe_autocast(device):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            sync_device(device)
            batch_time = time.perf_counter() - batch_start
            tokens = y.numel()

            step += 1
            interval_tokens += tokens
            interval_steps += 1
            interval_train_time += batch_time
            total_tokens_seen += tokens
            history["train_steps"].append(step)
            history["train_loss"].append(loss.item())
            history["step_time_sec"].append(batch_time)

            if show_progress_bar:
                iterator.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr:.1e}")

            if step % cfg.eval_every == 0 or step == total_steps:
                val_loss, eval_time_sec = evaluate(model, val_loader, device)
                elapsed_sec = time.perf_counter() - train_start
                avg_step_time_sec = interval_train_time / max(interval_steps, 1)
                tokens_per_sec = interval_tokens / max(interval_train_time, 1e-9)
                eta_sec = (elapsed_sec / max(step, 1)) * max(total_steps - step, 0)
                runtime_stats = get_runtime_stats(device)

                history["val_steps"].append(step)
                history["val_loss"].append(val_loss)
                history["val_elapsed_sec"].append(elapsed_sec)
                history["tokens_per_sec"].append(tokens_per_sec)
                history["runtime_stats"].append(runtime_stats)

                if show_progress_bar:
                    iterator.set_postfix(
                        loss=f"{loss.item():.3f}",
                        val=f"{val_loss:.3f}",
                        step=f"{avg_step_time_sec:.2f}s",
                    )

                print(
                    f"[{label}] step {step}/{total_steps} | train {loss.item():.4f} | val {val_loss:.4f} | "
                    f"elapsed {format_duration(elapsed_sec)} | eta {format_duration(eta_sec)} | "
                    f"step {avg_step_time_sec:.2f}s | {tokens_per_sec:.0f} tok/s | "
                    f"mem {runtime_stats.get('mem_allocated_gb', 0.0):.2f} GB"
                )

                if progress_callback is not None:
                    progress_callback({
                        "label": label, "step": step, "total_steps": total_steps,
                        "epoch": epoch, "lr": lr, "train_loss": loss.item(),
                        "val_loss": val_loss, "elapsed_sec": elapsed_sec,
                        "tokens_per_sec": tokens_per_sec, "history": history,
                    })

                interval_tokens = 0
                interval_steps = 0
                interval_train_time = 0.0

    final_val = history["val_loss"][-1]
    total_train_time_sec = time.perf_counter() - train_start
    avg_step_time_sec = sum(history["step_time_sec"]) / max(len(history["step_time_sec"]), 1)
    avg_tokens_per_sec = total_tokens_seen / max(sum(history["step_time_sec"]), 1e-9)
    history["summary"] = {
        "final_val_loss": final_val,
        "final_val_ppl": math.exp(final_val),
        "best_val_loss": min(history["val_loss"]),
        "total_train_time_sec": total_train_time_sec,
        "avg_step_time_sec": avg_step_time_sec,
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "final_runtime_stats": history["runtime_stats"][-1] if history["runtime_stats"] else {},
    }
    print(
        f"[{label}] done | final val {final_val:.4f} | ppl {math.exp(final_val):.2f} | "
        f"total {total_train_time_sec / 60.0:.1f} min"
    )
    return history


def build_base_full_model(cfg: ExperimentConfig) -> SmallLM:
    set_seed(cfg.seed)
    return SmallLM(cfg, FullAttention)


def build_model_from_spec(
    base_full: SmallLM,
    cfg: ExperimentConfig,
    spec: dict[str, str],
    device: torch.device,
) -> SmallLM:
    if spec["kind"] == "full":
        model = copy.deepcopy(base_full)
    else:
        model_cfg = replace(cfg, nsa_mode=spec["nsa_mode"])
        model = SmallLM(model_cfg, ReferenceNativeSparseAttention)
        copied = copy_matching_state(base_full, model)
        if cfg.zero_init_nsa_gates:
            init_nsa_gates(model)
        print(
            f"Copied {copied} shared tensors from full-attention init to {spec['label']} "
            f"(mode={spec['nsa_mode']})"
        )
    return model.to(device)


def plot_results(results: dict[str, Any], out_path: Optional[Path] = None):
    import matplotlib.pyplot as plt

    cfg_dict = results["config"]
    show_elapsed = cfg_dict.get("show_elapsed_time_plot", False)
    num_cols = 3 if show_elapsed else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 5))
    if num_cols == 1:
        axes = [axes]
    for key in results["model_order"]:
        history = results["models"][key]
        label = history["plot_label"]
        axes[0].plot(
            history["train_steps"],
            smooth_curve(history["train_loss"], cfg_dict["smoothing_window"]),
            label=label,
            alpha=0.9,
        )
        axes[1].plot(
            history["val_steps"],
            history["val_loss"],
            label=label,
            **validation_plot_kwargs(len(history["val_steps"])),
        )
        if show_elapsed:
            axes[2].plot(
                [elapsed / 60.0 for elapsed in history["val_elapsed_sec"]],
                history["val_loss"],
                label=label,
                **validation_plot_kwargs(len(history["val_elapsed_sec"])),
            )

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Validation Loss vs Step")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Validation Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if show_elapsed:
        axes[2].set_title("Validation Loss vs Elapsed Time")
        axes[2].set_xlabel("Elapsed time (minutes)")
        axes[2].set_ylabel("Validation Loss")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    return fig


def save_results(results: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "nsa_vs_full.png"
    json_path = out_dir / "results.json"

    fig = plot_results(results, fig_path)
    plt.close(fig)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    return fig_path, json_path


def run_experiment(
    cfg: Optional[ExperimentConfig] = None,
    output_dir: Optional[str | Path] = None,
    progress_callback: Optional[ProgressCallback] = None,
    show_progress_bar: bool = True,
) -> dict[str, Any]:
    cfg = cfg or ExperimentConfig()
    device = get_device(cfg.device)
    device_info = get_device_info(device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    print(f"Device: {device}")
    print(f"Device info: {json.dumps(device_info, indent=2)}")
    print(f"Config: {cfg}")

    train_tokens, val_tokens = load_data(cfg)
    val_loader = make_loader(val_tokens, cfg.batch_size, shuffle=False, seed=cfg.seed)
    specs = get_model_specs(cfg)
    base_full = build_base_full_model(cfg)
    results_models: dict[str, Any] = {}
    summary: dict[str, Any] = {}

    for spec in specs:
        print(f"\n========== {spec['label']} ==========")
        model = build_model_from_spec(base_full, cfg, spec, device)
        print(f"{spec['label']} params: {model.param_count():,}")

        train_loader = make_loader(train_tokens, cfg.batch_size, shuffle=True, seed=cfg.seed)
        result = train_model(
            model,
            train_loader,
            val_loader,
            cfg,
            device,
            label=spec["progress_label"],
            progress_callback=progress_callback,
            show_progress_bar=show_progress_bar,
        )
        result["plot_label"] = spec["label"]
        result["model_key"] = spec["key"]
        results_models[spec["key"]] = result
        summary[spec["key"]] = result["summary"]

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = {
        "config": asdict(cfg),
        "device": str(device),
        "device_info": device_info,
        "dataset": {
            "train_chunks": len(train_tokens),
            "val_chunks": len(val_tokens),
            "seq_len": cfg.max_seq_len,
        },
        "model_order": [spec["key"] for spec in specs],
        "models": results_models,
        "summary": summary,
    }
    if "full_attention" in results_models:
        results["full_attention"] = results_models["full_attention"]
    if "nsa" in results_models:
        results["nsa"] = results_models["nsa"]

    if output_dir is not None:
        fig_path, json_path = save_results(results, output_dir)
        print(f"Saved plot to {fig_path}")
        print(f"Saved metrics to {json_path}")

    return results


def main() -> None:
    output_dir = os.environ.get("NSA_OUTPUT_DIR", "outputs")
    preset = os.environ.get("NSA_EXPERIMENT_PRESET", "standard")
    cfg = make_preset_config(preset)
    run_experiment(cfg=cfg, output_dir=output_dir, show_progress_bar=True)


if __name__ == "__main__":
    main()
