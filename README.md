# Native Sparse Attention — Reproduction & Experiments

**Yentl Collin** · **Gaspard Beaudoin** · MSc Artificial Intelligence, Large Scale Models · Université Paris-Saclay · 2025–2026

---

## Overview

This repository contains my full reproduction and experimental study of **Native Sparse Attention (NSA)**, introduced in *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention* (Yuan et al., 2025, [arXiv:2502.11089](https://arxiv.org/abs/2502.11089)).

The goal is threefold: (1) understand the NSA mechanism deeply enough to re-implement it from scratch in pure PyTorch, (2) validate the claimed hardware efficiency gains through GPU benchmarking, and (3) evaluate the learning quality of NSA against dense attention on controlled tasks. All experiments are reproducible and documented in the notebooks and the accompanying report.

---

## Background and Motivation

Standard self-attention scales quadratically — both in FLOPs and memory — with sequence length. For a sequence of length *n*, the attention matrix is *n × n*, which becomes prohibitive beyond a few thousand tokens. Various sparse attention mechanisms have been proposed to address this (Longformer, BigBird, Reformer, etc.), but they all share a critical weakness: **irregular memory access patterns**.

Sparse attention that uses learned or random token subsets requires scatter/gather operations over non-contiguous memory locations. On modern GPUs, this breaks the coalesced memory access pattern that tensor cores and SRAM rely on, introducing stall cycles that make the sparse kernel slower in practice than the dense FlashAttention kernel — even when FLOPs are significantly reduced.

**NSA's key insight** is to constrain all sparse operations to act on **fixed-size contiguous blocks** of tokens. This makes the memory access pattern identical to dense attention tiling, natively compatible with GPU tensor cores and FlashAttention tile loops. The result is a sparse attention mechanism that is both theoretically efficient (linear in sequence length) and practically faster than dense attention at real GPU scale.

---

## The NSA Mechanism

NSA decomposes attention into three parallel branches that operate over different scales of context, then combines their outputs through learned per-head sigmoid gates.

```
              Input: Q, K, V
             /         |         \
    [Compression]  [Selection]  [Sliding Window]
          |              |              |
       × g_cmp        × g_slc       × g_swa
             \            |           /
      o_t = g_cmp·o_cmp + g_slc·o_slc + g_swa·o_swa

  Gates: g ∈ [0, 1]^(num_heads) — per-head sigmoid, end-to-end differentiable
```

### Branch 1 — Compression

K and V are mean-pooled into block summaries: each block of `Bs` consecutive tokens is reduced to a single representative key-value pair. The query then attends over these `⌈n/Bs⌉` compressed representations.

This branch provides **global context at low resolution**. Its attention weights double as block relevance scores: which blocks of the sequence are most relevant for each query position. These scores drive the Selection branch.

Causal masking is strict: query at position `t` can only attend to block `j` if the entire block has been observed, i.e. `t >= (j+1)·Bs - 1`.

### Branch 2 — Selection

Using the compression attention weights (detached from the gradient graph via stop-gradient), the top-K most relevant blocks are selected per query position. The query then attends with full precision over the `K·Bs` tokens in those selected blocks.

This branch provides **high-resolution attention over the most informative context**. Because the selected blocks are contiguous in memory, the attention kernel can operate with standard tile loops — no scatter/gather required.

### Branch 3 — Sliding Window

Standard causal local attention over the last `W` tokens. This captures high-frequency recency signals — information that is always relevant regardless of global context. It is the simplest and most computationally predictable branch.

### Gating

The three branch outputs are combined by per-head sigmoid gates `g_cmp, g_slc, g_swa ∈ [0,1]^H`, produced by a small linear projection of the input. This makes the model fully end-to-end differentiable (with the exception of the top-K index selection, which is detached). The gates allow each attention head to specialize: some heads may rely primarily on compressed global context, others on local recency.

### Complexity

| Branch | FLOPs | Memory |
|---|---|---|
| Compression | O(n · n/Bs) | O(n · n/Bs) |
| Selection | O(n · K·Bs) | O(n · K·Bs) |
| Sliding Window | O(n · W) | O(n · W) |
| **Total** | **O(n · n_act)** | **O(n · n_act)** |

where `n_act = K·Bs + W` is constant in `n`. The complexity is **linear in sequence length**, compared to O(n²) for dense attention.

---

## Repository Structure

```
native-sparse-attention/
│
├── native_sparse_attention/            # Main Python package
│   ├── __init__.py                     # Package init and version
│   ├── configuration_nsa.py            # NSA model configuration dataclass
│   ├── modeling_nsa.py                 # Full NSA language model (HuggingFace-compatible)
│   ├── pytorch_reference.py            # Reference implementation in pure PyTorch (no Triton)
│   └── ops/
│       ├── naive.py                    # Naive Python-loop implementation (correctness reference)
│       ├── parallel.py                 # Optimized Triton GPU kernels (forward + backward)
│       └── utils.py                    # Shared utilities
│
├── notebooks/                          # Main experiment notebooks
│   ├── axe1_attention_patterns.ipynb   # Attention pattern visualization
│   ├── axe2_arithmetic_intensity.ipynb # GPU benchmark: NSA vs FlashAttention-2
│   └── axe5_needle_haystack.ipynb      # Retrieval task (Needle In A Haystack)
│
├── experiments/
│   └── demo_nsa.py                     # 4 lightweight CPU experiments (FLOPs, speed, sparsity, quality)
│
├── benchmarks/
│   └── benchmark_nsa.py                # Forward/backward latency benchmarks
│
├── tests/
│   ├── test_nsa.py                     # NSA kernel correctness tests
│   └── test_nsa_with_compression.py    # Compression + selection correctness tests
│
├── train_nsa_vs_full.py                # WikiText-2 training: NSA vs Full Attention + branch ablation
├── configs/
│   └── nsa_340M.json                   # Model configuration for a 340M parameter model
│
└── report/
    ├── nsa_report.tex / .pdf           # Full written report
    └── nsa_poster.tex / .pdf           # Conference-style A0 landscape poster
```

---

## Implementation Details

### `pytorch_reference.py` — Pure PyTorch Reference

`ReferenceNativeSparseAttention` is a from-scratch implementation of all three NSA branches in standard PyTorch, with no Triton or FlashAttention dependency. It is designed for correctness, readability, and portability (runs on any device including CPU).

Key implementation choices:
- **RoPE embeddings** via `precompute_rope` / `apply_rope` — standard rotary position embeddings applied to Q and K before attention
- **RMSNorm** in the surrounding transformer architecture (via `train_nsa_vs_full.py`)
- **SwiGLU** MLP blocks (gated feedforward with SiLU activation)
- **Strict causal masking** in the compression branch: token `t` can only attend to block `j` if `t >= (j+1)·Bs - 1` — the full block must be in the past
- **Stop-gradient** on compression scores before top-K selection: `block_scores = w_cmp.detach().clone()`
- **Zero-initialized gates** when transferring from a full-attention checkpoint, so that the NSA model starts as a smoothed version of dense attention and diverges gradually during fine-tuning
- **Ablation support** via `nsa_mode ∈ {"all", "compression", "selection", "sliding"}` — any single branch or all three can be activated independently

This reference implementation is used in all learning experiments (WikiText-2 and NIAH), as it does not require a high-end GPU with Triton support.

### `native_sparse_attention/ops/naive.py` — Naive Reference Kernels

Loop-based Python implementation of the NSA operations, used to validate the correctness of the Triton kernels. Running `pytest tests/` compares naive and Triton outputs across random inputs.

### `native_sparse_attention/ops/parallel.py` — Triton GPU Kernels

Optimized Triton implementations of the NSA forward and backward passes:
- Grouped Query Attention (GQA) support with configurable KV-head ratio
- Fused sliding window + selected attention kernel
- Online top-K selection that avoids materializing the full attention matrix during selection
- Variable-length sequence support via `cu_seqlens` offsets
- Efficient tiled memory layout aligned with FlashAttention tile loops

### `train_nsa_vs_full.py` — WikiText-2 Training Script

A self-contained script that trains and compares Full Attention and NSA models on WikiText-2 (`wikitext-2-raw-v1`). Key features:
- Configurable via `ExperimentConfig` dataclass and presets (`quick`, `standard`, `representative`)
- Shared initialization: NSA models are copied from the Full Attention checkpoint, with only the gate projection zero-initialized
- Cosine LR schedule with warmup, gradient clipping, mixed-precision (fp16)
- Branch ablation via `compare_nsa_branches=True`, which trains separate models for each branch
- Results saved as JSON (full training history) and PNG (loss curves)

---

## Installation

```bash
git clone https://github.com/YentlCollin/native-sparse-attention.git
cd native-sparse-attention
pip install -e .
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.5, Triton ≥ 3.0, Transformers ≥ 4.45, Datasets ≥ 3.3.

For the GPU benchmarks and Triton kernels, a CUDA-capable GPU is required. The PyTorch reference implementation and the CPU experiments in `experiments/demo_nsa.py` run without a GPU.

---

## Experiments

### Experiment 1 — GPU Benchmark: NSA vs FlashAttention-2

**Notebook:** `notebooks/axe2_arithmetic_intensity.ipynb`

**Setup:** NSA selection kernel vs FlashAttention-2 (PyTorch SDPA) on an **A100-SXM4-40GB** GPU, BF16 precision, CUDA 12.8, GQA ratio 16 (64 query heads, 4 KV heads), head dimension 64, block size 64, K=4 selected blocks, sliding window 64. Measurements via Triton `do_bench` with 200 repetitions, 25 warmup iterations, median latency.

**Forward pass results:**

| Sequence length | Dense (ms) | NSA (ms) | Speedup |
|---|---|---|---|
| 8k | 5.97 | 5.01 | 1.2× |
| 16k | 22.2 | 9.96 | 2.2× |
| 32k | 87.0 | 20.0 | 4.4× |
| **64k** | **348** | **41.9** | **8.3×** |

**Backward pass results:**

| Sequence length | Dense (ms) | NSA (ms) | Speedup |
|---|---|---|---|
| 8k | 21.9 | 29.3 | 0.75× |
| 16k | 82.8 | 59.9 | 1.4× |
| 32k | 320 | 132 | 2.4× |
| **64k** | **1273** | **379** | **3.4×** |

**Key findings:**
- The measured 8.3× forward speedup exceeds the theoretical 5.8× FLOPs prediction. The surplus comes from L2 cache reuse: contiguous 64-token blocks eliminate the stall cycles caused by irregular scatter/gather memory access patterns. NSA's memory layout is identical to dense FlashAttention tiling, so the GPU can prefetch tiles efficiently.
- Forward breakeven (where NSA becomes faster than dense): ~6k tokens. Backward breakeven: ~14k tokens. The backward pass benefits less because gradient computation involves additional scatter operations.
- Peak VRAM at n=64k: 5.55 GB (NSA) vs 6.09 GB (Dense) — a 1.1× ratio. NSA's advantage is latency, not memory capacity: it does not enable longer sequences on a fixed GPU budget, but it processes the same sequences significantly faster.

### Experiment 2 — Retrieval Task: Needle In A Haystack (NIAH)

**Notebook:** `notebooks/axe5_needle_haystack.ipynb`

**Task setup:** Three identical small transformers (~665k parameters each) are trained on a synthetic retrieval task. Each example consists of a context of 128–320 tokens containing a "needle" phrase of the form `"today Mathieu <action> ."` at a random position, surrounded by filler tokens and 3 distractor phrases (same structure but with past temporal markers). The model receives the full context followed by the query `"What does Mathieu do today?"` and must predict the correct action (one of 20 possible verbs) at the query position. The training objective is a cross-entropy loss at a single position (the query token), not standard next-token prediction over the full sequence.

The three models differ only in their attention mechanism: Full Attention (dense causal), NSA (all three branches), and Sliding Window (local attention only). All other hyperparameters are identical.

**Results (evaluation accuracy):**

| Model | Eval accuracy | Train/eval gap |
|---|---|---|
| NSA | **97.6%** | 2.3% |
| Dense | 52.9% | 31% |
| Sliding Window | 5.9% | — |

**Recall vs. needle position:** NSA maintains near-perfect recall regardless of where in the context the needle appears. Dense attention is moderate but position-agnostic. Sliding Window recovers only in the last W tokens (~10% of positions).

**Analysis:**
- NSA outperforms dense attention by 44.7 percentage points despite having fewer effective attention connections. The compressed branch builds block-level representations that directly encode the retrieval signal: mean-pooled K/V pairs capture which block contains the needle, and the selection mechanism routes the query there.
- Dense attention has strictly more expressivity in theory, but the small model must learn to assign near-one-hot attention weights over >300 positions — a harder optimization problem. It overfits: ~84% train accuracy vs 52.9% eval (31% gap), compared to NSA's 2.3% gap.
- Sliding Window fails structurally. The needle falls outside the window ~90% of the time, making the task unsolvable for most examples. The model reaches 70% train accuracy (memorizing in-window cases) but 5.9% eval, with validation loss rising monotonically from epoch 50 — a clear sign of non-generalizing memorization.

### Experiment 3 — Language Modeling: WikiText-2

**Script:** `train_nsa_vs_full.py`

**Setup:** Same transformer architecture (RMSNorm, SwiGLU, RoPE), trained on real English text from WikiText-2 (~2M tokens). Vocabulary: GPT-2 BPE tokenizer (50,257 tokens). Model: 4 layers, 8 heads, hidden size 256, MLP hidden 1024, sequence length 256. NSA models are initialized from the Full Attention checkpoint; only the gate projection is new (zero-initialized), so the NSA model starts from a good initialization and fine-tunes the gating behavior.

Branch ablation: separate models are trained for each configuration — Full Attention, NSA (all branches), NSA Compression only, NSA Selection only, NSA Sliding only.

**Key findings:**
- NSA (all branches) matches Full Attention validation loss throughout training with no quality degradation — NSA does not sacrifice language modeling quality for speed.
- Sliding window alone is nearly sufficient on Wikipedia, which consists predominantly of short declarative sentences. Long-range dependencies are rare in this corpus.
- Combining all three branches is competitive with or slightly better than full attention, suggesting the three branches are complementary.
- The shared initialization strategy is critical: starting NSA gates at zero ensures the model begins as a smoothed version of the full-attention checkpoint and learns to specialize progressively.

### Experiment 4 — CPU Demonstrations

**Script:** `experiments/demo_nsa.py`

Four lightweight experiments that run entirely on CPU, useful for understanding and demonstration without GPU access:

1. **Theoretical FLOPs:** Plots FLOPs as a function of sequence length for full attention (O(T²)) and NSA (O(T·k)), showing the crossover point and asymptotic speedup ratio.
2. **Empirical scalability:** Measures wall-clock time for the naive Python implementations, confirming O(T²) vs O(T) growth slopes even in unoptimized code.
3. **Sparsity visualization:** For a random input, shows which blocks are selected by the compression branch for each query position, the resulting sparse attention mask, and the sparsity ratio compared to full causal attention.
4. **Output quality:** Compares NSA output to full attention output token-by-token via cosine similarity and relative L2 error, both for compression-driven block selection and random block selection (lower bound).

```bash
python experiments/demo_nsa.py
# Generates figures in experiments/figures/
```

---

## Tests

```bash
# Test NSA kernel correctness
pytest tests/test_nsa.py

# Test compression + top-K selection correctness
# Note: first run may be slow (Triton kernel compilation)
pytest tests/test_nsa_with_compression.py
```

---

## Report and Poster

The full written report (`report/nsa_report.pdf`) covers six axes of analysis:
1. Architecture and implementation details
2. Hardware efficiency (GPU benchmarks)
3. Arithmetic intensity analysis
4. Attention pattern visualization
5. NIAH retrieval task
6. WikiText-2 language modeling and branch ablation

The conference-style poster (`report/nsa_poster.pdf`) is an A0 landscape summary designed for a poster session, covering the main results on hardware efficiency and learning quality.

Both are compiled from LaTeX sources (`nsa_report.tex`, `nsa_poster.tex`) using TinyTeX/pdflatex.

---

## Reference

Yuan et al., *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*, arXiv:2502.11089, 2025.

---

## Acknowledgements

The Triton kernel implementation (`native_sparse_attention/ops/`) was inspired by the work of **Songlin Yang** and **Yu Zhang**.
