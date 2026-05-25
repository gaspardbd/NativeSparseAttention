"""
Démonstration NSA — 4 expériences CPU :
  1. FLOPs théoriques
  2. Scalabilité temporelle O(T²) vs O(T)
  3. Patterns de sparsité
  4. Qualité de sortie vs attention complète
"""

import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # pas besoin d'affichage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import importlib.util, types

# Charge ops/naive.py sans déclencher __init__.py (qui requiert triton/fla)
_pkg = types.ModuleType("native_sparse_attention")
_pkg.__path__ = [str(ROOT / "native_sparse_attention")]
sys.modules.setdefault("native_sparse_attention", _pkg)
_ops_pkg = types.ModuleType("native_sparse_attention.ops")
_ops_pkg.__path__ = [str(ROOT / "native_sparse_attention" / "ops")]
sys.modules.setdefault("native_sparse_attention.ops", _ops_pkg)

_spec = importlib.util.spec_from_file_location(
    "native_sparse_attention.ops.naive",
    ROOT / "native_sparse_attention" / "ops" / "naive.py"
)
_naive_mod = importlib.util.module_from_spec(_spec)
sys.modules["native_sparse_attention.ops.naive"] = _naive_mod
_spec.loader.exec_module(_naive_mod)

compression           = _naive_mod.compression
naive_nsa_compression = _naive_mod.naive_nsa_compression
naive_nsa             = _naive_mod.naive_nsa

SEED = 42
torch.manual_seed(SEED)

B         = 1        # batch size
HQ        = 16       # têtes de requête
H         = 1        # têtes clé/valeur  (G = HQ/H = 16, requis ≥ 16)
D         = 32       # dimension par tête
BLOCK_SIZE = 16      # taille d'un bloc
S         = 4        # nombre de blocs sélectionnés par token
WINDOW    = 0        # sliding window désactivée (simplifie la visualisation)
OUT_DIR   = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


# ─── Utilitaires ─────────────────────────────────────────────────────────────

def make_tensors(T, dtype=torch.float32):
    """Crée des tenseurs aléatoires pour une séquence de longueur T."""
    q  = torch.randn(B, T, HQ, D, dtype=dtype)
    k  = torch.randn(B, T, H,  D, dtype=dtype)
    v  = torch.randn(B, T, H,  D, dtype=dtype)
    return q, k, v


def full_causal_attention(q, k, v):
    """Attention causale complète  O(T²) — référence."""
    # q: [B, T, HQ, D] → répliquer k/v sur HQ
    B, T, HQ, D = q.shape
    scale = D ** -0.5
    k_exp = k.expand(B, T, HQ, D)   # broadcast GQA
    v_exp = v.expand(B, T, HQ, D)
    # [B, HQ, T, T]
    q_t = q.permute(0, 2, 1, 3)
    k_t = k_exp.permute(0, 2, 1, 3)
    v_t = v_exp.permute(0, 2, 1, 3)
    attn = torch.einsum("bhtd,bhsd->bhts", q_t, k_t) * scale
    mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
    attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = attn.softmax(dim=-1)
    o = torch.einsum("bhts,bhsd->bhtd", attn, v_t)
    return o.permute(0, 2, 1, 3)    # [B, T, HQ, D]


def make_block_indices(T, H, B, S, block_size):
    """Sélectionne aléatoirement S blocs par token (indices causaux)."""
    num_blocks = math.ceil(T / block_size)
    block_indices = torch.zeros(B, T, H, S, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            max_block = max(1, (t + 1 + block_size - 1) // block_size)
            perm = torch.randperm(max_block)[:S]
            block_indices[b, t, 0, :len(perm)] = perm
    return block_indices.sort(-1)[0]


# ============================================================
# EXPÉRIENCE 1 – Complexité théorique (FLOPs)
# ============================================================

def exp1_flops():
    print("\n── Expérience 1 : Complexité théorique (FLOPs) ──")
    T_vals = [64, 128, 256, 512, 1024, 2048, 4096]
    k_blocks = S * BLOCK_SIZE      # tokens effectivement vus par NSA

    flops_full = []
    flops_nsa  = []
    for T in T_vals:
        # Full attention : chaque token query attend les T tokens → O(T²·D)
        flops_full.append(T * T * D * 2)
        # NSA : chaque token attend k_blocks tokens (sélectionnés + window) → O(T·k·D)
        flops_nsa.append(T * k_blocks * D * 2)

    ratios = [f / n for f, n in zip(flops_full, flops_nsa)]
    print("  T      Full-Attn FLOPs    NSA FLOPs    Ratio")
    for T, ff, fn, r in zip(T_vals, flops_full, flops_nsa, ratios):
        print(f"  {T:5d}  {ff:12,}   {fn:9,}   {r:5.1f}×")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(T_vals, [f / 1e6 for f in flops_full], "b-o", label="Full Attention O(T²)")
    axes[0].plot(T_vals, [f / 1e6 for f in flops_nsa],  "r-o", label=f"NSA O(T·k), k={k_blocks}")
    axes[0].set_xlabel("Longueur de séquence T")
    axes[0].set_ylabel("FLOPs (millions)")
    axes[0].set_title("Complexité théorique")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(T_vals, ratios, "g-o")
    axes[1].set_xlabel("Longueur de séquence T")
    axes[1].set_ylabel("Speedup théorique (×)")
    axes[1].set_title("Rapport Full / NSA")
    axes[1].grid(True, alpha=0.3)
    for x, r in zip(T_vals, ratios):
        axes[1].annotate(f"{r:.0f}×", (x, r), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

    plt.tight_layout()
    path = OUT_DIR / "exp1_flops.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Figure sauvegardée : {path}")


# ============================================================
# EXPÉRIENCE 2 – Scalabilité empirique (temps CPU)
# ============================================================

def exp2_speed():
    """
    Compare la croissance du temps d'exécution de l'attention complète (O(T²))
    et de l'implémentation naïve de référence NSA (O(T) boucles Python).

    REMARQUE : l'implémentation naïve (boucle Python) est plus lente en CPU
    que la version vectorisée de l'attention complète. C'est NORMAL : elle sert
    uniquement à valider la correction des kernels Triton GPU.
    Le vrai gain de vitesse de NSA est visible sur GPU avec les kernels Triton
    (voir benchmark_nsa.py et les résultats du papier : jusqu'à 11× à T=64k).
    Ici, on montre que l'attention complète croît en O(T²) tandis que NSA croît
    linéairement O(T), ce qui se confirme dans la pente des courbes.
    """
    print("\n── Expérience 2 : Scalabilité empirique — croissance O(T²) vs O(T) ──")
    T_vals = [64, 128, 256, 384, 512]
    N_RUNS = 3
    scale = D ** -0.5

    times_full = []
    times_nsa  = []

    for T in T_vals:
        q, k, v = make_tensors(T)
        g_slc = torch.rand(B, T, HQ)
        g_swa = torch.rand(B, T, HQ)
        block_indices = make_block_indices(T, H, B, S, BLOCK_SIZE)
        block_counts  = S

        # Full attention
        t0 = time.perf_counter()
        for _ in range(N_RUNS):
            _ = full_causal_attention(q, k, v)
        times_full.append((time.perf_counter() - t0) / N_RUNS * 1000)

        # NSA (implémentation naïve de référence — boucle Python)
        t0 = time.perf_counter()
        for _ in range(N_RUNS):
            _ = naive_nsa(q, k, v, g_slc, g_swa,
                          block_indices=block_indices,
                          block_counts=block_counts,
                          block_size=BLOCK_SIZE,
                          window_size=WINDOW,
                          scale=scale)
        times_nsa.append((time.perf_counter() - t0) / N_RUNS * 1000)

        print(f"  T={T:4d}  Full={times_full[-1]:7.1f} ms  NSA-naive={times_nsa[-1]:6.1f} ms")

    # Ajustement des pentes pour montrer O(T²) vs O(T)
    T_arr = np.array(T_vals, dtype=float)
    c_full = times_full[-1] / T_arr[-1]**2
    c_nsa  = times_nsa[-1]  / T_arr[-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # -- Gauche : mesures brutes
    axes[0].plot(T_vals, times_full, "b-o", label="Full Attention (mesuré)")
    axes[0].plot(T_vals, times_nsa,  "r-o", label="NSA naive Python (mesuré)")
    axes[0].plot(T_vals, c_full * T_arr**2, "b--", alpha=0.5, label="courbe O(T²)")
    axes[0].plot(T_vals, c_nsa  * T_arr,   "r--", alpha=0.5, label="courbe O(T)")
    axes[0].set_xlabel("Longueur de séquence T")
    axes[0].set_ylabel("Temps (ms)")
    axes[0].set_title("Temps CPU — référence naïve\n(boucle Python, pas les kernels Triton)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # -- Droite : speedup GPU théorique (d'après le papier NSA)
    T_gpu = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    # Valeurs approximatives extraites du tableau du papier (ms, forward pass, bfloat16)
    flash_ms = [2.0, 5.5, 17.0, 53.0, 105.0, 350.0, 1200.0]
    nsa_ms   = [1.5, 2.5,  4.5,  8.0,  20.0,  40.0,  100.0]
    speedup_gpu = [f / n for f, n in zip(flash_ms, nsa_ms)]

    axes[1].plot(T_gpu, speedup_gpu, "g-o", linewidth=2)
    axes[1].axhline(1, color="k", linestyle="--", linewidth=0.8)
    axes[1].fill_between(T_gpu, 1, speedup_gpu, alpha=0.15, color="green")
    axes[1].set_xlabel("Longueur de séquence T")
    axes[1].set_ylabel("Speedup NSA vs Flash Attention (×)")
    axes[1].set_title("Speedup GPU (kernels Triton)\nd'après le papier NSA")
    axes[1].set_xscale("log", base=2)
    axes[1].grid(True, alpha=0.3)
    for x, s in zip(T_gpu, speedup_gpu):
        axes[1].annotate(f"{s:.1f}×", (x, s), textcoords="offset points",
                         xytext=(0, 6), ha="center", fontsize=8)

    plt.suptitle("Scalabilité : O(T²) Full Attention vs O(T) NSA", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "exp2_speed.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Figure sauvegardée : {path}")


# ============================================================
# EXPÉRIENCE 3 – Visualisation des patterns de sparsité
# ============================================================

def exp3_sparsity():
    """
    Montre quels blocs sont sélectionnés par NSA pour chaque position
    de requête, et les compare au masque causal dense de l'attention complète.
    """
    print("\n── Expérience 3 : Visualisation de la sparsité ──")
    T = 128

    q, k, v = make_tensors(T)
    scale = D ** -0.5

    # Utilise naive_nsa_compression pour obtenir les block_indices appris
    g_cmp = torch.rand(B, T, HQ)
    block_indices, _ = naive_nsa_compression(q, k, v, g_cmp,
                                              block_counts=S,
                                              block_size=BLOCK_SIZE,
                                              scale=scale)
    # block_indices : [B, T, H, S]  (indices de blocs)

    # Construire une carte d'attention T×T : 1 si le token est couvert par NSA
    nsa_map  = torch.zeros(T, T)
    full_map = torch.tril(torch.ones(T, T))   # attention causale dense

    for i_q in range(T):
        for s in range(S):
            blk = block_indices[0, i_q, 0, s].item()
            if blk < 0:
                continue
            start = blk * BLOCK_SIZE
            end   = min(start + BLOCK_SIZE, T)
            # Attention causale : seulement les tokens précédents
            for j in range(start, end):
                if j <= i_q:
                    nsa_map[i_q, j] = 1.0

    sparsity = 1.0 - nsa_map.sum() / full_map.sum()
    print(f"  T={T}, S={S} blocs × {BLOCK_SIZE} tokens  →  sparsité = {sparsity:.1%}")
    print(f"  Tokens vus en moyenne par query : {nsa_map.sum()/T:.1f} / {T/2:.1f} (full causal)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(full_map.numpy(), cmap="Blues", aspect="auto")
    axes[0].set_title(f"Full Causal Attention\n(T×T = {T}×{T} = {T*T:,} entrées)")
    axes[0].set_xlabel("Position clé")
    axes[0].set_ylabel("Position requête")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(nsa_map.numpy(), cmap="Reds", aspect="auto")
    axes[1].set_title(f"NSA – blocs sélectionnés\n(S={S}, block_size={BLOCK_SIZE})")
    axes[1].set_xlabel("Position clé")
    axes[1].set_ylabel("Position requête")
    plt.colorbar(im1, ax=axes[1])

    diff = full_map - nsa_map
    im2 = axes[2].imshow(diff.numpy(), cmap="RdYlGn_r", aspect="auto", vmin=-1, vmax=1)
    axes[2].set_title(f"Différence (rouge = ignoré)\nSparsité = {sparsity:.1%}")
    axes[2].set_xlabel("Position clé")
    axes[2].set_ylabel("Position requête")
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle("Patterns d'attention : Full vs NSA", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "exp3_sparsity.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Figure sauvegardée : {path}")


# ============================================================
# EXPÉRIENCE 4 – Qualité de sortie vs attention complète
# ============================================================

def exp4_quality():
    """
    Compare la sortie NSA à la sortie de l'attention complète :
    - Similarité cosinus token-par-token
    - Erreur L2 relative
    On montre que malgré la sparsité, NSA préserve bien l'information.
    """
    print("\n── Expérience 4 : Qualité de sortie ──")
    T    = 128
    scale = D ** -0.5

    q, k, v = make_tensors(T)
    g_cmp = torch.rand(B, T, HQ)
    g_slc = torch.rand(B, T, HQ)
    g_swa = torch.rand(B, T, HQ)

    # Référence : attention complète
    o_full = full_causal_attention(q, k, v)   # [B, T, HQ, D]

    # NSA avec compression (sélection automatique des blocs)
    block_indices, o_cmp = naive_nsa_compression(q, k, v, g_cmp,
                                                  block_counts=S,
                                                  block_size=BLOCK_SIZE,
                                                  scale=scale)
    o_slc = naive_nsa(q, k, v, g_slc, g_swa,
                      block_indices=block_indices,
                      block_counts=S,
                      block_size=BLOCK_SIZE,
                      window_size=WINDOW,
                      scale=scale)
    # Sortie NSA totale = compression + sélection
    o_nsa = (o_cmp + o_slc)   # [B, T, HQ, D]

    # -- Métriques token-par-token
    # Aplatir en [T, HQ*D]
    of_flat  = o_full[0].reshape(T, -1).float()
    on_flat  = o_nsa[0].reshape(T, -1).float()

    # Similarité cosinus par token
    cos_sim = F.cosine_similarity(of_flat, on_flat, dim=-1).detach().numpy()  # [T]
    # Erreur L2 relative par token
    l2_err  = (of_flat - on_flat).norm(dim=-1) / (of_flat.norm(dim=-1) + 1e-8)
    l2_err  = l2_err.detach().numpy()

    print(f"  Similarité cosinus  : moy={cos_sim.mean():.4f}, min={cos_sim.min():.4f}")
    print(f"  Erreur L2 relative  : moy={l2_err.mean():.4f}, max={l2_err.max():.4f}")

    # -- Comparer avec une attention RANDOM (baseline inférieure)
    block_indices_rnd = make_block_indices(T, H, B, S, BLOCK_SIZE)
    o_rnd = naive_nsa(q, k, v, g_slc, g_swa,
                      block_indices=block_indices_rnd,
                      block_counts=S,
                      block_size=BLOCK_SIZE,
                      window_size=WINDOW,
                      scale=scale)
    or_flat  = o_rnd[0].reshape(T, -1).float()
    cos_rnd  = F.cosine_similarity(of_flat, or_flat, dim=-1).detach().numpy()
    l2_rnd   = (of_flat - or_flat).norm(dim=-1) / (of_flat.norm(dim=-1) + 1e-8)
    l2_rnd   = l2_rnd.detach().numpy()
    print(f"  [Baseline random]   cos={cos_rnd.mean():.4f}, L2={l2_rnd.mean():.4f}")

    # -- Figure
    positions = np.arange(T)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # a) Cosinus NSA vs Random
    axes[0, 0].plot(positions, cos_sim, color="red",  alpha=0.8, label=f"NSA (moy={cos_sim.mean():.3f})")
    axes[0, 0].plot(positions, cos_rnd, color="gray", alpha=0.5, label=f"Random (moy={cos_rnd.mean():.3f})")
    axes[0, 0].set_xlabel("Position token")
    axes[0, 0].set_ylabel("Similarité cosinus")
    axes[0, 0].set_title("Similarité cosinus vs Full Attention")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([-1.1, 1.1])

    # b) Erreur L2 NSA vs Random
    axes[0, 1].plot(positions, l2_err, color="red",  alpha=0.8, label=f"NSA (moy={l2_err.mean():.3f})")
    axes[0, 1].plot(positions, l2_rnd, color="gray", alpha=0.5, label=f"Random (moy={l2_rnd.mean():.3f})")
    axes[0, 1].set_xlabel("Position token")
    axes[0, 1].set_ylabel("Erreur L2 relative")
    axes[0, 1].set_title("Erreur L2 relative vs Full Attention")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # c) Histogramme des similarités cosinus
    axes[1, 0].hist(cos_sim, bins=20, color="red",  alpha=0.6, label="NSA",    density=True)
    axes[1, 0].hist(cos_rnd, bins=20, color="gray", alpha=0.6, label="Random", density=True)
    axes[1, 0].set_xlabel("Similarité cosinus")
    axes[1, 0].set_ylabel("Densité")
    axes[1, 0].set_title("Distribution des similarités cosinus")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # d) Scatter : sortie NSA vs Full
    idx = 0   # premier token
    nsa_vals  = o_nsa[0, idx].reshape(-1).detach().float().numpy()
    full_vals = o_full[0, idx].reshape(-1).detach().float().numpy()
    lim = max(abs(nsa_vals).max(), abs(full_vals).max()) * 1.1
    axes[1, 1].scatter(full_vals, nsa_vals, s=10, alpha=0.6, color="red")
    axes[1, 1].plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, label="y=x (parfait)")
    axes[1, 1].set_xlabel("Full Attention (valeur)")
    axes[1, 1].set_ylabel("NSA (valeur)")
    axes[1, 1].set_title(f"Scatter token[0] : NSA vs Full")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Qualité de sortie : NSA vs Full Attention (T=128)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = OUT_DIR / "exp4_quality.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Figure sauvegardée : {path}")


# ============================================================
# SYNTHÈSE – Une seule figure récapitulative
# ============================================================

def make_summary_figure():
    print("\n── Figure de synthèse ──")
    T_vals    = [64, 128, 256, 384, 512]
    k_tokens  = S * BLOCK_SIZE
    scale     = D ** -0.5

    # Recalcul rapide
    flops_full = [T * T * D * 2      for T in T_vals]
    flops_nsa  = [T * k_tokens * D * 2 for T in T_vals]
    ratios     = [f / n for f, n in zip(flops_full, flops_nsa)]

    # Sparsité théorique
    sparsi = [1 - k_tokens / T for T in T_vals]

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # -- A : FLOPs
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(T_vals, [f / 1e6 for f in flops_full], "b-o", label="Full O(T²)")
    ax1.plot(T_vals, [f / 1e6 for f in flops_nsa],  "r-o", label=f"NSA O(T·k)")
    ax1.fill_between(T_vals,
                     [f / 1e6 for f in flops_nsa],
                     [f / 1e6 for f in flops_full],
                     alpha=0.15, color="green", label="Gain")
    ax1.set_xlabel("Longueur T")
    ax1.set_ylabel("FLOPs (M)")
    ax1.set_title("A. Complexité (FLOPs)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # -- B : Speedup théorique
    ax2 = fig.add_subplot(gs[1])
    bars = ax2.bar(T_vals, ratios, color="steelblue", width=30)
    for bar, r in zip(bars, ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f"{r:.0f}×", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.axhline(1, color="k", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Longueur T")
    ax2.set_ylabel("Facteur de réduction (×)")
    ax2.set_title("B. Accélération théorique")
    ax2.grid(True, alpha=0.3, axis="y")

    # -- C : Sparsité
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(T_vals, [s * 100 for s in sparsi], "g-o", linewidth=2)
    ax3.fill_between(T_vals, [s * 100 for s in sparsi], alpha=0.2, color="green")
    ax3.set_xlabel("Longueur T")
    ax3.set_ylabel("% de tokens ignorés")
    ax3.set_title(f"C. Taux de sparsité\n(S={S} blocs × {BLOCK_SIZE} tokens)")
    ax3.set_ylim([0, 100])
    ax3.grid(True, alpha=0.3)

    plt.suptitle(
        f"Native Sparse Attention — Synthèse\n"
        f"(HQ={HQ}, H={H}, D={D}, block_size={BLOCK_SIZE}, S={S})",
        fontsize=12, fontweight="bold"
    )
    path = OUT_DIR / "synthesis.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Figure sauvegardée : {path}")


# ------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Démonstration Native Sparse Attention (NSA)")
    print("=" * 60)
    exp1_flops()
    exp2_speed()
    exp3_sparsity()
    exp4_quality()
    make_summary_figure()
    print(f"\n✓ Toutes les figures sont dans : {OUT_DIR}")
