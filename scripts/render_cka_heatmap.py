"""Per-pair CKA heatmaps across depth.

The scalar integration index in compute_metrics.py averages over all condition
pairs, hiding *which* modalities fuse first. This renders that structure from
the already-computed results/metrics.json (no model or activations needed):

  figures/metrics/cka_pairs_vs_depth.png
      One row per condition pair, one column per layer. Rows sorted by fusion
      onset (earliest-fusing pair at top), so the reading order is the fusion
      order. Cell = linear CKA.

  figures/metrics/cka_matrix_by_layer.png
      The full 5x5 condition-by-condition CKA matrix at each layer, as a strip
      of small multiples — watch the off-diagonal warm up with depth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.viz import CONDITION_LABEL, CONDITIONS, INK  # noqa: E402

RESULTS = ROOT / "results" / "metrics.json"
OUT_FIG = ROOT / "figures" / "metrics"

CMAP = "viridis"
ONSET_THRESHOLD = 0.5  # CKA at which a pair counts as "fused" (for row sorting)


def _pair_label(key: str) -> str:
    a, b = key.split("|")
    return f"{CONDITION_LABEL[a]} ↔ {CONDITION_LABEL[b]}"


def _annotate(ax, data: np.ndarray) -> None:
    """Write each cell value, in a color that stays legible on the colormap."""
    cmap = matplotlib.colormaps[CMAP]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            r, g, b, _ = cmap(v)  # value is already in [0, 1]
            txt = "#0f172a" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.6 else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=txt)


def render_pairs_vs_depth(layers, pair_keys, matrix, out_path: Path) -> None:
    """matrix: (n_pairs, n_layers) CKA, rows already in display order."""
    n_pairs, n_layers = matrix.shape
    fig, ax = plt.subplots(figsize=(0.9 * n_layers + 4.2, 0.5 * n_pairs + 1.8),
                           dpi=130, facecolor="white")
    im = ax.imshow(matrix, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{L}" for L in layers], fontsize=10, color=INK)
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels([_pair_label(k) for k in pair_keys], fontsize=10, color=INK)
    ax.set_xlabel("Layer", fontsize=12, color=INK)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    _annotate(ax, matrix)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("linear CKA", fontsize=10, color=INK)

    ax.set_title("Per-pair representational alignment across ESM3 depth\n"
                 "(rows sorted by fusion onset — earliest-fusing pair on top)",
                 fontsize=13, fontweight="bold", color=INK, pad=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(ROOT)}")


def render_matrix_by_layer(layers, conditions, per_layer, out_path: Path) -> None:
    n = len(layers)
    short = {c: CONDITION_LABEL[c].replace("All modalities", "All") for c in conditions}
    ticks = [short[c] for c in conditions]
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n + 1.0, 2.8),
                             dpi=130, facecolor="white")
    im = None
    for ax, L in zip(axes, layers):
        m = np.array(per_layer[str(L)]["cka_matrix"])
        im = ax.imshow(m, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(f"L{L}", fontsize=12, fontweight="bold", color=INK, pad=6)
        ax.set_xticks(range(len(conditions)))
        ax.set_yticks(range(len(conditions)))
        ax.set_xticklabels(ticks, fontsize=7, rotation=90, color=INK)
        ax.set_yticklabels(ticks if ax is axes[0] else [], fontsize=7, color=INK)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Condition × condition CKA at each layer", fontsize=13,
                 fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    cbar.set_label("linear CKA", fontsize=9, color=INK)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(ROOT)}")


def main() -> None:
    d = json.loads(RESULTS.read_text())
    layers = d["meta"]["layers"]
    per_layer = d["per_layer"]
    conditions = [c for c in CONDITIONS if c in d["meta"]["conditions"]]

    # Collect the per-pair CKA series. Use the first layer's keys as canonical
    # (compute_metrics emits identical, CONDITIONS-ordered keys at every layer).
    pair_keys = list(per_layer[str(layers[0])]["cka_pairs"].keys())
    series = {k: np.array([per_layer[str(L)]["cka_pairs"][k] for L in layers])
              for k in pair_keys}

    # Sort rows by fusion onset: first layer reaching the threshold (earlier =
    # higher up). Pairs that never reach it sink to the bottom; ties broken by
    # mean CKA (more-aligned first).
    def onset(k: str) -> tuple[float, float]:
        vals = series[k]
        hit = np.where(vals >= ONSET_THRESHOLD)[0]
        first = layers[hit[0]] if hit.size else float("inf")
        return (first, -float(vals.mean()))

    pair_keys.sort(key=onset)
    matrix = np.vstack([series[k] for k in pair_keys])

    print(f"layers: {layers}")
    print(f"pairs ({len(pair_keys)}), sorted by fusion onset (threshold "
          f"CKA≥{ONSET_THRESHOLD}):")
    for k in pair_keys:
        o = onset(k)[0]
        tag = f"L{int(o)}" if o != float("inf") else "never"
        print(f"  {_pair_label(k):<28} onset {tag:>5}   "
              f"final L{layers[-1]} = {series[k][-1]:.3f}")

    render_pairs_vs_depth(layers, pair_keys, matrix,
                          OUT_FIG / "cka_pairs_vs_depth.png")
    render_matrix_by_layer(layers, conditions, per_layer,
                           OUT_FIG / "cka_matrix_by_layer.png")


if __name__ == "__main__":
    main()
