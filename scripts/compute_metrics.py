"""Quantitative companions to the modality-geometry atlas.

Reads the full-dimension (1536-d) mean-pooled residual stream from
``activations/by_layer/layer_{L:02d}.npz`` and quantifies, per layer, how
separated vs. fused the modality conditions are. All metrics are computed in
the native 1536-d space (not the 3-d PCA used for the GIFs), so they reflect
the true representation rather than its first three principal components.

Three companions (mirrors the README):

  1. Silhouette score across modality conditions.
       Labels = condition. High → conditions form tight, well-separated
       clusters (distinct subspaces). Drops toward 0 as modalities fuse.

  2. Linear CKA between every condition pair at each layer.
       Centered Kernel Alignment (Kornblith et al. 2019) on the per-protein
       representation matrices. Rotation/scale-invariant similarity of the
       two representational geometries. Rises toward 1 as conditions collapse
       into a shared subspace.

  3. Modality integration index (the headline scalar).
       integration_index = mean linear CKA over all unordered condition pairs.
       Ranges [0, 1]; monotonically rising index = progressive fusion.
       Reported alongside two geometric diagnostics for triangulation:
       centroid-separation eta^2 (fraction of total variance explained by the
       condition label) and the silhouette above.

Outputs:
  results/metrics.json                  full structured dump (per-layer + series)
  figures/metrics/metrics_vs_depth.png  silhouette / integration index vs depth
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.viz import CONDITION_COLOR, CONDITION_LABEL, CONDITIONS, INK  # noqa: E402

IN_DIR = ROOT / "activations" / "by_layer"
OUT_RESULTS = ROOT / "results"
OUT_FIG = ROOT / "figures" / "metrics"


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #
def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two (n_samples, n_features) matrices.

    Rows of x and y must correspond to the same samples (same proteins, same
    order). Invariant to orthogonal transforms and isotropic scaling.
    """
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    # ||Y^T X||_F^2 = <XX^T, YY^T>  (the linear-kernel HSIC numerator)
    hsic_xy = float(np.linalg.norm(x.T @ y, ord="fro") ** 2)
    norm_x = float(np.linalg.norm(x.T @ x, ord="fro"))
    norm_y = float(np.linalg.norm(y.T @ y, ord="fro"))
    denom = norm_x * norm_y
    return hsic_xy / denom if denom > 0 else 0.0


def centroid_separation_eta_sq(coords: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of total variance explained by the condition label (eta^2).

    eta^2 = trace(between-class scatter) / trace(total scatter). 1 → condition
    centroids are maximally spread (distinct subspaces); 0 → centroids coincide
    (fully fused). The complement (1 - eta^2) reads as a geometric integration.
    """
    grand = coords.mean(axis=0, keepdims=True)
    total = float(((coords - grand) ** 2).sum())
    between = 0.0
    for c in np.unique(labels):
        sub = coords[labels == c]
        diff = sub.mean(axis=0, keepdims=True) - grand
        between += sub.shape[0] * float((diff ** 2).sum())
    return between / total if total > 0 else 0.0


def condition_matrices(
    coords: np.ndarray, condition: np.ndarray, protein_id: np.ndarray
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Split a layer's points into one (n_proteins, d) matrix per condition,
    with rows aligned to a single canonical protein order across conditions.

    Returns (ordered_proteins, {condition: matrix}). Conditions are taken in
    canonical CONDITIONS order, restricted to those actually present.
    """
    present = [c for c in CONDITIONS if c in set(condition.tolist())]
    proteins = sorted(set(protein_id.tolist()))
    mats: dict[str, np.ndarray] = {}
    for c in present:
        mask = condition == c
        ids = protein_id[mask]
        rows = coords[mask]
        lookup = {pid: rows[i] for i, pid in enumerate(ids.tolist())}
        # Only keep proteins seen under this condition (should be all of them).
        mats[c] = np.stack([lookup[p] for p in proteins if p in lookup], axis=0)
    return present, mats


# --------------------------------------------------------------------------- #
# Per-layer driver
# --------------------------------------------------------------------------- #
def metrics_for_layer(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    coords = d["coords"].astype(np.float64)        # (N*C, 1536)
    condition = d["condition"].astype(str)
    protein_id = d["protein_id"].astype(str)

    # --- 1. Silhouette ---
    sil_overall = float(silhouette_score(coords, condition, metric="euclidean"))
    samples = silhouette_samples(coords, condition, metric="euclidean")
    sil_by_cond = {
        c: float(samples[condition == c].mean())
        for c in CONDITIONS
        if (condition == c).any()
    }

    # --- 2. CKA between every condition pair ---
    present, mats = condition_matrices(coords, condition, protein_id)
    n_present = len(present)
    cka_mat = np.eye(n_present)
    pairs: dict[str, float] = {}
    pair_vals: list[float] = []
    for i in range(n_present):
        for j in range(i + 1, n_present):
            v = linear_cka(mats[present[i]], mats[present[j]])
            cka_mat[i, j] = cka_mat[j, i] = v
            pairs[f"{present[i]}|{present[j]}"] = v
            pair_vals.append(v)

    mean_pairwise_cka = float(np.mean(pair_vals)) if pair_vals else 1.0

    # CKA of each single modality against the fused "all" condition.
    align_to_all = None
    if "all" in present:
        vs_all = [pairs.get(f"{c}|all", pairs.get(f"all|{c}"))
                  for c in present if c != "all"]
        vs_all = [v for v in vs_all if v is not None]
        align_to_all = float(np.mean(vs_all)) if vs_all else None

    # --- 3. Integration index + geometric diagnostics ---
    eta_sq = centroid_separation_eta_sq(coords, condition)

    return {
        "silhouette_overall": sil_overall,
        "silhouette_by_condition": sil_by_cond,
        "conditions_present": present,
        "cka_matrix": cka_mat.tolist(),
        "cka_pairs": pairs,
        "mean_pairwise_cka": mean_pairwise_cka,
        "alignment_to_all": align_to_all,
        "centroid_separation_eta_sq": eta_sq,
        "integration_index": mean_pairwise_cka,
    }


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def render_plot(layers: list[int], series: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(9.0, 5.0), dpi=130, facecolor="white")
    x = np.array(layers)

    # Integration index (headline) on the left axis.
    ax1.plot(x, series["integration_index"], "-o", color=INK, lw=2.4,
             markersize=6, label="Integration index (mean pairwise CKA)")
    ax1.plot(x, series["centroid_separation_eta_sq"], "--s",
             color="#94a3b8", lw=1.8, markersize=5,
             label="Centroid separation (eta²)")
    ax1.set_xlabel("Layer", fontsize=12, color=INK)
    ax1.set_ylabel("Integration  /  separation", fontsize=12, color=INK)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.25)

    # Silhouette on the right axis (different scale).
    ax2 = ax1.twinx()
    ax2.plot(x, series["silhouette"], "-^", color=CONDITION_COLOR["sequence"],
             lw=2.0, markersize=6, label="Silhouette (condition separation)")
    ax2.set_ylabel("Silhouette score", fontsize=12,
                   color=CONDITION_COLOR["sequence"])
    ax2.tick_params(axis="y", labelcolor=CONDITION_COLOR["sequence"])

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="center left",
               fontsize=9, framealpha=0.9)

    ax1.set_title("Modality fusion across ESM3 depth",
                  fontsize=15, fontweight="bold", color=INK, pad=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
def main() -> None:
    index = json.loads((IN_DIR / "index.json").read_text())
    layers: list[int] = index["layers"]
    print(f"layers: {layers}")
    print(f"conditions: {index['conditions']}")
    print(f"N = {index['n_proteins']} proteins x "
          f"{index['n_conditions']} conditions\n")

    per_layer: dict[str, dict] = {}
    series = {
        "layers": layers,
        "silhouette": [],
        "mean_pairwise_cka": [],
        "integration_index": [],
        "alignment_to_all": [],
        "centroid_separation_eta_sq": [],
    }

    print(f"{'layer':>6}  {'silhouette':>11}  {'integration':>12}  "
          f"{'align→all':>10}  {'eta^2':>7}")
    for L in layers:
        m = metrics_for_layer(IN_DIR / f"layer_{L:02d}.npz")
        per_layer[str(L)] = m
        series["silhouette"].append(m["silhouette_overall"])
        series["mean_pairwise_cka"].append(m["mean_pairwise_cka"])
        series["integration_index"].append(m["integration_index"])
        series["alignment_to_all"].append(m["alignment_to_all"])
        series["centroid_separation_eta_sq"].append(
            m["centroid_separation_eta_sq"])
        a2a = m["alignment_to_all"]
        print(f"{L:>6}  {m['silhouette_overall']:>11.3f}  "
              f"{m['integration_index']:>12.3f}  "
              f"{(a2a if a2a is not None else float('nan')):>10.3f}  "
              f"{m['centroid_separation_eta_sq']:>7.3f}")

    out = {
        "meta": {
            "n_proteins": index["n_proteins"],
            "conditions": index["conditions"],
            "conditions_order": list(CONDITIONS),
            "layers": layers,
            "d_model": 1536,
            "space": "full (1536-d mean-pooled residual stream)",
            "distance": "euclidean",
            "cka": "linear (Kornblith et al. 2019)",
            "integration_index": "mean linear CKA over all unordered condition pairs",
        },
        "per_layer": per_layer,
        "series": series,
    }

    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = OUT_RESULTS / "metrics.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_json.relative_to(ROOT)}")

    render_plot(layers, series, OUT_FIG / "metrics_vs_depth.png")


if __name__ == "__main__":
    main()
