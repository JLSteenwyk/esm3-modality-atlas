"""The honest LDA companion — modality identity is decodable at every depth.

This is the deliberate counterpoint to the fusion hero. Supervised LDA finds the
linear projection that MAXIMALLY separates the five modality labels, so it draws
crisp, well-separated clusters at *every* layer — including the final layers
where the unsupervised geometry has fully fused (silhouette ~0.05) and the
depth-sweep shows one blob.

That is not a contradiction: it visualizes the probe finding (compute_diagnostics)
that a thin additive "modality tag" keeps the conditions linearly separable
throughout. LDA seizes that tag; PCA (variance-following) ignores it and shows
the fusion. Shown side by side, the pair makes the point precisely:

    geometry fuses (PCA / depth-sweep)  ·  identity persists (LDA)

Outputs:
  figures/metrics/lda_separability.png
  figures/embed/lda/layer_{L:02d}.npz   (LDA-3D coords, for reproducibility)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.viz import (  # noqa: E402
    CONDITION_COLOR,
    CONDITIONS,
    INK,
    build_scatters,
    draw_grouped_legend_horizontal,
    strip_axes,
)

IN_DIR = ROOT / "activations" / "by_layer"
OUT_EMBED = ROOT / "figures" / "embed" / "lda"
OUT_FIG = ROOT / "figures" / "metrics" / "lda_separability.png"

VIEW_ELEV, VIEW_AZIM = 20.0, -60.0


def main() -> None:
    layers = sorted(int(p.stem.split("_")[-1]) for p in IN_DIR.glob("layer_*.npz"))
    OUT_EMBED.mkdir(parents=True, exist_ok=True)

    coords3d, labels_by_layer, sils = {}, {}, {}
    for L in layers:
        d = np.load(IN_DIR / f"layer_{L:02d}.npz", allow_pickle=True)
        X = d["coords"].astype(np.float64)
        y = d["condition"].astype(str)
        # 5 classes -> at most 4 discriminants; take the leading 3 for a 3-D view.
        Z = LinearDiscriminantAnalysis(n_components=3).fit_transform(X, y)
        coords3d[L] = Z
        labels_by_layer[L] = y
        sils[L] = float(silhouette_score(Z, y))
        np.savez_compressed(OUT_EMBED / f"layer_{L:02d}.npz",
                            coords3d=Z.astype(np.float32), condition=y,
                            protein_id=d["protein_id"], layer_idx=np.int32(L))
        print(f"  L{L:02d}  LDA-3D silhouette = {sils[L]:.3f}")

    # ----- multi-panel figure, one LDA-3D panel per layer -----
    n = len(layers)
    fig = plt.figure(figsize=(18.0, 6.6), dpi=110, facecolor="white")
    fig.text(0.5, 0.975, "Modality identity stays linearly decodable at every depth",
             ha="center", va="top", fontsize=21, fontweight="bold", color=INK,
             family="DejaVu Sans")
    fig.text(0.5, 0.918, "Supervised LDA — the projection that *maximizes* "
             "condition separation. Clusters stay distinct even at L47, where the "
             "unsupervised geometry has fused:\nthis is the persistent additive "
             "“modality tag”, not unfused geometry (contrast the PCA depth-sweep).",
             ha="center", va="top", fontsize=11, color=INK,
             family="DejaVu Sans", alpha=0.78)

    pad_l = pad_r = 0.012
    gap = 0.005
    width = (1.0 - pad_l - pad_r - gap * (n - 1)) / n
    bottom, height = 0.215, 0.585
    for i, L in enumerate(layers):
        left = pad_l + i * (width + gap)
        fig.text(left + width / 2, bottom + height + 0.030, f"Layer {L}",
                 ha="center", va="bottom", fontsize=14, fontweight="bold",
                 color=INK, family="DejaVu Sans")
        fig.text(left + width / 2, bottom + height + 0.008,
                 f"LDA sil = {sils[L]:.2f}", ha="center", va="bottom",
                 fontsize=9, color="#64748b", family="DejaVu Sans")
        ax = fig.add_axes([left, bottom, width, height], projection="3d")
        strip_axes(ax, zoom=7.4)
        scatters = build_scatters(ax, coords3d[L], labels_by_layer[L], 16)
        for _c, sc, _n in scatters:
            if sc is not None:
                sc.set_alpha(0.9)
        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    draw_grouped_legend_horizontal(fig, bottom=0.03, height=0.12,
                                   dot_size=100, state_fontsize=12, cat_fontsize=11)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG)
    plt.close(fig)
    print(f"\nwrote {OUT_FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
