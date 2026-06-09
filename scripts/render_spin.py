"""Render the per-layer spinning-modality GIFs.

Reads the per-layer PCA coords from figures/embed/per_layer/ and produces:

  figures/gifs/spin_layer_{L:02d}.gif   one square panel per layer
  figures/gifs/spin_all_layers.gif      multi-panel grid showing all 7 layers
                                        spinning in sync (the hero figure)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.viz import (  # noqa: E402
    CONDITION_COLOR,
    CONDITION_LABEL,
    CONDITIONS,
    FPS,
    INK,
    N_FRAMES,
    apply_appearance,
    build_scatters,
    camera_for_frame,
    draw_grouped_legend_horizontal,
    draw_grouped_legend_vertical,
    strip_axes,
)

IN_DIR = ROOT / "figures" / "embed" / "per_layer"
OUT_DIR = ROOT / "figures" / "gifs"


def _load(layer_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(IN_DIR / f"layer_{layer_idx:02d}.npz", allow_pickle=True)
    return d["coords3d"], d["condition"].astype(object), d["evr"]


def render_single(layer_idx: int, dpi: int = 110) -> Path:
    """Square 1:1 single-layer GIF — clean composition with vertical legend."""
    coords, labels, evr = _load(layer_idx)

    fig = plt.figure(figsize=(8.0, 8.0), dpi=dpi, facecolor="white")
    ax = fig.add_axes([-0.02, 0.04, 0.78, 0.82], projection="3d")
    strip_axes(ax, zoom=5.5)

    fig.text(0.50, 0.965, "The Geometry of ESM3 Modalities",
             ha="center", va="top", fontsize=20, fontweight="bold",
             color=INK, family="DejaVu Sans")
    fig.text(0.50, 0.918, f"Layer {layer_idx}  ·  residual stream  ·  "
             f"PC1–3 = {evr.sum():.0%}",
             ha="center", va="top", fontsize=13,
             color=INK, family="DejaVu Sans")

    scatters = build_scatters(ax, coords, labels, target_size=44)
    draw_grouped_legend_vertical(fig, left=0.74, right=0.99, top=0.82, bottom=0.10)

    def update(i: int):
        apply_appearance(scatters, i, target_size=44)
        elev, azim = camera_for_frame(i)
        ax.view_init(elev=elev, azim=azim)
        return [s for _, s, _ in scatters if s is not None]

    anim = FuncAnimation(fig, update, frames=N_FRAMES,
                         interval=1000.0 / FPS, blit=False)
    out = OUT_DIR / f"spin_layer_{layer_idx:02d}.gif"
    anim.save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")
    return out


def render_all_layers_hero(layers: list[int], dpi: int = 95) -> Path:
    """16:9 multi-panel hero — one 3D panel per layer, synchronized spin.

    Layout: 1 row × 7 columns at the top, evr+layer labels above each panel,
    horizontal grouped legend at the bottom.
    """
    fig = plt.figure(figsize=(18.0, 7.0), dpi=dpi, facecolor="white")

    fig.text(0.50, 0.975, "ESM3 Modality Geometry Across Depth",
             ha="center", va="top", fontsize=22, fontweight="bold",
             color=INK, family="DejaVu Sans")
    fig.text(0.50, 0.915, "Mean-pooled residual stream under single-modality forward "
             "passes  ·  PCA(3) fit independently per layer",
             ha="center", va="top", fontsize=11, color=INK,
             family="DejaVu Sans", alpha=0.75)

    n = len(layers)
    pad_l = 0.012
    pad_r = 0.012
    gap = 0.005
    width = (1.0 - pad_l - pad_r - gap * (n - 1)) / n
    bottom = 0.20
    height = 0.60

    target_size = 18
    per_panel_scatters: list[list[tuple]] = []
    axes = []

    for i, L in enumerate(layers):
        left = pad_l + i * (width + gap)
        coords, labels, evr = _load(L)

        # Layer label sits just above the panel; EVR just below the label.
        fig.text(
            left + width / 2, bottom + height + 0.028,
            f"Layer {L}",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=INK,
            family="DejaVu Sans",
        )
        fig.text(
            left + width / 2, bottom + height + 0.006,
            f"PC1–3 = {evr.sum():.0%}",
            ha="center", va="bottom", fontsize=9,
            color=INK, family="DejaVu Sans", alpha=0.65,
        )

        ax = fig.add_axes([left, bottom, width, height], projection="3d")
        strip_axes(ax, zoom=7.2)
        scatters = build_scatters(ax, coords, labels, target_size)
        per_panel_scatters.append(scatters)
        axes.append(ax)

    draw_grouped_legend_horizontal(fig, bottom=0.03, height=0.13,
                                   dot_size=100, state_fontsize=12,
                                   cat_fontsize=11)

    def update(i: int):
        elev, azim = camera_for_frame(i)
        for ax, scatters in zip(axes, per_panel_scatters):
            apply_appearance(scatters, i, target_size)
            ax.view_init(elev=elev, azim=azim)
        flat = []
        for scs in per_panel_scatters:
            flat.extend(s for _, s, _ in scs if s is not None)
        return flat

    anim = FuncAnimation(fig, update, frames=N_FRAMES,
                         interval=1000.0 / FPS, blit=False)
    out = OUT_DIR / "spin_all_layers.gif"
    anim.save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    layers = sorted(int(p.stem.split("_")[-1])
                    for p in IN_DIR.glob("layer_*.npz"))
    if not layers:
        raise SystemExit(f"No per-layer NPZs in {IN_DIR}; run embed_3d.py first.")
    print(f"layers: {layers}")

    print("\nrendering per-layer GIFs ...")
    for L in layers:
        render_single(L)

    print("\nrendering multi-panel hero ...")
    render_all_layers_hero(layers)


if __name__ == "__main__":
    main()
