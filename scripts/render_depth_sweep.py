"""The depth-sweep hero — watch the modalities fuse.

A single fixed-camera 3D panel whose frames step through layers L0 -> L47. All
points live in ONE shared coordinate frame (the joint-PCA basis in
figures/embed/joint/, fit across every layer at once), so the camera can stay
locked and the clusters genuinely morph in place: five distinct modality clouds
at the input collapse into a single fused blob by the final layers.

Point order is identical across the joint-layer NPZs, so a point's position is
linearly interpolated between consecutive layers for a smooth morph. A small
"separation" meter (per-layer silhouette) and a depth track narrate where we are.

Output: figures/gifs/depth_sweep.gif
"""

from __future__ import annotations

import json
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
    CONDITIONS,
    FPS,
    INK,
    draw_grouped_legend_horizontal,
    strip_axes,
)

JOINT_DIR = ROOT / "figures" / "embed" / "joint"
METRICS = ROOT / "results" / "metrics.json"
OUT = ROOT / "figures" / "gifs" / "depth_sweep.gif"

HOLD_START = 20     # frames dwelling on L0 before the sweep
MORPH = 24          # frames morphing between adjacent layers
HOLD = 16           # frames dwelling on each layer
HOLD_END = 56       # extra dwell + admire-the-blob rotation at the last layer
TARGET_SIZE = 16


def _smoothstep(p: float) -> float:
    return p * p * (3.0 - 2.0 * p)


def _build_schedule(n_layers: int) -> list[float]:
    """Continuous layer-index s per frame: dwell, eased morph, dwell, ... ."""
    s = [0.0] * HOLD_START
    for i in range(n_layers - 1):
        s += [i + _smoothstep((f + 1) / MORPH) for f in range(MORPH)]
        s += [float(i + 1)] * HOLD
    s += [float(n_layers - 1)] * HOLD_END
    return s


def main() -> None:
    layers = sorted(int(p.stem.split("_")[-1]) for p in JOINT_DIR.glob("layer_*.npz"))
    if not layers:
        raise SystemExit(f"No joint embeddings in {JOINT_DIR}; run embed_3d.py first.")

    coords = {L: np.load(JOINT_DIR / f"layer_{L:02d}.npz", allow_pickle=True)["coords3d"]
              for L in layers}
    labels = np.load(JOINT_DIR / f"layer_{layers[0]:02d}.npz",
                     allow_pickle=True)["condition"].astype(str)
    masks = {c: labels == c for c in CONDITIONS}

    metrics = json.loads(METRICS.read_text())
    sil = dict(zip(metrics["series"]["layers"], metrics["series"]["silhouette"]))
    sil_arr = np.array([sil[L] for L in layers])
    sil_lo, sil_hi = float(sil_arr.min()), float(sil_arr.max())

    # Fixed camera frame: robust (percentile) limits across all layers, padded.
    allc = np.concatenate([coords[L] for L in layers], axis=0)
    lo = np.percentile(allc, 1.5, axis=0)
    hi = np.percentile(allc, 98.5, axis=0)
    pad = 0.08 * (hi - lo)
    lims = list(zip(lo - pad, hi + pad))

    sched = _build_schedule(len(layers))
    n_frames = len(sched)

    fig = plt.figure(figsize=(9.6, 9.2), dpi=110, facecolor="white")
    fig.text(0.5, 0.965, "Watch the modalities fuse", ha="center", va="top",
             fontsize=23, fontweight="bold", color=INK, family="DejaVu Sans")
    fig.text(0.5, 0.928, "ESM3 residual stream in a fixed joint-PCA frame  ·  "
             "each frame is a deeper layer  ·  L0 → L47",
             ha="center", va="top", fontsize=11.5, color=INK,
             family="DejaVu Sans", alpha=0.75)

    # separation meter (top)
    ax_m = fig.add_axes([0.28, 0.862, 0.44, 0.022])
    ax_m.set_xlim(0, 1); ax_m.set_ylim(0, 1); ax_m.axis("off")
    ax_m.add_patch(plt.Rectangle((0, 0.25), 1.0, 0.5, fc="#e2e8f0", ec="none"))
    meter_fill = ax_m.add_patch(plt.Rectangle((0, 0.25), 0.0, 0.5,
                                              fc=CONDITION_COLOR["sequence"], ec="none"))
    ax_m.text(0.0, 1.6, "fused", ha="left", va="bottom", fontsize=8.5, color="#64748b")
    ax_m.text(1.0, 1.6, "distinct", ha="right", va="bottom", fontsize=8.5, color="#64748b")
    meter_txt = ax_m.text(0.5, -0.8, "", ha="center", va="top", fontsize=9,
                          color=INK, family="DejaVu Sans")

    # big layer readout in the empty upper-left of the panel
    layer_txt = fig.text(0.12, 0.82, "", ha="left", va="top", fontsize=17,
                         fontweight="bold", color=INK, family="DejaVu Sans")

    # depth track (just above legend)
    ax_t = fig.add_axes([0.18, 0.155, 0.64, 0.018])
    ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1); ax_t.axis("off")
    ax_t.plot([0, 1], [0.5, 0.5], color="#cbd5e1", lw=2, solid_capstyle="round")
    for i, L in enumerate(layers):
        xx = i / (len(layers) - 1)
        ax_t.scatter(xx, 0.5, s=22, color="#cbd5e1", edgecolors="none", zorder=2)
        ax_t.text(xx, -1.0, f"L{L}", ha="center", va="top", fontsize=8, color="#94a3b8")
    track_dot = ax_t.scatter([0], [0.5], s=70, color=INK, edgecolors="white",
                             linewidths=1.0, zorder=3)

    # main 3D panel
    ax = fig.add_axes([0.03, 0.18, 0.94, 0.66], projection="3d")
    strip_axes(ax, zoom=8.0)
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    scatters = {}
    for c in CONDITIONS:
        m = masks[c]
        c0 = coords[layers[0]][m]
        scatters[c] = ax.scatter(c0[:, 0], c0[:, 1], c0[:, 2],
                                 c=CONDITION_COLOR[c], s=TARGET_SIZE,
                                 edgecolors="none", alpha=0.9, depthshade=True)

    draw_grouped_legend_horizontal(fig, bottom=0.015, height=0.085,
                                   dot_size=90, state_fontsize=11, cat_fontsize=10)

    def update(fi: int):
        s = sched[fi]
        lo_i = int(np.floor(s))
        hi_i = min(lo_i + 1, len(layers) - 1)
        t = s - lo_i
        cur = (1 - t) * coords[layers[lo_i]] + t * coords[layers[hi_i]]
        for c in CONDITIONS:
            m = masks[c]
            scatters[c]._offsets3d = (cur[m, 0], cur[m, 1], cur[m, 2])

        # locked camera with a slow drift so depth reads
        prog = fi / max(1, n_frames - 1)
        ax.view_init(elev=20.0 + 4.0 * np.sin(2 * np.pi * prog),
                     azim=-72.0 + 96.0 * prog)

        s_now = float((1 - t) * sil_arr[lo_i] + t * sil_arr[hi_i])
        frac = (s_now - sil_lo) / (sil_hi - sil_lo + 1e-9)
        meter_fill.set_width(max(0.0, min(1.0, frac)))
        meter_txt.set_text(f"modality separation (silhouette) = {s_now:.2f}")
        nearest = layers[int(round(s))]
        layer_txt.set_text(f"Layer {nearest}\nof {layers[-1]}")
        track_dot.set_offsets([[s / (len(layers) - 1), 0.5]])
        return list(scatters.values())

    print(f"layers: {layers}  frames: {n_frames}  (~{n_frames / FPS:.1f}s @ {FPS}fps)")
    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000.0 / FPS, blit=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"  wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
