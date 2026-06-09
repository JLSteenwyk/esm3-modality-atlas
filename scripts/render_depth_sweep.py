"""The depth-sweep hero — a polished, share-ready "watch the modalities fuse" GIF.

Composite, dark-theme, 16:9:

  · main panel  — fixed-camera depth-sweep. All points live in ONE shared
    joint-PCA frame (figures/embed/joint/), so the camera stays locked and the
    cloud morphs in place: five distinct modality clouds at L0 collapse into one
    fused blob by L47. Point positions are linearly interpolated between layers.
  · two insets  — small, continuously spinning per-layer views of L0 ("distinct")
    and L47 ("fused"), so the separation-vs-convergence contrast is legible at a
    glance.
  · a separation meter (per-layer silhouette) and a depth track narrate progress.

Run with PREVIEW=<frame> to dump a single composite frame to /tmp for fast
layout iteration instead of rendering the whole GIF.

Output: figures/gifs/depth_sweep.gif
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.viz import CONDITION_COLOR, CONDITION_LABEL, CONDITIONS, FPS, strip_axes  # noqa: E402

JOINT_DIR = ROOT / "figures" / "embed" / "joint"
PER_DIR = ROOT / "figures" / "embed" / "per_layer"
METRICS = ROOT / "results" / "metrics.json"
OUT = ROOT / "figures" / "gifs" / "depth_sweep.gif"

# --- dark "AI-lab" theme ---
BG = "#0A0E1A"
TEXT = "#EaEef6"
MUTED = "#8C9AB4"
FAINT = "#2A3349"
ACCENT = "#7AA2FF"

HOLD_START = 14
MORPH = 18
HOLD = 10
HOLD_END = 40
MAIN_SIZE = 15
INSET_SIZE = 11


def _smoothstep(p: float) -> float:
    return p * p * (3.0 - 2.0 * p)


def _build_schedule(n_layers: int) -> list[float]:
    s = [0.0] * HOLD_START
    for i in range(n_layers - 1):
        s += [i + _smoothstep((f + 1) / MORPH) for f in range(MORPH)]
        s += [float(i + 1)] * HOLD
    s += [float(n_layers - 1)] * HOLD_END
    return s


def _scatters(ax, coords, cond, size):
    """One scatter per condition, vivid on dark (no depthshade)."""
    out = {}
    for c in CONDITIONS:
        m = cond == c
        if not m.any():
            out[c] = None
            continue
        out[c] = ax.scatter(coords[m, 0], coords[m, 1], coords[m, 2],
                            c=CONDITION_COLOR[c], s=size, edgecolors="none",
                            alpha=0.92, depthshade=False)
    return out


def _limits(coords, lo=1.5, hi=98.5, pad=0.08):
    a, b = np.percentile(coords, lo, axis=0), np.percentile(coords, hi, axis=0)
    p = pad * (b - a)
    return list(zip(a - p, b + p))


def main() -> None:
    layers = sorted(int(p.stem.split("_")[-1]) for p in JOINT_DIR.glob("layer_*.npz"))
    coords = {L: np.load(JOINT_DIR / f"layer_{L:02d}.npz", allow_pickle=True)["coords3d"]
              for L in layers}
    labels = np.load(JOINT_DIR / f"layer_{layers[0]:02d}.npz",
                     allow_pickle=True)["condition"].astype(str)
    masks = {c: labels == c for c in CONDITIONS}

    metrics = json.loads(METRICS.read_text())
    sil = dict(zip(metrics["series"]["layers"], metrics["series"]["silhouette"]))
    sil_arr = np.array([sil[L] for L in layers])
    sil_lo, sil_hi = float(sil_arr.min()), float(sil_arr.max())

    main_lims = _limits(np.concatenate([coords[L] for L in layers]))

    # insets: best standalone per-layer PCA view of the two endpoints
    def load_per(L):
        d = np.load(PER_DIR / f"layer_{L:02d}.npz", allow_pickle=True)
        return d["coords3d"], d["condition"].astype(str)
    c0, cond0 = load_per(layers[0])
    c47, cond47 = load_per(layers[-1])
    lim0, lim47 = _limits(c0), _limits(c47)

    sched = _build_schedule(len(layers))
    n_frames = len(sched)

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)

    fig.text(0.035, 0.95, "Watch the modalities fuse", ha="left", va="top",
             fontsize=25, fontweight="bold", color=TEXT, family="DejaVu Sans")
    fig.text(0.036, 0.895, "ESM3's input modalities collapse into one shared "
             "subspace with depth", ha="left", va="top", fontsize=12.5,
             color=MUTED, family="DejaVu Sans")

    # main panel
    ax = fig.add_axes([0.005, 0.16, 0.60, 0.66], projection="3d")
    ax.set_facecolor(BG)
    strip_axes(ax, zoom=7.0)
    ax.set_xlim(*main_lims[0]); ax.set_ylim(*main_lims[1]); ax.set_zlim(*main_lims[2])
    main_sc = _scatters(ax, coords[layers[0]], labels, MAIN_SIZE)

    layer_txt = fig.text(0.055, 0.79, "", ha="left", va="top", fontsize=19,
                         fontweight="bold", color=TEXT, family="DejaVu Sans")

    # separation meter (under the main panel)
    axm = fig.add_axes([0.055, 0.115, 0.40, 0.02]); axm.set_xlim(0, 1); axm.set_ylim(0, 1); axm.axis("off")
    axm.add_patch(plt.Rectangle((0, 0.2), 1.0, 0.6, fc=FAINT, ec="none"))
    meter_fill = axm.add_patch(plt.Rectangle((0, 0.2), 0.0, 0.6, fc=ACCENT, ec="none"))
    axm.text(0.0, 1.9, "fused", ha="left", va="bottom", fontsize=8.5, color=MUTED)
    axm.text(1.0, 1.9, "distinct", ha="right", va="bottom", fontsize=8.5, color=MUTED)
    meter_txt = axm.text(0.5, -1.1, "", ha="center", va="top", fontsize=9, color=MUTED)

    # depth track
    axt = fig.add_axes([0.055, 0.07, 0.40, 0.015]); axt.set_xlim(0, 1); axt.set_ylim(0, 1); axt.axis("off")
    axt.plot([0, 1], [0.5, 0.5], color=FAINT, lw=2, solid_capstyle="round")
    for i, L in enumerate(layers):
        xx = i / (len(layers) - 1)
        axt.scatter(xx, 0.5, s=16, color=FAINT, edgecolors="none", zorder=2)
        axt.text(xx, -1.2, f"L{L}", ha="center", va="top", fontsize=7.5, color=MUTED)
    track_dot = axt.scatter([0], [0.5], s=60, color=ACCENT, edgecolors=BG, linewidths=1.2, zorder=3)

    # ---- insets ----
    def make_inset(rect, coords_i, cond_i, lim_i):
        a = fig.add_axes(rect, projection="3d"); a.set_facecolor(BG)
        strip_axes(a, zoom=7.6)
        a.set_xlim(*lim_i[0]); a.set_ylim(*lim_i[1]); a.set_zlim(*lim_i[2])
        return a, _scatters(a, coords_i, cond_i, INSET_SIZE)

    ax_top, sc_top = make_inset([0.63, 0.45, 0.345, 0.345], c0, cond0, lim0)
    ax_bot, sc_bot = make_inset([0.63, 0.075, 0.345, 0.345], c47, cond47, lim47)
    fig.text(0.802, 0.815, f"LAYER {layers[0]}", ha="center", va="top", fontsize=12,
             fontweight="bold", color=TEXT, family="DejaVu Sans")
    fig.text(0.802, 0.788, "distinct subspaces", ha="center", va="top",
             fontsize=9.5, color=ACCENT, family="DejaVu Sans")
    fig.text(0.802, 0.44, f"LAYER {layers[-1]}", ha="center", va="top", fontsize=12,
             fontweight="bold", color=TEXT, family="DejaVu Sans")
    fig.text(0.802, 0.413, "fused into one", ha="center", va="top",
             fontsize=9.5, color="#DC267F", family="DejaVu Sans")

    # legend (custom, light text) along the bottom-left
    axl = fig.add_axes([0.035, 0.0, 0.58, 0.045]); axl.set_xlim(0, 1); axl.set_ylim(0, 1); axl.axis("off")
    for j, c in enumerate(CONDITIONS):
        x = 0.5 / len(CONDITIONS) + j / len(CONDITIONS)
        axl.scatter(x - 0.045, 0.5, s=70, c=CONDITION_COLOR[c], edgecolors="none", clip_on=False)
        axl.text(x - 0.022, 0.5, CONDITION_LABEL[c], ha="left", va="center",
                 fontsize=9.5, color=TEXT, family="DejaVu Sans")

    fig.text(0.965, 0.02, "esm3-sm-open-v1  ·  199 human proteins  ·  joint PCA(3)",
             ha="right", va="bottom", fontsize=8, color=MUTED, family="DejaVu Sans")

    def update(fi: int):
        s = sched[fi]
        lo_i = int(np.floor(s)); hi_i = min(lo_i + 1, len(layers) - 1); t = s - lo_i
        cur = (1 - t) * coords[layers[lo_i]] + t * coords[layers[hi_i]]
        for c in CONDITIONS:
            if main_sc[c] is not None:
                m = masks[c]; main_sc[c]._offsets3d = (cur[m, 0], cur[m, 1], cur[m, 2])

        prog = fi / max(1, n_frames - 1)
        ax.view_init(elev=19.0 + 3.0 * np.sin(2 * np.pi * prog), azim=-70.0 + 90.0 * prog)
        spin = -60.0 + 360.0 * 1.4 * prog
        ax_top.view_init(elev=18.0, azim=spin)
        ax_bot.view_init(elev=18.0, azim=spin)

        s_now = float((1 - t) * sil_arr[lo_i] + t * sil_arr[hi_i])
        meter_fill.set_width(max(0.0, min(1.0, (s_now - sil_lo) / (sil_hi - sil_lo + 1e-9))))
        meter_txt.set_text(f"modality separation (silhouette) = {s_now:.2f}")
        nearest = layers[int(round(s))]
        layer_txt.set_text(f"LAYER {nearest}\nof {layers[-1]}")
        track_dot.set_offsets([[s / (len(layers) - 1), 0.5]])
        arts = [a for a in main_sc.values() if a is not None]
        return arts

    preview = os.environ.get("PREVIEW")
    if preview is not None:
        update(int(preview))
        p = Path(f"/tmp/sweep_preview_{preview}.png")
        fig.savefig(p, facecolor=BG)
        print(f"preview -> {p}")
        return

    print(f"layers: {layers}  frames: {n_frames}  (~{n_frames / FPS:.1f}s @ {FPS}fps)")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000.0 / FPS, blit=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=FPS), savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print(f"  wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
