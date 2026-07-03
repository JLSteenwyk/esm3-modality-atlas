"""Reusable pieces for the reveal-then-spin GIF aesthetic.

Parameterized over the ESM-3 modality conditions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .palette import (
    CATEGORY_COLOR,
    CATEGORY_GROUPS,
    CONDITION_COLOR,
    CONDITION_LABEL,
    CONDITIONS,
    INK,
)

# Timing.
FPS = 24
N_INTRO = 40          # ~1.7 s intro reveal
N_SPIN = 144          # ~6 s eased revolution
N_FRAMES = N_INTRO + N_SPIN
FADE_FRAMES = 10
TARGET_ALPHA = 0.92


def _ease_cos(t: float) -> float:
    """Cosine smoothstep over [0, 1] — slow start and end."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * t)


def _ease_out_cubic(p: float) -> float:
    return 1.0 - (1.0 - p) ** 3


def strip_axes(ax, zoom: float = 7.5) -> None:
    """Pure transparent 3D axes — no panes, grid, ticks, or labels."""
    ax.set_facecolor("none")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        try:
            axis.line.set_visible(False)
        except Exception:
            pass
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.dist = zoom


def _cluster_factor(condition_idx: int, frame: int) -> float:
    """Per-condition reveal factor at a given frame.

    Each condition begins fading in at evenly spaced slots across the intro,
    each fade lasts FADE_FRAMES, then sits at 1.0 through the spin phase.
    """
    if frame >= N_INTRO:
        return 1.0
    n = len(CONDITIONS)
    stride = max(1.0, (N_INTRO - FADE_FRAMES) / max(n - 1, 1))
    start = condition_idx * stride
    if frame < start:
        return 0.0
    if frame >= start + FADE_FRAMES:
        return 1.0
    return _ease_out_cubic((frame - start) / FADE_FRAMES)


def build_scatters(
    ax,
    coords: np.ndarray,
    labels: np.ndarray,
    target_size: float,
) -> list[tuple[str, "plt.Artist | None", int]]:
    """Build one scatter per condition. Returns [(name, scatter, n_pts), ...]."""
    out: list[tuple[str, plt.Artist | None, int]] = []
    for cond in CONDITIONS:
        mask = labels == cond
        if not mask.any():
            out.append((cond, None, 0))
            continue
        sc = ax.scatter(
            coords[mask, 0], coords[mask, 1], coords[mask, 2],
            c=CONDITION_COLOR[cond],
            s=target_size,
            edgecolors="none",
            alpha=TARGET_ALPHA,
            depthshade=True,
        )
        out.append((cond, sc, int(mask.sum())))
    return out


def apply_appearance(
    scatters: list[tuple[str, "plt.Artist | None", int]],
    frame: int,
    target_size: float,
) -> None:
    """Set per-condition alpha + size based on the reveal schedule."""
    for idx, (_cond, sc, n_pts) in enumerate(scatters):
        if sc is None or n_pts == 0:
            continue
        f = _cluster_factor(idx, frame)
        sc.set_alpha(TARGET_ALPHA * f)
        overshoot = 1.0 + 0.08 * np.sin(np.pi * f)
        size = target_size * (0.35 + 0.65 * f) * overshoot
        sc.set_sizes(np.full(n_pts, size))


def camera_for_frame(i: int) -> tuple[float, float]:
    """Camera (elev, azim) at frame i — drift during intro, eased spin after."""
    if i < N_INTRO:
        p = i / max(1, N_INTRO - 1)
        eased = _ease_out_cubic(p)
        return 22.0, -90.0 + 30.0 * eased
    spin_t = (i - N_INTRO) / max(1, N_SPIN)
    azim = 360.0 * _ease_cos(spin_t) - 60.0
    elev = 22.0 + 3.0 * np.sin(2 * np.pi * spin_t)
    return elev, azim


def draw_grouped_legend_horizontal(
    fig,
    *,
    bottom: float = 0.0,
    height: float = 0.10,
    dot_size: float = 80,
    state_fontsize: float = 11,
    cat_fontsize: float = 10,
) -> None:
    """Bottom-of-figure grouped horizontal legend.

    Category headers sit above their member dots and are colored by category
    (so the eye can pick out groupings at a glance).
    """
    ax = fig.add_axes([0.02, bottom, 0.96, height])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    total = sum(len(g[1]) for g in CATEGORY_GROUPS)
    group_gap = 0.05
    n_gaps = len(CATEGORY_GROUPS) - 1
    slot_w = (1.0 - group_gap * n_gaps) / total

    x = 0.0
    for cat_name, conditions in CATEGORY_GROUPS:
        group_w = slot_w * len(conditions)
        cx = x + group_w / 2
        ax.text(
            cx, 0.93, cat_name,
            ha="center", va="top",
            fontsize=cat_fontsize, fontweight="bold",
            color=CATEGORY_COLOR[cat_name],
            family="DejaVu Sans",
        )
        ax.plot(
            [x + 0.012, x + group_w - 0.012], [0.75, 0.75],
            color=CATEGORY_COLOR[cat_name], linewidth=1.3, solid_capstyle="round",
        )
        for j, c in enumerate(conditions):
            item_x = x + (j + 0.5) * slot_w
            ax.scatter(item_x, 0.45, c=CONDITION_COLOR[c], s=dot_size,
                       edgecolors="none", clip_on=False)
            ax.text(item_x, 0.18, CONDITION_LABEL[c], ha="center", va="top",
                    fontsize=state_fontsize, color=INK, family="DejaVu Sans")
        x += group_w + group_gap


def draw_grouped_legend_vertical(
    fig,
    *,
    left: float = 0.74,
    right: float = 0.99,
    top: float = 0.85,
    bottom: float = 0.12,
    dot_size: float = 80,
    state_fontsize: float = 11,
    cat_fontsize: float = 10,
) -> None:
    """Right-rail grouped vertical legend (for single-panel square layouts)."""
    ax = fig.add_axes([left, bottom, right - left, top - bottom])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cat_header_h = 0.07
    item_h = 0.075
    gap_h = 0.05
    y = 1.0
    for cat_name, conditions in CATEGORY_GROUPS:
        ax.text(0.0, y, cat_name, ha="left", va="top",
                fontsize=cat_fontsize, fontweight="bold",
                color=CATEGORY_COLOR[cat_name], family="DejaVu Sans")
        ax.plot([0.0, 0.55], [y - cat_header_h + 0.012, y - cat_header_h + 0.012],
                color=CATEGORY_COLOR[cat_name], linewidth=1.3, solid_capstyle="round")
        y -= cat_header_h
        for c in conditions:
            ax.scatter(0.05, y - item_h / 2 + 0.015, c=CONDITION_COLOR[c],
                       s=dot_size, edgecolors="none", clip_on=False)
            ax.text(0.13, y - item_h / 2 + 0.012, CONDITION_LABEL[c],
                    ha="left", va="center",
                    fontsize=state_fontsize, color=INK, family="DejaVu Sans")
            y -= item_h
        y -= gap_h
