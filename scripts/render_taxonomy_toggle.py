"""Toggle GIF: the same cloud is organised by modality, not by taxonomy.

Every point is a (protein, condition) pair, so it carries both a modality label
and a superkingdom label. The cloud spins while the point colours cross-fade
between the two labellings. Coloured by modality the cloud is structured; coloured
by superkingdom the colours reshuffle into noise, which shows that ESM3 represents
proteins by their biochemistry rather than their source organism.

Reads the diverse joint-PCA embedding (figures/diverse/embed/joint).
Output: figures/diverse/gifs/modality_vs_taxonomy.gif
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
from matplotlib.colors import to_rgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.viz import CONDITION_COLOR, CONDITION_LABEL, CONDITIONS, FPS  # noqa: E402

JOINT = ROOT / "figures" / "diverse" / "embed" / "joint"
CURATION = ROOT / "data" / "diverse" / "curation_summary.json"
OUT = ROOT / "figures" / "diverse" / "gifs" / "modality_vs_taxonomy.gif"
LAYER = 24

BG, TEXT, MUTED = "#0A0E1A", "#EaEef6", "#8C9AB4"
KCOL = {"eukaryota": "#648FFF", "bacteria": "#FE6100", "archaea": "#009E73"}
HOLD, FADE = 58, 28


def _smooth(p):
    return p * p * (3 - 2 * p)


def main() -> None:
    d = np.load(JOINT / f"layer_{LAYER:02d}.npz", allow_pickle=True)
    xyz = d["coords3d"]
    cond = d["condition"].astype(str)
    org = d["category"].astype(str)
    o2k = {o: v["superkingdom"] for o, v in
           json.loads(CURATION.read_text())["per_organism"].items()}
    king = np.array([o2k.get(o, "eukaryota") for o in org])

    rng = np.random.default_rng(0)
    if len(xyz) > 4000:
        keep = rng.choice(len(xyz), 4000, replace=False)
        xyz, cond, king = xyz[keep], cond[keep], king[keep]
    c_mod = np.array([to_rgb(CONDITION_COLOR[c]) for c in cond])
    c_kng = np.array([to_rgb(KCOL[k]) for k in king])
    lim = [(np.percentile(xyz[:, i], 1), np.percentile(xyz[:, i], 99)) for i in range(3)]

    sched = ([0.0] * HOLD + [_smooth((i + 1) / FADE) for i in range(FADE)]
             + [1.0] * HOLD + [1 - _smooth((i + 1) / FADE) for i in range(FADE)])
    n = len(sched)

    fig = plt.figure(figsize=(8.0, 8.4), dpi=110, facecolor=BG)
    fig.text(0.5, 0.965, "Organised by modality, not by taxonomy", ha="center",
             va="top", fontsize=20, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.925, f"ESM3 residual stream, layer {LAYER}  ·  the same cloud, "
             "two labellings  ·  tree of life", ha="center", va="top",
             fontsize=11, color=MUTED)
    label = fig.text(0.5, 0.875, "", ha="center", va="top", fontsize=13,
                     fontweight="bold", color=TEXT)

    ax = fig.add_axes([0.02, 0.10, 0.96, 0.76], projection="3d")
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis.line.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=3, c=c_mod,
                    edgecolors="none", depthshade=False, alpha=0.85)

    # two persistent legend rows, toggled by visibility (no per-frame redraw)
    axl = fig.add_axes([0.05, 0.0, 0.90, 0.09]); axl.axis("off")
    axl.set_xlim(0, 1); axl.set_ylim(0, 1)

    def build_legend(items):
        arts = []
        xs = np.linspace(0.5 / len(items), 1 - 0.5 / len(items), len(items))
        for x, (col, lab) in zip(xs, items):
            arts.append(axl.scatter(x - 0.045, 0.5, s=70, c=col, edgecolors="none",
                                    clip_on=False))
            arts.append(axl.text(x - 0.028, 0.5, lab, ha="left", va="center",
                                 color=TEXT, fontsize=9))
        return arts

    mod_leg = build_legend([(CONDITION_COLOR[c], CONDITION_LABEL[c]) for c in CONDITIONS])
    kng_leg = build_legend([(KCOL[k], k) for k in ["eukaryota", "bacteria", "archaea"]])

    def update(i):
        f = sched[i]
        sc.set_facecolors((1 - f) * c_mod + f * c_kng)
        ax.view_init(elev=18 + 3 * np.sin(2 * np.pi * i / n),
                     azim=-60 + 360 * i / n)
        is_mod = f < 0.5
        label.set_text("coloured by  MODALITY" if is_mod else "coloured by  SUPERKINGDOM")
        label.set_color(TEXT if is_mod else "#7AA2FF")
        for a in mod_leg:
            a.set_visible(is_mod)
        for a in kng_leg:
            a.set_visible(not is_mod)
        return [sc]

    print(f"frames: {n}  (~{n / FPS:.1f}s)  points: {len(xyz)}")
    anim = FuncAnimation(fig, update, frames=n, interval=1000 / FPS, blit=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=FPS), savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
