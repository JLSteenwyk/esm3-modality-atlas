"""Assemble the publication figure suite (four multi-panel main figures).

Re-plots every panel from the committed result JSONs (and the joint-PCA
embeddings) into consistent, publication-styled composites with panel letters,
written to paper/figures/figure{1..4}.png at 300 dpi.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.viz import CONDITION_COLOR, CONDITION_LABEL, CONDITIONS, INK  # noqa: E402

OUT = ROOT / "paper" / "figures"
R = ROOT / "results"
JOINT = ROOT / "figures" / "scaled" / "embed" / "joint"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "figure.dpi": 300,
})
SEQc = CONDITION_COLOR["sequence"]


def panel(ax, letter, dx=-24, dy=6):
    """Panel letter anchored to the axes' top-left corner with a fixed point
    offset, so it sits in a consistent outer corner regardless of titles, axis
    labels, or layout (dx, dy are offsets in points)."""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points", fontsize=12,
                fontweight="bold", va="bottom", ha="right", color=INK,
                annotation_clip=False)


def below(ax, ncol=2, y=-0.32, handles=None, labels=None):
    """Place a legend below the axes (outside the plot) to avoid overlap."""
    kw = dict(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
              frameon=False, fontsize=7, handlelength=1.6, columnspacing=1.2)
    if handles is not None:
        ax.legend(handles, labels, **kw)
    else:
        ax.legend(**kw)


def jload(p):
    return json.loads((R / p).read_text())


# --------------------------------------------------------------------------- #
# Figure 1 — modalities fuse into a shared subspace with depth
# --------------------------------------------------------------------------- #
def figure1():
    snaps = [0, 12, 24, 35, 47]
    views = [(18, -62), (10, 28), (34, -120)]   # three camera orientations
    coords = {L: np.load(JOINT / f"layer_{L:02d}.npz", allow_pickle=True)
              for L in snaps}
    cond = coords[0]["condition"].astype(str)
    allc = np.concatenate([coords[L]["coords3d"] for L in snaps])
    lim = [(np.percentile(allc[:, i], 1.5), np.percentile(allc[:, i], 98.5))
           for i in range(3)]
    nrow = len(views)

    fig = plt.figure(figsize=(7.2, 1.45 * nrow + 2.8))
    gs = gridspec.GridSpec(nrow + 2, len(snaps),
                           height_ratios=[1] * nrow + [0.35, 1.7],
                           hspace=0.12, wspace=0.04)
    for r, (el, az) in enumerate(views):
        for j, L in enumerate(snaps):
            ax = fig.add_subplot(gs[r, j], projection="3d")
            c = coords[L]["coords3d"]
            for k in CONDITIONS:
                m = cond == k
                ax.scatter(c[m, 0], c[m, 1], c[m, 2], s=1.6, c=CONDITION_COLOR[k],
                           alpha=0.55, edgecolors="none", depthshade=False)
            ax.view_init(el, az)
            ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            if r == 0:
                ax.set_title(f"layer {L}", fontsize=8, pad=-1)
            if j == 0:
                ax.text2D(-0.08, 0.5, f"view {r + 1}", transform=ax.transAxes,
                          rotation=90, va="center", ha="center", fontsize=7,
                          color="#64748b")
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.set_visible(False)
    fig.text(0.065, 0.985, "a", fontsize=12, fontweight="bold", color=INK,
             va="top")

    # modality colour key for panel a, in its own row directly below the snapshots
    handles = [plt.Line2D([0], [0], marker="o", ls="", ms=5,
               mfc=CONDITION_COLOR[k], mec="none", label=CONDITION_LABEL[k])
               for k in CONDITIONS]
    ax_key = fig.add_subplot(gs[nrow, :]); ax_key.axis("off")
    ax_key.legend(handles=handles, loc="center", ncol=6, frameon=False,
                  fontsize=8, handletextpad=0.4, columnspacing=1.4)

    # panel b — fusion curve (dual axis; line identity given by the coloured axes)
    m = jload("scaled/metrics.json")["series"]
    x = np.array(m["layers"])
    axb = fig.add_subplot(gs[nrow + 1, :])
    axb.plot(x, m["silhouette"], "-o", color=SEQc, ms=2.5, lw=1.6)
    axb2 = axb.twinx()
    axb2.plot(x, m["integration_index"], "-s", color=INK, ms=2.5, lw=1.6)
    axb.set_xlabel("layer")
    axb.set_ylabel("condition separation\n(silhouette)", color=SEQc)
    axb2.set_ylabel("integration index\n(mean pairwise CKA)", color=INK)
    axb.tick_params(axis="y", labelcolor=SEQc)
    axb2.tick_params(axis="y", labelcolor=INK)
    axb.axvspan(25, 35, color="0.85", alpha=0.5, zorder=0)
    axb.set_xticks(range(0, 48, 4))
    axb2.spines["top"].set_visible(False)
    panel(axb, "b")
    fig.savefig(OUT / "figure1.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote figure1.png")


# --------------------------------------------------------------------------- #
# Figure 2 — ordered fusion; the functional channel stays separate
# --------------------------------------------------------------------------- #
def figure2():
    d = jload("scaled/metrics.json")
    layers = d["meta"]["layers"]
    pairs = list(d["per_layer"][str(layers[0])]["cka_pairs"])
    mat = np.array([[d["per_layer"][str(L)]["cka_pairs"][p] for L in layers]
                    for p in pairs])
    onset = [next((layers[i] for i, v in enumerate(row) if v >= 0.5), 99)
             for row in mat]
    order = np.argsort(onset)
    mat, pairs = mat[order], [pairs[i] for i in order]

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.25, 1.0], hspace=0.42)
    axa = fig.add_subplot(gs[0])
    im = axa.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axa.set_yticks(range(len(pairs)))
    axa.set_yticklabels([" ".join(CONDITION_LABEL[c] for c in p.split("|"))
                         .replace("All modalities", "All") for p in pairs], fontsize=6.5)
    axa.set_xticks(range(0, len(layers), 4))
    axa.set_xticklabels([layers[i] for i in range(0, len(layers), 4)])
    axa.set_xlabel("layer")
    for s in axa.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=axa, fraction=0.025, pad=0.02)
    cb.set_label("linear CKA", fontsize=7)
    panel(axa, "a")

    rc = jload("scaled/residue_annotation_compare.json")["series"]
    rl = jload("scaled/residue_annotation_compare.json")["layers"]
    x = np.array(rl)
    axb = fig.add_subplot(gs[1])
    axb.plot(x, rc["structure->all"], "-", color="#FE6100", lw=1.8,
             label="structure to all (fusion reference)")
    axb.plot(x, rc["residue_annotation->physical_mean"], "-o", color=INK, ms=2.5,
             lw=1.8, label="residue-annotation to physical")
    axb.plot(x, rc["function->physical_mean"], "--s", color="#009E73", ms=2.5,
             lw=1.6, label="function to physical")
    axb.set_xlabel("layer"); axb.set_ylabel("linear CKA"); axb.set_ylim(-0.03, 1.0)
    axb.set_xticks(range(0, 48, 4))
    below(axb, ncol=3, y=-0.30)
    panel(axb, "b")
    fig.savefig(OUT / "figure2.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote figure2.png")


# --------------------------------------------------------------------------- #
# Figure 3 — learned, holds per-residue, a re-organization of variance
# --------------------------------------------------------------------------- #
def figure3():
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.2, 2.6))
    t = jload("scaled/metrics.json")["series"]
    r = jload("scaled_randinit/metrics.json")["series"]
    x = np.array(t["layers"])
    axA.plot(x, t["silhouette"], "-", color=INK, lw=1.8, label="trained")
    axA.plot(x, r["silhouette"], "--", color="#94a3b8", lw=1.8, label="random init")
    axA.set_xlabel("layer"); axA.set_ylabel("silhouette")
    axA.set_xticks(range(0, 48, 8)); below(axA, ncol=2)
    panel(axA, "a")

    pr = jload("scaled/per_residue_validation.json")
    xp = np.array(pr["layers"])
    axB.plot(xp, pr["mean_pool"]["integration"], "-o", color=INK, ms=3, lw=1.8,
             label="mean-pool")
    axB.plot(xp, pr["residue"]["integration"], "--s", color=SEQc, ms=3, lw=1.8,
             label="per-residue")
    axB.set_xlabel("layer"); axB.set_ylabel("integration index")
    axB.set_xticks(xp); below(axB, ncol=2)
    panel(axB, "b")

    dg = jload("scaled/diagnostics.json")["per_layer"]
    L = jload("scaled/diagnostics.json")["meta"]["layers"]
    ov = [dg[str(l)]["dimensionality"]["overall_effective_rank"] for l in L]
    wi = [dg[str(l)]["dimensionality"]["within_effective_rank_mean"] for l in L]
    axC.plot(L, ov, "-", color=INK, lw=1.8, label="whole cloud")
    axC.plot(L, wi, "--", color="#94a3b8", lw=1.6, label="per-condition")
    axC.set_xlabel("layer"); axC.set_ylabel("effective rank")
    axC.set_xticks(range(0, 48, 8)); axC.set_ylim(0, 118)
    below(axC, ncol=2)
    panel(axC, "c")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "figure3.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote figure3.png")


# --------------------------------------------------------------------------- #
# Figure 4 — fusion depth tracks secondary structure; universal across life
# --------------------------------------------------------------------------- #
def figure4():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    b = jload("scaled/biology_breakdown.json")
    cs = b["category_onset"]
    order = sorted(cs, key=lambda k: cs[k]["mean_onset"])
    means = [cs[k]["mean_onset"] for k in order]
    err = [cs[k]["std_onset"] / np.sqrt(cs[k]["n"]) for k in order]
    axA.barh(range(len(order)), means, xerr=err, color=SEQc, ecolor="#475569",
             edgecolor="none", height=0.7)
    axA.set_yticks(range(len(order))); axA.set_yticklabels(order, fontsize=7)
    axA.set_xlabel("fusion-onset layer"); axA.set_xlim(20, 30)
    panel(axA, "a")

    s = jload("diverse/stratified.json")
    x = np.array(s["layers"])
    kcol = {"all": INK, "eukaryota": "#648FFF", "bacteria": "#FE6100",
            "archaea": "#009E73"}
    for g in ["all", "eukaryota", "bacteria", "archaea"]:
        axB.plot(x, s["series"][g]["silhouette"], "-", color=kcol[g],
                 lw=2.2 if g == "all" else 1.4,
                 label=f"{g} (n={s['n_by_group'][g]})")
    axB.set_xlabel("layer"); axB.set_ylabel("silhouette")
    axB.set_xticks(range(0, 48, 8)); below(axB, ncol=2)
    axB.axvspan(34, 36, color="0.85", alpha=0.6, zorder=0)
    panel(axB, "b")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "figure4.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote figure4.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figure1(); figure2(); figure3(); figure4()
