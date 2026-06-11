"""Assemble the publication figure suite (four multi-panel main figures).

Re-plots every panel from the committed result JSONs (and the joint-PCA
embeddings) into consistent, publication-styled composites with panel letters,
written to paper/figures/figure{1..4}.png at 300 dpi. Captions live in
paper/figure_captions.md.
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


def panel(ax, letter, dx=-0.10, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="right", color=INK)


def jload(p):
    return json.loads((R / p).read_text())


# --------------------------------------------------------------------------- #
# Figure 1 — modalities fuse into a shared subspace with depth
# --------------------------------------------------------------------------- #
def figure1():
    snaps = [0, 12, 24, 35, 47]
    coords = {L: np.load(JOINT / f"layer_{L:02d}.npz", allow_pickle=True)
              for L in snaps}
    cond = coords[0]["condition"].astype(str)
    allc = np.concatenate([coords[L]["coords3d"] for L in snaps])
    lim = [(np.percentile(allc[:, i], 1.5), np.percentile(allc[:, i], 98.5))
           for i in range(3)]

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = gridspec.GridSpec(2, len(snaps), height_ratios=[1.0, 1.15],
                           hspace=0.32, wspace=0.05)
    for j, L in enumerate(snaps):
        ax = fig.add_subplot(gs[0, j], projection="3d")
        c = coords[L]["coords3d"]
        for k in CONDITIONS:
            m = cond == k
            ax.scatter(c[m, 0], c[m, 1], c[m, 2], s=2, c=CONDITION_COLOR[k],
                       alpha=0.55, edgecolors="none", depthshade=False)
        ax.view_init(18, -62)
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f"layer {L}", fontsize=8, pad=-2)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_visible(False)
    fig.text(0.07, 0.93, "a", fontsize=12, fontweight="bold", color=INK)
    # shared legend
    handles = [plt.Line2D([0], [0], marker="o", ls="", ms=5,
               mfc=CONDITION_COLOR[k], mec="none", label=CONDITION_LABEL[k])
               for k in CONDITIONS]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.50), fontsize=7.5)

    # panel b — fusion curve
    m = jload("scaled/metrics.json")["series"]
    x = np.array(m["layers"])
    axb = fig.add_subplot(gs[1, :])
    axb.plot(x, m["silhouette"], "-o", color=SEQc, ms=2.5, lw=1.6,
             label="condition separation (silhouette)")
    axb2 = axb.twinx()
    axb2.plot(x, m["integration_index"], "-s", color=INK, ms=2.5, lw=1.6,
              label="integration index (mean pairwise CKA)")
    axb.set_xlabel("layer"); axb.set_ylabel("silhouette", color=SEQc)
    axb2.set_ylabel("integration index", color=INK)
    axb.tick_params(axis="y", labelcolor=SEQc)
    axb.axvspan(25, 35, color="0.85", alpha=0.5, zorder=0)
    axb.set_xticks(range(0, 48, 4))
    l1, la1 = axb.get_legend_handles_labels()
    l2, la2 = axb2.get_legend_handles_labels()
    axb.legend(l1 + l2, la1 + la2, loc="lower left", frameon=False)
    axb2.spines["top"].set_visible(False)
    panel(axb, "b", dx=-0.06, dy=1.10)
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
    panel(axa, "a", dx=-0.16, dy=1.08)

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
    axb.legend(loc="upper left", frameon=False)
    panel(axb, "b", dx=-0.09, dy=1.08)
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
    axA.set_xticks(range(0, 48, 8)); axA.legend(frameon=False, loc="upper right")
    panel(axA, "a")

    pr = jload("scaled/per_residue_validation.json")
    xp = np.array(pr["layers"])
    axB.plot(xp, pr["mean_pool"]["integration"], "-o", color=INK, ms=3, lw=1.8,
             label="mean-pool")
    axB.plot(xp, pr["residue"]["integration"], "--s", color=SEQc, ms=3, lw=1.8,
             label="per-residue")
    axB.set_xlabel("layer"); axB.set_ylabel("integration index")
    axB.set_xticks(xp); axB.legend(frameon=False, loc="upper left")
    panel(axB, "b")

    dg = jload("scaled/diagnostics.json")["per_layer"]
    L = jload("scaled/diagnostics.json")["meta"]["layers"]
    ov = [dg[str(l)]["dimensionality"]["overall_effective_rank"] for l in L]
    wi = [dg[str(l)]["dimensionality"]["within_effective_rank_mean"] for l in L]
    axC.plot(L, ov, "-", color=INK, lw=1.8, label="whole cloud")
    axC.plot(L, wi, "--", color="#94a3b8", lw=1.6, label="per-condition")
    axC.set_xlabel("layer"); axC.set_ylabel("effective rank")
    axC.set_xticks(range(0, 48, 8))
    axC.text(0.5, 0.92, "never isotropic (peak ~85 of 1536)", transform=axC.transAxes,
             ha="center", fontsize=6.3, style="italic", color="#64748b")
    axC.legend(frameon=False, loc="center right")
    panel(axC, "c")
    fig.tight_layout()
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
    cor = {c["vs"]: c for c in b["correlations"]}
    axA.text(0.97, 0.04,
             "onset vs disorder (coil) r = +0.22\n"
             "onset vs length r = -0.04 (n.s.)\n"
             "onset vs pLDDT r = +0.00 (n.s.)",
             transform=axA.transAxes, ha="right", va="bottom", fontsize=6.2,
             color="#334155")
    panel(axA, "a", dx=-0.32)

    s = jload("diverse/stratified.json")
    x = np.array(s["layers"])
    kcol = {"all": INK, "eukaryota": "#648FFF", "bacteria": "#FE6100",
            "archaea": "#009E73"}
    for g in ["all", "eukaryota", "bacteria", "archaea"]:
        axB.plot(x, s["series"][g]["silhouette"], "-", color=kcol[g],
                 lw=2.2 if g == "all" else 1.4,
                 label=f"{g} (n={s['n_by_group'][g]})")
    axB.set_xlabel("layer"); axB.set_ylabel("silhouette")
    axB.set_xticks(range(0, 48, 8)); axB.legend(frameon=False, loc="lower left")
    axB.axvspan(34, 36, color="0.85", alpha=0.6, zorder=0)
    panel(axB, "b")
    fig.tight_layout()
    fig.savefig(OUT / "figure4.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote figure4.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figure1(); figure2(); figure3(); figure4()
