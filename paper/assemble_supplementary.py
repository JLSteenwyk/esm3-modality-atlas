"""Assemble the supplementary figure suite (figureS1..S6).

Re-plots robustness and breakdown analyses that support, but do not lead, the main
figures. Output: paper/figures/figureS{1..6}.png at 300 dpi; captions in
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from compute_metrics import condition_matrices, linear_cka  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from src.viz import CONDITION_COLOR, CONDITION_LABEL, CONDITIONS, INK  # noqa: E402

OUT = ROOT / "paper" / "figures"
R = ROOT / "results"
BYLAYER = ROOT / "activations" / "scaled" / "by_layer"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 9, "legend.fontsize": 7, "figure.dpi": 300,
})
SEQc = CONDITION_COLOR["sequence"]


def jload(p):
    return json.loads((R / p).read_text())


def panel(ax, letter, dx=-0.12, dy=1.07):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="right", color=INK)


# S1 — sample-size convergence -------------------------------------------------
def figS1():
    rng = np.random.default_rng(0)
    Ns = [100, 200, 400, 600, 892]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, L, title in [(axes[0], 24, "layer 24 (separation peak)"),
                         (axes[1], 40, "layer 40 (fusion zone)")]:
        d = np.load(BYLAYER / f"layer_{L:02d}.npz", allow_pickle=True)
        coords = d["coords"].astype(np.float64)
        cond = d["condition"].astype(str); pid = d["protein_id"].astype(str)
        nprot = len(set(pid))
        present, mats = condition_matrices(coords, cond, pid)
        sil_m, sil_s, ig_m, ig_s = [], [], [], []
        for N in Ns:
            sres, ires = [], []
            reps = 12 if N < nprot else 1
            for _ in range(reps):
                idx = rng.choice(nprot, min(N, nprot), replace=False)
                sub = {c: mats[c][idx] for c in present}
                vals = [linear_cka(sub[present[i]], sub[present[j]])
                        for i in range(len(present)) for j in range(i + 1, len(present))]
                ires.append(np.mean(vals))
                keep = np.isin(pid, np.array(sorted(set(pid)))[idx])
                sres.append(silhouette_score(coords[keep], cond[keep]))
            sil_m.append(np.mean(sres)); sil_s.append(np.std(sres))
            ig_m.append(np.mean(ires)); ig_s.append(np.std(ires))
        ax.errorbar(Ns, sil_m, yerr=sil_s, fmt="-o", color=SEQc, ms=3, capsize=2,
                    label="silhouette")
        ax.errorbar(Ns, ig_m, yerr=ig_s, fmt="-s", color=INK, ms=3, capsize=2,
                    label="integration index")
        ax.set_xlabel("number of proteins"); ax.set_title(title)
        ax.legend(frameon=False)
    axes[0].set_ylabel("metric value")
    fig.tight_layout(); fig.savefig(OUT / "figureS1.png", bbox_inches="tight")
    plt.close(fig); print("wrote figureS1.png")


# S2 — significance: bootstrap CIs vs permutation nulls ------------------------
def figS2():
    dg = jload("scaled/diagnostics.json")["per_layer"]
    Ls = [int(k) for k, v in dg.items() if v.get("significance")]
    Ls.sort(); x = np.array(Ls)
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for met, col, lab in [("silhouette", SEQc, "silhouette"),
                          ("integration_index", INK, "integration index")]:
        obs = [dg[str(l)]["significance"][met]["observed"] for l in Ls]
        lo = [dg[str(l)]["significance"][met]["ci95"][0] for l in Ls]
        hi = [dg[str(l)]["significance"][met]["ci95"][1] for l in Ls]
        nm = [dg[str(l)]["significance"][met]["null_mean"] for l in Ls]
        ax.fill_between(x, lo, hi, color=col, alpha=0.18)
        ax.plot(x, obs, "-o", color=col, ms=3, lw=1.6, label=f"{lab} (95% CI)")
        ax.plot(x, nm, ":", color=col, lw=1.3, label=f"{lab} permutation null")
    ax.set_xlabel("layer"); ax.set_ylabel("metric value"); ax.set_ylim(-0.05, 1.0)
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig(OUT / "figureS2.png", bbox_inches="tight")
    plt.close(fig); print("wrote figureS2.png")


# S3 — modality-identity probe ------------------------------------------------
def figS3():
    dg = jload("scaled/diagnostics.json")
    L = dg["meta"]["layers"]; per = dg["per_layer"]; x = np.array(L)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    full = [per[str(l)]["probe"]["fulldim"]["accuracy_mean"] for l in L]
    p3 = [per[str(l)]["probe"]["pca3d"]["accuracy_mean"] for l in L]
    ax.plot(x, full, "-", color="#64748b", lw=1.8, label="full 1536-d")
    ax.plot(x, p3, "-o", color=INK, ms=2.5, lw=2.0, label="top-3 PCA")
    for c in CONDITIONS:
        r = [per[str(l)]["probe"]["pca3d"]["recall_by_condition"].get(c) for l in L]
        if all(v is not None for v in r):
            ax.plot(x, r, "-", color=CONDITION_COLOR[c], lw=1.0, alpha=0.8,
                    label=f"recall: {CONDITION_LABEL[c]}")
    chance = per[str(L[0])]["probe"]["pca3d"]["chance"]
    ax.axhline(chance, ls=":", color="#94a3b8", lw=1.2)
    ax.text(L[-1], chance + 0.02, "chance", ha="right", fontsize=6.5, color="#94a3b8")
    ax.set_ylim(0, 1.03); ax.set_xlabel("layer"); ax.set_ylabel("decodability")
    ax.set_xticks(range(0, 48, 8)); ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "figureS3.png", bbox_inches="tight")
    plt.close(fig); print("wrote figureS3.png")


# S4 — per-organism universality ----------------------------------------------
def figS4():
    s = jload("diverse/stratified.json"); L = np.array(s["layers"])
    cur = json.loads((ROOT / "data/diverse/curation_summary.json").read_text())
    o2k = {o: v["superkingdom"] for o, v in cur["per_organism"].items()}
    kcol = {"eukaryota": "#648FFF", "bacteria": "#FE6100", "archaea": "#009E73"}
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    seen = set()
    for g in sorted(k for k in s["series"] if k.startswith("org:")):
        org = g[4:]; king = o2k.get(org, "eukaryota")
        lab = king if king not in seen else None
        seen.add(king)
        ax.plot(L, s["series"][g]["silhouette"], "-", color=kcol[king], lw=0.9,
                alpha=0.7, label=lab)
    ax.plot(L, s["series"]["all"]["silhouette"], "-", color=INK, lw=2.4,
            label="all (5,555)")
    ax.axvline(35, ls=":", color="0.5", lw=1)
    ax.set_xlabel("layer"); ax.set_ylabel("silhouette"); ax.set_xticks(range(0, 48, 8))
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "figureS4.png", bbox_inches="tight")
    plt.close(fig); print("wrote figureS4.png")


# S5 — per-layer condition x condition CKA ------------------------------------
def figS5():
    d = jload("scaled/metrics.json")
    conds = [c for c in CONDITIONS if c in d["meta"]["conditions"]]
    short = [CONDITION_LABEL[c].replace("All modalities", "All") for c in conds]
    snaps = [0, 8, 16, 24, 32, 40, 47]
    fig, axes = plt.subplots(1, len(snaps), figsize=(7.2, 1.7))
    im = None
    for ax, L in zip(axes, snaps):
        m = np.array(d["per_layer"][str(L)]["cka_matrix"])
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"L{L}", fontsize=8)
        ax.set_xticks(range(len(conds))); ax.set_yticks(range(len(conds)))
        ax.set_xticklabels(short, fontsize=4.5, rotation=90)
        ax.set_yticklabels(short if ax is axes[0] else [], fontsize=4.5)
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, fraction=0.01, pad=0.01).set_label("CKA", fontsize=7)
    fig.savefig(OUT / "figureS5.png", bbox_inches="tight")
    plt.close(fig); print("wrote figureS5.png")


# S6 — diverse dataset composition --------------------------------------------
def figS6():
    meta = json.loads((ROOT / "data/diverse/metadata.json").read_text())
    kcol = {"eukaryota": "#648FFF", "bacteria": "#FE6100", "archaea": "#009E73"}
    from collections import Counter, defaultdict
    cnt = Counter(v["organism"] for v in meta.values())
    o2k = {v["organism"]: v["superkingdom"] for v in meta.values()}
    lens = defaultdict(list)
    for v in meta.values():
        lens[v["superkingdom"]].append(v["length"])
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    orgs = sorted(cnt, key=lambda o: (o2k[o], o))
    axA.barh(range(len(orgs)), [cnt[o] for o in orgs],
             color=[kcol[o2k[o]] for o in orgs], edgecolor="none")
    axA.set_yticks(range(len(orgs))); axA.set_yticklabels(orgs, fontsize=7)
    axA.set_xlabel("proteins curated"); panel(axA, "a", dx=-0.34)
    for ki, king in enumerate(["eukaryota", "bacteria", "archaea"]):
        axB.hist(lens[king], bins=30, color=kcol[king], alpha=0.55,
                 label=f"{king} (n={len(lens[king])})")
    axB.set_xlabel("protein length (residues)"); axB.set_ylabel("count")
    axB.legend(frameon=False); panel(axB, "b")
    fig.tight_layout(); fig.savefig(OUT / "figureS6.png", bbox_inches="tight")
    plt.close(fig); print("wrote figureS6.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figS1(); figS2(); figS3(); figS4(); figS5(); figS6()
