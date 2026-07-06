"""Assemble the supplementary figure suite (figureS1..S11).

Re-plots robustness and breakdown analyses that support, but do not lead, the main
figures. Output: paper/figures/figureS{1..11}.png at 300 dpi.
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


def panel(ax, letter, dx=-24, dy=6):
    """Panel letter anchored to the axes' top-left corner with a fixed point
    offset, so it sits in a consistent outer corner regardless of titles, axis
    labels, or layout (dx, dy are offsets in points)."""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points", fontsize=12,
                fontweight="bold", va="bottom", ha="right", color=INK,
                annotation_clip=False)


def below(ax, ncol=2, y=-0.30, fontsize=7):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
              frameon=False, fontsize=fontsize, handlelength=1.6, columnspacing=1.2)


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
        below(ax, ncol=2)
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
    below(ax, ncol=2, fontsize=6.5)
    fig.tight_layout(rect=(0,0.04,1,1)); fig.savefig(OUT / "figureS2.png", bbox_inches="tight")
    plt.close(fig); print("wrote figureS2.png")


# S4 — modality-identity probe ------------------------------------------------
def figS4():
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
    ax.set_xticks(range(0, 48, 8)); below(ax, ncol=4, fontsize=6)
    fig.tight_layout(rect=(0,0.05,1,1)); fig.savefig(OUT / "figureS4.png", bbox_inches="tight")
    plt.close(fig); print("wrote figureS4.png")


# S6 — per-organism universality ----------------------------------------------
def figS6():
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
    below(ax, ncol=4, fontsize=7)
    fig.tight_layout(rect=(0,0.05,1,1)); fig.savefig(OUT / "figureS6.png", bbox_inches="tight")
    plt.close(fig); print("wrote figureS6.png")


# S3 — per-layer condition x condition CKA ------------------------------------
def figS3():
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
    fig.savefig(OUT / "figureS3.png", bbox_inches="tight")
    plt.close(fig); print("wrote figureS3.png")


# S5 — diverse dataset composition --------------------------------------------
def figS5():
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
    axA.set_xlabel("proteins curated"); panel(axA, "a")
    for ki, king in enumerate(["eukaryota", "bacteria", "archaea"]):
        axB.hist(lens[king], bins=30, color=kcol[king], alpha=0.55,
                 label=f"{king} (n={len(lens[king])})")
    axB.set_xlabel("protein length (residues)"); axB.set_ylabel("count")
    below(axB, ncol=3); panel(axB, "b")
    fig.tight_layout(); fig.savefig(OUT / "figureS5.png", bbox_inches="tight")
    plt.close(fig); print("wrote figureS5.png")


# S7 — the representation is organism-agnostic --------------------------------
def figS7():
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    BY = ROOT / "activations" / "diverse" / "by_layer"
    cur = json.loads((ROOT / "data/diverse/curation_summary.json").read_text())
    o2k = {o: v["superkingdom"] for o, v in cur["per_organism"].items()}
    kcol = {"eukaryota": "#648FFF", "bacteria": "#FE6100", "archaea": "#009E73"}
    layers = sorted(int(p.stem.split("_")[-1]) for p in BY.glob("layer_*.npz"))

    cache = R / "diverse" / "superkingdom_sil.json"
    if cache.exists():
        sk = json.loads(cache.read_text())
    else:
        sk = {"layers": layers, "silhouette": []}
        for L in layers:
            d = np.load(BY / f"layer_{L:02d}.npz", allow_pickle=True)
            m = d["condition"].astype(str) == "all"
            king = np.array([o2k.get(o, "?") for o in d["category"][m].astype(str)])
            sk["silhouette"].append(float(silhouette_score(d["coords"][m].astype(np.float64), king)))
        cache.write_text(json.dumps(sk))

    mod = jload("diverse/metrics.json")["series"]
    fig = plt.figure(figsize=(8.4, 3.5))
    axA = fig.add_subplot(1, 2, 1, projection="3d")
    d = np.load(BY / "layer_40.npz", allow_pickle=True)
    m = d["condition"].astype(str) == "all"
    Z = PCA(3).fit_transform(d["coords"][m].astype(np.float64))
    king = np.array([o2k.get(o, "?") for o in d["category"][m].astype(str)])
    for k in ["eukaryota", "bacteria", "archaea"]:
        mm = king == k
        axA.scatter(Z[mm, 0], Z[mm, 1], Z[mm, 2], s=2, c=kcol[k], alpha=0.35,
                    edgecolors="none", depthshade=False, label=k)
    axA.set_xticks([]); axA.set_yticks([]); axA.set_zticks([])
    axA.set_title("layer-40 representation by superkingdom", fontsize=8.5)
    axA.view_init(18, -60)
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
               frameon=False, fontsize=7)
    fig.text(0.07, 0.95, "a", fontsize=12, fontweight="bold", color=INK)

    axB = fig.add_subplot(1, 2, 2)
    axB.plot(mod["layers"], mod["silhouette"], "-", color=INK, lw=2,
             label="modality (condition)")
    axB.plot(sk["layers"], sk["silhouette"], "-", color="#94a3b8", lw=2,
             label="superkingdom")
    axB.axhline(0, ls=":", color="0.6", lw=1)
    axB.set_xlabel("layer"); axB.set_ylabel("silhouette")
    axB.set_xticks(range(0, 48, 8))
    below(axB, ncol=2)
    panel(axB, "b")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "figureS7.png", bbox_inches="tight")
    plt.close(fig); print("wrote figureS7.png")


def figS10():
    """Orphan test: function stays orthogonal whether or not it is redundant."""
    from scipy.stats import mannwhitneyu, spearmanr
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    MP = ROOT / "activations" / "scaled" / "per_protein"
    meta = json.loads((ROOT / "data/pilot/annotations/metadata.json").read_text())
    accs = sorted(p.stem for p in MP.glob("*.npz"))
    head = np.load(MP / f"{accs[0]}.npz", allow_pickle=True)
    ci = {str(c): i for i, c in enumerate(head["conditions"])}
    layers = [int(l) for l in head["layers"]]
    A = np.stack([np.load(MP / f"{a}.npz", allow_pickle=True)["mean_pool"]
                  for a in accs]).astype(np.float64)
    A = A - A.mean(axis=0, keepdims=True)

    phys = ["sequence", "structure", "ss8", "sasa"]
    align = np.zeros(len(accs))
    for L in (32, 40, 47):
        li = layers.index(L)
        f = A[:, ci["function"], li, :]
        p = A[:, [ci[c] for c in phys], li, :].mean(axis=1)
        fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        pn = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
        align += (fn * pn).sum(1)
    align /= 3.0

    n_ipr = np.array([len(meta[a].get("interpro") or []) for a in accs])
    ec = np.array([int(e[0].split(".")[0]) if (e := meta[a].get("ec_numbers"))
                   and e[0].split(".")[0].isdigit() else 0 for a in accs])
    keep = np.isin(ec, [1, 2, 3])
    Xs = A[keep, ci["structure"], layers.index(40), :]
    y = ec[keep].astype(str)
    pipe = make_pipeline(StandardScaler(), PCA(50, random_state=0),
                         LogisticRegression(max_iter=2000))
    pred = cross_val_predict(pipe, Xs, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=0))
    correct = pred == y
    a_red, a_non = align[keep][correct], align[keep][~correct]
    _, p_mw = mannwhitneyu(a_non, a_red, alternative="greater")
    r_ipr, _ = spearmanr(align, n_ipr)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.0, 3.3))
    bx = axA.boxplot([a_red, a_non], widths=0.5, showfliers=False,
                     patch_artist=True,
                     labels=[f"redundant\n(structure decodes EC)\nn={len(a_red)}",
                             f"non-redundant\n(structure fails)\nn={len(a_non)}"])
    for b in bx["boxes"]:
        b.set(facecolor="#e2e8f0", edgecolor=INK)
    for med in bx["medians"]:
        med.set(color=INK)
    axA.axhline(0, ls=":", color="0.6", lw=1)
    axA.set_ylabel("function-physical alignment (deep layers)")
    axA.set_title(f"alignment by redundancy ($p$={p_mw:.2f})")

    axB.scatter(n_ipr, align, s=6, alpha=0.3, color=INK, edgecolors="none")
    axB.axhline(0, ls=":", color="0.6", lw=1)
    axB.set_xlabel("InterPro domain count")
    axB.set_ylabel("function-physical alignment")
    axB.set_title(f"alignment by characterisation ($r$={r_ipr:+.2f})")
    panel(axA, "a"); panel(axB, "b")
    fig.tight_layout()
    fig.savefig(OUT / "figureS10.png", bbox_inches="tight")
    plt.close(fig); print("wrote figureS10.png")


def figS8():
    """Information vs geometry: function content lives in the physical subspace."""
    d = jload("scaled/subspace_decode.json")
    res, L = d["results"], d["layers"]
    ml = d["matrix_layer"]; mi = L.index(ml)
    struc_c, func_c = CONDITION_COLOR["structure"], CONDITION_COLOR["function"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.2, 3.5))

    x = np.arange(len(CONDITIONS)); w = 0.38
    axA.bar(x - w / 2, [res["fold"][c][mi] for c in CONDITIONS], w,
            color=struc_c, label="fold class (structure target)")
    axA.bar(x + w / 2, [res["ec"][c][mi] for c in CONDITIONS], w,
            color=func_c, label="enzyme class (function target)")
    axA.axhline(d["chance"], ls=":", color="0.5", lw=1)
    axA.set_xticks(x)
    axA.set_xticklabels([CONDITION_LABEL[c].replace("All modalities", "All")
                         for c in CONDITIONS], rotation=35, ha="right", fontsize=7)
    axA.set_ylabel("balanced accuracy")
    axA.set_title(f"what each channel encodes (layer {ml})")
    below(axA, ncol=1, y=-0.42)
    panel(axA, "a")

    xa = np.array(L)
    axB.plot(xa, res["ec"]["structure"], "-", color=func_c, lw=2,
             label="enzyme class from structure")
    axB.plot(xa, res["ec"]["function"], "--", color=func_c, lw=1.6,
             label="enzyme class from function channel")
    axB.plot(xa, res["fold"]["structure"], "-", color=struc_c, lw=2,
             label="fold class from structure")
    axB.plot(xa, res["fold"]["function"], "--", color=struc_c, lw=1.6,
             label="fold class from function channel")
    axB.axhline(d["chance"], ls=":", color="0.5", lw=1)
    axB.set_xlabel("layer"); axB.set_ylabel("balanced accuracy")
    axB.set_xticks(range(0, 48, 8))
    axB.set_title("function content lives in the physical subspace")
    below(axB, ncol=1, y=-0.42)
    panel(axB, "b")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "figureS8.png", bbox_inches="tight")
    plt.close(fig); print("wrote figureS8.png")


def figS9():
    """Causal ablation: withholding function barely moves the fused representation."""
    d = jload("scaled/ablation.json")
    layers, series, box = d["layers"], d["series"], d["box_fusion"]
    fuse = d["fusion_layer"]
    mods = ["sequence", "structure", "ss8", "sasa", "function"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for m in mods:
        axA.plot(layers, series[m]["magnitude"], "-", color=CONDITION_COLOR[m],
                 lw=2.4 if m == "function" else 1.6, label=CONDITION_LABEL[m])
    axA.set_xlabel("layer")
    axA.set_ylabel(r"relative displacement  $||d_X||/||h_\mathrm{all}||$")
    axA.set_title("withholding each modality, across depth")
    axA.set_xticks(range(0, 48, 8))
    below(axA, ncol=3, y=-0.30)
    panel(axA, "a")

    order = ["structure", "sequence", "sasa", "ss8", "function"]
    bp = axB.boxplot([box[m] for m in order], widths=0.6, showfliers=False,
                     patch_artist=True, labels=[CONDITION_LABEL[m] for m in order])
    for patch, m in zip(bp["boxes"], order):
        patch.set(facecolor=CONDITION_COLOR[m], alpha=0.55, edgecolor=INK)
    for med in bp["medians"]:
        med.set(color=INK)
    axB.set_ylabel(f"relative displacement at layer {fuse}")
    axB.set_title("per-protein footprint at the fusion layer")
    axB.tick_params(axis="x", labelrotation=30, labelsize=7)
    panel(axB, "b")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "figureS9.png", bbox_inches="tight")
    plt.close(fig); print("wrote figureS9.png")


def figS11():
    """Fusion replicates on experimental structures (RCSB) vs AlphaFold."""
    exp = jload("experimental/metrics.json")["series"]
    af = jload("scaled/metrics.json")["series"]
    AFc, EXc = "#94a3b8", INK
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.2, 3.4))
    axA.plot(af["layers"], af["silhouette"], "-", color=AFc, lw=2,
             label="AlphaFold (n=892)")
    axA.plot(exp["layers"], exp["silhouette"], "-", color=EXc, lw=2,
             label="experimental (n=177)")
    axA.axvline(35, ls=":", color="0.6", lw=1)
    axA.set_xlabel("layer"); axA.set_ylabel("condition separation (silhouette)")
    axA.set_xticks(range(0, 48, 8)); below(axA, ncol=2); panel(axA, "a")
    axB.plot(af["layers"], af["integration_index"], "-", color=AFc, lw=2,
             label="AlphaFold (n=892)")
    axB.plot(exp["layers"], exp["integration_index"], "-", color=EXc, lw=2,
             label="experimental (n=177)")
    axB.set_xlabel("layer"); axB.set_ylabel("integration index (mean pairwise CKA)")
    axB.set_xticks(range(0, 48, 8)); below(axB, ncol=2); panel(axB, "b")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "figureS11.png", bbox_inches="tight")
    plt.close(fig); print("wrote figureS11.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figS1(); figS2(); figS3(); figS4(); figS5(); figS6(); figS7(); figS8(); figS9()
    figS10(); figS11()
