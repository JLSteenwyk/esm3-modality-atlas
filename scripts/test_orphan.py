"""Test the orphan hypothesis: does function fuse more when it is non-redundant?

H1 holds that function stays orthogonal because it is redundant with the physical
representation (function is derivable from sequence/structure). The prediction is
that the function channel should align MORE with the physical subspace for proteins
whose function is NOT derivable from structure (orphans / novel folds).

Per protein, function-physical alignment is the cosine between its centred
function vector and its centred mean-physical vector, averaged over deep layers.
Redundancy is measured three ways:
  primary   - whether EC class is correctly decoded from the structure
              representation (correct = redundant, wrong = non-redundant);
  secondary - InterPro-domain count and GO-term count (more = better
              characterised, with the caveat that they also set function input).

H1 predicts higher alignment for the non-redundant (EC-wrong, low-count) proteins.

Outputs: results/scaled/orphan_test.json, figures/scaled/metrics/orphan_test.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.viz import CONDITIONS, INK  # noqa: E402

MP_DIR = ROOT / "activations" / "scaled" / "per_protein"
META = ROOT / "data" / "pilot" / "annotations" / "metadata.json"
OUT_JSON = ROOT / "results" / "scaled" / "orphan_test.json"
OUT_FIG = ROOT / "figures" / "scaled" / "metrics" / "orphan_test.png"

DEEP = [32, 40, 47]
PHYS = ["sequence", "structure", "ss8", "sasa"]


def main() -> None:
    accs = sorted(p.stem for p in MP_DIR.glob("*.npz"))
    meta = json.loads(META.read_text())
    head = np.load(MP_DIR / f"{accs[0]}.npz", allow_pickle=True)
    conds = [str(c) for c in head["conditions"]]
    ci = {c: i for i, c in enumerate(conds)}
    layers = [int(l) for l in head["layers"]]
    A = np.stack([np.load(MP_DIR / f"{a}.npz", allow_pickle=True)["mean_pool"]
                  for a in accs]).astype(np.float64)          # (N,6,48,1536)
    A = A - A.mean(axis=0, keepdims=True)                     # centre per cond/layer

    # per-protein function-physical alignment over deep layers
    align = np.zeros(len(accs))
    for L in DEEP:
        li = layers.index(L)
        f = A[:, ci["function"], li, :]
        p = A[:, [ci[c] for c in PHYS], li, :].mean(axis=1)
        fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        pn = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
        align += (fn * pn).sum(1)
    align /= len(DEEP)

    # redundancy proxies
    n_ipr = np.array([len(meta[a].get("interpro") or []) for a in accs])
    n_go = np.array([len(meta[a].get("go_terms") or []) for a in accs])

    # primary: can structure decode EC? correct = redundant, wrong = non-redundant
    ec = np.array([int(e[0].split(".")[0]) if (e := meta[a].get("ec_numbers"))
                   and e[0].split(".")[0].isdigit() else 0 for a in accs])
    keep = np.isin(ec, [1, 2, 3])
    li40 = layers.index(40)
    Xs = A[keep, ci["structure"], li40, :]
    y = ec[keep].astype(str)
    pipe = make_pipeline(StandardScaler(), PCA(50, random_state=0),
                         LogisticRegression(max_iter=2000))
    pred = cross_val_predict(pipe, Xs, y, cv=StratifiedKFold(5, shuffle=True,
                             random_state=0))
    correct = pred == y
    al_keep = align[keep]
    a_red = al_keep[correct]        # function redundant (structure predicts EC)
    a_non = al_keep[~correct]       # function non-redundant (structure fails)
    U, p_mw = mannwhitneyu(a_non, a_red, alternative="greater")  # H1: non > red

    r_ipr, p_ipr = spearmanr(align, n_ipr)
    r_go, p_go = spearmanr(align, n_go)

    out = {
        "n": len(accs), "deep_layers": DEEP,
        "alignment_mean": float(align.mean()), "alignment_sd": float(align.std()),
        "primary_structure_decodes_EC": {
            "n_ec": int(keep.sum()),
            "n_redundant_correct": int(correct.sum()),
            "n_nonredundant_wrong": int((~correct).sum()),
            "align_redundant_mean": float(a_red.mean()),
            "align_nonredundant_mean": float(a_non.mean()),
            "mannwhitney_p_nonred_gt_red": float(p_mw),
        },
        "secondary": {
            "spearman_align_vs_interpro_count": [float(r_ipr), float(p_ipr)],
            "spearman_align_vs_go_count": [float(r_go), float(p_go)],
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.0, 3.3), dpi=300)
    axA.boxplot([a_red, a_non], labels=[f"redundant\n(structure decodes EC)\nn={len(a_red)}",
                f"non-redundant\n(structure fails)\nn={len(a_non)}"],
                widths=0.5, showfliers=False)
    axA.set_ylabel("function-physical alignment (deep layers)")
    axA.set_title(f"primary test (Mann-Whitney p={p_mw:.2f})", fontsize=9)
    axA.text(-0.16, 1.05, "a", transform=axA.transAxes, fontsize=12, fontweight="bold")

    axB.scatter(n_ipr, align, s=6, alpha=0.3, color=INK, edgecolors="none")
    axB.set_xlabel("InterPro domain count"); axB.set_ylabel("function-physical alignment")
    axB.set_title(f"vs characterisation (r={r_ipr:+.2f})", fontsize=9)
    axB.text(-0.16, 1.05, "b", transform=axB.transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig); print(f"wrote {OUT_FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
