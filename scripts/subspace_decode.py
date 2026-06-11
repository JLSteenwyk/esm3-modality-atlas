"""Information versus geometry: what each modality subspace encodes.

The modality-integration result is geometric, that the function condition is
orthogonal to the physical modalities by CKA. This probe asks the complementary
informational question, namely what biology is linearly decodable from each modality
condition at each layer. For every condition a probe decodes a structural target
(fold class from SS8) and a functional target (EC enzyme class) from the mean-pooled
representation.

Geometric orthogonality does not mean functional content is absent. The physical
conditions decode both fold class and enzyme class, and enzyme class becomes more
decodable from the physical subspace with depth. The explicit function channel, by
contrast, decodes fold class and enzyme class only weakly. The network therefore
absorbs functional content into the physical subspace while holding the discrete
function channel on a separate, informationally thin axis. There is no division of
representational labour, which makes the persistence of geometric orthogonality a
representational choice rather than an information-theoretic necessity.

Outputs:
  results/scaled/subspace_decode.json
  figures/scaled/metrics/subspace_decode.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.viz import CONDITION_COLOR, CONDITION_LABEL, CONDITIONS, INK  # noqa: E402

MP_DIR = ROOT / "activations" / "scaled" / "per_protein"
META = ROOT / "data" / "pilot" / "annotations" / "metadata.json"
ANNO = ROOT / "data" / "scaled" / "annotations" / "structure_annotations.json"
OUT_JSON = ROOT / "results" / "scaled" / "subspace_decode.json"
OUT_FIG = ROOT / "figures" / "scaled" / "metrics" / "subspace_decode.png"

LAYERS = list(range(0, 48, 3)) + [47]
MATRIX_LAYER = 39          # representative post-fusion layer for the bar panel (in LAYERS)
STRUC_c, FUNC_c = "#FE6100", "#009E73"   # structure target / function target


def fold_class(ss: str) -> str:
    n = max(1, len(ss))
    h = sum(ss.count(c) for c in "HGI") / n
    s = sum(ss.count(c) for c in "EB") / n
    if h > 0.3 and h >= s:
        return "alpha"
    if s > 0.2 and s > h:
        return "beta"
    return "mixed"


def probe(X, y):
    pipe = make_pipeline(StandardScaler(), PCA(n_components=50, random_state=0),
                         LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    s = cross_val_score(pipe, X, y, cv=cv,
                        scoring=make_scorer(balanced_accuracy_score))
    return float(np.mean(s))


def main() -> None:
    accs = sorted(p.stem for p in MP_DIR.glob("*.npz"))
    meta = json.loads(META.read_text())
    anno = json.loads(ANNO.read_text())
    head = np.load(MP_DIR / f"{accs[0]}.npz", allow_pickle=True)
    conds = [str(c) for c in head["conditions"]]
    cidx = {c: i for i, c in enumerate(conds)}
    layers_all = [int(l) for l in head["layers"]]

    mp = np.stack([np.load(MP_DIR / f"{a}.npz", allow_pickle=True)["mean_pool"]
                   for a in accs]).astype(np.float64)   # (N, C, 48, 1536)

    # labels
    fold = np.array([fold_class(anno[a]["ss8"]) for a in accs])
    ec_raw = []
    for a in accs:
        e = meta[a].get("ec_numbers") or []
        d = e[0].split(".")[0] if e else ""
        ec_raw.append(int(d) if d.isdigit() else 0)
    ec_raw = np.array(ec_raw)
    ec_keep = np.isin(ec_raw, [1, 2, 3])     # three major enzyme classes
    print(f"fold n={len(fold)} dist={dict(zip(*np.unique(fold, return_counts=True)))}")
    print(f"EC  n={ec_keep.sum()} dist={dict(zip(*np.unique(ec_raw[ec_keep], return_counts=True)))}")

    targets = {"fold": (fold, np.ones(len(accs), bool)),
               "ec": (ec_raw.astype(str), ec_keep)}
    results = {t: {c: [] for c in CONDITIONS} for t in targets}

    print(f"{'layer':>5}  " + "  ".join(f"{c[:4]}:{t}" for c in CONDITIONS
                                        for t in ("fold", "ec")))
    for L in LAYERS:
        li = layers_all.index(L)
        row = []
        for c in CONDITIONS:
            for t, (y, mask) in targets.items():
                X = mp[mask, cidx[c], li, :]
                acc = probe(X, y[mask])
                results[t][c].append(acc)
                row.append(f"{acc:.2f}")
        print(f"{L:>5}  " + "  ".join(row))

    out = {"layers": LAYERS, "matrix_layer": MATRIX_LAYER,
           "chance": 1 / 3, "results": results}
    _render(out)


def _render(out):
    results, LAYERS = out["results"], out["layers"]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    mi = LAYERS.index(MATRIX_LAYER)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.2, 3.4), dpi=300)

    # A: division of labour at the fused layer
    x = np.arange(len(CONDITIONS)); w = 0.38
    fold_acc = [results["fold"][c][mi] for c in CONDITIONS]
    ec_acc = [results["ec"][c][mi] for c in CONDITIONS]
    axA.bar(x - w / 2, fold_acc, w, color=STRUC_c, label="fold class (structure)")
    axA.bar(x + w / 2, ec_acc, w, color=FUNC_c, label="EC class (function)")
    axA.axhline(1 / 3, ls=":", color="0.5", lw=1)
    axA.set_xticks(x)
    axA.set_xticklabels([CONDITION_LABEL[c].replace("All modalities", "All")
                         for c in CONDITIONS], rotation=35, ha="right", fontsize=7)
    axA.set_ylabel("balanced accuracy")
    axA.set_title(f"What each channel encodes (layer {MATRIX_LAYER})", fontsize=9)
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False)
    axA.text(-0.13, 1.06, "a", transform=axA.transAxes, fontsize=12, fontweight="bold")

    # B: where each target is decodable across depth (solid = from structure,
    # dashed = from the function channel)
    xa = np.array(LAYERS)
    axB.plot(xa, results["ec"]["structure"], "-", color=FUNC_c, lw=2,
             label="enzyme class from structure")
    axB.plot(xa, results["ec"]["function"], "--", color=FUNC_c, lw=1.6,
             label="enzyme class from function channel")
    axB.plot(xa, results["fold"]["structure"], "-", color=STRUC_c, lw=2,
             label="fold class from structure")
    axB.plot(xa, results["fold"]["function"], "--", color=STRUC_c, lw=1.6,
             label="fold class from function channel")
    axB.axhline(1 / 3, ls=":", color="0.5", lw=1)
    axB.set_xlabel("layer"); axB.set_ylabel("balanced accuracy")
    axB.set_title("Function content lives in the physical subspace", fontsize=9)
    axB.set_xticks(range(0, 48, 8))
    axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False)
    axB.text(-0.13, 1.06, "b", transform=axB.transAxes, fontsize=12, fontweight="bold")

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
