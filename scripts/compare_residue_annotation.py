"""The granularity control: does per-residue functional annotation fuse?

Compares the residue_annotation condition (per-residue functional info, from real
InterPro residue sites) against the existing conditions on the SAME proteins, via
linear CKA across all 48 layers.

The question this settles:
  · if residue_annotation FUSES (its CKA with the physical modalities / "all"
    rises with depth, like structure<->all) -> the whole-protein function track's
    orthogonality was a GRANULARITY effect.
  · if residue_annotation stays ORTHOGONAL (CKA ~0 throughout, like function<->all)
    -> functional information is kept in a separate subspace regardless of
    granularity (the strong, content-driven claim).

Outputs:
  results/scaled/residue_annotation_compare.json
  figures/scaled/metrics/residue_annotation_compare.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from compute_metrics import linear_cka  # noqa: E402
from src.viz import CONDITION_COLOR, INK  # noqa: E402

RA_DIR = ROOT / "activations" / "scaled" / "residue_annotation"
MP_DIR = ROOT / "activations" / "scaled" / "per_protein"
OUT_JSON = ROOT / "results" / "scaled" / "residue_annotation_compare.json"
OUT_FIG = ROOT / "figures" / "scaled" / "metrics" / "residue_annotation_compare.png"


def main() -> None:
    accs = sorted(p.stem for p in RA_DIR.glob("*.npz") if not p.stem.startswith("_"))
    print(f"{len(accs)} proteins with residue_annotation activations")

    head = np.load(MP_DIR / f"{accs[0]}.npz", allow_pickle=True)
    conditions = [str(c) for c in head["conditions"]]
    layers = [int(l) for l in head["layers"]]
    n_layers = len(layers)
    cidx = {c: i for i, c in enumerate(conditions)}

    # Load aligned matrices: residue_annotation + every existing condition.
    ra = np.stack([np.load(RA_DIR / f"{a}.npz")["mean_pool"] for a in accs])  # (N,48,1536)
    mp = np.stack([np.load(MP_DIR / f"{a}.npz", allow_pickle=True)["mean_pool"]
                   for a in accs])  # (N,6,48,1536)

    physical = ["sequence", "structure", "ss8", "sasa"]
    series = {f"residue_annotation->{c}": [] for c in conditions}
    series["residue_annotation->physical_mean"] = []
    # reference: the whole-protein function track vs the same targets
    series.update({f"function->{c}": [] for c in ["all"] + physical})
    series["function->physical_mean"] = []
    # and a fusion reference
    series["structure->all"] = []

    for ki in range(n_layers):
        ra_L = ra[:, ki, :].astype(np.float64)
        cond_L = {c: mp[:, cidx[c], ki, :].astype(np.float64) for c in conditions}
        # residue_annotation vs everything
        ra_phys = []
        for c in conditions:
            v = linear_cka(ra_L, cond_L[c])
            series[f"residue_annotation->{c}"].append(v)
            if c in physical:
                ra_phys.append(v)
        series["residue_annotation->physical_mean"].append(float(np.mean(ra_phys)))
        # function references
        fn_phys = []
        for c in ["all"] + physical:
            v = linear_cka(cond_L["function"], cond_L[c])
            series[f"function->{c}"].append(v)
            if c in physical:
                fn_phys.append(v)
        series["function->physical_mean"].append(float(np.mean(fn_phys)))
        series["structure->all"].append(linear_cka(cond_L["structure"], cond_L["all"]))

    out = {"n_proteins": len(accs), "layers": layers, "series": series}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")

    ra_all = series["residue_annotation->all"]
    fn_all = series["function->all"]
    print("\nCKA -> 'all' across depth (does residue_annotation fuse?):")
    for L in [0, 8, 16, 24, 32, 40, 47]:
        i = layers.index(L)
        print(f"  L{L:2d}  residue_annotation={ra_all[i]:.3f}  function={fn_all[i]:.3f}  "
              f"(structure={series['structure->all'][i]:.3f})")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.array(layers)
    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=130, facecolor="white")
    ax.plot(x, series["structure->all"], "-", color="#FE6100", lw=2.0, alpha=0.8,
            label="structure ↔ all  (fusion reference)")
    ax.plot(x, series["residue_annotation->physical_mean"], "-o", color=INK, lw=2.6,
            markersize=4, label="residue-annotation ↔ physical (mean)")
    ax.plot(x, series["function->physical_mean"], "--s", color="#009E73", lw=2.2,
            markersize=4, label="function ↔ physical (mean)  [orthogonal ref]")
    ax.axhspan(-0.02, 0.1, color="#94a3b8", alpha=0.08)
    ax.set_xlabel("Layer", color=INK)
    ax.set_ylabel("linear CKA", color=INK)
    ax.set_ylim(-0.03, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_title("Does per-residue functional annotation fuse?\n"
                 f"residue-annotation vs the whole-protein function track "
                 f"(n={len(accs)})", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
