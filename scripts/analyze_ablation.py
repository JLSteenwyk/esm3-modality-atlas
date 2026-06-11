"""Quantify each modality's contribution to the fused representation.

For every protein and layer, the contribution of modality X is the displacement
    d_X = h_all - h_(all without X)
between the full "all" representation and the representation with X withheld. Three
properties of d_X are reported across depth, averaged over proteins:

  magnitude     ||d_X|| / ||h_all||              how far withholding X moves the point
  off-manifold  fraction of ||d_X|| outside the   whether the move leaves the physical
                top-k subspace of protein         manifold (orthogonal) or travels
                variation in h_all                along it (reshapes physical identity)
  common-mode   ||mean_p d_X|| / mean_p ||d_X||   whether withholding X is a shared
                                                  translation (an addressable tag) or
                                                  a protein-specific rearrangement

Result: magnitude is the discriminating signal. Withholding function moves the fused
representation about five times less than withholding structure, sequence, or SASA,
so the physical representation is largely insensitive to the function input. The
off-manifold and common-mode terms are recorded for completeness but do not separate
the modalities cleanly (most displacement sits off the low-dimensional protein-
variation subspace for every modality), so they are not over-interpreted. A small
magnitude on its own is ambiguous, because ss8 is also small but for redundancy with
structure rather than orthogonality; the geometric CKA result and the decoding result
attribute function's small footprint to its orthogonal organisation.

Outputs: results/scaled/ablation.json, figures/scaled/metrics/ablation.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.viz import CONDITION_COLOR, CONDITION_LABEL, INK  # noqa: E402

ABL = ROOT / "activations" / "ablation" / "per_protein"
SCALED = ROOT / "activations" / "scaled" / "per_protein"
OUT_JSON = ROOT / "results" / "scaled" / "ablation.json"
OUT_FIG = ROOT / "figures" / "scaled" / "metrics" / "ablation.png"
K = 10            # dimensionality of the physical-manifold subspace
MODS = ["sequence", "structure", "ss8", "sasa", "function"]


def main() -> None:
    accs = sorted(p.stem for p in ABL.glob("*.npz"))
    head = np.load(ABL / f"{accs[0]}.npz", allow_pickle=True)
    abl_conds = [str(c) for c in head["conditions"]]
    layers = [int(l) for l in head["layers"]]
    ai = {c: i for i, c in enumerate(abl_conds)}
    print(f"{len(accs)} proteins | ablation conds {abl_conds} | {len(layers)} layers")

    abl = np.stack([np.load(ABL / f"{a}.npz", allow_pickle=True)["mean_pool"]
                    for a in accs]).astype(np.float64)        # (N,5,48,1536)
    allmp = []
    for a in accs:
        d = np.load(SCALED / f"{a}.npz", allow_pickle=True)
        ci = [str(c) for c in d["conditions"]].index("all")
        allmp.append(d["mean_pool"][ci])
    allmp = np.stack(allmp).astype(np.float64)                # (N,48,1536)

    series = {m: {"magnitude": [], "off_manifold": [], "common_mode": []}
              for m in MODS}
    for li, L in enumerate(layers):
        hall = allmp[:, li, :]                                # (N,1536)
        hn = np.linalg.norm(hall, axis=1)
        mu = hall.mean(0)
        _, _, Vt = np.linalg.svd(hall - mu, full_matrices=False)
        Pk = Vt[:K]                                           # (K,1536) manifold basis
        for m in MODS:
            d = hall - abl[:, ai[f"all_no_{m}"], li, :]       # (N,1536)
            dn = np.linalg.norm(d, axis=1)
            on = d @ Pk.T @ Pk                                # on-manifold part
            off = np.linalg.norm(d - on, axis=1) / (dn + 1e-9)
            series[m]["magnitude"].append(float(np.mean(dn / hn)))
            series[m]["off_manifold"].append(float(np.mean(off)))
            series[m]["common_mode"].append(
                float(np.linalg.norm(d.mean(0)) / (np.mean(dn) + 1e-9)))

    # per-protein relative displacement at the fusion layer, for the box panel
    FUSE = 35
    fl = layers.index(FUSE)
    hall = allmp[:, fl, :]; hn = np.linalg.norm(hall, axis=1)
    box = {m: ((np.linalg.norm(hall - abl[:, ai[f"all_no_{m}"], fl, :], axis=1))
               / hn).tolist() for m in MODS}

    out = {"layers": layers, "k_manifold": K, "n": len(accs),
           "fusion_layer": FUSE, "series": series, "box_fusion": box}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    canon = [32, 35, 40]
    cl = [layers.index(L) for L in canon]
    print(f"\n{'modality':>10}  {'magnitude':>20}  {'off-manifold':>20}  {'common-mode':>20}")
    print(f"{'':>10}  {'L32   L35   L40':>20}  {'L32   L35   L40':>20}  {'L32   L35   L40':>20}")
    for m in MODS:
        s = series[m]
        g = "  ".join(f"{s['magnitude'][i]:.3f}" for i in cl)
        o = "  ".join(f"{s['off_manifold'][i]:.3f}" for i in cl)
        c = "  ".join(f"{s['common_mode'][i]:.3f}" for i in cl)
        print(f"{m:>10}  {g:>20}  {o:>20}  {c:>20}")
    _render(out)


def _render(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    layers, series, box = out["layers"], out["series"], out["box_fusion"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.2, 3.4), dpi=300)
    for m in MODS:
        c = CONDITION_COLOR[m]
        lw = 2.4 if m == "function" else 1.6
        axA.plot(layers, series[m]["magnitude"], "-", color=c, lw=lw,
                 label=CONDITION_LABEL[m])
    axA.set_xlabel("layer"); axA.set_ylabel("relative displacement  $||d_X||/||h_{all}||$")
    axA.set_title("withholding each modality, across depth", fontsize=8.5)
    axA.set_xticks(range(0, 48, 8))
    axA.text(-0.14, 1.06, "a", transform=axA.transAxes, fontsize=12, fontweight="bold")
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
               frameon=False, fontsize=7.5, columnspacing=1.2, handlelength=1.4)

    order = ["structure", "sequence", "sasa", "ss8", "function"]
    bp = axB.boxplot([box[m] for m in order], widths=0.6, showfliers=False,
                     patch_artist=True,
                     labels=[CONDITION_LABEL[m] for m in order])
    for patch, m in zip(bp["boxes"], order):
        patch.set(facecolor=CONDITION_COLOR[m], alpha=0.55, edgecolor=INK)
    for med in bp["medians"]:
        med.set(color=INK)
    axB.set_ylabel("relative displacement at layer 35")
    axB.set_title("per-protein footprint at the fusion layer", fontsize=8.5)
    axB.tick_params(axis="x", labelrotation=30, labelsize=7)
    axB.text(-0.14, 1.06, "b", transform=axB.transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig); print(f"wrote {OUT_FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
