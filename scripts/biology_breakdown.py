"""Does modality fusion depend on protein biology?

Defines a per-protein fusion trajectory and asks whether the depth at which a
protein's modalities fuse varies with its category, length, or disorder.

Per-protein fusion measure
--------------------------
For each protein, at each layer, take its four PHYSICAL modality vectors
(sequence, structure, ss8, sasa; function is excluded — it never fuses), centre
each by that condition's population mean at that layer (removing the shared
component so we measure protein-specific alignment), and compute the mean
pairwise cosine similarity. This 'alignment' rises with depth as the modalities
fuse. The per-protein 'fusion-onset layer' is the first layer reaching the
half-way point between its early baseline (L0-7 mean) and its peak.

Biology
-------
  category      — the 10 curation classes
  length        — residues
  disorder/SS   — coil/helix/strand fraction from SS8

Outputs:
  results/scaled/biology_breakdown.json
  figures/scaled/metrics/biology_breakdown.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.viz import CONDITIONS, INK  # noqa: E402

MP_DIR = ROOT / "activations" / "scaled" / "per_protein"
META = ROOT / "data" / "pilot" / "annotations" / "metadata.json"
ANNO = ROOT / "data" / "scaled" / "annotations" / "structure_annotations.json"
OUT_JSON = ROOT / "results" / "scaled" / "biology_breakdown.json"
OUT_FIG = ROOT / "figures" / "scaled" / "metrics" / "biology_breakdown.png"

PHYSICAL = ["sequence", "structure", "ss8", "sasa"]


STRUCT_DIR = ROOT / "data" / "scaled" / "structures"


def ss_fracs(ss: str) -> tuple[float, float, float]:
    n = max(1, len(ss))
    helix = sum(ss.count(c) for c in "HGI") / n
    strand = sum(ss.count(c) for c in "EB") / n
    coil = sum(ss.count(c) for c in "CTSP") / n
    return helix, strand, coil


def mean_plddt(acc: str) -> float:
    """Mean per-residue AlphaFold confidence (pLDDT) from the CA B-factor column."""
    p = STRUCT_DIR / f"{acc}.pdb"
    if not p.exists():
        return float("nan")
    vals = []
    for line in p.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                vals.append(float(line[60:66]))
            except ValueError:
                pass
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    accs = sorted(p.stem for p in MP_DIR.glob("*.npz"))
    meta = json.loads(META.read_text())
    anno = json.loads(ANNO.read_text())
    phys_idx = [CONDITIONS.index(c) for c in PHYSICAL]

    # Load mean-pool tensor (N, C, 48, 1536).
    first = np.load(MP_DIR / f"{accs[0]}.npz", allow_pickle=True)
    layers = [int(l) for l in first["layers"]]
    n_layers = len(layers)
    A = np.zeros((len(accs), len(PHYSICAL), n_layers, 1536), dtype=np.float32)
    for i, a in enumerate(accs):
        mp = np.load(MP_DIR / f"{a}.npz", allow_pickle=True)["mean_pool"]
        A[i] = mp[phys_idx]
    print(f"loaded {len(accs)} proteins x {len(PHYSICAL)} physical conditions x {n_layers} layers")

    # Centre per (condition, layer) by population mean; per-protein alignment.
    A = A - A.mean(axis=0, keepdims=True)
    align = np.zeros((len(accs), n_layers), dtype=np.float64)
    for L in range(n_layers):
        B = A[:, :, L, :]                                   # (N, 4, D)
        Bn = B / (np.linalg.norm(B, axis=-1, keepdims=True) + 1e-8)
        sims = [(Bn[:, i] * Bn[:, j]).sum(-1)
                for i in range(len(PHYSICAL)) for j in range(i + 1, len(PHYSICAL))]
        align[:, L] = np.mean(sims, axis=0)
    print("population mean alignment by layer (should rise = fusion):")
    print("  " + "  ".join(f"L{layers[k]}={align[:, k].mean():.2f}"
                           for k in range(0, n_layers, 8)))

    # Per-protein fusion-onset layer: half-rise from L0-7 baseline to peak.
    onset = np.zeros(len(accs))
    for i in range(len(accs)):
        traj = align[i]
        base = traj[:8].mean()
        peak = traj.max()
        thresh = base + 0.5 * (peak - base)
        hit = np.where(traj >= thresh)[0]
        onset[i] = layers[hit[0]] if len(hit) else layers[-1]

    # Biology properties.
    length = np.array([meta[a]["length"] for a in accs], dtype=float)
    cats = [meta[a]["category"] for a in accs]
    helix, strand, coil = np.array([ss_fracs(anno[a]["ss8"]) for a in accs]).T
    plddt = np.array([mean_plddt(a) for a in accs])  # AFDB structure confidence

    def corr(x, name):
        ok = np.isfinite(x)
        r, p = spearmanr(onset[ok], x[ok])
        return {"vs": name, "spearman_r": float(r), "p_value": float(p),
                "n": int(ok.sum())}

    correlations = [corr(length, "length"), corr(coil, "coil_fraction"),
                    corr(helix, "helix_fraction"), corr(strand, "strand_fraction"),
                    corr(plddt, "mean_plddt")]
    print("\nfusion-onset Spearman correlations:")
    for c in correlations:
        print(f"  vs {c['vs']:<16} r={c['spearman_r']:+.3f}  p={c['p_value']:.1e}")

    # Per-category onset.
    cat_names = sorted(set(cats))
    by_cat = {c: onset[[i for i, x in enumerate(cats) if x == c]] for c in cat_names}
    cat_stats = {c: {"n": int(len(v)), "mean_onset": float(v.mean()),
                     "std_onset": float(v.std())} for c, v in by_cat.items()}
    print("\nmean fusion-onset by category:")
    for c in sorted(cat_stats, key=lambda k: cat_stats[k]["mean_onset"]):
        print(f"  {c:<20} n={cat_stats[c]['n']:>3}  onset={cat_stats[c]['mean_onset']:.1f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "n_proteins": len(accs), "layers": layers,
        "population_alignment": [float(align[:, k].mean()) for k in range(n_layers)],
        "correlations": correlations, "category_onset": cat_stats,
    }, indent=2))
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = matplotlib.colormaps["tab10"]
    x = np.array(layers)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=120, facecolor="white")

    # A: alignment curves by category
    axA = axes[0, 0]
    for ci, c in enumerate(cat_names):
        idx = [i for i, v in enumerate(cats) if v == c]
        axA.plot(x, align[idx].mean(0), color=cmap(ci % 10), lw=1.8,
                 label=f"{c} (n={len(idx)})")
    axA.set_title("Modality alignment across depth, by category",
                  fontsize=12, fontweight="bold", color=INK)
    axA.set_xlabel("Layer"); axA.set_ylabel("physical-modality alignment (cosine)")
    axA.legend(fontsize=7, ncol=2); axA.grid(True, alpha=0.25)

    # B: onset vs length
    axB = axes[0, 1]
    rL = correlations[0]["spearman_r"]
    axB.scatter(length, onset, s=10, alpha=0.4, color=INK, edgecolors="none")
    axB.set_title(f"Fusion onset vs length  (Spearman r={rL:+.2f})",
                  fontsize=12, fontweight="bold", color=INK)
    axB.set_xlabel("length (residues)"); axB.set_ylabel("fusion-onset layer")
    axB.grid(True, alpha=0.25)

    # C: onset vs pLDDT (structure confidence) — coil & helix correlations are in the JSON
    axC = axes[1, 0]
    rP = correlations[4]["spearman_r"]
    okp = np.isfinite(plddt)
    axC.scatter(plddt[okp], onset[okp], s=10, alpha=0.4, color="#DC267F", edgecolors="none")
    axC.set_title(f"Fusion onset vs structure confidence (pLDDT)  (r={rP:+.2f})",
                  fontsize=12, fontweight="bold", color=INK)
    axC.set_xlabel("mean pLDDT (AlphaFold confidence)"); axC.set_ylabel("fusion-onset layer")
    axC.grid(True, alpha=0.25)

    # D: onset by category (sorted)
    axD = axes[1, 1]
    order = sorted(cat_names, key=lambda c: by_cat[c].mean())
    means = [by_cat[c].mean() for c in order]
    stds = [by_cat[c].std() / np.sqrt(len(by_cat[c])) for c in order]
    axD.barh(range(len(order)), means, xerr=stds, color="#648FFF",
             edgecolor="none", error_kw={"ecolor": "#475569"})
    axD.set_yticks(range(len(order)))
    axD.set_yticklabels(order, fontsize=9)
    axD.set_title("Mean fusion-onset layer by category (±SE)",
                  fontsize=12, fontweight="bold", color=INK)
    axD.set_xlabel("fusion-onset layer"); axD.grid(True, alpha=0.25, axis="x")

    fig.suptitle("Does modality fusion depend on protein biology? "
                 f"(n={len(accs)})", fontsize=15, fontweight="bold", color=INK, y=1.0)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
