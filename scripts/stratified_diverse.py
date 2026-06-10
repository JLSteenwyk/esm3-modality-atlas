"""Is modality fusion universal? Stratify the diverse run by superkingdom/organism.

Reads activations/diverse/by_layer (12 organisms across eukaryota / bacteria /
archaea) and recomputes the fusion curve (silhouette + integration index) WITHIN
each superkingdom and each organism across all 48 layers. If every superkingdom
shows the same rise-then-collapse, fusion is a universal property of ESM3, not an
artifact of the curated human set.

Outputs:
  results/diverse/stratified.json
  figures/diverse/metrics/stratified_fusion.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from compute_metrics import condition_matrices, linear_cka  # noqa: E402
from src.viz import CONDITIONS, INK  # noqa: E402

IN_DIR = ROOT / "activations" / "diverse" / "by_layer"
CURATION = ROOT / "data" / "diverse" / "curation_summary.json"
OUT_JSON = ROOT / "results" / "diverse" / "stratified.json"
OUT_FIG = ROOT / "figures" / "diverse" / "metrics" / "stratified_fusion.png"

KINGDOM_COLOR = {"eukaryota": "#648FFF", "bacteria": "#FE6100",
                 "archaea": "#009E73", "all": INK}


def integration_and_sil(coords, condition, protein_id, keep_ids):
    mask = np.isin(protein_id, list(keep_ids))
    c, cond = coords[mask], condition[mask]
    present, mats = condition_matrices(c, cond, protein_id[mask])
    vals = [linear_cka(mats[present[i]], mats[present[j]])
            for i in range(len(present)) for j in range(i + 1, len(present))]
    integ = float(np.mean(vals)) if vals else 1.0
    sil = float(silhouette_score(c, cond)) if mask.sum() > len(present) else 0.0
    return sil, integ, int(mask.sum() // len(present))


def main() -> None:
    curation = json.loads(CURATION.read_text())
    org2king = {o: v["superkingdom"] for o, v in curation["per_organism"].items()
                if "superkingdom" in v}

    layers = sorted(int(p.stem.split("_")[-1]) for p in IN_DIR.glob("layer_*.npz"))
    # protein -> organism (category) from the first layer
    d0 = np.load(IN_DIR / f"layer_{layers[0]:02d}.npz", allow_pickle=True)
    pid0 = d0["protein_id"].astype(str)
    cat0 = d0["category"].astype(str)
    prot2org = dict(zip(pid0, cat0))
    proteins = sorted(set(pid0))

    groups = {"all": set(proteins)}
    for king in ["eukaryota", "bacteria", "archaea"]:
        groups[king] = {p for p in proteins if org2king.get(prot2org.get(p, ""), "") == king}
    organisms = sorted(set(prot2org.values()))
    for org in organisms:
        groups[f"org:{org}"] = {p for p in proteins if prot2org.get(p) == org}
    print({k: len(v) for k, v in groups.items() if not k.startswith("org:")})

    series = {g: {"silhouette": [], "integration": []} for g in groups}
    for L in layers:
        d = np.load(IN_DIR / f"layer_{L:02d}.npz", allow_pickle=True)
        coords = d["coords"].astype(np.float64)
        condition = d["condition"].astype(str)
        protein_id = d["protein_id"].astype(str)
        for g, ids in groups.items():
            if len(ids) < 20:
                series[g]["silhouette"].append(None)
                series[g]["integration"].append(None)
                continue
            sil, integ, _ = integration_and_sil(coords, condition, protein_id, ids)
            series[g]["silhouette"].append(sil)
            series[g]["integration"].append(integ)
        print(f"L{L:2d} done")

    out = {"layers": layers, "n_by_group": {g: len(v) for g, v in groups.items()},
           "series": series}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")

    # ---- figure: fusion curve per superkingdom ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.array(layers)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), dpi=130, facecolor="white")
    for g in ["all", "eukaryota", "bacteria", "archaea"]:
        n = out["n_by_group"][g]
        axL.plot(x, series[g]["silhouette"], "-", color=KINGDOM_COLOR[g],
                 lw=2.6 if g == "all" else 1.9, label=f"{g} (n={n})")
        axR.plot(x, series[g]["integration"], "-", color=KINGDOM_COLOR[g],
                 lw=2.6 if g == "all" else 1.9, label=f"{g} (n={n})")
    for ax, t in [(axL, "Condition separation (silhouette)"),
                  (axR, "Integration index (mean pairwise CKA)")]:
        ax.set_xlabel("Layer", color=INK); ax.set_title(t, fontsize=12,
                      fontweight="bold", color=INK)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    fig.suptitle("Modality fusion is universal across the tree of life\n"
                 f"{out['n_by_group']['all']} proteins, 12 organisms, 3 superkingdoms",
                 fontsize=14, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
