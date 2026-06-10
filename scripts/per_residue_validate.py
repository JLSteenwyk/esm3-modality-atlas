"""Per-residue validation: does modality fusion hold below the mean-pool?

All the atlas metrics are computed on mean-pooled vectors (one per protein x
condition), so fusion could in principle be an averaging artifact. This recomputes
the fusion signal at the residue level on the stored per-residue subset
(activations/scaled/per_residue/, 100 proteins x 7 layers x fp16) and overlays it
on the mean-pool signal for the SAME proteins and layers.

For each stored layer we compute, per condition:
  · residue-level: stack every residue (all proteins) into one matrix; mean
    pairwise linear CKA across conditions (residue-aligned, so row r is the same
    residue of the same protein under each condition) + a subsampled silhouette.
  · mean-pool-level: one row per protein; the same two metrics.

If both curves show the same rise/fall, fusion is real below the pool.

Outputs:
  results/scaled/per_residue_validation.json
  figures/scaled/metrics/per_residue_validation.png
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
from src.viz import CONDITION_COLOR, CONDITIONS, INK  # noqa: E402

PR_DIR = ROOT / "activations" / "scaled" / "per_residue"
MP_DIR = ROOT / "activations" / "scaled" / "per_protein"
OUT_JSON = ROOT / "results" / "scaled" / "per_residue_validation.json"
OUT_FIG = ROOT / "figures" / "scaled" / "metrics" / "per_residue_validation.png"

SIL_SUBSAMPLE = 6000
SEED = 0


def mean_pairwise_cka(mats: dict) -> float:
    conds = list(mats)
    vals = [linear_cka(mats[conds[i]], mats[conds[j]])
            for i in range(len(conds)) for j in range(i + 1, len(conds))]
    return float(np.mean(vals)) if vals else 1.0


def subsampled_silhouette(points: np.ndarray, labels: np.ndarray, rng) -> float:
    from sklearn.metrics import silhouette_score
    if len(points) > SIL_SUBSAMPLE:
        idx = rng.choice(len(points), SIL_SUBSAMPLE, replace=False)
        points, labels = points[idx], labels[idx]
    return float(silhouette_score(points, labels))


def main() -> None:
    rng = np.random.default_rng(SEED)
    files = sorted(PR_DIR.glob("*.npz"))
    if not files:
        raise SystemExit(f"no per-residue NPZs in {PR_DIR}")
    accs = [f.stem for f in files]
    print(f"{len(files)} proteins with per-residue activations")

    # Load per-residue arrays once: each (C, K, L, 1536) fp16.
    head = np.load(files[0], allow_pickle=True)
    conditions = [str(c) for c in head["conditions"]]
    layers = [int(l) for l in head["layers"]]
    res_arrays = [np.load(f, allow_pickle=True)["residue"] for f in files]
    print(f"conditions {conditions} | layers {layers}")

    # Mean-pool for the same accessions, restricted to these layers.
    mp = {a: np.load(MP_DIR / f"{a}.npz", allow_pickle=True) for a in accs}
    mp_layer_idx = {L: list(mp[accs[0]]["layers"]).index(L) for L in layers}

    out = {"layers": layers, "conditions": conditions,
           "residue": {"integration": [], "silhouette": []},
           "mean_pool": {"integration": [], "silhouette": []}}

    for ki, L in enumerate(layers):
        # --- residue level ---
        res_mats, res_pts, res_lab = {}, [], []
        for ci, c in enumerate(conditions):
            stacked = np.concatenate([a[ci, ki] for a in res_arrays], axis=0).astype(np.float64)
            res_mats[c] = stacked
            res_pts.append(stacked)
            res_lab.append(np.full(len(stacked), c))
        res_integ = mean_pairwise_cka(res_mats)
        res_sil = subsampled_silhouette(np.concatenate(res_pts),
                                        np.concatenate(res_lab), rng)

        # --- mean-pool level (same proteins) ---
        mp_mats = {}
        for ci, c in enumerate(conditions):
            mp_mats[c] = np.stack(
                [mp[a]["mean_pool"][ci, mp_layer_idx[L]] for a in accs]).astype(np.float64)
        mp_integ = mean_pairwise_cka(mp_mats)
        mp_pts = np.concatenate([mp_mats[c] for c in conditions])
        mp_lab = np.concatenate([np.full(len(accs), c) for c in conditions])
        mp_sil = subsampled_silhouette(mp_pts, mp_lab, rng)

        out["residue"]["integration"].append(res_integ)
        out["residue"]["silhouette"].append(res_sil)
        out["mean_pool"]["integration"].append(mp_integ)
        out["mean_pool"]["silhouette"].append(mp_sil)
        print(f"L{L:2d}  integration  residue={res_integ:.3f} pool={mp_integ:.3f}   "
              f"silhouette  residue={res_sil:.3f} pool={mp_sil:.3f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)}")

    # --- figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.array(layers)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=130, facecolor="white")
    for ax, key, title in [(axL, "integration", "Integration index (mean pairwise CKA)"),
                           (axR, "silhouette", "Condition separation (silhouette)")]:
        ax.plot(x, out["mean_pool"][key], "-o", color=INK, lw=2.4, label="mean-pool")
        ax.plot(x, out["residue"][key], "--s", color=CONDITION_COLOR["sequence"],
                lw=2.2, label="per-residue")
        ax.set_xlabel("Layer", color=INK); ax.set_xticks(x)
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    fig.suptitle("Fusion holds below the mean-pool — per-residue vs pooled "
                 f"(same {len(accs)} proteins)", fontsize=14, fontweight="bold",
                 color=INK, y=1.02)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
