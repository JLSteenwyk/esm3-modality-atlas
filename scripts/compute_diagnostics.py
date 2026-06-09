"""Representational diagnostics that defend the fusion claim.

compute_metrics.py establishes *that* the modality conditions fuse (silhouette
falls, CKA rises). This script answers the three questions a skeptic asks next,
all from the cached 1536-d activations (no model, no GPU):

  1. DECODABILITY — modality-identity linear probe across depth.
     A logistic probe (protein-grouped 5-fold CV, so train/test proteins are
     disjoint) predicting the condition label from the residual stream. Overall
     accuracy should fall from ~1.0 toward chance (1/5 = 0.20) exactly where CKA
     spikes; per-condition recall shows *which* modality stays decodable longest.

  2. DIMENSIONALITY — effective dimensionality per layer.
     Rules out the confound that "fusion" is just the cloud going isotropic.
     We compare the effective rank / participation ratio of the *whole* cloud
     against the mean of the *per-condition* sub-clouds. If conditions truly
     share a subspace, overall ≈ within (the condition label adds no extra
     spread); if they were merely spreading into noise, overall would balloon.

  3. SIGNIFICANCE — null baselines and bootstrap CIs.
     Label-permutation nulls and protein-bootstrap 95% CIs for both silhouette
     and the mean-pairwise-CKA integration index, so every layer's number comes
     with an error bar and a p-value.

Outputs:
  results/diagnostics.json
  figures/metrics/diagnostics.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compute_metrics import condition_matrices, linear_cka  # noqa: E402
from src.viz import CONDITION_COLOR, CONDITION_LABEL, CONDITIONS, INK  # noqa: E402

IN_DIR = ROOT / "activations" / "by_layer"
EMBED_DIR = ROOT / "figures" / "embed" / "per_layer"
METRICS = ROOT / "results" / "metrics.json"
OUT_RESULTS = ROOT / "results"
OUT_FIG = ROOT / "figures" / "metrics"

N_PERM = 200      # label-permutation null draws
N_BOOT = 200      # protein-bootstrap resamples
SEED = 0
FORCE_SIG = False  # recompute the expensive significance block (else reuse json)


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def eig_spectrum(x: np.ndarray) -> np.ndarray:
    """Covariance eigenvalues (descending) of (n, d) data, via SVD on the
    centred matrix. Returns squared singular values (proportional to variance)."""
    xc = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(xc, full_matrices=False, compute_uv=False)
    return (s ** 2)


def participation_ratio(lam: np.ndarray) -> float:
    """(Σλ)² / Σλ²  — a smooth 'effective number of dimensions'."""
    lam = lam[lam > 0]
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if lam.size else 0.0


def effective_rank(lam: np.ndarray) -> float:
    """exp(entropy of the normalised eigenspectrum) — Roy & Vetterli (2007)."""
    lam = lam[lam > 0]
    if lam.size == 0:
        return 0.0
    p = lam / lam.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


# --------------------------------------------------------------------------- #
# 1. Modality-identity probe
# --------------------------------------------------------------------------- #
def _grouped_cv_probe(x, condition, protein_id, present):
    """Protein-grouped 5-fold logistic probe; out-of-fold acc + per-class recall."""
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    gkf = GroupKFold(n_splits=5)
    oof = np.empty(len(condition), dtype=object)
    fold_acc = []
    for tr, te in gkf.split(x, condition, groups=protein_id):
        pipe.fit(x[tr], condition[tr])
        pred = pipe.predict(x[te])
        oof[te] = pred
        fold_acc.append(accuracy_score(condition[te], pred))
    recall = recall_score(condition, oof, labels=present, average=None,
                          zero_division=0)
    return {
        "accuracy_mean": float(np.mean(fold_acc)),
        "accuracy_std": float(np.std(fold_acc)),
        "chance": 1.0 / len(present),
        "recall_by_condition": {c: float(r) for c, r in zip(present, recall)},
    }


def probe_layer(coords, coords3d, condition, protein_id, present):
    """Probe modality identity in two spaces:

      fulldim — the native 1536-d stream. With n<<d this stays trivially
                separable (a thin additive 'modality-tag' offset), so it tracks
                *whether any* linear identity signal survives, not its strength.
      pca3d   — the top-3 PCA geometry the GIFs render. Separability here is
                non-trivial and falls as the dominant geometry fuses.
    """
    return {
        "fulldim": _grouped_cv_probe(coords, condition, protein_id, present),
        "pca3d": _grouped_cv_probe(coords3d, condition, protein_id, present),
    }


# --------------------------------------------------------------------------- #
# 2. Effective dimensionality
# --------------------------------------------------------------------------- #
def dimensionality_layer(coords, condition, present):
    lam_all = eig_spectrum(coords)
    within_pr, within_er = [], []
    by_cond = {}
    for c in present:
        lam_c = eig_spectrum(coords[condition == c])
        pr_c, er_c = participation_ratio(lam_c), effective_rank(lam_c)
        within_pr.append(pr_c)
        within_er.append(er_c)
        by_cond[c] = {"participation_ratio": pr_c, "effective_rank": er_c}
    return {
        "overall_participation_ratio": participation_ratio(lam_all),
        "overall_effective_rank": effective_rank(lam_all),
        "within_participation_ratio_mean": float(np.mean(within_pr)),
        "within_effective_rank_mean": float(np.mean(within_er)),
        "by_condition": by_cond,
    }


# --------------------------------------------------------------------------- #
# 3. Significance — nulls + bootstrap
# --------------------------------------------------------------------------- #
def mean_pairwise_cka(mats, present, row_idx=None, perm_rng=None):
    """Mean linear CKA over all condition pairs. row_idx resamples proteins
    (bootstrap); perm_rng independently permutes each matrix's rows (null)."""
    use = {}
    for c in present:
        m = mats[c]
        if row_idx is not None:
            m = m[row_idx]
        if perm_rng is not None:
            m = m[perm_rng.permutation(m.shape[0])]
        use[c] = m
    vals = [linear_cka(use[present[i]], use[present[j]])
            for i in range(len(present)) for j in range(i + 1, len(present))]
    return float(np.mean(vals))


def significance_layer(coords, condition, protein_id, present,
                       obs_sil, obs_cka, rng):
    n = len(condition)
    dist = pairwise_distances(coords, metric="euclidean")

    # protein -> row indices (each protein contributes all conditions)
    proteins = sorted(set(protein_id.tolist()))
    p_to_rows = {p: np.where(protein_id == p)[0] for p in proteins}
    _, mats = condition_matrices(coords.astype(np.float64), condition, protein_id)
    n_prot = mats[present[0]].shape[0]

    # --- silhouette: permutation null + bootstrap CI ---
    sil_null = np.array([
        silhouette_score(dist, rng.permutation(condition), metric="precomputed")
        for _ in range(N_PERM)
    ])
    sil_boot = []
    cka_boot = []
    for _ in range(N_BOOT):
        samp = rng.choice(proteins, size=len(proteins), replace=True)
        idx = np.concatenate([p_to_rows[p] for p in samp])
        sil_boot.append(silhouette_score(
            dist[np.ix_(idx, idx)], condition[idx], metric="precomputed"))
        ridx = rng.integers(0, n_prot, size=n_prot)
        cka_boot.append(mean_pairwise_cka(mats, present, row_idx=ridx))
    sil_boot = np.array(sil_boot)
    cka_boot = np.array(cka_boot)

    # --- CKA: row-shuffle null ---
    cka_null = np.array([
        mean_pairwise_cka(mats, present, perm_rng=rng) for _ in range(N_PERM)
    ])

    def pval(null, obs):  # one-sided: P(null >= obs)
        return float((1 + np.sum(null >= obs)) / (len(null) + 1))

    return {
        "silhouette": {
            "observed": float(obs_sil),
            "null_mean": float(sil_null.mean()),
            "null_std": float(sil_null.std()),
            "p_value": pval(sil_null, obs_sil),
            "ci95": [float(np.percentile(sil_boot, 2.5)),
                     float(np.percentile(sil_boot, 97.5))],
        },
        "integration_index": {
            "observed": float(obs_cka),
            "null_mean": float(cka_null.mean()),
            "null_std": float(cka_null.std()),
            "p_value": pval(cka_null, obs_cka),
            "ci95": [float(np.percentile(cka_boot, 2.5)),
                     float(np.percentile(cka_boot, 97.5))],
        },
    }


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def render_plot(layers, diag, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array(layers)
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.5, 4.6),
                                        dpi=130, facecolor="white")

    # --- A. probe ---
    full = np.array([diag[str(L)]["probe"]["fulldim"]["accuracy_mean"] for L in layers])
    acc3 = np.array([diag[str(L)]["probe"]["pca3d"]["accuracy_mean"] for L in layers])
    std3 = np.array([diag[str(L)]["probe"]["pca3d"]["accuracy_std"] for L in layers])
    chance = diag[str(layers[0])]["probe"]["pca3d"]["chance"]
    axA.plot(x, full, "-o", color="#64748b", lw=2.0, alpha=0.9,
             label="full 1536-d (identity tag persists)")
    axA.fill_between(x, acc3 - std3, acc3 + std3, color=INK, alpha=0.15)
    axA.plot(x, acc3, "-o", color=INK, lw=2.4, label="top-3 PCA (overall)")
    for c in CONDITIONS:
        r = [diag[str(L)]["probe"]["pca3d"]["recall_by_condition"].get(c)
             for L in layers]
        if all(v is not None for v in r):
            axA.plot(x, r, "-", color=CONDITION_COLOR[c], lw=1.2, alpha=0.8,
                     label=f"3D recall: {CONDITION_LABEL[c]}")
    axA.axhline(chance, ls=":", color="#94a3b8", lw=1.5)
    axA.text(x[-1], chance + 0.02, "chance", ha="right", fontsize=8,
             color="#94a3b8")
    axA.set_ylim(0, 1.03)
    axA.set_xticks(x)
    axA.set_xlabel("Layer", color=INK)
    axA.set_ylabel("modality-identity decodability", color=INK)
    axA.set_title("1 · Probe: can we still tell modalities apart?",
                  fontsize=11, fontweight="bold", color=INK)
    axA.legend(fontsize=7, loc="center left", framealpha=0.9)
    axA.grid(True, alpha=0.25)

    # --- B. effective dim ---
    ov = [diag[str(L)]["dimensionality"]["overall_effective_rank"] for L in layers]
    wi = [diag[str(L)]["dimensionality"]["within_effective_rank_mean"] for L in layers]
    axB.plot(x, ov, "-o", color=INK, lw=2.4, label="whole cloud")
    axB.plot(x, wi, "--s", color="#94a3b8", lw=2.0,
             label="per-condition (mean)")
    axB.fill_between(x, wi, ov, color=CONDITION_COLOR["sequence"], alpha=0.10)
    axB.set_xticks(x)
    axB.set_xlabel("Layer", color=INK)
    axB.set_ylabel("effective rank (dimensions)", color=INK)
    axB.set_title("2 · Dimensionality: shared subspace, not isotropy?",
                  fontsize=11, fontweight="bold", color=INK)
    axB.annotate("isotropic would be ≈1536 — peak is ~70,\n"
                 "so fusion is collapse into a low-d subspace",
                 xy=(0.50, 0.82), xycoords="axes fraction", ha="center",
                 va="center", fontsize=8, color="#64748b", style="italic")
    axB.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    axB.grid(True, alpha=0.25)

    # --- C. significance ---
    def series(metric, key):
        return np.array([diag[str(L)]["significance"][metric][key] for L in layers])

    for metric, color, lab in [
        ("silhouette", CONDITION_COLOR["sequence"], "silhouette"),
        ("integration_index", INK, "integration index"),
    ]:
        obs = series(metric, "observed")
        lo = np.array([diag[str(L)]["significance"][metric]["ci95"][0] for L in layers])
        hi = np.array([diag[str(L)]["significance"][metric]["ci95"][1] for L in layers])
        nm = series(metric, "null_mean")
        axC.fill_between(x, lo, hi, color=color, alpha=0.18)
        axC.plot(x, obs, "-o", color=color, lw=2.2, label=f"{lab} (95% CI)")
        axC.plot(x, nm, ":", color=color, lw=1.4, alpha=0.8,
                 label=f"{lab} null")
    axC.set_xticks(x)
    axC.set_ylim(-0.05, 1.03)
    axC.set_xlabel("Layer", color=INK)
    axC.set_ylabel("score", color=INK)
    axC.set_title("3 · Significance: bootstrap CIs vs permutation null",
                  fontsize=11, fontweight="bold", color=INK)
    axC.legend(fontsize=8, loc="center right", framealpha=0.9)
    axC.grid(True, alpha=0.25)

    fig.suptitle("ESM3 modality-fusion diagnostics", fontsize=15,
                 fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
def main() -> None:
    index = json.loads((IN_DIR / "index.json").read_text())
    layers = index["layers"]
    metrics = json.loads(METRICS.read_text())
    obs_sil = dict(zip(layers, metrics["series"]["silhouette"]))
    obs_cka = dict(zip(layers, metrics["series"]["integration_index"]))
    rng = np.random.default_rng(SEED)

    # The significance block (permutation + bootstrap) is the only expensive part
    # (~minutes/layer). Reuse it from a previous run unless forced.
    cached_sig = {}
    prev = OUT_RESULTS / "diagnostics.json"
    if prev.exists() and not FORCE_SIG:
        old = json.loads(prev.read_text()).get("per_layer", {})
        cached_sig = {k: v["significance"] for k, v in old.items()
                      if "significance" in v}
        if cached_sig:
            print("reusing cached significance (set FORCE_SIG=True to recompute)")

    print(f"layers: {layers}  (N_PERM={N_PERM}, N_BOOT={N_BOOT})\n")
    print(f"{'layer':>5}  {'acc_full':>8}  {'acc_3d':>7}  "
          f"{'eff_rank(all/within)':>21}  {'sil p':>6}  {'cka p':>6}")

    diag = {}
    for L in layers:
        d = np.load(IN_DIR / f"layer_{L:02d}.npz", allow_pickle=True)
        coords = d["coords"].astype(np.float64)
        condition = d["condition"].astype(str)
        protein_id = d["protein_id"].astype(str)
        present = [c for c in CONDITIONS if (condition == c).any()]

        emb = np.load(EMBED_DIR / f"layer_{L:02d}.npz", allow_pickle=True)
        coords3d = emb["coords3d"].astype(np.float64)

        probe = probe_layer(coords, coords3d, condition, protein_id, present)
        dim = dimensionality_layer(coords, condition, present)
        if str(L) in cached_sig:
            sig = cached_sig[str(L)]
        else:
            sig = significance_layer(coords, condition, protein_id, present,
                                     obs_sil[L], obs_cka[L], rng)
        diag[str(L)] = {"probe": probe, "dimensionality": dim, "significance": sig}

        print(f"{L:>5}  {probe['fulldim']['accuracy_mean']:>8.3f}  "
              f"{probe['pca3d']['accuracy_mean']:>7.3f}  "
              f"{dim['overall_effective_rank']:>9.1f} /"
              f"{dim['within_effective_rank_mean']:>9.1f}  "
              f"{sig['silhouette']['p_value']:>6.3f}  "
              f"{sig['integration_index']['p_value']:>6.3f}")

    out = {
        "meta": {
            "n_proteins": index["n_proteins"],
            "conditions_order": list(CONDITIONS),
            "layers": layers,
            "probe": "logistic regression, protein-grouped 5-fold CV, "
                     "standardized 1536-d features",
            "effective_rank": "exp(entropy of covariance eigenspectrum)",
            "n_perm": N_PERM,
            "n_boot": N_BOOT,
            "seed": SEED,
        },
        "per_layer": diag,
    }
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    (OUT_RESULTS / "diagnostics.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {(OUT_RESULTS / 'diagnostics.json').relative_to(ROOT)}")

    render_plot(layers, diag, OUT_FIG / "diagnostics.png")


if __name__ == "__main__":
    main()
