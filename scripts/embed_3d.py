"""Embed per-layer activations to 3D with PCA.

Produces two NPZ artifacts per layer:

  figures/embed/per_layer/layer_{L:02d}.npz
      coords3d:    float32 (N*C, 3)   — per-layer PCA (best in-layer separation)
      condition:   str    (N*C,)
      protein_id:  str    (N*C,)
      category:    str    (N*C,)
      evr:         float32 (3,)       — explained variance ratio
      layer_idx:   int32  ()

  figures/embed/joint/layer_{L:02d}.npz   (shares axes across all layers)
      coords3d, condition, protein_id, category, evr_joint, layer_idx

  figures/embed/joint/basis.npz
      components:  float32 (3, 1536)
      layer_means: dict-equivalent (one (1536,) per layer)
      evr_joint:   float32 (3,)
      layer_scale: float
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.embed import fit_joint_pca, fit_per_layer_pca, project_joint  # noqa: E402


IN_DIR = ROOT / "activations" / "by_layer"
OUT_PER = ROOT / "figures" / "embed" / "per_layer"
OUT_JOINT = ROOT / "figures" / "embed" / "joint"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="activations",
                    help="activations base (reads <base>/by_layer)")
    ap.add_argument("--tag", default="",
                    help="output subdir under figures/ (e.g. 'scaled')")
    ap.add_argument("--subsample", type=int, default=0,
                    help="randomly keep this many proteins before PCA (0 = all); "
                         "for tractable visualisation of very large sets")
    args = ap.parse_args()
    global IN_DIR, OUT_PER, OUT_JOINT
    IN_DIR = ROOT / args.base / "by_layer"
    emb = ROOT / "figures" / args.tag / "embed" if args.tag else ROOT / "figures" / "embed"
    OUT_PER, OUT_JOINT = emb / "per_layer", emb / "joint"

    OUT_PER.mkdir(parents=True, exist_ok=True)
    OUT_JOINT.mkdir(parents=True, exist_ok=True)

    index = json.loads((IN_DIR / "index.json").read_text())
    layers: list[int] = index["layers"]
    print(f"layers: {layers}")

    # Optional protein subsample (same proteins across all layers), for tractable
    # PCA on very large sets. Built from the first layer's protein order.
    keep_mask = None
    if args.subsample:
        d0 = np.load(IN_DIR / f"layer_{layers[0]:02d}.npz", allow_pickle=True)
        pid0 = d0["protein_id"].astype(str)
        proteins = np.array(sorted(set(pid0)))
        rng = np.random.default_rng(0)
        keep = set(rng.choice(proteins, min(args.subsample, len(proteins)),
                              replace=False).tolist())
        keep_mask = np.isin(pid0, list(keep))
        print(f"subsampling to {len(keep)} proteins ({keep_mask.sum()} rows/layer)")

    # Load every layer's coords into memory once. Each is (N*C, 1536) — small.
    coords_by_layer: dict[int, np.ndarray] = {}
    labels_by_layer: dict[int, dict] = {}
    for L in layers:
        d = np.load(IN_DIR / f"layer_{L:02d}.npz", allow_pickle=True)
        m = keep_mask if keep_mask is not None else slice(None)
        coords_by_layer[L] = d["coords"][m]
        labels_by_layer[L] = {
            "condition":  d["condition"][m],
            "protein_id": d["protein_id"][m],
            "category":   d["category"][m],
        }

    # --- Per-layer PCA: independent basis per layer ---
    print("\nfitting per-layer PCA ...")
    per_layer = fit_per_layer_pca(coords_by_layer, n_components=3)
    for L in layers:
        out_path = OUT_PER / f"layer_{L:02d}.npz"
        lab = labels_by_layer[L]
        np.savez_compressed(
            out_path,
            coords3d=per_layer[L]["coords3d"],
            evr=per_layer[L]["evr"],
            condition=lab["condition"],
            protein_id=lab["protein_id"],
            category=lab["category"],
            layer_idx=np.int32(L),
        )
        evr = per_layer[L]["evr"]
        print(f"  L{L:02d}  evr = {evr[0]:.3f} {evr[1]:.3f} {evr[2]:.3f}   "
              f"({evr.sum():.3f} cumulative)  -> {out_path.name}")

    # --- Joint PCA: shared basis across layers ---
    print("\nfitting joint PCA (axes shared across layers) ...")
    joint = fit_joint_pca(coords_by_layer, n_components=3)
    print(f"  joint evr = {joint.explained_variance_ratio[0]:.3f} "
          f"{joint.explained_variance_ratio[1]:.3f} "
          f"{joint.explained_variance_ratio[2]:.3f}   "
          f"({joint.explained_variance_ratio.sum():.3f} cumulative)")
    print("  per-layer scales (joint basis is shared; magnitudes vary by layer):")
    for L, s in joint.layer_scales.items():
        print(f"    L{L:02d}  scale = {s:.4f}")

    joint_coords = project_joint(joint, coords_by_layer)
    for L in layers:
        out_path = OUT_JOINT / f"layer_{L:02d}.npz"
        lab = labels_by_layer[L]
        np.savez_compressed(
            out_path,
            coords3d=joint_coords[L],
            evr=joint.explained_variance_ratio,
            condition=lab["condition"],
            protein_id=lab["protein_id"],
            category=lab["category"],
            layer_idx=np.int32(L),
        )
        print(f"  wrote {out_path.name}  range: "
              f"x={joint_coords[L][:,0].min():.2f}..{joint_coords[L][:,0].max():.2f}  "
              f"y={joint_coords[L][:,1].min():.2f}..{joint_coords[L][:,1].max():.2f}  "
              f"z={joint_coords[L][:,2].min():.2f}..{joint_coords[L][:,2].max():.2f}")

    # Save the joint basis for later use (re-projection, ablation experiments)
    np.savez_compressed(
        OUT_JOINT / "basis.npz",
        components=joint.components,
        evr=joint.explained_variance_ratio,
        layer_means=np.stack([joint.layer_means[L] for L in layers], axis=0),
        layer_scales=np.array([joint.layer_scales[L] for L in layers], dtype=np.float32),
        layers=np.array(layers, dtype=np.int32),
    )
    print(f"\nwrote joint basis -> {OUT_JOINT / 'basis.npz'}")


if __name__ == "__main__":
    main()
