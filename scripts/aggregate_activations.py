"""Aggregate per-protein NPZs into per-layer NPZs ready for embedding.

Each per-protein NPZ has shape (C conditions, K layers, 1536). We reshape
into K separate (N*C, 1536) arrays — one row per (protein, condition) point —
which is the natural input for PCA/UMAP and for the GIF renderer.

Outputs:
  activations/by_layer/layer_{L:02d}.npz
      coords:      float32 (N*C, 1536)
      condition:   str    (N*C,)         — one of MODALITY_CONDITIONS
      protein_id:  str    (N*C,)
      category:    str    (N*C,)
      length:      int32  (N*C,)
      layer_idx:   int32  ()
  activations/by_layer/index.json
      proteins (ordered), conditions (ordered), layers, n_points
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="activations",
                    help="activations base dir (e.g. 'activations' or "
                         "'activations/scaled'); reads <base>/per_protein, "
                         "writes <base>/by_layer")
    args = ap.parse_args()
    base = ROOT / args.base
    global IN_DIR, OUT_DIR
    IN_DIR = base / "per_protein"
    OUT_DIR = base / "by_layer"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_protein_files = sorted(IN_DIR.glob("*.npz"))
    if not per_protein_files:
        raise SystemExit(f"No per-protein NPZs found in {IN_DIR}")
    print(f"loading {len(per_protein_files)} per-protein NPZs")

    # Inspect the first one to fix the schema (conditions order, layers).
    head = np.load(per_protein_files[0], allow_pickle=True)
    conditions = [str(c) for c in head["conditions"]]
    layers = [int(l) for l in head["layers"]]
    C, K = len(conditions), len(layers)
    print(f"  C={C} conditions = {conditions}")
    print(f"  K={K} layers     = {layers}")

    # Collect per-layer arrays.
    coords_by_layer = {l: [] for l in layers}
    protein_ids: list[str] = []
    categories: list[str] = []
    lengths: list[int] = []

    for fp in per_protein_files:
        npz = np.load(fp, allow_pickle=True)
        mp = npz["mean_pool"]  # (C, K, 1536)
        if mp.shape != (C, K, 1536):
            raise ValueError(f"{fp.name}: unexpected shape {mp.shape}")
        for li, layer_idx in enumerate(layers):
            coords_by_layer[layer_idx].append(mp[:, li, :])  # (C, 1536) per protein
        sid = fp.stem
        cat = str(npz["category"])
        L = int(npz["length"])
        protein_ids.append(sid)
        categories.append(cat)
        lengths.append(L)

    # Build the per-row labels — same order as coords concatenation:
    # outer loop over proteins, inner loop over conditions.
    cond_arr = np.array(conditions * len(protein_ids), dtype=object).reshape(
        len(protein_ids), C
    ).flatten()
    protein_arr = np.array(
        [pid for pid in protein_ids for _ in range(C)], dtype=object
    )
    category_arr = np.array(
        [cat for cat in categories for _ in range(C)], dtype=object
    )
    length_arr = np.array(
        [L for L in lengths for _ in range(C)], dtype=np.int32
    )

    for layer_idx in layers:
        # Stack per-protein (C, 1536) → (N, C, 1536) → (N*C, 1536)
        per_prot = np.stack(coords_by_layer[layer_idx], axis=0)  # (N, C, 1536)
        flat = per_prot.reshape(-1, per_prot.shape[-1])           # (N*C, 1536)
        out = OUT_DIR / f"layer_{layer_idx:02d}.npz"
        np.savez_compressed(
            out,
            coords=flat.astype(np.float32),
            condition=cond_arr,
            protein_id=protein_arr,
            category=category_arr,
            length=length_arr,
            layer_idx=np.int32(layer_idx),
        )
        print(f"  wrote {out.relative_to(ROOT)}  shape={flat.shape}")

    index = {
        "proteins": protein_ids,
        "conditions": conditions,
        "layers": layers,
        "n_proteins": len(protein_ids),
        "n_conditions": C,
        "n_points_per_layer": len(protein_ids) * C,
    }
    (OUT_DIR / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nwrote index → {OUT_DIR / 'index.json'}")
    print(f"N={len(protein_ids)} proteins × C={C} conditions = "
          f"{len(protein_ids) * C} points per layer")


if __name__ == "__main__":
    main()
