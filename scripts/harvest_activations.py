"""Harvest ESM-3 residual-stream activations under each modality condition.

For each pilot protein and each condition in
``{sequence, structure, ss8, sasa, all}``, runs one ESM-3 forward pass and
caches the mean-pooled residual stream at the requested layers.

Outputs:
  activations/per_protein/{accession}.npz
      conditions:    array of condition names (C,)
      layers:        array of layer indices (K,)
      mean_pool:     float32 (C, K, 1536)  — residue-averaged hidden state,
                     special tokens excluded
      length:        int                   — sequence length (no specials)
      category:      str
  activations/manifest.json
      sampled proteins, conditions, layers, runtime stats

After harvest, run ``scripts/aggregate_activations.py`` to fan the per-protein
files out into per-layer NPZs ready for embedding.

Resumable: skips proteins whose per-protein NPZ already exists.

Usage:
    python scripts/harvest_activations.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import iter_pilot_proteins  # noqa: E402
from src.models import (  # noqa: E402
    load_esm3,
    run_modality_condition,
    tokenize_protein,
)

# Layers to extract. ESM-3 has 48 transformer blocks (0..47). 7 evenly-spaced
# slices give us early/integration/late views without ballooning storage.
LAYERS: tuple[int, ...] = (0, 8, 16, 24, 32, 40, 47)

# Conditions to extract. Function-token wiring is deferred — see README.
CONDITIONS: tuple[str, ...] = ("sequence", "structure", "ss8", "sasa", "all")

OUT_DIR = ROOT / "activations" / "per_protein"
MANIFEST_PATH = ROOT / "activations" / "manifest.json"
PILOT_DIR = ROOT / "data" / "pilot"


def _coords_from_pdb(pdb_path: Path) -> torch.Tensor:
    """Backbone (N, CA, C) coords as (L, 3, 3) float32 tensor."""
    from esm.utils.structure.protein_chain import ProteinChain

    chain = ProteinChain.from_pdb(str(pdb_path))
    atom37 = torch.from_numpy(chain.atom37_positions).float()
    return atom37[:, :3, :]


def _mean_pool_residues(hidden: torch.Tensor) -> np.ndarray:
    """Mean across residue positions, excluding BOS/EOS special tokens.

    hidden shape: (L+2, 1536) → (1536,) float32 numpy.
    """
    return hidden[1:-1].mean(dim=0).cpu().numpy().astype(np.float32)


def harvest_protein(
    protein,
    model,
    tokenizers,
    structure_encoder,
    device: str,
) -> dict:
    """Run all conditions for one protein, return arrays for NPZ."""
    coords = _coords_from_pdb(protein.pdb_path)
    if coords.shape[0] != protein.length:
        # Pilot PDB occasionally has a different residue count than the
        # canonical UniProt sequence (e.g. trimmed termini). Skip rather
        # than risk misaligned residue-level features.
        raise ValueError(
            f"{protein.accession}: PDB length {coords.shape[0]} != seq length {protein.length}"
        )

    inputs = tokenize_protein(
        sequence_id=protein.accession,
        sequence=protein.sequence,
        tokenizers=tokenizers,
        structure_encoder=structure_encoder,
        coordinates=coords,
        secondary_structure=protein.ss8,
        sasa=protein.sasa,
        function_annotations=None,  # deferred
        device=device,
    )

    available = set(inputs.available_conditions())
    missing = [c for c in CONDITIONS if c not in available]
    if missing:
        raise ValueError(f"{protein.accession}: missing required conditions {missing}")

    mean_pool = np.zeros((len(CONDITIONS), len(LAYERS), 1536), dtype=np.float32)
    for ci, cond in enumerate(CONDITIONS):
        cache = run_modality_condition(
            model, inputs, cond, layers=list(LAYERS), device=device
        )
        for li, layer_idx in enumerate(LAYERS):
            mean_pool[ci, li] = _mean_pool_residues(cache.residual_stream[layer_idx])

    return {
        "conditions": np.array(CONDITIONS, dtype=object),
        "layers": np.array(LAYERS, dtype=np.int32),
        "mean_pool": mean_pool,
        "length": np.int32(protein.length),
        "category": str(protein.category),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N eligible proteins (smoke test).")
    ap.add_argument("--min-length", type=int, default=30)
    ap.add_argument("--max-length", type=int, default=800)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    proteins = list(iter_pilot_proteins(
        PILOT_DIR,
        require_structure=True,
        min_length=args.min_length,
        max_length=args.max_length,
    ))
    if args.limit is not None:
        proteins = proteins[:args.limit]
    print(f"eligible proteins = {len(proteins)}")

    # Skip already-harvested
    todo = [p for p in proteins if not (OUT_DIR / f"{p.accession}.npz").exists()]
    print(f"already cached    = {len(proteins) - len(todo)}")
    print(f"to harvest        = {len(todo)}")

    if not todo:
        print("Nothing to do.")
        _write_manifest(proteins)
        return

    print("Loading ESM3 ...")
    model, tokenizers = load_esm3(device=device)
    structure_encoder = model.get_structure_encoder()
    print(f"model loaded; running {len(CONDITIONS)} conditions × "
          f"{len(LAYERS)} layers on {len(todo)} proteins")

    t0 = time.time()
    errors: dict[str, str] = {}
    pbar = tqdm(todo, desc="harvest")
    for protein in pbar:
        try:
            payload = harvest_protein(
                protein, model, tokenizers, structure_encoder, device
            )
        except Exception as exc:  # noqa: BLE001
            errors[protein.accession] = f"{type(exc).__name__}: {exc}"
            pbar.write(f"  ! skip {protein.accession}: {errors[protein.accession]}")
            continue
        np.savez_compressed(OUT_DIR / f"{protein.accession}.npz", **payload)

    elapsed = time.time() - t0
    n_ok = len(todo) - len(errors)
    print(f"\nharvest complete: {n_ok} ok, {len(errors)} failed, "
          f"{elapsed/60:.1f} min total ({elapsed/max(n_ok,1):.2f}s / protein)")

    _write_manifest(proteins, errors=errors, elapsed_sec=elapsed)


def _write_manifest(proteins, errors: dict | None = None, elapsed_sec: float | None = None) -> None:
    manifest = {
        "n_proteins": len(proteins),
        "accessions": [p.accession for p in proteins],
        "conditions": list(CONDITIONS),
        "layers": list(LAYERS),
        "d_model": 1536,
        "errors": errors or {},
        "elapsed_sec": elapsed_sec,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
