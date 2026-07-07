"""Leave-one-out modality ablation harvest.

For each protein, run ESM3 under five conditions, each presenting every modality
of the "all" condition except one withheld modality. Comparing the resulting
representation against the existing "all" representation (in
activations/scaled/per_protein) isolates the contribution of each modality to the
fused residual stream.

The test of interest: withholding function should perturb the representation far
less than withholding a physical modality, and should move it along an axis
orthogonal to the physical manifold rather than within it.

Restricted to the canonical 892 proteins already harvested under the scaled atlas.
Resumable. Mean-pool only, all 48 layers.

Output: activations/ablation/per_protein/{acc}.npz (conditions, layers, mean_pool)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.protein_loader import iter_scaled_proteins  # noqa: E402
from src.models import load_esm3  # noqa: E402

from harvest_scaled import harvest_protein  # noqa: E402

CONDITIONS = ["all_no_sequence", "all_no_structure", "all_no_ss8",
              "all_no_sasa", "all_no_function"]
SCALED = ROOT / "activations" / "scaled" / "per_protein"
OUT = ROOT / "activations" / "ablation" / "per_protein"


def main() -> None:
    cfg = json.loads((ROOT / "config" / "harvest.json").read_text())
    ds = cfg["dataset"]
    layers = list(cfg["layers"])
    pr_cfg = cfg["pooling"]["per_residue"]   # passed through but per-residue off
    OUT.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    canonical = {p.stem for p in SCALED.glob("*.npz")}
    print(f"device = {device} | {len(canonical)} canonical proteins | "
          f"{len(CONDITIONS)} conditions | {len(layers)} layers")

    proteins = list(iter_scaled_proteins(ROOT / ds["root"], ROOT / ds["metadata"],
                                         accessions=canonical))
    model, tokenizers = load_esm3(device=device)
    structure_encoder = model.get_structure_encoder()
    known_ipr = set(tokenizers.function.interpro_to_index.keys())

    errors: dict[str, str] = {}
    done = 0
    t0 = time.time()
    for protein in tqdm(proteins, desc="ablation"):
        dst = OUT / f"{protein.accession}.npz"
        if dst.exists():
            done += 1
            continue
        try:
            rec, _ = harvest_protein(
                protein, model, tokenizers, structure_encoder, device,
                CONDITIONS, layers, pr_cfg, False, known_ipr)
            np.savez_compressed(dst, **rec)
            done += 1
        except Exception as e:  # noqa: BLE001
            errors[protein.accession] = f"{type(e).__name__}: {e}"

    manifest = {
        "n_proteins": done, "n_requested": len(proteins),
        "conditions": CONDITIONS, "layers": layers,
        "reference": "activations/scaled/per_protein (condition 'all')",
        "n_errors": len(errors), "errors": errors,
        "elapsed_sec": time.time() - t0,
    }
    (ROOT / "activations" / "ablation" / "manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"\ndone: {done}/{len(proteins)}  errors: {len(errors)}  "
          f"({time.time() - t0:.0f}s)")
    if errors:
        for a, e in list(errors.items())[:5]:
            print(f"  {a}: {e}")


if __name__ == "__main__":
    main()
