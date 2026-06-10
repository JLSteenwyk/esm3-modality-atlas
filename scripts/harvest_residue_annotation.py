"""Harvest the residue_annotation condition (the granularity control).

The function modality stays orthogonal at every layer, but it is the only
WHOLE-PROTEIN modality. This harvests ESM3's per-residue functional-annotation
modality, driven by real InterPro residue-site annotations (fetched via
fetch_residue_annotations.py), so we can ask whether per-residue functional
information FUSES (like the physical modalities) or stays orthogonal like the
whole-protein function track.

Only the residue_annotation condition is run (it carries L via its own tokens,
so no structure/coords are needed). Mean-pooled, all 48 layers, on every protein
whose InterPro sites produce a non-degenerate annotation signal.

Output: activations/scaled/residue_annotation/{acc}.npz
        mean_pool (48, 1536) f32, length, category
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

from src.models import load_esm3, run_modality_condition, tokenize_protein  # noqa: E402

SITES = ROOT / "data" / "scaled" / "annotations" / "residue_sites.json"
META = ROOT / "data" / "pilot" / "annotations" / "metadata.json"
OUT_DIR = ROOT / "activations" / "scaled" / "residue_annotation"
LAYERS = list(range(48))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sites = json.loads(SITES.read_text())
    meta = json.loads(META.read_text())
    accs = [a for a in sorted(sites) if a in meta]
    print(f"{len(accs)} proteins with InterPro residue sites")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizers = load_esm3(device=device)

    done, skipped, errors = 0, 0, {}
    t0 = time.time()
    for acc in tqdm(accs, desc="residue_annotation"):
        dst = OUT_DIR / f"{acc}.npz"
        if dst.exists():
            done += 1
            continue
        try:
            s = sites[acc]
            sample = {
                "interpro_site_descriptions": s["descriptions"],
                "interpro_site_starts": s["starts"],
                "interpro_site_ends": s["ends"],
                "interpro_site_residues": s["residues"],
            }
            seq = meta[acc]["sequence"]
            inputs = tokenize_protein(
                sequence_id=acc, sequence=seq, tokenizers=tokenizers,
                residue_annotation_sample=sample, device=device)
            if "residue_annotation" not in inputs.available_conditions():
                skipped += 1   # all sites mismatched / out of vocab -> degenerate
                continue
            cache = run_modality_condition(
                model, inputs, "residue_annotation", layers=LAYERS, device=device)
            mean_pool = np.zeros((len(LAYERS), 1536), dtype=np.float32)
            for li, L in enumerate(LAYERS):
                h = cache.residual_stream[L]
                mean_pool[li] = h[1:-1].mean(dim=0).cpu().numpy().astype(np.float32)
            np.savez_compressed(
                dst, mean_pool=mean_pool, length=np.int32(len(seq)),
                category=str(meta[acc].get("category", "unknown")))
            done += 1
        except Exception as e:  # noqa: BLE001
            errors[acc] = f"{type(e).__name__}: {e}"

    print(f"\ndone {done} | skipped(degenerate) {skipped} | errors {len(errors)} "
          f"({time.time() - t0:.0f}s)")
    if errors:
        print("first errors:", dict(list(errors.items())[:3]))
    (OUT_DIR / "_manifest.json").write_text(json.dumps(
        {"n_done": done, "n_skipped": skipped, "n_errors": len(errors),
         "errors": errors, "layers": LAYERS}, indent=2))


if __name__ == "__main__":
    main()
