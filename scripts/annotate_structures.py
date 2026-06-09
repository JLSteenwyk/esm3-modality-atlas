"""Compute SS8 + SASA for every fetched structure, uniformly.

Uses ESM's own ProteinChain so the annotations are exactly what ESM3 expects:
  · SS8  — DSSP 8-class (mkdssp, via biotite); out-of-vocab symbols left as-is
           and normalised to 'C' at tokenisation time by the modality driver.
  · SASA — per-residue solvent-accessible area in Å² (biotite Shrake-Rupley).

Requires `mkdssp` on PATH (we install it in an isolated env; append its bin):
  PATH="$PATH:$HOME/anaconda3/envs/dssp/bin" python scripts/annotate_structures.py [N]

Writes data/scaled/annotations/structure_annotations.json:
  { "<ACC>": {"ss8": "...", "sasa": [..], "length": L}, ... }
and a logs/annotate_summary.json with coverage + any failures.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from esm.utils.structure.protein_chain import ProteinChain

ROOT = Path(__file__).resolve().parent.parent
STRUCT_DIR = ROOT / "data" / "scaled" / "structures"
OUT = ROOT / "data" / "scaled" / "annotations" / "structure_annotations.json"
LOG = ROOT / "data" / "scaled" / "logs" / "annotate_summary.json"


def annotate_one(pdb_path: Path) -> dict:
    pc = ProteinChain.from_pdb(str(pdb_path))
    ss8 = "".join(map(str, pc.dssp()))
    sasa = [round(float(x), 3) for x in pc.sasa()]
    seq = pc.sequence
    if not (len(ss8) == len(sasa) == len(seq)):
        raise ValueError(f"length mismatch ss8={len(ss8)} sasa={len(sasa)} seq={len(seq)}")
    return {"ss8": ss8, "sasa": sasa, "length": len(seq)}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    pdbs = sorted(STRUCT_DIR.glob("*.pdb"))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(pdbs)
    pdbs = pdbs[:limit]
    print(f"annotating {len(pdbs)} structures")

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    failed: list[dict] = []
    t0 = time.time()
    for i, p in enumerate(pdbs, 1):
        acc = p.stem
        if acc in out:
            continue
        try:
            out[acc] = annotate_one(p)
        except Exception as e:  # noqa: BLE001
            failed.append({"accession": acc, "error": f"{type(e).__name__}: {e}"})
        if i % 50 == 0 or i == len(pdbs):
            print(f"  {i}/{len(pdbs)}  ok={len(out)} failed={len(failed)} "
                  f"({time.time() - t0:.0f}s)")

    OUT.write_text(json.dumps(out))
    summary = {"n_structures": len(pdbs), "n_annotated": len(out),
               "n_failed": len(failed), "failed": failed,
               "elapsed_sec": time.time() - t0}
    LOG.write_text(json.dumps(summary, indent=2))
    print(f"\nannotated {len(out)} -> {OUT.relative_to(ROOT)}")
    if failed:
        print(f"{len(failed)} failed (see {LOG.relative_to(ROOT)})")


if __name__ == "__main__":
    main()
