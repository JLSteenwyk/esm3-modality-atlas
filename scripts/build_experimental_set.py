"""Build a replication set backed by experimental (X-ray/cryo-EM) structures.

For canonical scaled proteins that carry experimental PDB entries, fetch the
resolved chain from the RCSB, keep only fully modelled chains whose sequence is a
fragment of the UniProt sequence (so the parent InterPro annotations still apply),
and write them as a self-contained dataset under data/experimental/ in the same
layout the AlphaFold pipeline uses. SS8/SASA are then produced by the unchanged
annotate_structures.py, and activations by the unchanged harvest_scaled.py, so the
only thing that differs from the main atlas is the source of the 3D coordinates.

Output: data/experimental/{structures/*.pdb, metadata.json, accessions.txt}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

META = ROOT / "data" / "pilot" / "annotations" / "metadata.json"
SCALED = ROOT / "activations" / "scaled" / "per_protein"
OUTDIR = ROOT / "data" / "experimental"


def best_identity(chain: str, full: str) -> float:
    """Max per-residue identity of `chain` slid against `full` (chain <= full)."""
    if chain in full:
        return 1.0
    if len(chain) > len(full):
        chain, full = full, chain
    c = np.frombuffer(chain.encode(), dtype=np.uint8)
    f = np.frombuffer(full.encode(), dtype=np.uint8)
    best = 0.0
    for off in range(len(f) - len(c) + 1):
        best = max(best, float((c == f[off:off + len(c)]).mean()))
        if best == 1.0:
            break
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=180,
                    help="stop once this many structures are written")
    ap.add_argument("--scan-limit", type=int, default=600,
                    help="max canonical proteins to scan")
    args = ap.parse_args()

    from esm.utils.structure.protein_chain import ProteinChain

    meta = json.loads(META.read_text())
    canon = sorted(p.stem for p in SCALED.glob("*.npz"))
    (OUTDIR / "structures").mkdir(parents=True, exist_ok=True)

    out_meta: dict = {}
    existing = {p.stem for p in (OUTDIR / "structures").glob("*.pdb")}
    if (OUTDIR / "metadata.json").exists():
        out_meta = json.loads((OUTDIR / "metadata.json").read_text())

    scanned = 0
    for acc in canon:
        if len(out_meta) >= args.target:
            break
        if acc in existing:
            continue
        if scanned >= args.scan_limit:
            break
        scanned += 1
        uni = meta[acc]["sequence"]
        ipr = meta[acc].get("interpro") or []
        if not ipr:
            continue
        for pdb_id in (meta[acc].get("pdb_ids") or [])[:4]:
            try:
                pc = ProteinChain.from_rcsb(pdb_id, "detect")
            except Exception:
                continue
            seq = pc.sequence
            ca = pc.atom37_positions[:, 1, :]
            n_gap = int(np.isnan(ca).any(axis=1).sum())
            if not (50 <= len(seq) <= 800) or n_gap > 0:
                continue
            if best_identity(seq, uni) < 0.95:
                continue
            try:
                pc.to_pdb(str(OUTDIR / "structures" / f"{acc}.pdb"))
            except Exception:
                continue
            out_meta[acc] = {
                "accession": acc, "sequence": seq, "length": len(seq),
                "category": meta[acc].get("category", "unknown"),
                "interpro": ipr, "pdb_ids": meta[acc].get("pdb_ids"),
                "source_pdb": pdb_id,
            }
            print(f"  {acc}  <- {pdb_id}  len {len(seq)}  ({len(out_meta)})")
            break

    (OUTDIR / "metadata.json").write_text(json.dumps(out_meta, indent=2))
    (OUTDIR / "accessions.txt").write_text("\n".join(sorted(out_meta)) + "\n")
    print(f"\nwrote {len(out_meta)} experimental structures "
          f"(scanned {scanned}) -> {OUTDIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
