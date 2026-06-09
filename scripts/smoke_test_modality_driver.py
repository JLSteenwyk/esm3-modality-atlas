"""End-to-end smoke test: load ESM3, tokenize one pilot protein with all
available modalities, and run each single-modality condition. Prints residual
stream shape and a quick L2-distance summary so we can eyeball that
conditions are producing distinct activations.

Run from the project root:
    python scripts/smoke_test_modality_driver.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import (  # noqa: E402
    MODALITY_CONDITIONS,
    load_esm3,
    run_modality_condition,
    tokenize_protein,
)


LAYERS = [0, 8, 16, 24, 32, 40, 47]


def _load_one_pilot_protein() -> tuple[str, str, dict | None, Path | None]:
    """Pick a protein from the pilot set that has DSSP + PDB on disk."""
    pilot = ROOT / "data" / "pilot"
    seqs_json = pilot / "sequences" / "sequences.json"
    dssp_json = pilot / "annotations" / "dssp_annotations.json"
    struct_dir = pilot / "structures"

    seqs = json.loads(seqs_json.read_text())
    dssp = json.loads(dssp_json.read_text())

    for sid, seq in seqs.items():
        pdb = struct_dir / f"{sid}.pdb"
        if pdb.exists() and sid in dssp and 30 < len(seq) < 400:
            return sid, seq, dssp[sid], pdb
    raise RuntimeError("No suitable pilot protein found")


def _coords_from_pdb(pdb_path: Path) -> torch.Tensor:
    """Load backbone Cα/N/C coordinates as a (L, 3, 3) tensor.

    ESM3's tokenize_structure accepts atom14/atom37/(L,3,3) — backbone-only is
    enough for the structure VQ encoder.
    """
    from esm.utils.structure.protein_chain import ProteinChain

    chain = ProteinChain.from_pdb(str(pdb_path))
    # atom37 → take N, CA, C (indices 0, 1, 2)
    atom37 = torch.from_numpy(chain.atom37_positions).float()
    return atom37[:, :3, :]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    sid, seq, dssp_entry, pdb_path = _load_one_pilot_protein()
    print(f"protein = {sid}  L={len(seq)}  pdb={pdb_path.name}")

    coords = _coords_from_pdb(pdb_path)
    print(f"coords shape = {tuple(coords.shape)}")

    # dssp_entry is a list of per-residue dicts with keys
    # {resnum, aa, ss8, ss3, asa, rsa}
    ss8 = "".join(r["ss8"] for r in dssp_entry)
    sasa = [r["asa"] for r in dssp_entry]
    if len(ss8) != len(seq):
        print(f"WARN ss8 length {len(ss8)} != seq length {len(seq)}; dropping ss8")
        ss8 = None
    if sasa is not None and len(sasa) != len(seq):
        print(f"WARN sasa length {len(sasa)} != seq length {len(seq)}; dropping sasa")
        sasa = None

    print("Loading ESM3 ...")
    model, tokenizers = load_esm3(device=device)
    structure_encoder = model.get_structure_encoder()
    print(f"layers in model = {len(model.transformer.blocks)}")

    inputs = tokenize_protein(
        sequence_id=sid,
        sequence=seq,
        tokenizers=tokenizers,
        structure_encoder=structure_encoder,
        coordinates=coords,
        secondary_structure=ss8,
        sasa=sasa,
        function_annotations=None,  # not wired yet
        device=device,
    )
    print(f"available conditions = {inputs.available_conditions()}")

    # Reference: sequence_only condition
    ref = run_modality_condition(
        model, inputs, "sequence", layers=LAYERS, device=device
    )
    print(f"\nresidual stream shape per layer = {tuple(ref.residual_stream[LAYERS[0]].shape)}")

    print("\nMean-pool L2 distance from sequence-only baseline:")
    print(f"  {'cond':<10} " + " ".join(f"L{l:>2}" for l in LAYERS))
    ref_means = {l: ref.residual_stream[l].mean(dim=0) for l in LAYERS}

    for cond in inputs.available_conditions():
        if cond == "sequence":
            continue
        acts = run_modality_condition(
            model, inputs, cond, layers=LAYERS, device=device
        )
        dists = []
        for l in LAYERS:
            v = acts.residual_stream[l].mean(dim=0)
            d = float(torch.linalg.vector_norm(v - ref_means[l]))
            dists.append(d)
        print(f"  {cond:<10} " + " ".join(f"{d:5.2f}" for d in dists))


if __name__ == "__main__":
    main()
