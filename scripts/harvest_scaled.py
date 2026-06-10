"""Scaled, config-driven harvest of ESM3 residual-stream activations.

The publication-scale version of harvest_activations.py:
  · reads config/harvest.json (dataset paths, conditions, layers, pooling)
  · 6 conditions incl. the function modality (InterPro whole-protein annotations)
  · all 48 layers, mean-pooled (cheap, the core atlas)
  · per-residue activations for a reduced subset (config), fp16, to validate
    that fusion holds below the mean-pool

Outputs (under activations/scaled/):
  per_protein/{acc}.npz   conditions, layers, mean_pool (C,48,1536) f32, length, category
  per_residue/{acc}.npz   conditions, layers (subset), residue (C,Kpr,L,1536) f16   [subset only]
  manifest.json

Resumable: skips proteins whose per_protein NPZ already exists.

Usage:
    python scripts/harvest_scaled.py [--config config/harvest.json] [--limit N]
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

from esm.utils.types import FunctionAnnotation  # noqa: E402

from src.data.pilot_loader import iter_scaled_proteins  # noqa: E402
from src.models import load_esm3, run_modality_condition, tokenize_protein  # noqa: E402


def _coords_from_pdb(pdb_path: Path) -> torch.Tensor:
    from esm.utils.structure.protein_chain import ProteinChain

    chain = ProteinChain.from_pdb(str(pdb_path))
    atom37 = torch.from_numpy(chain.atom37_positions).float()
    return atom37[:, :3, :]  # (L, 3, 3) backbone N, CA, C


@torch.no_grad()
def randomize_trunk(model) -> tuple[int, int]:
    """Reinitialise every weight EXCEPT the structure VQ encoder.

    The random-init control: inputs stay meaningful (trained tokenizers +
    structure encoder still produce the same token streams), but the modality
    embeddings, transformer trunk and heads are untrained. If the trained model
    fuses modalities and this one doesn't, fusion is a learned property rather
    than an artifact of the architecture summing modality embeddings.
    """
    import torch.nn as nn

    se_param_ids = {id(p) for p in model._structure_encoder.parameters()}
    n_reset = 0
    for name, m in model.named_modules():
        if name == "_structure_encoder" or name.startswith("_structure_encoder."):
            continue
        own = list(m.parameters(recurse=False))
        if not own or any(id(p) in se_param_ids for p in own):
            continue
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
            n_reset += 1
    # param-bearing modules without reset_parameters (e.g. geometric attention)
    n_manual = 0
    for name, m in model.named_modules():
        if name.startswith("_structure_encoder") or hasattr(m, "reset_parameters"):
            continue
        for p in m.parameters(recurse=False):
            if id(p) in se_param_ids:
                continue
            nn.init.normal_(p, 0.0, 0.02) if p.dim() >= 2 else p.zero_()
            n_manual += 1
    return n_reset, n_manual


def harvest_protein(protein, model, tokenizers, structure_encoder, device,
                    conditions, layers, pr_cfg, want_per_residue, known_ipr):
    coords = _coords_from_pdb(protein.pdb_path)
    if coords.shape[0] != protein.length:
        raise ValueError(
            f"{protein.accession}: PDB length {coords.shape[0]} != seq length {protein.length}")

    # Keep only InterPro IDs in ESM3's function vocabulary (newer entries 404 the
    # tokenizer); proteins with none were filtered out upstream.
    valid_ipr = [ipr for ipr in protein.interpro_ids if ipr in known_ipr]
    func_anns = [FunctionAnnotation(label=ipr, start=1, end=protein.length)
                 for ipr in valid_ipr] or None

    inputs = tokenize_protein(
        sequence_id=protein.accession,
        sequence=protein.sequence,
        tokenizers=tokenizers,
        structure_encoder=structure_encoder,
        coordinates=coords,
        secondary_structure=protein.ss8,
        sasa=protein.sasa,
        function_annotations=func_anns,
        device=device,
    )
    available = set(inputs.available_conditions())
    missing = [c for c in conditions if c not in available]
    if missing:
        raise ValueError(f"{protein.accession}: missing conditions {missing}")

    C, K, L = len(conditions), len(layers), protein.length
    mean_pool = np.zeros((C, K, 1536), dtype=np.float32)
    pr_layers = pr_cfg["layers"] if want_per_residue else []
    per_res = (np.zeros((C, len(pr_layers), L, 1536), dtype=np.float16)
               if want_per_residue else None)

    for ci, cond in enumerate(conditions):
        cache = run_modality_condition(model, inputs, cond, layers=layers, device=device)
        for li, layer_idx in enumerate(layers):
            hidden = cache.residual_stream[layer_idx]  # (L+2, 1536)
            mean_pool[ci, li] = hidden[1:-1].mean(dim=0).cpu().numpy().astype(np.float32)
        if want_per_residue:
            for pj, layer_idx in enumerate(pr_layers):
                per_res[ci, pj] = cache.residual_stream[layer_idx][1:-1].cpu().numpy().astype(np.float16)

    out = {
        "conditions": np.array(conditions, dtype=object),
        "layers": np.array(layers, dtype=np.int32),
        "mean_pool": mean_pool,
        "length": np.int32(L),
        "category": str(protein.category),
    }
    pr_out = None
    if want_per_residue:
        pr_out = {
            "conditions": np.array(conditions, dtype=object),
            "layers": np.array(pr_layers, dtype=np.int32),
            "residue": per_res,
            "length": np.int32(L),
        }
    return out, pr_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/harvest.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--random-init", action="store_true",
                    help="randomize the trunk (control); writes to "
                         "activations/scaled_randinit, no per-residue")
    args = ap.parse_args()

    cfg = json.loads((ROOT / args.config).read_text())
    ds = cfg["dataset"]
    conditions = list(cfg["conditions"])
    layers = list(cfg["layers"])
    pr_cfg = cfg["pooling"]["per_residue"]
    do_per_residue = bool(pr_cfg.get("enabled")) and not args.random_init

    # output base follows the dataset (data/scaled -> activations/scaled,
    # data/diverse -> activations/diverse), with a _randinit suffix for the control
    out_base = Path(ds["root"]).name
    if args.random_init:
        out_base += "_randinit"
    out_dir = ROOT / "activations" / out_base / "per_protein"
    pr_dir = ROOT / "activations" / out_base / "per_residue"
    out_dir.mkdir(parents=True, exist_ok=True)
    if do_per_residue:
        pr_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    proteins = list(iter_scaled_proteins(ROOT / ds["root"], ROOT / ds["metadata"]))
    if args.limit:
        proteins = proteins[:args.limit]
    print(f"{len(proteins)} proteins | {len(conditions)} conditions | {len(layers)} layers")
    print(f"per-residue: {do_per_residue} (first {pr_cfg.get('max_proteins')} proteins, "
          f"layers {pr_cfg.get('layers')})")

    model, tokenizers = load_esm3(device=device)
    structure_encoder = model.get_structure_encoder()
    if args.random_init:
        n_reset, n_manual = randomize_trunk(model)
        model.eval()
        print(f"RANDOM-INIT control: reinitialised {n_reset} modules + "
              f"{n_manual} extra params (structure encoder preserved)")

    # Filter to proteins that support the function modality (>=1 InterPro ID in
    # ESM3's vocab) so every harvested protein supports all conditions uniformly.
    known_ipr = set(tokenizers.function.interpro_to_index.keys())
    need_function = "function" in conditions or "all" in conditions
    if need_function:
        before = len(proteins)
        proteins = [p for p in proteins
                    if any(i in known_ipr for i in p.interpro_ids)]
        print(f"function modality: kept {len(proteins)}/{before} proteins with "
              f">=1 known InterPro ({before - len(proteins)} dropped)")

    errors: dict[str, str] = {}
    done = 0
    t0 = time.time()
    for idx, protein in enumerate(tqdm(proteins, desc="harvest")):
        dst = out_dir / f"{protein.accession}.npz"
        if dst.exists():
            done += 1
            continue
        want_pr = do_per_residue and idx < int(pr_cfg.get("max_proteins", 0))
        try:
            rec, pr_rec = harvest_protein(
                protein, model, tokenizers, structure_encoder, device,
                conditions, layers, pr_cfg, want_pr, known_ipr)
            np.savez_compressed(dst, **rec)
            if pr_rec is not None:
                np.savez_compressed(pr_dir / f"{protein.accession}.npz", **pr_rec)
            done += 1
        except Exception as e:  # noqa: BLE001
            errors[protein.accession] = f"{type(e).__name__}: {e}"

    manifest = {
        "n_proteins": done,
        "n_requested": len(proteins),
        "conditions": conditions,
        "layers": layers,
        "d_model": 1536,
        "per_residue": {"enabled": do_per_residue, **pr_cfg},
        "n_errors": len(errors),
        "errors": errors,
        "elapsed_sec": time.time() - t0,
    }
    (ROOT / "activations" / out_base / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\ndone: {done}/{len(proteins)}  errors: {len(errors)}  "
          f"({time.time() - t0:.0f}s)")
    if errors:
        print("first errors:", dict(list(errors.items())[:3]))


if __name__ == "__main__":
    main()
