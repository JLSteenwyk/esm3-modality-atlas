"""Run ESM-3 in single-modality input conditions.

ESM-3 fuses up to six discrete input modalities — sequence, structure tokens,
SS8, SASA, function, residue_annotation — into one residual stream via additive
embedding. Its ``forward`` auto-fills mask/pad tokens for any modality you pass
as ``None``. So a "structure-only" forward pass is just: build structure tokens,
hand them to the model, let everything else default to masks.

This module wraps that pattern:

  * ``tokenize_protein`` precomputes tokens for whichever modalities are
    available for a given protein (sequence + DSSP secondary structure + SASA +
    optional structure coordinates + optional GO/InterPro annotations).
  * ``run_modality_condition`` runs ESM-3 in one of the six conditions and
    returns the cached residual stream at the requested layers.

We intentionally start with five "primary" single-modality conditions plus an
``all`` baseline. Residue-annotation tokens require an annotation source we
don't yet have wired up; deferred for a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from esm.utils.encoding import (
    tokenize_function_annotations,
    tokenize_sasa,
    tokenize_secondary_structure,
    tokenize_sequence,
    tokenize_structure,
)
from esm.utils.types import FunctionAnnotation

from .esm3_hooks import ESM3ActivationCache, ESM3HookManager

# The six conditions we extract activations for. Each condition is a single
# forward pass through ESM-3 with the named modality (or all of them) supplied
# as real tokens and everything else left to ESM-3's mask/pad defaults.
MODALITY_CONDITIONS: tuple[str, ...] = (
    "sequence",
    "structure",
    "ss8",
    "sasa",
    "function",
    "residue_annotation",
    "all",
    # leave-one-out ablations: every modality of "all" except the named one. Used
    # to measure each modality's contribution to the fused representation, by
    # comparing "all" against "all" with that modality withheld.
    "all_no_sequence",
    "all_no_structure",
    "all_no_ss8",
    "all_no_sasa",
    "all_no_function",
)

# ESM-3's SS8 vocabulary is {G, H, I, T, E, B, S, C}. DSSP can additionally emit
# 'P' (polyproline II) and '-'/' ' for undefined residues. We collapse anything
# out-of-vocab to 'C' (coil), the standard biophysical fallback.
_ESM3_SS8_VOCAB = frozenset("GHITEBSC")


def _normalize_ss8(ss8: str) -> str:
    return "".join(c if c in _ESM3_SS8_VOCAB else "C" for c in ss8)


@dataclass
class ProteinInputs:
    """Pre-tokenized inputs for a single protein, ready to feed to ESM-3.

    Each tensor (if present) has shape (1, L+2) with BOS/EOS special tokens
    added — except ``function_tokens`` which is (1, L+2, 8). Any field set to
    ``None`` means that modality is unavailable for this protein and the
    corresponding single-modality condition will be skipped.
    """

    sequence_id: str
    length: int  # residue count, excluding special tokens
    sequence_tokens: torch.Tensor
    structure_tokens: Optional[torch.Tensor] = None
    ss8_tokens: Optional[torch.Tensor] = None
    sasa_tokens: Optional[torch.Tensor] = None
    function_tokens: Optional[torch.Tensor] = None
    residue_annotation_tokens: Optional[torch.Tensor] = None

    def available_conditions(self) -> list[str]:
        out = ["sequence"]
        if self.structure_tokens is not None:
            out.append("structure")
        if self.ss8_tokens is not None:
            out.append("ss8")
        if self.sasa_tokens is not None:
            out.append("sasa")
        if self.function_tokens is not None:
            out.append("function")
        if self.residue_annotation_tokens is not None:
            out.append("residue_annotation")
        # "all" requires sequence (always present) and at least one other modality
        if len(out) > 1:
            out.append("all")
            # leave-one-out conditions: "all" with one present modality withheld,
            # available as long as withholding it leaves at least one modality
            present = [m for m in ("sequence", "structure", "ss8", "sasa",
                                   "function") if m in out]
            if len(present) > 1:
                out.extend(f"all_no_{m}" for m in present)
        return out


def tokenize_protein(
    sequence_id: str,
    sequence: str,
    tokenizers: object,
    *,
    structure_encoder: Optional[object] = None,
    coordinates: Optional[torch.Tensor] = None,
    secondary_structure: Optional[str] = None,
    sasa: Optional[Sequence[Optional[float]]] = None,
    function_annotations: Optional[Sequence[FunctionAnnotation]] = None,
    residue_annotation_sample: Optional[dict] = None,
    device: str = "cuda",
) -> ProteinInputs:
    """Tokenize as many modalities as we have data for.

    Args:
        sequence_id: stable identifier, used downstream for caching.
        sequence: amino acid string (canonical 20 + 'X' tolerated).
        tokenizers: collection returned by ESM3.tokenizers.
        structure_encoder: ESM3's VQ structure encoder (model.get_structure_encoder()).
            Required to produce structure_tokens from coordinates.
        coordinates: (L, 37, 3) or (L, 14, 3) or (L, 3, 3) backbone-only.
        secondary_structure: DSSP 8-class string of length L (e.g. "HHHHHEEE...").
        sasa: per-residue SASA in Å² (length L), missing values as None.
        function_annotations: GO/InterPro annotation ranges.
        device: target torch device.

    Returns:
        ProteinInputs with tokens populated for whichever modalities were
        supplied. The sequence tokens are always present.
    """
    seq_tok = tokenize_sequence(sequence, tokenizers.sequence, add_special_tokens=True)
    seq_tok = seq_tok.to(device).unsqueeze(0)  # (1, L+2)
    length = len(sequence)

    inputs = ProteinInputs(
        sequence_id=sequence_id,
        length=length,
        sequence_tokens=seq_tok,
    )

    if coordinates is not None and structure_encoder is not None:
        coords = coordinates.to(device)
        # tokenize_structure returns (coordinates, plddt, structure_tokens)
        _coords, _plddt, struct_tok = tokenize_structure(
            coords,
            structure_encoder,
            tokenizers.structure,
            reference_sequence=sequence,
            add_special_tokens=True,
        )
        inputs.structure_tokens = struct_tok.to(device).unsqueeze(0)

    if secondary_structure is not None:
        ss_tok = tokenize_secondary_structure(
            _normalize_ss8(secondary_structure),
            tokenizers.secondary_structure,
            add_special_tokens=True,
        )
        inputs.ss8_tokens = ss_tok.to(device).unsqueeze(0)

    if sasa is not None:
        sasa_tok = tokenize_sasa(sasa, tokenizers.sasa, add_special_tokens=True)
        inputs.sasa_tokens = sasa_tok.to(device).unsqueeze(0)

    if function_annotations is not None and len(function_annotations) > 0:
        func_tok, _res_tok = tokenize_function_annotations(
            function_annotations,
            reference_sequence=sequence,
            function_tokenizer=tokenizers.function,
            residue_annotation_tokenizer=tokenizers.residue_annotations,
            add_special_tokens=True,
        )
        inputs.function_tokens = func_tok.to(device).unsqueeze(0)

    if residue_annotation_sample is not None:
        ra_tok_str = tokenizers.residue_annotations.tokenize(
            residue_annotation_sample, sequence)
        ra_tok = tokenizers.residue_annotations.encode(
            ra_tok_str, add_special_tokens=True)  # (L+2, max_annotations)
        # keep only if it carries real annotations: any <unk> (2) or <ra:..> (>=4)
        # token, i.e. not all <pad>(0)/<none>(3).
        annotated = ((ra_tok == 2) | (ra_tok >= 4)).sum().item()
        if annotated > 0:
            inputs.residue_annotation_tokens = ra_tok.to(device).unsqueeze(0)

    return inputs


def _build_forward_kwargs(inputs: ProteinInputs, condition: str) -> dict:
    """Pick which modality tokens to forward to ESM3, given a condition.

    Anything we don't pass gets defaulted to mask/pad tokens by ESM3.forward.
    """
    if condition == "sequence":
        return {"sequence_tokens": inputs.sequence_tokens}
    if condition == "structure":
        if inputs.structure_tokens is None:
            raise ValueError(f"{inputs.sequence_id}: no structure tokens")
        # ESM3 needs L from at least one input; structure_tokens carries it.
        return {"structure_tokens": inputs.structure_tokens}
    if condition == "ss8":
        if inputs.ss8_tokens is None:
            raise ValueError(f"{inputs.sequence_id}: no ss8 tokens")
        return {"ss8_tokens": inputs.ss8_tokens}
    if condition == "sasa":
        if inputs.sasa_tokens is None:
            raise ValueError(f"{inputs.sequence_id}: no sasa tokens")
        return {"sasa_tokens": inputs.sasa_tokens}
    if condition == "function":
        if inputs.function_tokens is None:
            raise ValueError(f"{inputs.sequence_id}: no function tokens")
        return {"function_tokens": inputs.function_tokens}
    if condition == "residue_annotation":
        if inputs.residue_annotation_tokens is None:
            raise ValueError(f"{inputs.sequence_id}: no residue_annotation tokens")
        return {"residue_annotation_tokens": inputs.residue_annotation_tokens}
    if condition == "all" or condition.startswith("all_no_"):
        drop = condition[len("all_no_"):] if condition.startswith("all_no_") else ""
        tok = {
            "sequence": inputs.sequence_tokens,
            "structure": inputs.structure_tokens,
            "ss8": inputs.ss8_tokens,
            "sasa": inputs.sasa_tokens,
            "function": inputs.function_tokens,
        }
        kw = {f"{m}_tokens": t for m, t in tok.items()
              if t is not None and m != drop}
        if not kw:
            raise ValueError(f"{inputs.sequence_id}: no tokens left for {condition!r}")
        return kw
    raise ValueError(f"Unknown condition: {condition!r}")


@torch.no_grad()
def run_modality_condition(
    model: torch.nn.Module,
    inputs: ProteinInputs,
    condition: str,
    *,
    layers: Sequence[int],
    autocast_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> ESM3ActivationCache:
    """Run one forward pass under the given modality condition.

    Returns a cache mapping ``layer_idx -> (L+2, 1536)`` residual-stream
    activations in float32, with the batch dimension squeezed out.
    """
    if condition not in MODALITY_CONDITIONS:
        raise ValueError(f"condition must be one of {MODALITY_CONDITIONS}; got {condition!r}")

    kwargs = _build_forward_kwargs(inputs, condition)

    hm = ESM3HookManager(model, layers=list(layers))
    with hm:
        with torch.autocast(device_type=device.split(":")[0], dtype=autocast_dtype):
            model(**kwargs)

    out = ESM3ActivationCache()
    for layer_idx, tensor in hm.cache.residual_stream.items():
        # (1, L+2, 1536) -> (L+2, 1536), float32 for downstream PCA
        out.residual_stream[layer_idx] = tensor[0].float()
    return out
