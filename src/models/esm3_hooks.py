"""Activation extraction hooks for ESM-3 (esm3-sm-open-v1, 1.4B).

ESM-3 architecture:
    ESM3
    ├── encoder (EncodeInputs) — additive multimodal embedding
    ├── transformer (TransformerStack)
    │   ├── blocks (ModuleList of UnifiedTransformerBlock × 48)
    │   └── norm (LayerNorm)
    └── output_heads

d_model=1536, n_heads=24, n_layers=48, RoPE.

Lifted from INTERPRETABILITY/src/models/esm3_hooks.py and trimmed to
the residual-stream extraction path we need here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ESM3ActivationCache:
    """Stores extracted residual-stream activations from an ESM-3 forward pass."""

    residual_stream: dict[int, torch.Tensor] = field(default_factory=dict)
    model_name: str = "esm3"

    def clear(self) -> None:
        self.residual_stream.clear()


class ESM3HookManager:
    """Register forward hooks on a subset of ESM-3 transformer blocks.

    Usage:
        model, tokenizers = load_esm3()
        with ESM3HookManager(model, layers=[0, 8, 16, 24, 32, 40, 47]) as hm:
            model(sequence_tokens=tokens)
            acts = hm.cache.residual_stream  # {layer_idx: (B, L, 1536)}
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[list[int]] = None,
    ):
        self.model = model
        self.cache = ESM3ActivationCache()
        self._hooks: list = []
        self._blocks = self._find_blocks()
        self.num_layers = len(self._blocks)
        if layers is None:
            self.layers = list(range(self.num_layers))
        else:
            self.layers = [l if l >= 0 else self.num_layers + l for l in layers]

    def _find_blocks(self) -> nn.ModuleList:
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "blocks"):
            return self.model.transformer.blocks
        raise ValueError(
            "ESM3 transformer blocks not found. "
            f"Top-level attrs: {[n for n, _ in self.model.named_children()]}"
        )

    def _make_residual_hook(self, layer_idx: int):
        def hook_fn(module, _input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.cache.residual_stream[layer_idx] = hidden.detach().to("cpu")
        return hook_fn

    def register(self) -> None:
        self.remove()
        self.cache.clear()
        for layer_idx in self.layers:
            if layer_idx >= self.num_layers:
                raise IndexError(
                    f"Layer {layer_idx} requested but model has {self.num_layers} layers"
                )
            block = self._blocks[layer_idx]
            self._hooks.append(block.register_forward_hook(self._make_residual_hook(layer_idx)))

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self) -> "ESM3HookManager":
        self.register()
        return self

    def __exit__(self, *_args) -> None:
        self.remove()


def load_esm3(
    model_name: str = "esm3_sm_open_v1",
    device: str = "cuda",
) -> tuple[nn.Module, object]:
    """Load esm3-sm-open-v1 and return (model, tokenizer_collection).

    The tokenizer collection exposes .sequence, .structure, .secondary_structure,
    .sasa, .function, .residue_annotations.
    """
    from esm.models.esm3 import ESM3

    model = ESM3.from_pretrained(model_name).to(device)
    model.eval()
    return model, model.tokenizers
