"""Color palette and legend taxonomy for ESM-3 modality conditions.

Grouped into three semantic categories so the legend reads as a taxonomy:

  INPUT MODALITY   — primary, model-native token streams (sequence, structure)
  DERIVED MODALITY — geometric annotations derived from structure (SS8, SASA)
  FUSED            — all modalities supplied jointly
"""

# Canonical condition order. Drives the reveal sequence in animations
# (clusters appear left-to-right in this order).
CONDITIONS: tuple[str, ...] = (
    "sequence",
    "structure",
    "ss8",
    "sasa",
    "all",
)

CONDITION_LABEL: dict[str, str] = {
    "sequence":  "Sequence",
    "structure": "Structure",
    "ss8":       "SS8",
    "sasa":      "SASA",
    "all":       "All modalities",
}

CONDITION_COLOR: dict[str, str] = {
    "sequence":  "#2563eb",  # blue — canonical amino acid sequence
    "structure": "#16a34a",  # green — 3D structure tokens
    "ss8":       "#84cc16",  # lime — secondary structure
    "sasa":      "#06b6d4",  # cyan — solvent accessibility
    "all":       "#1e293b",  # ink — multimodal fusion
}

CATEGORY_GROUPS: list[tuple[str, list[str]]] = [
    ("INPUT MODALITY",   ["sequence", "structure"]),
    ("DERIVED MODALITY", ["ss8", "sasa"]),
    ("FUSED",            ["all"]),
]

CATEGORY_COLOR: dict[str, str] = {
    "INPUT MODALITY":   "#1d4ed8",   # deep blue
    "DERIVED MODALITY": "#15803d",   # deep green
    "FUSED":            "#0f172a",   # ink
}

INK = "#0f172a"
