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
    "function",
    "all",
)

CONDITION_LABEL: dict[str, str] = {
    "sequence":  "Sequence",
    "structure": "Structure",
    "ss8":       "SS8",
    "sasa":      "SASA",
    "function":  "Function",
    "all":       "All modalities",
}

# pypubfigs `nickel_five` — the IBM colorblind-safe 5-colour palette — for the
# original five conditions; the scaled atlas adds the function modality, coloured
# with the Wong/ito green (#009E73, also a pypubfigs colour) so all six stay
# distinct and colorblind-safe. ss8/sasa (the two derived modalities that
# overlapped worst) keep the most-separated hues (purple/amber).
CONDITION_COLOR: dict[str, str] = {
    "sequence":  "#648FFF",  # blue    — canonical amino acid sequence
    "structure": "#FE6100",  # orange  — 3D structure tokens
    "ss8":       "#785EF0",  # purple  — secondary structure
    "sasa":      "#FFB000",  # amber   — solvent accessibility
    "function":  "#009E73",  # teal    — GO/InterPro function annotation
    "all":       "#DC267F",  # magenta — multimodal fusion
}

CATEGORY_GROUPS: list[tuple[str, list[str]]] = [
    ("INPUT MODALITY",   ["sequence", "structure"]),
    ("DERIVED MODALITY", ["ss8", "sasa"]),
    ("FUNCTION",         ["function"]),
    ("FUSED",            ["all"]),
]

# Neutral slate for all headers: hue no longer encodes the taxonomy (each
# condition has its own distinct colour), so the grouping is conveyed by the
# legend layout alone rather than by header colour.
CATEGORY_COLOR: dict[str, str] = {
    "INPUT MODALITY":   "#475569",
    "DERIVED MODALITY": "#475569",
    "FUNCTION":         "#475569",
    "FUSED":            "#475569",
}

INK = "#0f172a"
