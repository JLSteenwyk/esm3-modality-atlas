"""PCA helpers for activation 3D embedding.

Two modes:

* ``fit_per_layer_pca`` — independent 3D PCA per layer. Best visual separation
  for self-contained per-layer GIFs; axes are not comparable across layers.

* ``fit_joint_pca`` + ``project_joint`` — fit one PCA on stacked, per-layer
  centred activations. Axes are shared, so the same point at layer L₁ and
  layer L₂ has comparable coordinates — required for the layer-sweep GIF where
  we want the camera fixed and frames to be layers.

Per-layer centering before joint PCA keeps the global energy comparable across
layers (ESM3 residual stream norms grow with depth) while still letting joint
PCA discover one orthonormal basis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class JointPCA:
    """A 3D PCA basis shared across layers, with per-layer center + scale.

    ESM3 residual stream magnitudes grow with depth (the L47 norm dwarfs L0
    by orders of magnitude), so to make a layer-sweep animation readable we
    project on the shared basis but rescale each layer to its own 99th-
    percentile box. Cluster *shape* is then directly comparable across
    layers; absolute position is not (and shouldn't be inspected here).
    """

    layer_means: dict[int, np.ndarray]    # (1536,) per layer
    layer_scales: dict[int, float]        # scalar per layer (post-projection)
    components: np.ndarray                # (3, 1536)
    explained_variance_ratio: np.ndarray  # (3,)

    def project(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        centred = x - self.layer_means[layer_idx]
        return (centred @ self.components.T) * self.layer_scales[layer_idx]


def fit_per_layer_pca(
    layers_to_coords: dict[int, np.ndarray],
    n_components: int = 3,
) -> dict[int, dict]:
    """Fit independent PCA per layer.

    Returns dict mapping layer_idx -> {coords3d, evr, components, mean}.
    """
    out: dict[int, dict] = {}
    for layer_idx, x in layers_to_coords.items():
        pca = PCA(n_components=n_components, svd_solver="auto")
        coords3 = pca.fit_transform(x.astype(np.float32))
        out[layer_idx] = {
            "coords3d": coords3.astype(np.float32),
            "components": pca.components_.astype(np.float32),
            "mean": pca.mean_.astype(np.float32),
            "evr": pca.explained_variance_ratio_.astype(np.float32),
        }
    return out


def fit_joint_pca(
    layers_to_coords: dict[int, np.ndarray],
    n_components: int = 3,
) -> JointPCA:
    """Fit one PCA on the concatenated, per-layer-centered activations.

    Axes are stable across layers, suitable for layer-sweep animations.
    """
    centered_blocks: dict[int, np.ndarray] = {}
    means: dict[int, np.ndarray] = {}
    for layer_idx, x in layers_to_coords.items():
        m = x.mean(axis=0, keepdims=True)
        means[layer_idx] = m.squeeze(0).astype(np.float32)
        centered_blocks[layer_idx] = (x - m).astype(np.float32)

    big = np.concatenate(list(centered_blocks.values()), axis=0)
    pca = PCA(n_components=n_components, svd_solver="auto")
    pca.fit(big)

    # Per-layer post-projection rescaling so each layer fills a unit-ish box.
    layer_scales: dict[int, float] = {}
    for layer_idx, centred in centered_blocks.items():
        proj = centred @ pca.components_.T
        layer_scales[layer_idx] = float(
            1.0 / max(1e-6, np.percentile(np.abs(proj), 99))
        )

    return JointPCA(
        layer_means=means,
        layer_scales=layer_scales,
        components=pca.components_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )


def project_joint(joint: JointPCA, layers_to_coords: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Project each layer's activations into the joint PCA basis."""
    return {l: joint.project(x, l).astype(np.float32) for l, x in layers_to_coords.items()}
