# MECH_INTERP_ESM3_MODALITIES

A geometry-first atlas of how ESM3 (esm3-sm-open-v1, 1.4B) organizes its
**multimodal residual stream** across its 48 transformer layers.

ESM3 ingests up to six discrete modalities — sequence, structure tokens,
SS8, SASA, function, and residue annotations — and sums their embeddings
into a single residual stream per residue. The natural mech-interp question
is: *do those modalities live in different subspaces, and when do they get
fused?*

## The hero artifact

A spinning-cluster GIF in the visual style of `LLM_MECH_INTERP/paper/figures/distinct_clusters/spin_*.gif`,
but where each cluster is a **modality condition** (sequence-only,
structure-only, …, all-modalities), and the animation axis is either rotation
(per-layer spin GIFs) or **depth** (layer-sweep GIF — camera fixed, frames
are layers, watch modalities fuse).

## Experimental design

- **Subjects:** ~500 proteins subsampled from the 938-protein pilot in
  `INTERPRETABILITY/data/pilot/` (sequence + PDB structure + DSSP-derived
  SS8/SASA + GO annotations all present).
- **Conditions (the "modalities"):** for each protein, run ESM3 with
  inputs masked to a single modality at a time, plus an all-modalities
  reference:
  1. `sequence_only`
  2. `structure_only`
  3. `ss8_only`
  4. `sasa_only`
  5. `function_only`
  6. `all_modalities`
- **Activation extraction:** residual stream at layers
  `{0, 8, 16, 24, 32, 40, 47}` (7 depth slices across 48 blocks).
  Mean-pool over residue dimension to get one vector per
  (protein × condition × layer).
- **Embedding:** PCA → 3D per layer (UMAP as a comparison).
- **Visualization:** matplotlib `FuncAnimation` + `PillowWriter`, with
  reveal-then-spin phasing lifted from
  `LLM_MECH_INTERP/scripts/render_spinning_clusters.py`.

## Quantitative companions

- Per-layer silhouette score across modality conditions.
- Centered Kernel Alignment between condition pairs at each layer.
- A scalar "modality integration index" that captures when conditions
  collapse into a shared subspace.

## Layout

```
src/
  models/      ESM3 loader, hook manager, modality-condition driver
  data/        Pilot dataset loader (reads from INTERPRETABILITY/data/pilot/)
  embed/       PCA/UMAP wrappers, CKA, silhouette
  viz/         Animation utilities, color palettes, legend layout
scripts/
  harvest_activations.py   forward passes → NPZ per (condition, layer)
  embed_3d.py              activations → 3D coords
  render_spin.py           per-layer spin GIFs
  render_layer_sweep.py    layer-as-time GIF
  compute_metrics.py       silhouette, CKA, integration index
  render_cka_heatmap.py    per-pair CKA across depth (which modalities fuse first)
activations/                NPZ cache (gitignored)
figures/                    GIF and PNG outputs
results/                    JSON metric dumps
data/                       symlink to INTERPRETABILITY/data/pilot/
```

## Sibling projects we lift from

- `LLM_MECH_INTERP/scripts/render_spinning_clusters.py` — animation pipeline
- `INTERPRETABILITY/src/models/esm3_hooks.py` — ESM3 hook manager
- `INTERPRETABILITY/data/pilot/` — curated multimodal protein set
- `MECH_INTERP_ESMFOLD/src/mi_esmfold/hooks.py` — generic hook patterns

## Status

Pilot complete (199 proteins × 5 conditions × 7 layers). The hero spin GIFs
(`figures/gifs/`) and the quantitative companions (`results/metrics.json`,
`figures/metrics/metrics_vs_depth.png`) are both rendered. The geometry and the
metrics agree: modality conditions sit in distinct subspaces through L24
(silhouette ≈ 0.63 → 0.44) and fuse sharply from L32 onward (silhouette → 0.05,
mean-pairwise-CKA → 0.96 at L40), with a partial re-separation at the final
layer (L47). The per-pair heatmap (`figures/metrics/cka_pairs_vs_depth.png`)
localizes *which* modalities fuse first: the structure family (structure, SS8,
SASA) is mutually aligned from L0, while **sequence is the holdout** — every
`sequence ↔ *` pair stays near CKA ≈ 0.2 through L24 and only joins the shared
subspace at L32. Not a paper yet — this is the geometry-first atlas (Act 1 of a
larger program). Causal steering, SAE dictionaries, and cross-architecture
comparison are deferred.
