# When does ESM3 fuse its modalities? A geometry-first atlas of the multimodal residual stream

*Working draft. Figures referenced by their committed paths under `figures/`.*

## Abstract

ESM3 ingests a protein through several distinct channels at once — amino-acid
sequence, 3D structure, secondary structure (SS8), solvent accessibility (SASA),
and discrete functional annotations — summing their embeddings into a single
residual stream. We ask a geometry-first question: do these modalities occupy
separate subspaces, and if so, when along the network's depth do they fuse? Using
single-modality forward passes and representational-similarity analysis across all
48 layers of `esm3-sm-open-v1`, we find that the four *physical* modalities
(sequence, structure, SS8, SASA) begin in distinct subspaces, remain maximally
separated through roughly the first half of the network, and then **fuse sharply
into a shared low-dimensional subspace** between layers ~25 and ~35. The fusion is
ordered — the structure-derived modalities (structure, SS8, SASA) are mutually
aligned from the input, while sequence is the last to join (~L28–33). Strikingly,
the **functional-annotation modality never fuses**: it remains representationally
orthogonal to the physical modalities at every layer, and this holds whether the
annotation is provided whole-protein or per-residue, identifying it as a
content-driven separation rather than a tokenisation artifact. We show the fusion
is a *learned* property (absent in a random-initialised model of the same
architecture), holds below the mean-pool (at the residue level), is statistically
robust at our sample size, and is a representational *re-organisation* — between-
condition variance is converted into within-condition variance while the stream
never approaches isotropy. Fusion depth is independent of protein length but is
delayed by structural disorder. [Universality across the tree of life — eukaryota,
bacteria, archaea — established on a 12-organism set; *result pending the diverse
run*.]

## 1. Introduction

Multimodal protein language models such as ESM3 are trained to reason jointly over
sequence and structure. Architecturally, each input modality has its own embedding
table, and the per-residue embeddings are summed before the transformer trunk. This
makes a natural mechanistic-interpretability question available: are the modalities
kept in separate parts of the representation, or does the network blend them — and
if it blends them, *where*?

We take a geometry-first, descriptive approach. For each protein we run ESM3 once
per modality in isolation (all other modalities masked), plus an all-modalities
reference, and read out the residual stream at every layer. The collection of these
per-(protein, modality, layer) representations is an atlas of how multimodal
information is organised across depth. We characterise it with standard
representational-geometry tools — silhouette score, linear CKA, effective rank,
and linear probes — and a battery of controls.

This is **Act 1** (a descriptive atlas); causal steering, SAE dictionaries, and
cross-architecture comparison are deferred to / handled in companion work.

## 2. Methods

**Model and modalities.** `esm3-sm-open-v1` (1.4B parameters, 48 transformer
blocks, d_model = 1536). We study six *conditions*: five single-modality inputs —
`sequence`, `structure`, `ss8`, `sasa`, `function` — and `all` (every modality
supplied jointly). Structure is provided as VQ structure tokens from coordinates;
SS8/SASA are computed by DSSP from the structure; `function` is provided as
whole-protein InterPro annotations.

**Datasets.** A 199-protein human pilot for development; a **892-protein** human set
(the main results); and a **5,984-protein, 12-organism set spanning eukaryota,
bacteria and archaea** for the universality test. Structures are AlphaFold-DB models
(uniform release), SS8/SASA from DSSP (mkdssp 4.6.1 via ESM's `ProteinChain`),
InterPro/GO from UniProt.

**Harvesting.** For each (protein, condition) we run one forward pass and cache the
residual stream at all 48 layers, mean-pooled over residues (BOS/EOS excluded). A
100-protein subset additionally stores per-residue activations (fp16, 7 layers) for
the pooling control.

**Metrics.** Per layer: (i) **silhouette** of the conditions (Euclidean, full
1536-d); (ii) **linear CKA** (Kornblith et al. 2019) between every condition pair,
protein-aligned; (iii) an **integration index** = mean pairwise CKA; (iv)
**effective rank** = exp(entropy of the covariance eigenspectrum), whole-cloud vs.
per-condition; (v) a protein-grouped 5-fold logistic **probe** of modality identity,
in full-dim and in the top-3 PCA geometry. Embeddings for visualisation use PCA
(per-layer and a shared "joint" basis across layers).

**Controls.** A random-init model (trunk + modality embeddings reinitialised,
structure encoder preserved); a per-residue replication; a subsampling-convergence
analysis; label-permutation nulls and protein-bootstrap CIs.

## 3. Results

### R1 — Physical modalities fuse sharply at mid-depth
Condition separation (silhouette) *rises* from 0.32 at the input to a peak of **0.42
at layer 24**, then collapses to a minimum of **0.156 at layers 35–38**, with a
partial re-separation at the final layers. The integration index moves inversely.
The transition is sharp and localised (knee ~L25–26), not gradual.
*(`figures/scaled/metrics/metrics_vs_depth.png`; hero: `figures/scaled/gifs/depth_sweep.gif`.)*

### R2 — The fusion is ordered; sequence is the last physical modality to join
Per-pair CKA shows the structure-derived modalities (structure, SS8, SASA) are
mutually aligned from layer 0, whereas every `sequence ↔ *` pair stays near CKA ≈
0.2 until **~L28–33**, then rises. Sequence — the one channel carrying information
not derivable from the structure — integrates last.
*(`figures/scaled/metrics/cka_pairs_vs_depth.png`.)*

### R3 — The functional-annotation modality never fuses (content-driven)
Every `function ↔ *` CKA stays ≈ 0.01–0.05 across all 48 layers while structure ↔
all reaches ≈ 0.99. We verify this is not a degenerate representation (the function
cloud has real cross-protein spread). We further rule out a *granularity* artifact —
function is the only whole-protein modality — by adding ESM3's **per-residue**
`residue_annotation` modality, driven by real InterPro residue-site annotations
(632/892 proteins). Per-residue functional annotation also stays largely orthogonal
(CKA ↔ physical ≈ 0.1, transient peak ≈ 0.5, never the ≈ 0.99 of fusion). Thus ESM3
keeps functional information in a separate subspace **regardless of granularity** —
a content-driven separation.
*(`figures/scaled/metrics/residue_annotation_compare.png`.)*

### R4 — The fusion is learned, not architectural
A random-initialised model of the same architecture (inputs unchanged) shows **no
fusion**: silhouette is flat at ≈ 0.62 and the integration index flat at ≈ 0.71
across all 48 layers, versus the trained model's peak-then-collapse. Fusion is
therefore a learned property, not a consequence of summing modality embeddings.
*(`figures/scaled/metrics/randinit_control.png`.)*

### R5 — Fusion holds below the mean-pool
At the residue level (100-protein subset), the integration index rises with depth
(0.19 → 0.59), mirroring the pooled curve. Fusion is not an averaging artifact.
*(`figures/scaled/metrics/per_residue_validation.png`.)*

### R6 — Fusion is a re-organisation, not a collapse
Effective rank collapses to ≈ 3 at the L24 separation peak, then **expands to ≈ 85
in the fusion zone (L36–38)** before contracting. Whole-cloud and per-condition
ranks converge at the fused layers. So fusion converts between-condition variance
into within-condition variance while the stream **never approaches isotropy** (peak
≈ 85 of 1536). The full-dim identity probe stays at 1.0 throughout (a thin "modality
tag" persists), while the top-3 PCA probe dips to ≈ 0.68 in the fusion zone — driven
by SASA (recall ≈ 0.33) — even as `function` recall stays ≈ 1.0.
*(`figures/scaled/metrics/diagnostics.png`.)*

### R7 — Fusion depth depends on disorder, not length
Across 892 proteins, fusion-onset is independent of length (Spearman r = −0.04,
p = 0.26) but is delayed by structural disorder (coil fraction r = +0.22,
p = 3×10⁻¹¹; helix fraction r = −0.20, p = 3×10⁻⁹): well-folded proteins fuse
earlier. *(`figures/scaled/metrics/biology_breakdown.png`.)*

### R8 — Fusion is universal across the tree of life
*[Pending the diverse run: 5,984 proteins × 12 organisms × 3 superkingdoms. The
per-superkingdom fusion curves will go here; `figures/diverse/metrics/stratified_fusion.png`.]*

## 4. Discussion

ESM3 organises its inputs into two representational regimes: a **shared physical
subspace** that the four geometry/sequence modalities collapse into at mid-depth,
and a **persistent functional-annotation subspace** that stays orthogonal
throughout. The mid-network fusion of physical modalities, the late integration of
sequence specifically, and the disorder-dependence of fusion depth together suggest
the network first builds modality-specific features and then commits to a unified
physical representation once structure is "resolved," while keeping discrete
functional labels on a separate axis.

## 5. Limitations

- **Model scale.** Only one model — `esm3-sm-open-v1` is the sole openly available
  *multimodal* ESM3; larger ESM3 is API-gated and ESM2 is sequence-only — so we
  cannot test whether (fractional) fusion depth scales with capacity.
- **Structures** are AlphaFold predictions; the disorder effect (R7) partly tracks
  prediction confidence.
- **Descriptive, not causal.** We characterise the geometry; we do not intervene.

## 6. Reproducibility

The full pipeline is scripted and config-driven: curation
(`curate_diverse.py`), structure fetch (`fetch_structures.py`), DSSP annotation
(`annotate_structures.py`), harvest (`harvest_scaled.py`, `config/*.json`),
aggregation (`aggregate_activations.py`), metrics (`compute_metrics.py`,
`compute_diagnostics.py`, `render_cka_heatmap.py`), controls
(`per_residue_validate.py`, `plot_randinit_control.py`,
`compare_residue_annotation.py`), and the visual heroes (`render_depth_sweep.py`).
Colours use the colorblind-safe `pypubfigs` palette.

## Figure index

| # | Figure | File |
|---|--------|------|
| 1 | Depth-sweep hero | `figures/scaled/gifs/depth_sweep.gif` |
| 2 | Fusion curve (silhouette / CKA / integration) | `figures/scaled/metrics/metrics_vs_depth.png` |
| 3 | Per-pair CKA across depth | `figures/scaled/metrics/cka_pairs_vs_depth.png` |
| 4 | Function vs residue-annotation (granularity) | `figures/scaled/metrics/residue_annotation_compare.png` |
| 5 | Random-init control | `figures/scaled/metrics/randinit_control.png` |
| 6 | Per-residue validation | `figures/scaled/metrics/per_residue_validation.png` |
| 7 | Diagnostics (probe / dim / significance) | `figures/scaled/metrics/diagnostics.png` |
| 8 | Biology breakdown | `figures/scaled/metrics/biology_breakdown.png` |
| 9 | Universality across superkingdoms | `figures/diverse/metrics/stratified_fusion.png` *(pending)* |
