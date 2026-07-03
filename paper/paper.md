# When does ESM3 fuse its modalities? A geometry-first atlas of the multimodal residual stream

*Markdown companion to `manuscript.tex` (the LaTeX manuscript is canonical). Figures in `paper/figures/`.*

## Abstract

ESM3 ingests a protein through several channels at once, including amino-acid
sequence, three-dimensional structure, secondary structure (SS8), solvent
accessibility (SASA), and discrete functional annotations, summing their embeddings
into a single residual stream. Whether these modalities occupy separate subspaces,
and at what depth they fuse, has not been characterised. The present analysis runs
`esm3-sm-open-v1` once per modality in isolation and applies representational-similarity
analysis across all 48 layers. The four physical modalities (sequence, structure,
SS8, SASA) begin in distinct subspaces, remain maximally separated through roughly
the first half of the network, and then fuse into a shared low-dimensional subspace
between layers 25 and 35. The fusion is ordered. The structure-derived modalities are
mutually aligned from the input, whereas sequence joins last, after layer 28. The
functional-annotation modality never fuses. It remains representationally orthogonal
to the physical modalities at every layer, and this orthogonality holds whether the
annotation is supplied whole-protein or per-residue, which identifies it as a
content-driven separation rather than a tokenisation artifact. The fusion is a learned
property, absent in a randomly initialised model of the same architecture, holds at the
residue level below the mean-pool, and reorganises variance while the stream never
approaches isotropy. Fusion depth is independent of protein length but is delayed by
structural disorder. The phenomenon is universal across the tree of life. Across 5,555
proteins from 12 organisms spanning eukaryota, bacteria, and archaea, every
superkingdom, and every individual organism, reaches peak modality fusion at the same
network depth (layer 35).

## 1. Introduction

Multimodal protein language models such as ESM3 reason jointly over sequence and
structure. Each input modality has its own embedding table, and the per-residue
embeddings are summed before the transformer trunk. This architecture raises a
mechanistic question. The modalities may be kept in separate parts of the
representation, or the network may blend them, and if it blends them, the depth at
which it does so is unknown. Prior interpretability work on protein language models
has examined attention heads and learned features, but how the several input
modalities of a multimodal model are organised geometrically across depth has not
been characterised.

The present analysis takes a geometry-first, descriptive approach. For each protein,
ESM3 runs once per modality in isolation, with all other modalities masked, together
with an all-modalities reference, and the residual stream is read out at every layer.
The resulting collection of per-(protein, modality, layer) representations forms an
atlas of how multimodal information is organised across depth. Standard
representational-geometry tools, namely silhouette score, linear CKA, effective rank,
and linear probes, characterise the atlas, supported by several controls. This work
is the first stage of a larger program. A companion study applies sparse autoencoders
to ESM3 and ESM2 and, among other results, reports that the sequence and
sequence-plus-structure representations of ESM3 converge between layers 28 and 38. The
present analysis extends that two-condition observation to a full multimodal atlas,
characterising the ordered fusion of the four physical modalities, the persistent
orthogonality of the functional channel, the universality of the pattern across the
tree of life, and a battery of controls; causal steering and sparse-autoencoder
dictionaries are developed in that companion work.

## 2. Methods

**Model and modalities.** The model is `esm3-sm-open-v1` (1.4 billion parameters, 48
transformer blocks, model dimension 1536). Six conditions are studied, comprising five
single-modality inputs (sequence, structure, SS8, SASA, and function) and an
all-modalities condition in which every modality is supplied jointly. Structure is
provided as vector-quantised structure tokens derived from coordinates, SS8 is computed
by DSSP and SASA by the Shrake-Rupley algorithm from the structure, and function is
provided as whole-protein
InterPro annotations filtered to the 29,026-entry vocabulary of the ESM3 function
tokeniser.

**Datasets.** Three datasets were used, namely a 199-protein human pilot for
development, an 892-protein human set for the main results, and a set of 5,984 proteins
curated from 12 organisms spanning eukaryota, bacteria, and archaea, of which the 5,555
with complete structure and annotation were analysed, for the universality test.
Structures are AlphaFold-DB models of a single release, SS8 and SASA are computed
with DSSP (mkdssp 4.6.1, via ESM's `ProteinChain`), and InterPro and Gene Ontology
annotations are taken from UniProt. A fourth dataset of 177 proteins backed by
experimental (X-ray and cryo-EM) structures from the RCSB replicates the main result
with non-predicted coordinates.

**Harvesting.** For each (protein, condition) pair, ESM3 runs one forward pass and the
residual stream is cached at all 48 layers, mean-pooled over residues with the special
tokens excluded. A 100-protein subset additionally stores per-residue activations at
seven layers for the pooling control.

**Metrics.** Each layer is summarised by the silhouette score of the conditions
(Euclidean, in the full 1536-dimensional space), by the linear CKA of Kornblith et al.
between every protein-aligned condition pair, by an integration index defined as the
mean pairwise CKA, by the effective rank computed as the exponential of the entropy of
the covariance eigenspectrum for the whole cloud and for each condition, and by a
protein-grouped five-fold logistic probe of modality identity evaluated both in the
full space and in the top-3 PCA geometry. Visualisations use PCA, fit per layer and in
a shared basis across layers.

**Controls.** The controls comprise a randomly initialised model whose trunk and
modality embeddings are reset while the structure encoder is preserved, a per-residue
replication, a subsampling-convergence analysis, label-permutation nulls, and
protein-bootstrap confidence intervals.

## 3. Results

### Physical modalities fuse sharply at mid-depth
Condition separation, measured by silhouette score, rises from 0.32 at the input to a
maximum of 0.42 at layer 24, then falls to a minimum of 0.156 at layer 35, with
a partial re-separation at the final layers (Figure 1b). The integration index moves
inversely. The transition is sharp and localised, with a knee near layer 25, rather
than gradual. The accompanying joint-PCA projection shows five separated clusters
through layer 24 that collapse into one cloud by layer 35 (Figure 1a). These metrics
are converged at the sample size used (Figure S1) and sit far above label-permutation
nulls at every layer (Figure S2).

### The fusion is ordered, and sequence joins last
Per-pair CKA shows that the structure-derived modalities (structure, SS8, and SASA) are
mutually aligned from layer 0, whereas every pairing that involves sequence stays near
a CKA of 0.2 until layer 28 and rises thereafter (Figure 2a; the full
condition-by-condition matrix at seven depths is in Figure S5). Sequence, the one channel
carrying information that cannot be derived from the structure, integrates last.

### The functional-annotation modality never fuses
Every pairing of the function condition with a physical modality stays near a CKA of
0.01 to 0.07 across all 48 layers, while structure reaches a CKA of about 0.99 with the
all-modalities reference. This orthogonality does not stem from a degenerate
representation, because the function cloud retains genuine cross-protein spread, with an
effective rank of 2.8 to 218 across layers. One alternative explanation is granularity,
because function is the only whole-protein modality whereas the physical modalities vary
residue by residue. To test this explanation, the per-residue `residue_annotation`
modality of ESM3 was added, driven by InterPro residue-site annotations available for
631 of the 892 proteins. Per-residue functional annotation also stays largely
orthogonal, with a CKA against the physical modalities near 0.1 and a transient maximum
near 0.5, never approaching the CKA of about 0.99 that marks fusion (Figure 2b).
Functional information is therefore held in a separate subspace irrespective of
granularity, a content-driven separation.

### The fusion is learned, not architectural
A model with the same architecture but randomly initialised trunk and modality
embeddings, with the structure encoder preserved so that inputs remain meaningful,
shows no fusion. Condition separation stays near 0.62 and the integration index near
0.71 across all 48 layers, against the trained model's peak-then-collapse (Figure 3a).
Fusion is therefore a learned property and not a consequence of summing modality
embeddings.

### Fusion holds below the mean-pool
Recomputed on residue-level representations of a 100-protein subset, the integration
index rises with depth from 0.18 to 0.59, mirroring the pooled curve (Figure 3b).
Fusion is not an averaging artifact.

### Fusion reorganises variance without approaching isotropy
Effective rank of the whole cloud collapses to about 3 at the layer-24 separation peak,
then expands to about 85 of a possible 1536 in the fusion zone before contracting, and
the whole-cloud and per-condition ranks converge once the conditions are fused
(Figure 3c). Fusion therefore converts between-condition variance into within-condition
variance while the stream never approaches isotropy. A logistic probe of modality
identity reaches an accuracy of 1.0 in the full 1536-dimensional stream at every layer,
which indicates that a thin additive identity signal persists, whereas the same probe
in the top-3 PCA geometry falls to about 0.68 in the fusion zone, driven mainly by
SASA, even as the function condition stays decodable throughout (Figure S3).

### Fusion depth tracks secondary-structure content
Across the 892 proteins, fusion-onset depth is independent of protein length (Spearman
r of -0.04, p of 0.26) but is delayed by structural disorder, with a coil-fraction
correlation of +0.22 (p of 3e-11) and a helix-fraction correlation of -0.20 (p of
3e-9), so that well-folded, helical proteins fuse earlier (Figure 4a). Fusion-onset has
no relationship with AlphaFold confidence (mean pLDDT correlation of +0.003, p of 0.93).
The disorder effect therefore reflects genuine secondary-structure content rather than
the model's structural uncertainty.

### Fusion is universal across the tree of life
Across 5,555 proteins from 12 organisms spanning all three superkingdoms (Figure S6), the pooled
fusion curve is nearly identical to the human-only result, with a silhouette maximum of
0.42 at layer 23 and a minimum of 0.152 at layer 35 against 0.42 at layer 24 and 0.156
at layer 35. Stratified by superkingdom, the eukaryota, bacteria, and archaea curves
show the same peak-then-collapse and reach minimum separation at layer 35 (Figure 4b),
with maxima of 0.42, 0.48, and 0.50 and minima of 0.164, 0.151, and 0.149. Every one of
the 12 organisms reaches its minimum at layer 35, with a standard deviation of 0
(Figure S4). Superkingdoms are intermixed in the representation rather than separated,
so the organisation is by modality and not by source organism (Figure S7). Multimodal
fusion is therefore a universal, depth-locked property of ESM3
and not an artifact of the curated human set.

## 4. Discussion

ESM3 organises its inputs into two representational regimes. The four physical modalities collapse into a shared, low-dimensional subspace at
mid-depth, whereas the functional-annotation modality occupies a subspace that stays
orthogonal throughout the network. The mid-network timing of physical fusion, the late
entry of sequence, and the dependence of fusion depth on secondary-structure content
together suggest that the network first builds modality-specific features and then
commits to a unified physical representation once secondary structure is resolved,
while holding discrete functional labels on a separate axis. The depth-locked
universality across the tree of life indicates that this organisation is intrinsic to
the trained model rather than a property of any particular proteome.

Geometric orthogonality does not mean that functional content is absent from the fused
representation. A linear probe recovers enzyme class from the structure-derived
representation with a balanced accuracy that rises from 0.56 to about 0.79 by layer 39,
and it recovers fold class from the physical conditions at about 0.84 throughout (Figure
S9). The explicit function channel behaves in the opposite way, decoding fold class only
near 0.40 and enzyme class near 0.55, with the latter declining across depth. The network
therefore extracts functional content into the physical subspace while holding the
discrete function channel on a separate and informationally thin axis, so the
orthogonality is a representational choice rather than an absence of functional
information. A causal intervention agrees. Withholding the function input from the full
model displaces the fused representation about six times less than withholding
structure, sequence, or SASA, with a median relative displacement of 0.02 against 0.07
to 0.12 at the fusion layer (Figure S10), so the fused physical representation is
largely insensitive to the function input.

One mechanism that could keep functional annotation orthogonal is redundancy. Because
enzyme class is decodable from the structure-derived representation, the network may
already recover function from the physical modalities and face little pressure to
integrate the explicit functional channel. This account predicts that the function channel should
align more closely with the physical subspace for proteins whose function is not
recoverable from structure. The prediction fails (Figure S8). Per-protein alignment
between the function vector and the physical subspace is near zero for every protein,
with a mean of 0.0005 and a standard deviation of 0.15, and it does not track
redundancy. Proteins for which the structure representation misclassifies enzyme class
align no more strongly than proteins it classifies correctly, with means of 0.014 and
0.006 and a Mann-Whitney p of 0.35, and alignment is flat against both the number of
InterPro domains, with a Spearman correlation of -0.02, and the number of gene-ontology
terms, with a Spearman correlation of 0.09. Functional annotation therefore stays
orthogonal whether or not it is redundant with the physical modalities, which points
away from redundancy and toward a categorical organisation in which discrete functional
labels are held on a separate axis irrespective of their recoverability from structure.

## 5. Limitations

Several limitations bound these conclusions. Only one model was studied, because
`esm3-sm-open-v1` is the sole openly available multimodal ESM3, larger ESM3 models are
reachable only through a gated interface, and the ESM2 family is sequence-only and
cannot support the experiment. Whether the fractional fusion depth scales with model
capacity therefore remains open. The main structures are AlphaFold predictions, but the
fusion signature is not an artifact of predicted coordinates, because the disorder
effect is uncorrelated with prediction confidence and the silhouette dip at layer 35,
the integration-index rise, and the orthogonality of the function channel all reproduce
on 177 proteins with experimental coordinates (Figure S11). Finally, the present work
characterises the geometry of
the residual stream and does not intervene on it, so the analysis is descriptive rather
than causal.

## 6. Data and code availability

All analysis code is available at `https://github.com/ORGANISATION/REPOSITORY`. The
pipeline is configuration-driven and provides, as separate scripts, dataset curation,
structure retrieval from AlphaFold-DB and the RCSB, secondary-structure and
solvent-accessibility annotation, activation harvesting, the leave-one-out modality
ablation, aggregation, the metric and control analyses, and figure assembly. Random
seeds are fixed throughout.

Representations were extracted from `esm3-sm-open-v1`, accessed through the
EvolutionaryScale `esm` package (version 3.2.1) under its open licence. The principal
dependencies are Python 3.10, PyTorch 2.6, scikit-learn 1.7, SciPy 1.15, and NumPy 1.26.
Secondary structure is computed with DSSP (mkdssp 4.6.1) and solvent accessibility with
the Shrake-Rupley implementation in ESM's `ProteinChain`. Figures use the
colorblind-safe `pypubfigs` palette.

Input data are public. Structure models are from AlphaFold-DB and experimental
structures from the RCSB, and sequence, InterPro, and Gene Ontology annotations are from
UniProt. Accession lists for every dataset, including the exact RCSB entries used for the
experimental replication, are provided in the repository. The mean-pooled residual-stream
activations, about 24 gigabytes, are regenerated deterministically by the harvesting
scripts and are also deposited at figshare (`https://doi.org/FIGSHARE.DOI`).

## Figure index

Main figures (`paper/figures/figure{1..4}.png`) and supplementary figures
(`figureS{1..6}.png`) are described in `paper/figure_captions.md`.

| # | Figure |
|---|--------|
| 1 | Joint-PCA depth snapshots and the 48-layer fusion curve |
| 2 | Per-pair CKA across depth, and the function vs residue-annotation control |
| 3 | Random-init control, per-residue validation, and effective rank |
| 4 | Fusion depth vs secondary structure, and universality across superkingdoms |
| S1 | Sample-size convergence |
| S2 | Significance against permutation nulls |
| S3 | Modality-identity probe |
| S4 | Per-organism universality |
| S5 | Per-layer condition-by-condition CKA |
| S6 | Diverse dataset composition |
