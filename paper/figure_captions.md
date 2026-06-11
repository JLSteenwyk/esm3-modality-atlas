# Figure captions

Captions follow the project style. The present work is the subject; first person,
em-dashes, and colons in running prose are avoided (colons appear only in these
captions and in headings).

---

**Figure 1. Modality conditions occupy distinct subspaces early and fuse into a
shared subspace at mid-depth.** (a) Mean-pooled residual-stream representations of
892 human proteins under each single-modality condition, projected with a joint
PCA fit across all 48 layers and shown at five depths from three camera
orientations (rows). Each point is one
(protein, modality) representation, coloured by modality. The five conditions form
separated clusters through layer 24 and collapse into one cloud by layer 35, while
the functional-annotation condition (green) remains a distinct island. (b) Across
all 48 layers, condition separation (silhouette, left axis) rises to a maximum of
0.42 at layer 24, then falls sharply to a minimum of 0.156 at layer 35, while the
integration index (mean pairwise CKA, right axis) moves inversely. The shaded band
marks the fusion transition (layers 25 to 35).

**Figure 2. Fusion of the physical modalities is ordered, and the functional
channel never joins.** (a) Linear CKA between every condition pair at every layer,
with rows ordered by fusion onset (the first layer reaching CKA of 0.5). The three
structure-derived modalities (structure, SS8, SASA) are mutually aligned from
layer 0, the sequence pairs warm from near CKA 0.2 to alignment only after
layer 28, and all five function pairs (bottom rows) stay near CKA 0.01 throughout.
(b) The whole-protein function track and the per-residue residue-annotation track
behave alike. Across depth, structure reaches CKA of about 0.99 with the
all-modalities reference (orange), whereas residue-annotation (black) and function
(green) both stay below CKA 0.5 with the physical modalities. The orthogonality of
functional information therefore reflects its content rather than its whole-protein
granularity.

**Figure 3. The fusion is learned, holds below the mean-pool, and re-organizes
variance without approaching isotropy.** (a) A model with the same architecture but
randomly initialised weights (structure encoder preserved) shows no fusion: condition
separation stays near 0.62 at every layer, against the trained model's
peak-then-collapse. (b) Measured on residue-level representations of a 100-protein
subset, the integration index rises with depth (0.19 to 0.59), mirroring the pooled
curve. (c) Effective rank of the whole cloud collapses to about 3 at the layer-24
separation peak, then expands to about 85 (of 1536) in the fusion zone before
contracting, and the whole-cloud and per-condition ranks converge once fused. The
stream never approaches isotropy.

**Figure 4. Fusion depth tracks secondary-structure content and is universal across
the tree of life.** (a) Per-protein fusion-onset layer averaged within each of ten
protein categories (bars, mean and standard error). Onset is delayed by structural
disorder (Spearman r of +0.22 against coil fraction, p of 3e-11) but is unrelated to
protein length (r of -0.04, n.s.) or to AlphaFold confidence (mean pLDDT r of +0.003,
n.s.), indicating that the effect reflects genuine secondary-structure content rather
than prediction uncertainty. (b) Condition separation across depth for 5,555 proteins
from 12 organisms, stratified by superkingdom. The eukaryota, bacteria, and archaea
curves are nearly superimposable and reach minimum separation at layer 35 (shaded);
every one of the 12 organisms reaches its minimum at layer 35 (standard deviation of 0).

---

# Supplementary figure captions

**Figure S1. The metrics are converged at the sample size used.** Silhouette and
integration index computed on random subsamples of the 892-protein set (mean and
standard deviation over 12 draws per size) at layer 24 (separation peak) and
layer 40 (fusion zone). Silhouette is stable from 100 proteins, and the integration
index settles by 400 to 600 proteins, so 892 proteins lie within the converged
regime for every reported quantity.

**Figure S2. Every fusion metric is significant against a permutation null.** For
the canonical layers, the observed silhouette and integration index (points, with
protein-bootstrap 95 percent confidence bands) sit far above their label-permutation
null means (dotted), with permutation p of 0.005 at every layer tested.

**Figure S3. Modality identity stays decodable, and the functional channel stays
decodable longest.** A protein-grouped logistic probe of modality identity. In the
full 1536-dimensional stream, decodability is 1.0 at every layer (an additive
modality tag persists). In the top-3 PCA geometry, overall accuracy falls to about
0.68 in the fusion zone, driven by SASA (per-condition recall near 0.33), whereas the
function condition stays near 1.0 throughout. Chance is 1 of 6.

**Figure S4. Every individual organism reaches peak fusion at the same depth.**
Condition separation across depth for each of the 12 organisms (thin lines, coloured
by superkingdom) and for the pooled set (thick line). All 12 organism curves and the
pooled curve reach minimum separation at layer 35 (dotted line).

**Figure S5. The condition-by-condition CKA matrix warms with depth except along the
function axis.** Linear CKA between all six conditions at seven depths. The structure,
SS8, and SASA block is warm from layer 0, the sequence entries warm after layer 24,
and the function row and column stay near CKA 0 at every depth.

**Figure S6. Composition of the 12-organism diverse set.** (a) Proteins curated per
organism (about 500 each), coloured by superkingdom. (b) Protein-length distributions
by superkingdom over the 50 to 800 residue range.

**Figure S7. The representation is organised by modality, not by source organism.**
(a) The layer-40 all-modalities representation of 5,555 proteins, projected to three
dimensions and coloured by superkingdom. Eukaryota, bacteria, and archaea are
intermixed rather than separated. (b) Across depth, condition separation by modality
(silhouette, reaching 0.42 at layer 24) far exceeds separation by superkingdom, which
stays near 0 at every layer (peaking at only 0.08 at layer 33). ESM3 therefore
represents proteins by their biochemistry rather than their taxonomy.
