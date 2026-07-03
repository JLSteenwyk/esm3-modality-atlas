# Modality geometry in the ESM3 residual stream

Analysis code for a geometry-first atlas of how ESM3 (`esm3-sm-open-v1`, 1.4B
parameters, 48 transformer layers, d_model 1536) organizes its multimodal
residual stream across depth.

ESM3 ingests up to six discrete modalities (sequence, structure tokens, SS8,
SASA, function, and residue annotations) and sums their embeddings into a single
residual stream per residue. This repository measures where those modalities live
in the stream and when they fuse. The central findings are that the physical
modalities occupy distinct subspaces early, fuse into one shared subspace at
mid-depth (condition separation peaks at layer 24 and reaches its minimum at
layer 35), and that the functional-annotation channel stays geometrically
orthogonal at every depth while its information remains recoverable from the
physical subspace.

This code accompanies the manuscript in `paper/`. A companion sparse-autoencoder
study is referenced there.

## Repository layout

```
src/
  models/   ESM3 loader, residual-stream hooks, modality-condition driver
  data/     dataset loader
  embed/    PCA and CKA helpers
  viz/      animation utilities, colorblind-safe palette
scripts/    the pipeline (curation, retrieval, annotation, harvesting,
            aggregation, metrics, controls, figure assembly)
config/     configuration-driven run specs (harvest.json and variants)
paper/      manuscript.tex, paper.md, references.bib, captions, build scripts
figures/    assembled figures and hero GIFs
results/    metric and control JSON emitted by the analyses
data/        accession lists for every dataset (structures are fetched, not stored)
```

Large artifacts are not tracked. The residual-stream activations (about 24 GB)
and the fetched structure trees are regenerated deterministically by the scripts;
the activations are also deposited at figshare (see the manuscript). Random seeds
are fixed throughout.

## Datasets

Three protein sets are analysed, each curated from public sources.

- **Scaled** (892 human canonical proteins) is the primary set for the depth
  atlas and the ablation and decoding controls.
- **Diverse** (5,555 analysed proteins from 12 organisms across the three
  superkingdoms) tests universality across the tree of life.
- **Experimental** (177 proteins) replicates the fusion signature on X-ray and
  cryo-EM structures from the RCSB rather than AlphaFold predictions, ruling out
  a predicted-coordinate artifact.

Structure models come from AlphaFold-DB and the RCSB. Sequence, InterPro, and
Gene Ontology annotations come from UniProt. Accession lists for every dataset,
including the exact RCSB entries used for the experimental replication, are in
`data/`.

## Modality conditions

For each protein the residual stream is harvested under each single-modality
condition (`sequence`, `structure`, `ss8`, `sasa`, `function`), an
all-modalities reference (`all`), a per-residue `residue_annotation` condition,
and five leave-one-out conditions (`all_no_sequence`, `all_no_structure`,
`all_no_ss8`, `all_no_sasa`, `all_no_function`) used for the causal ablation.
Representations are mean-pooled over residues to one vector per
(protein, condition, layer).

## Metrics

- Condition separation by silhouette score in the full 1536-dimensional stream.
- Linear centered kernel alignment (CKA) between every condition pair at every
  layer.
- A mean-pairwise-CKA integration index that captures collapse into a shared
  subspace.
- Effective rank of the residual-stream covariance spectrum.
- A protein-grouped logistic probe of modality identity and of structural and
  functional targets.

## Reproducing the analysis

Install the environment (Python 3.10; principal dependencies PyTorch 2.6,
scikit-learn 1.7, SciPy 1.15, NumPy 1.26):

```
pip install -r requirements.txt
```

Secondary structure is computed with DSSP (`mkdssp` 4.6.1) and solvent
accessibility with the Shrake-Rupley implementation in ESM's `ProteinChain`;
`mkdssp` must be on `PATH` for the annotation step.

The pipeline is configuration-driven. Each stage is a standalone script, and the
end-to-end runs are wrapped in shell drivers:

- `scripts/run_diverse_pipeline.sh` builds and analyses the 12-organism set.
- `scripts/run_experimental.sh` runs the experimental-structure replication
  (build set, annotate, harvest, aggregate, metrics).

The individual stages, in order, are dataset curation
(`curate_diverse.py`, `build_experimental_set.py`), structure retrieval
(`fetch_structures.py`), annotation (`annotate_structures.py`), activation
harvesting (`harvest_scaled.py`, `harvest_ablation.py`), aggregation
(`aggregate_activations.py`), metrics and controls (`compute_metrics.py`,
`compute_diagnostics.py`, `subspace_decode.py`, `analyze_ablation.py`,
`test_orphan.py`, `plot_randinit_control.py`), and figure assembly
(`embed_3d.py`, `render_*.py`, and `paper/assemble_figures.py` /
`paper/assemble_supplementary.py`).

## Building the paper

```
bash paper/build.sh
```

compiles both `manuscript.pdf` and the supplementary PDF from the tracked
figures and captions.

## License

MIT. See `LICENSE`.
