# DG-VLM Equitable Glioma Molecular Diagnosis

DG-VLM predicts IDH mutation, 1p/19q codeletion, MGMT promoter methylation, and WHO 2021 integrated glioma subtype from hematoxylin-and-eosin whole-slide images. Frozen 512-dimensional CONCH patch representations feed feature-space domain randomization, text-modulated gated attention, domain-consistency regularization, and four molecular classification heads. Evaluation uses leave-one-dataset-out folds across TCGA, EBRAINS, CGGA, GLASS, and REMBRANDT.

## Environment

The reference environment is Python 3.10, PyTorch 2.1.2, CUDA 12.1, and an NVIDIA A100 80 GB GPU. OpenSlide system libraries are required for whole-slide image access.

Pip installation:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Conda installation:

```bash
conda env create -f environment.yml
conda activate dgvlm-glioma
pip install -e .
```

Container build:

```bash
docker build -t dgvlm-glioma:1.0 .
```

## Data

Verified dataset entry points and access conditions are listed in `datasets.txt`. Some files require registration, a data-use agreement, or controlled-access approval. No patient data are distributed with this package.

The manifest is a CSV with these columns:

```text
slide_id,patient_id,domain,slide_path,idh,codeletion,mgmt,subtype
```

Unavailable labels remain empty. Binary labels use `0` and `1`; integrated subtype uses `0` for astrocytoma IDH-mutant, `1` for oligodendroglioma IDH-mutant and 1p/19q-codeleted, and `2` for glioblastoma IDH-wildtype. Patient identifiers should be study-local pseudonyms.

Whole-slide processing uses non-overlapping 256×256 pixel patches at 20× magnification and 0.5 µm/pixel. Otsu tissue detection is followed by morphological closing, fragment removal, and hole filling. Tiles are rejected when background exceeds 70%, Laplacian variance is below 50, or detected pen area exceeds the configured threshold. Macenko normalization is the primary setting; raw H&E and Reinhard normalization support the reported stain analysis.

Each slide feature file is stored at:

```text
features/<domain>/<slide_id>.h5
```

The HDF5 file contains a `features` matrix with shape `[number_of_patches, 512]` and a matching `tile_ids` vector. The prototype tensor has shape `[9, 512]` and follows the exact descriptor order defined in `representation/encoder.py`.

## Training

The primary experiment holds out CGGA:

```bash
dgvlm-train \
  --manifest data/manifest.csv \
  --features data/features \
  --prototypes data/conch_text_prototypes.pt \
  --output outputs/cgga \
  --held-out CGGA
```

Run the remaining LODO folds by changing `--held-out` to `TCGA`, `EBRAINS`, `GLASS`, or `REMBRANDT`. Repeat every fold with seeds 42, 123, 256, 389, and 512 through `seed=<value>` overrides.

The default configuration uses AdamW with learning rate `1e-4`, weight decay `1e-2`, cosine annealing with warm restarts, one slide per batch, eight-step gradient accumulation, up to 200 epochs, and validation-AUC early stopping with patience 20. DG parameters are FSDR `α=0.3`, DA-MIL text boost `β=0.5`, DCR `λ=0.1`, attention entropy `γ=1e-3`, and learnable temperature initialized at `0.07`.

Feature extraction takes approximately 2–4 hours per dataset. MIL training takes approximately 3.2 hours per LODO fold on one A100 80 GB GPU and peaks near 48 GB device memory. The complete five-fold, five-seed primary matrix requires approximately 80 GPU-hours after features are cached. Storage depends on tissue area; reserve at least 1 TB for source slides, tiles, and 512-dimensional feature files.

Component settings are listed in `configs/ablation_components.yaml`. Sensitivity grids for α, β, λ, learning rate, and pooling are listed in `configs/ablation_sensitivity.yaml`. Configuration values can be overridden as `name=value` arguments.

## Evaluation

Generate per-domain binary metrics:

```bash
dgvlm-evaluate \
  --manifest data/manifest.csv \
  --features data/features \
  --prototypes data/conch_text_prototypes.pt \
  --weights outputs/cgga/best.pt \
  --output results/cgga.csv
```

The primary IDH AUC targets are `0.955 ± 0.012` on TCGA-Combined, `0.918 ± 0.018` on EBRAINS, and `0.907 ± 0.021` on CGGA across five seeds. Secondary targets are 1p/19q AUC `0.921 ± 0.016` on TCGA and `0.885 ± 0.025` on CGGA; MGMT AUC `0.808 ± 0.024` on TCGA and `0.748 ± 0.032` on CGGA; WHO integrated subtype AUC `0.965 ± 0.010` on TCGA and `0.952 ± 0.014` on EBRAINS.

Evaluation routines provide AUC-ROC, AUC-PR, weighted F1, balanced accuracy, sensitivity, specificity, Youden threshold, Cohen kappa, Brier score, stratified bootstrap intervals, DeLong comparisons, Bonferroni correction, Cochran Q, I², random-effects prediction intervals, coefficient of variation, Gini coefficient, and minimum-to-maximum site ratio.

Results can vary with slide revisions, molecular-label harmonization, CONCH weight revision, and vendor decoding. The manifest digest and package versions should accompany every reported run.

## Package map

`cohorts` handles manifests, LODO partitions, feature files, tissue masks, tile quality, and stain normalization. `representation` contains the frozen encoder boundary, FSDR profile, text alignment, domain-aware attention, and molecular heads. `learning` contains the multi-task objective, EMA domain moments, optimization engine, distributed helpers, validation, and atomic state persistence. `statistics` contains diagnostic, comparison, meta-analytic, equity, calibration, and clinical-impact calculations. `commands` provides preparation, training, and evaluation entry points.

## Intended use

This software supports retrospective computational research. It is not a medical device and must not determine patient care without prospective validation, local calibration, pathology review, and confirmatory molecular testing where available. Users are responsible for institutional review, data governance, demographic subgroup analysis, and reporting performance by healthcare infrastructure tier.
