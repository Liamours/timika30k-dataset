# timika50k_dataset

Reproducibility pipeline for timika-50k: assemble it from its 7 already-downloaded source datasets, drive labels/pseudo-labels, and produce the preprocessing variants below. Scaffolded 2026-09-03; the dataset session owns building this out, per this project's existing split between dataset construction/convention work and experiment execution.

**Documentation**: [DATASHEET.md](DATASHEET.md) (Gebru et al.'s 7-section
format, populated with this project's own real facts, not a template) and
[croissant.json](croissant.json) (MLCommons Croissant metadata, machine-
readable dataset description).

## Reproducibility checklist

Adapted from Papers with Code's ML Code Completeness Checklist (the
NeurIPS 2020 code-submission standard), mapped onto a dataset repo rather
than a model repo:

| Item | Status |
|---|---|
| Dependency spec (`pyproject.toml`) | Done |
| Build code: assembly (phase 2 below) | Done, `src/timika50k_dataset/assembly/build_data.py` |
| Build code: labels/pseudo-labels | Done, in `repo/timika50k_pseudolabels` (called into, not reimplemented here) |
| Build code: preprocessing | 1 of 5 variants done, `analyses/timika50k_preprocessing` |
| Evaluation/QC code | Done: 3-agent independent verification pattern used throughout (leakage, official-split fidelity, ratio integrity, box-mask geometry, confirmed-negative source rules), embedding-based anomaly detection (`src/timika50k_dataset/anomaly/`) |
| "Pretrained model" equivalent (the finished artifact) | Done, `E:\dataset\timika-50k\` + `dataset/timika-50k/` project copy |
| README with exact reproduction commands | Done, `src/timika50k_dataset/orchestrate.py` + Usage below |
| Results table | N/A, this is a dataset repo, not a results-reporting one; see per-source counts in DATASHEET.md instead |

## Scope, and what already exists elsewhere

Download has no code, by design (see below). Assembly's real code lives in this repo. Labels and preprocessing already have real code in 2 other repos; check there before writing anything new.

- **Download**: the 7 sources (Shenzhen, Montgomery, TBX11K, ChestX-Det, COVID-19 Radiography Database, BIMCV-CAAXR, SIIM-ACR Pneumothorax) were each acquired separately over time, heterogeneous source-specific methods (Kaggle API, direct NLM download, an access-gated custom tool, an authorized community rehost); see `dataset/manifest.md`'s per-source rows for how and from where. Not scripted, and likely never will be: 4 different access methods, 1 access-request gate, no single API covers all 7. Real per-source acquisition detail lives in `dataset/manifest.md`, not duplicated here.
- **Assembly** (phase 2, `src/timika50k_dataset/assembly/build_data.py`): the real, scripted gap this repo closes. Takes the 7 sources as already downloaded/canonicalized at `E:\dataset\{name}\` (assumed present, not fetched by this script) and applies timika-50k's own eligibility rule (`dataset/manifest.md`'s timika-50k row): drop TBX11K's own `data/extra/` (bonus augmentation, not part of its core release) and drop any file either source's own `labels/quality_flags.csv.bz2` marks `status=="duplicate"`. `--dry-run` reports per-source counts with no file copy; verified 2026-09-03 against the real quality_flags data to match the documented 51,659-image total (662/138/11,490/3,577/21,165/2,539/12,088) exactly.
- **Labels / pseudo-labels**: `repo/timika50k_pseudolabels` already generates the disease pseudo-label layer per source group (DS group built, 27,631 files; DB and OS groups blocked on the disease model retrain). Call into that repo, do not reimplement it here.
- **Preprocessing variants**: `analyses/timika50k_preprocessing` already builds one variant, a per-image min/max rescale to 8-bit matching `DiseaseSegDataset`'s own training-time preprocessing, plus `splits_official.csv`/`splits_fair.csv`. The variants below are new, and should reuse its split manifests rather than re-deriving train/val/test membership.

## Usage

`orchestrate.py` runs every phase already scripted, in dependency order,
each via `uv run` inside that phase's own uv project (assembly and
metadata here, labels in `repo/timika50k_pseudolabels`, preprocessing in
`analyses/timika50k_preprocessing` -- 4 separate dependency sets, so this
script shells out rather than importing across them):

```
uv run python -m timika50k_dataset.orchestrate --dry-run
uv run python -m timika50k_dataset.orchestrate
uv run python -m timika50k_dataset.orchestrate --phases assembly,sync_to_project_copy
```

`--dry-run` prints the exact command sequence, runs nothing. `--phases`
restricts to a comma-separated subset of: `assembly`,
`organ_region_labels`, `organ_region_manifest`,
`disease_confirmed_negatives`, `disease_caaxr_box_masks`,
`disease_tb_box_masks`, `disease_ds_group`, `disease_manifest`,
`sync_to_project_copy`, `preprocessing`, `splits`, `view_position`, the
same order a full run uses.

`assembly` through `disease_manifest` (the first 8 phases) read and write
`E:\dataset\timika-50k\`, the canonical copy. `sync_to_project_copy`
mirrors `data/` and `labels/` from there into `dataset/timika-50k/`, this
project's own second copy (`dataset/manifest.md`'s timika-50k row).
`preprocessing`, `splits`, and `view_position` (the last 3 phases) read
and write only that C: copy, never E:, per the 2026-09-02 decision to
keep `preprocessed/` project-local. Every phase script is independently
resumable (an existing output is treated as done and skipped), so
re-running the orchestrator after a partial or complete prior run costs
only the time to check what already exists, not a full rebuild -- verified
2026-09-03: `assembly --dry-run` matches the documented per-source counts
exactly with no file I/O, and `sync_to_project_copy --dry-run` found the
existing C: copy already 223,955/223,955 files in sync with E:, 0 pending.

## Preprocessing variants to build

1. Raw image: no processing.
2. Crop and normalize, no leakage across split: any learned statistic (crop bounds, normalization constants) fit on the training split only, per `splits_official.csv`/`splits_fair.csv`.
3. Bone suppression: `external/CXR-bone-suppression` is vendored (Rajaraman et al. 2021, *Diagnostics* 11:840, ResNet-BS architecture, Keras/TensorFlow, weights at `weights/resnet-bonesuppression-jsrt/`), never evaluated on this project's own images. It ships as a Jupyter notebook (`bone_suppression.ipynb`), not an importable package; inference code needs extracting. Runs on 256x256 grayscale input per its own README.
4. Bone suppression + CLAHE: variant 3's output through CLAHE (`skimage.exposure.equalize_adapthist` or `cv2.createCLAHE`).
5. CLAHE only: the same CLAHE step applied directly to the rescaled image, no bone suppression.

## Status

Scaffolded 2026-09-03; the core pipeline (assembly through view_position)
finished the same day:

- `src/timika50k_dataset/assembly/build_data.py`: phase 2, the eligibility
  filter and copy into `data/`, see Usage above.
- `src/timika50k_dataset/assembly/sync_to_project_copy.py`: mirrors
  `data/`+`labels/` from the canonical E: copy into this project's own C:
  copy.
- `src/timika50k_dataset/orchestrate.py`: runs every phase, this repo's
  and the other 3 projects', in order.
- `src/timika50k_dataset/anomaly/`: PSPNet/DenseNet embedding extraction,
  UMAP, Isolation Forest, HDBSCAN clustering, for image-quality QC. Not
  part of the core pipeline, a QC tool that surfaced a real finding (a
  CAAXR lateral-projection cluster) along the way.
- `src/timika50k_dataset/metadata/add_view_position.py`: added a
  `view_position` column to `preprocessed/manifest.csv`.
- `DATASHEET.md`, `croissant.json`: this repo's own documentation.

Real gap left: the orchestrator's own `--dry-run`/`--phases` logic and
the assembly/sync phases are verified directly (see Usage above); the 6
phases that shell out to the other 3 projects are verified only in that
each was independently run to completion by hand earlier this session,
not by an end-to-end orchestrator run since data/labels/preprocessed all
already exist correctly and a real rerun would mean an unneeded ~30GB
rebuild. The 4 remaining preprocessing variants (bone suppression, CLAHE,
and their combination, plus a no-leakage crop+normalize variant) are
separate, additive work, not started.
