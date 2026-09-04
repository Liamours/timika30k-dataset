# timika30k_dataset

Reproducibility pipeline for timika-30k: assemble it from its 6 already-downloaded source datasets, drive labels/pseudo-labels, and produce the preprocessing variants below. Scaffolded 2026-09-03; the dataset session owns building this out, per this project's existing split between dataset construction/convention work and experiment execution. Renamed 2026-09-04 from timika-50k/timika50k_dataset after dropping COVID-19 Radiography Database, see Status below.

![2 examples per source: raw image, organ-region overlay (cyan), disease overlay (orange) where a label exists](sample_grid.png)

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
| Build code: assembly (phase 2 below) | Done, `src/timika30k_dataset/assembly/build_data.py` |
| Build code: labels/pseudo-labels | Done, in `repo/timika30k_pseudolabels` (called into, not reimplemented here) |
| Build code: preprocessing | 4 of 5 variants done, `analyses/timika30k_preprocessing` + `src/timika30k_dataset/bone_suppression/` + `src/timika30k_dataset/clahe/` |
| Evaluation/QC code | Done: 3-agent independent verification pattern used throughout (leakage, official-split fidelity, ratio integrity, box-mask geometry, confirmed-negative source rules), embedding-based anomaly detection (`src/timika30k_dataset/anomaly/`) |
| "Pretrained model" equivalent (the finished artifact) | Done, `E:\dataset\timika-30k\` + `dataset/timika-30k/` project copy |
| README with exact reproduction commands | Done, `src/timika30k_dataset/orchestrate.py` + Usage below |
| Results table | N/A, this is a dataset repo, not a results-reporting one; see per-source counts in DATASHEET.md instead |

## Scope, and what already exists elsewhere

Download has no code, by design (see below). Assembly's real code lives in this repo. Labels and preprocessing already have real code in 2 other repos; check there before writing anything new.

- **Download**: the 6 sources (Shenzhen, Montgomery, TBX11K, ChestX-Det, BIMCV-CAAXR, SIIM-ACR Pneumothorax) were each acquired separately over time, heterogeneous source-specific methods (Kaggle API, direct NLM download, an access-gated custom tool, an authorized community rehost); see `dataset/manifest.md`'s per-source rows for how and from where. Not scripted, and likely never will be: 4 different access methods, 1 access-request gate, no single API covers all 6. Real per-source acquisition detail lives in `dataset/manifest.md`, not duplicated here.
- **Assembly** (phase 2, `src/timika30k_dataset/assembly/build_data.py`): the real, scripted gap this repo closes. Takes the 6 sources as already downloaded/canonicalized at `E:\dataset\{name}\` (assumed present, not fetched by this script) and applies timika-30k's own eligibility rule (`dataset/manifest.md`'s timika-30k row): drop TBX11K's own `data/extra/` (bonus augmentation, not part of its core release) and drop any file either source's own `labels/quality_flags.csv.bz2` marks `status=="duplicate"`. `--dry-run` reports per-source counts with no file copy; verified 2026-09-04 (after dropping COVID-19 Radiography Database) against the real quality_flags data to match the documented 30,494-image total (662/138/11,490/3,577/2,539/12,088) exactly.
- **Labels / pseudo-labels**: `repo/timika30k_pseudolabels` already generates the disease pseudo-label layer per source group (DS group built, 27,631 files; DB and OS groups blocked on the disease model retrain). Call into that repo, do not reimplement it here.
- **Preprocessing variants**: `analyses/timika30k_preprocessing` already builds one variant, a per-image min/max rescale to 8-bit matching `DiseaseSegDataset`'s own training-time preprocessing, plus `splits_official.csv`/`splits_fair.csv`. The variants below are new, and should reuse its split manifests rather than re-deriving train/val/test membership.

## Usage

`orchestrate.py` runs every phase already scripted, in dependency order,
each via `uv run` inside that phase's own uv project (assembly and
metadata here, labels in `repo/timika30k_pseudolabels`, preprocessing in
`analyses/timika30k_preprocessing` -- 4 separate dependency sets, so this
script shells out rather than importing across them):

```
uv run python -m timika30k_dataset.orchestrate --dry-run
uv run python -m timika30k_dataset.orchestrate
uv run python -m timika30k_dataset.orchestrate --phases assembly,sync_to_project_copy
```

`--dry-run` prints the exact command sequence, runs nothing. `--phases`
restricts to a comma-separated subset of: `assembly`,
`organ_region_labels`, `organ_region_manifest`,
`disease_confirmed_negatives`, `disease_caaxr_box_masks`,
`disease_tb_box_masks`, `disease_ds_group`, `disease_manifest`,
`sync_to_project_copy`, `preprocessing`, `splits`, `view_position`, the
same order a full run uses.

`assembly` through `disease_manifest` (the first 8 phases) read and write
`E:\dataset\timika-30k\`, the canonical copy. `sync_to_project_copy`
mirrors `data/` and `labels/` from there into `dataset/timika-30k/`, this
project's own second copy (`dataset/manifest.md`'s timika-30k row).
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

1. Raw image: no processing. Already satisfied by `data/` itself, nothing to build.
2. Crop and normalize, no leakage across split: `analyses/timika30k_preprocessing`, done.
3. Bone suppression: done, `src/timika30k_dataset/bone_suppression/`. Ported the vendored Keras ResNet-BS (`external/CXR-bone-suppression`, Rajaraman et al. 2021) to PyTorch, reusing an existing validated port (`analyses/bone_suppression_sample/pytorch_port/`) rather than re-extracting from the notebook. Runs directly on the already-512x512 preprocessed image (fully convolutional, no fixed input size despite the original notebook always resizing to 256x256 first), so it needs no geometry of its own for labels to track. 30,494/30,494, `E:\dataset\timika-30k\preprocessed_bone_suppressed\`.
4. Bone suppression + CLAHE: done, `src/timika30k_dataset/clahe/build_bone_suppression_clahe_variant.py`. 30,494/30,494, `E:\dataset\timika-30k\preprocessed_bone_suppression_clahe\`.
5. CLAHE only: done, `src/timika30k_dataset/clahe/build_clahe_variant.py`. `cv2.createCLAHE`, `clip_limit=2.0`/`tile_grid_size=(8,8)` (`clahe/apply.py`'s own docstring has the literature basis, a corroborated range rather than one pinned citation). 30,494/30,494, `E:\dataset\timika-30k\preprocessed_clahe\`.

## Status

Scaffolded 2026-09-03; the core pipeline (assembly through view_position)
finished the same day:

- `src/timika30k_dataset/assembly/build_data.py`: phase 2, the eligibility
  filter and copy into `data/`, see Usage above.
- `src/timika30k_dataset/assembly/sync_to_project_copy.py`: mirrors
  `data/`+`labels/` from the canonical E: copy into this project's own C:
  copy.
- `src/timika30k_dataset/orchestrate.py`: runs every phase, this repo's
  and the other 3 projects', in order.
- `src/timika30k_dataset/anomaly/`: PSPNet/DenseNet embedding extraction,
  UMAP, Isolation Forest, HDBSCAN clustering, for image-quality QC. Not
  part of the core pipeline, a QC tool that surfaced a real finding (a
  CAAXR lateral-projection cluster) along the way.
- `src/timika30k_dataset/metadata/add_view_position.py`: added a
  `view_position` column to `preprocessed/manifest.csv`.
- `DATASHEET.md`, `croissant.json`: this repo's own documentation.
- `src/timika30k_dataset/samples/build_readme_grid.py`: builds
  `sample_grid.png` above.
- `src/timika30k_dataset/bone_suppression/`: PyTorch ResNet-BS
  (Rajaraman et al. 2021), ported from the vendored Keras model and
  verified against its predictions (`analyses/bone_suppression_sample/`);
  runs directly on the already-512x512 preprocessed image, no extra
  resize, output written to `E:\dataset\timika-30k\preprocessed_bone_suppressed\`
  since C: was near capacity when this was built.
- `src/timika30k_dataset/label_matching/`: disease labels re-registered
  to `preprocessed/`'s own 512x512 grid (real masks: geometric transform
  only; SAM/box-derived masks: re-segmented against the bone-suppressed
  pixels rather than warped, since a stale mask would carry over whatever
  bone confounding it had before suppression). Output at
  `E:\dataset\timika-30k\preprocessed_labels\`. Includes
  `reproducibility_check.py`: real masks/RLE/confirmed-negatives are
  deterministic by construction, so this re-runs SAM (the one method
  family without a determinism guarantee) on a seeded sample and reports
  IoU against the saved mask; 20/20 sampled rows across 4 source/method
  groups landed at 0.9996-1.0 (found and fixed a real bug in the check
  script itself along the way, see its own module docstring).
- `src/timika30k_dataset/clahe/`: variants 4 and 5, see below.

The orchestrator has been run end to end for real, not only
dry-run-verified: every phase, 2026-09-04. All 5 preprocessing variants
are now built. **2026-09-04, later**: COVID-19 Radiography Database
dropped entirely (real basis: no disease-level box or mask for any of
its 3 abnormal classes, never mapped onto this project's own taxonomy;
costs zero disease-class coverage) and this repo renamed
timika50k_dataset -> timika30k_dataset, matching the dataset's own
timika-50k -> timika-30k rename (30,494 images); see
`context/intent.md`'s 2026-09-04 entry for the full rationale. Still
open: the disease-label layer itself is incomplete (the
disease-segmentation model needed for ~8,500 images, see DATASHEET.md's
Composition section, remains unusable after 2 failed retrains; a fix
(foreground-oversampled batch sampling, `repo/timika_segmentation_models`)
is built and smoke-tested but the actual retrain, an up-to-~28-hour run,
hasn't been launched).
