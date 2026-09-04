# Datasheet for timika-30k

Following Gebru et al., "Datasheets for Datasets" (arXiv:1803.09010, CACM
2021), 7 sections: Motivation, Composition, Collection Process,
Preprocessing/Cleaning/Labeling, Uses, Distribution, Maintenance. Every
fact below is sourced from this project's own `dataset/manifest.md`,
`E:\dataset\timika-30k\README.md`/`manifest.yaml`, and
`context/intent.md`'s dated log, not invented for this document. Where a
question has no real answer yet, that is stated directly, not glossed
over, per this project's own standing rule against overclaiming.

## Motivation

**For what purpose was the dataset created?** To train and evaluate
segmentation models for the Timika chest X-ray severity score (Ralph et
al. 2010; Chakraborty et al. 2018), specifically organ-region (6 lung
zones) and disease-finding segmentation, for a tuberculosis-severity
scoring pipeline. See `context/intent.md`'s research question.

**Who created the dataset and on whose behalf?** Assembled by this
project (research-cxr-timika, TelU/Unpad/RSHS collaboration), not
downloaded from a single upstream source. It pools 7 already-public
datasets, each independently created by its own original authors (cited
per-source in `dataset/manifest.md`).

**Who funded it?** Not tracked by this project; each of the 7 component
datasets states its own funding independently (see each source's
`manifest.yaml`).

## Composition

**What do the instances represent?** Frontal (and a minority lateral)
chest radiographs, one image per instance, plus per-image labels where
available.

**How many instances?** 30,494 images across 6 sources (Shenzhen 662,
Montgomery 138, TBX11K 11,490, ChestX-Det 3,577, BIMCV-CAAXR 2,539,
SIIM-ACR Pneumothorax 12,088). COVID-19 Radiography Database (21,165
images) was dropped 2026-09-04, see Maintenance below.

**Does the dataset contain all possible instances, or a sample?** A
filtered sample of each source: only images already carrying some form
of localization (box, mask, or polygon) qualify; 3 further public
datasets available to this project (`chestxray_pneumonia_covid_tb`,
`chestxray_covid19_pneumonia`, `tb4200`, 17,767 images combined) were
excluded outright for being classification-only. Within qualifying
sources, confirmed internal/cross-source duplicates were also excluded
(TBX11K 216, ChestX-Det 1, BIMCV-CAAXR 52, SIIM-ACR 1), never deleted
from their own canonical copies, only excluded from timika-30k's `data/`.

**What data does each instance consist of?** A single grayscale
radiograph (native resolution/bit-depth varies by source, see
`preprocessed/manifest.csv`'s `orig_height`/`orig_width` columns), plus
whatever label layers exist for it (below).

**Labels.** Two layers, both with a per-file provenance manifest, not
just images with an implicit folder-encoded label:

- `labels/organ_region/`: 30,494/30,494 images, one 6-zone lung-region
  map each, entirely model-derived (PSPNet + fixed geometry, see
  Preprocessing below), never real ground truth for any image. 6
  `confidence: failed` (valid image, model didn't segment it), 0
  `confidence: invalid` (the only invalid case, a blank placeholder
  frame, was itself a COVID-19 Radiography Database image, dropped along
  with the rest of that source 2026-09-04).
- `labels/disease/`: 59,482 label files across 7 finding classes for a
  varying subset of images, per-file `method`/`confidence` in
  `labels/disease/manifest.csv`: real radiologist masks for 3 sources
  (Shenzhen, ChestX-Det, SIIM-ACR positives), confirmed negatives (a real
  label, not a guess) for whole-image-normal reads, SAM box-prompted
  masks (`confidence: high`) for sources with a real box and a real
  finding class, and a small `confidence: low` tier (854 rows) where the
  box exists but no finding class does. **Not every image has a disease
  label**: roughly 8,500 images (TBX11K `sick`, some BIMCV-CAAXR images,
  both sources' withheld competition test splits) have none at all,
  blocked on a disease segmentation model that does not yet work
  (`repo/timika30k_pseudolabels/README.md`, "Current status"). Smaller
  than before 2026-09-04's drop of COVID-19 Radiography Database, whose
  own 10,973 unlabelable images (3 of its 4 classes) were most of the
  prior ~19,500 gap.

**Is any information missing from individual instances?** Yes, tracked
explicitly, not silently: `patient_id`/`study_id` are `"-"` for sources
with no such concept in their release (SIIM-ACR's filenames are DICOM
instance UIDs, not patient identifiers); `view_position` is `"-"` for 4
of 6 sources with no real per-image or per-source projection info
available (checked directly against each source's own files, not
assumed absent).

**Are relationships between instances made explicit?** Patient/subject
grouping exists for sources that carry it (`patient_id` in
`preprocessed/manifest.csv`) and is respected by the split logic (below),
but is not resolved across sources (e.g. TB-4200's documented overlap
with Montgomery/Shenzhen does not apply here, since neither TB-4200 nor
its overlap partners are in timika-30k).

**Errors, noise, redundancies?** Known and marked, never silently
dropped: 2 BIMCV-CAAXR `BoundingBoxes.csv` rows have zero-area boxes
(degenerate, flagged in the disease manifest); 398 of BIMCV-CAAXR's own
study-level annotation rows name images this store's own filtered
download never received (an acquisition gap, not a labeling error); a
2026-09-03 embedding-based QC pass (`src/timika30k_dataset/anomaly/`)
found a real cluster of ~50 BIMCV-CAAXR images with degraded/unusual
framing, partially explained by lateral projection (27.8% of that
cluster vs. 2.3% of CAAXR overall) and partially still unexplained; not
yet excluded or corrected, flagged here as an open item.

## Collection Process

Not a single collection process: 7 independently-collected source
datasets, acquired by this project via heterogeneous, source-specific
methods over 2026-08-26 through 2026-08-29 (Kaggle API downloads,
direct NLM download, an access-request-gated custom download tool for
BIMCV-CAAXR, an authorized community rehost for SIIM-ACR after the
original host was shut down). Full per-source detail, including exact
URLs/mirrors and access gating, is in `dataset/manifest.md`, not
duplicated here since it would drift out of sync.

## Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling done?** Yes, in stages, each
with its own real script, not hand-applied:

1. Per-source canonicalization (grayscale/channel/bit-depth
   standardization, deduplication, quality flagging) at each source's own
   `C:\dataset\{name}\`/`E:\dataset\{name}\`, before timika-30k assembly.
2. `labels/organ_region/`: PSPNet (torchxrayvision `chestx_det.PSPNet`,
   validated 0.870 combined lung Dice on Montgomery ground truth) plus a
   fixed 6-zone geometry (`analyses/timika30k_organ_labels/`).
3. `labels/disease/`: real masks kept as-is for 3 sources; SAM ViT-B
   box-prompted masks (0.779 IoU vs. ground-truth boxes, this project's
   own measurement) for sources with real boxes; explicit confirmed
   negatives for whole-image-normal reads (`repo/timika30k_pseudolabels/`).
4. `preprocessed/`: one materialized preprocessing variant so far (of 5
   planned, see this repo's own README), grayscale load with own-image
   min/max rescale for 16-bit or under-windowed sources, center-crop to
   square, resize to 512x512 via the exact bilinear call the training
   pipeline itself uses (`analyses/timika30k_preprocessing/`).
5. `splits_official.csv`/`splits_fair.csv`: patient-grouped 8:1:1 splits,
   respecting each source's real official split where one exists
   (ChestX-Det, SIIM-ACR, TBX11K), self-split otherwise
   (`analyses/timika30k_preprocessing/build_splits.py`).

**Is the raw data also available?** Yes: each source's own canonical
copy (pre-filtering, pre-labeling) remains at `C:\dataset\{name}\`/
`E:\dataset\{name}\`, untouched by timika-30k's own exclusions (mark, not
delete, is this project's standing policy).

**Is the preprocessing software available?** Yes, every script named
above is in this project, most now consolidated under
`repo/timika30k_dataset/` per this repo's own scope.

## Uses

**What tasks has the dataset been used for?** Organ-region and (partial)
disease segmentation model training/evaluation, `repo/timika_score`.

**Is there anything about the composition or collection that might
impact future uses?** Yes, stated plainly: `labels/organ_region/` is
100% model-derived, never real ground truth, for every image, by
deliberate design (one consistent method beats a patchwork of real masks
for some images and predictions for others, see
`E:\dataset\timika-30k\README.md`'s "What each source's original labels
contribute"). Anyone needing real organ-region ground truth for
evaluation should use Montgomery's or ChestX-Det's own original masks
directly, not this layer. `labels/disease/`'s `confidence` column must be
read before trusting any given row, a `low`-confidence row (854 of
59,482) is a real box with no verified finding class, not a
radiologist-equivalent label.

**Are there tasks for which the dataset should not be used?** Any use
implying `labels/organ_region/` reflects real radiologist ground truth.
Any use redistributing the dataset itself, see Distribution below.

## Distribution

**Will the dataset be distributed to third parties?** No. Restricted,
`no_redistribution: true` at the derived-dataset level (4 of 6 sources
carry that flag individually, the most restrictive term governs the
whole pooled set). This is an accepted, closed decision as of 2026-09-02,
not an open question: this project does not pursue public redistribution
of timika-30k.

**How will it be distributed?** Internal only, to this project's own
sessions/collaborators, via its canonical location
(`E:\dataset\timika-30k\`) and a project-local copy
(`dataset/timika-30k/`).

**Licensing.** No single license; inherits each of its 6 sources' own
terms (`dataset/manifest.md`'s timika-30k row). One real, deliberately
closed exception: BIMCV-CAAXR's own annotation layer carries a
CC BY-NC-SA vs. CC0 conflict between its paper and its OSF metadata,
closed 2026-09-02 as moot for this project's purposes, since the
underlying image license already governs regardless, and this project
never redistributes.

## Maintenance

**Who maintains the dataset?** The dataset session on this project
(`research-cxr-timika`), current identity subject to change across
session restarts, see `context/intent.md`'s own session-continuity notes.

**How can errors be reported / will the dataset be updated?** Via this
project's own dataset-quality-marking convention: found problems get a
`quality_flags.csv.bz2` row at the affected source (mark, never silently
delete), and a dated `context/intent.md` entry. No external issue tracker
exists for this internal dataset.

**Will older versions continue to be supported?** No formal versioning
yet; this dataset's own composition has changed three times (37,033 to
51,659 images across 2 source folds-in, then 51,659 to 30,494 after
dropping COVID-19 Radiography Database and renaming timika-50k to
timika-30k, 2026-09-04), each change logged in
`E:\dataset\timika-30k\manifest.yaml`'s own notes rather than tagged as a
separate release.
