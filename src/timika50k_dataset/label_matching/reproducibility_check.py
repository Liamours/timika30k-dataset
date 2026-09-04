"""Reproducibility check for the disease labels that are not exact by
construction. Real masks (Shenzhen/ChestX-Det/SIIM-ACR), confirmed
negatives, and RLE-decoded masks are deterministic transforms of fixed
input data, re-running them byte-for-byte reproduces the same file, no
check needed. The SAM-box-prompted classes (CAAXR, TBX11K/Montgomery
tb_lesion, and this repo's own bone-suppressed re-segmentation) are the
one method family without a determinism guarantee: floating-point
reduction order on GPU is not guaranteed identical run to run, so this
re-runs SAM on a small, seeded sample already on disk and reports IoU
between the original mask and the rerun, per source.

Seed 42, 5 images per group. IOU_BAR=0.98 is a different measurement
than sam_masks.py's own DEGENERATE_LOW/HIGH (those check a fresh mask's
area against its own prompting box; this checks a rerun's mask against
the mask already saved, two different masks compared to each other).
High IoU here means the label is safe to treat as fixed; anything lower
is reported plainly, not averaged away.
"""
from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
from PIL import Image

from timika50k_dataset.label_matching.transform import crop_box, transform_box
from timika50k_pseudolabels.build_tb_box_masks import load_montgomery, load_tbx11k
from timika50k_pseudolabels.caaxr_source import build_stem_index, load_boxes
from timika50k_pseudolabels.sam_masks import load_predictor, load_rgb_uint8, mask_from_box

TIMIKA_ROOT = Path(r"E:\dataset\timika-50k")
DATA_ROOT = TIMIKA_ROOT / "data"
LABEL_ROOT = TIMIKA_ROOT / "labels" / "disease"
BONE_SUPPRESSED_ROOT = Path(r"E:\dataset\timika-50k\preprocessed_bone_suppressed")
PREPROCESSED_LABELS_ROOT = Path(r"E:\dataset\timika-50k\preprocessed_labels\disease")
PREPROCESSED_LABELS_MANIFEST = PREPROCESSED_LABELS_ROOT / "manifest.csv"
PREPROCESSED_MANIFEST = Path(r"C:\research\research-cxr-timika\dataset\timika-50k\preprocessed\manifest.csv")

LOG_PATH = Path(r"C:\research\research-cxr-timika\logs\timika50k_reproducibility_check.log")
REPORT_PATH = LOG_PATH.with_suffix(".report.json")

SEED = 42
N_PER_GROUP = 5
IOU_BAR = 0.98


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return 1.0 if union == 0 else inter / union


def check_caaxr(predictor, rng: random.Random) -> list[dict]:
    boxes_by_stem = load_boxes(mappable_only=True)
    stem_to_rel = build_stem_index(DATA_ROOT / "caaxr")
    candidates = [(s, b) for s, b in boxes_by_stem.items() if s in stem_to_rel]
    sample = rng.sample(candidates, min(N_PER_GROUP, len(candidates)))

    results = []
    for stem, boxes in sample:
        rel = stem_to_rel[stem]
        image = load_rgb_uint8(DATA_ROOT / "caaxr" / rel)
        h, w = image.shape[:2]
        predictor.set_image(image)
        by_class: dict[str, list] = {}
        for b in boxes:
            by_class.setdefault(b.rshs_class, []).append((b.x1, b.y1, b.x2, b.y2))
        for rshs_class, box_list in by_class.items():
            saved_path = LABEL_ROOT / rshs_class / "caaxr" / rel.with_suffix(".png")
            if not saved_path.exists():
                continue
            saved = np.array(Image.open(saved_path)) > 127
            union = np.zeros((h, w), dtype=bool)
            for x1, y1, x2, y2 in box_list:
                union |= mask_from_box(predictor, (x1, y1, x2, y2), (h, w)).mask
            results.append({"source": "caaxr", "id": str(rel), "class": rshs_class, "iou": iou(saved, union)})
    return results


def check_tb(predictor, rng: random.Random, source: str, loader) -> list[dict]:
    boxes_by_rel = loader()
    items = list(boxes_by_rel.items())
    sample = rng.sample(items, min(N_PER_GROUP, len(items)))

    results = []
    for rel, boxes in sample:
        saved_path = LABEL_ROOT / "tb_lesion" / source / rel.with_suffix(".png")
        if not saved_path.exists():
            continue
        saved = np.array(Image.open(saved_path)) > 127
        image = load_rgb_uint8(DATA_ROOT / source / rel)
        h, w = image.shape[:2]
        predictor.set_image(image)
        union = np.zeros((h, w), dtype=bool)
        for b in boxes:
            union |= mask_from_box(predictor, (b.x1, b.y1, b.x2, b.y2), (h, w)).mask
        results.append({"source": source, "id": str(rel), "class": "tb_lesion", "iou": iou(saved, union)})
    return results


def check_bone_suppressed(predictor, rng: random.Random) -> list[dict]:
    with PREPROCESSED_LABELS_MANIFEST.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["method"] == "sam_box_mask_bone_suppressed_resegmented"]
    with PREPROCESSED_MANIFEST.open(encoding="utf-8", newline="") as f:
        by_id = {r["id"]: r for r in csv.DictReader(f)}

    caaxr_boxes = load_boxes(mappable_only=True)
    stem_to_rel = build_stem_index(DATA_ROOT / "caaxr")
    # Path.__str__ uses backslashes on Windows; normalize every key to
    # forward-slash so this matches source_rel_for()'s own normalized
    # lookup key, not silently miss every row via a separator mismatch.
    rel_to_stem = {str(v).replace("\\", "/"): k for k, v in stem_to_rel.items()}
    tb_boxes = {
        **{f"tbx11k/{k.as_posix()}": v for k, v in load_tbx11k().items()},
        **{f"montgomery/{k.as_posix()}": v for k, v in load_montgomery().items()},
    }

    sample = rng.sample(rows, min(N_PER_GROUP, len(rows)))
    results = []
    for row in sample:
        pre_row = by_id.get(row["id"])
        if pre_row is None:
            continue
        h, w = int(pre_row["orig_height"]), int(pre_row["orig_width"])
        y0, x0, crop = crop_box(h, w)
        bone_img_path = BONE_SUPPRESSED_ROOT / pre_row["preprocessed_relative_path"]
        if not bone_img_path.exists():
            continue
        gray = np.array(Image.open(bone_img_path))
        rgb = np.stack([gray, gray, gray], axis=-1)
        predictor.set_image(rgb)

        source = pre_row["source"]
        boxes_orig = source_rel_for(source, pre_row["original_relative_path"], row["class"], caaxr_boxes, rel_to_stem, tb_boxes)
        if not boxes_orig:
            continue

        saved_path = PREPROCESSED_LABELS_ROOT / row["label_relative_path"]
        if not saved_path.exists():
            continue
        saved = np.array(Image.open(saved_path)) > 127

        union = np.zeros((512, 512), dtype=bool)
        for x1, y1, x2, y2 in boxes_orig:
            tx1, ty1, tx2, ty2, area_kept = transform_box(x1, y1, x2, y2, y0, x0, crop)
            if area_kept < 0.1:
                continue
            union |= mask_from_box(predictor, (round(tx1), round(ty1), round(tx2), round(ty2)), (512, 512)).mask
        results.append({"source": f"{source}_bone_suppressed", "id": row["id"], "class": row["class"], "iou": iou(saved, union)})
    return results


def source_rel_for(source: str, original_relative_path: str, target_class: str, caaxr_boxes, rel_to_stem, tb_boxes) -> list:
    if source == "caaxr":
        # A CAAXR image can carry boxes for several classes; build_sam_labels.py's
        # own caaxr_jobs() groups by class before unioning, so the rerun must
        # filter to target_class too, or a check for one class silently pulls
        # in another class's boxes from the same image and produces a bogus
        # low IoU against the target class's own saved mask.
        stem = rel_to_stem.get(original_relative_path.replace("\\", "/"))
        boxes = caaxr_boxes.get(stem, []) if stem else []
        return [(b.x1, b.y1, b.x2, b.y2) for b in boxes if b.rshs_class == target_class]
    # tb_lesion (and Montgomery's single named extra class, when present)
    # always uses the image's full TB box set regardless of which of the
    # two classes is being checked, matching build_tb_box_masks.py's own
    # build_source(): no per-class filtering needed here.
    key = f"{source}/{original_relative_path.replace(chr(92), '/')}"
    boxes = tb_boxes.get(key, [])
    return [(b.x1, b.y1, b.x2, b.y2) for b in boxes]


def main() -> None:
    log = setup_logging()
    rng = random.Random(SEED)
    predictor = load_predictor()

    all_results = []
    all_results += check_caaxr(predictor, rng)
    all_results += check_tb(predictor, rng, "tbx11k", load_tbx11k)
    all_results += check_tb(predictor, rng, "montgomery", load_montgomery)
    all_results += check_bone_suppressed(predictor, rng)

    REPORT_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    by_source: dict[str, list[float]] = {}
    for r in all_results:
        by_source.setdefault(r["source"], []).append(r["iou"])

    log.info(f"reproducibility check: {len(all_results)} samples, seed={SEED}, bar={IOU_BAR}")
    for source, ious in by_source.items():
        below_bar = [i for i in ious if i < IOU_BAR]
        log.info(f"  {source}: n={len(ious)} mean_iou={sum(ious)/len(ious):.4f} min_iou={min(ious):.4f} below_bar={len(below_bar)}")
    overall_below = [r for r in all_results if r["iou"] < IOU_BAR]
    if overall_below:
        log.info(f"below-bar rows: {json.dumps(overall_below, indent=2)}")
    log.info(f"full report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
