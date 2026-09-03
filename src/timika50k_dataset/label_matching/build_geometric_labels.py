"""Disease labels for the preprocessed/ (and, sharing the same grid,
preprocessed_bone_suppressed/) 512x512 coordinate frame, for every class
whose method needs only a geometric transform, not a re-segmentation:

- method starts "real_mask": a human drew this from the original pixels,
  the finding itself doesn't move when the image is cropped and resized,
  so center_crop_resize()'s own crop+resize (label_matching/transform.py's
  transform_mask, nearest-neighbor so the mask stays exact 0/255) is the
  whole job.
- method is a confirmed-negative kind (real_confirmed_negative_rle,
  confirmed_negative_whole_image, annotated_negative_for_class): the
  source label is already all-zero, so this writes a fresh 512x512 zero
  array rather than transforming actual zero content.

SAM/box-derived classes (sam_box_mask_*) are out of scope here: those get
re-segmented against the bone-suppressed pixels instead of geometrically
warped, see build_sam_labels.py.

Output: E:\\dataset\\timika-50k\\preprocessed_labels\\disease\\{class}\\{id}.png,
plus a manifest.csv mirroring labels/disease/manifest.csv's own schema
(same columns, image_relative_path now pointing at preprocessed/'s own
relative path instead of data/). Resumable: an id+class already in the
output manifest is skipped.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from timika50k_dataset.label_matching.transform import crop_box, transform_mask

TIMIKA_ROOT = Path(r"E:\dataset\timika-50k")
DISEASE_MANIFEST = TIMIKA_ROOT / "labels" / "disease" / "manifest.csv"
PREPROCESSED_MANIFEST = Path(r"C:\research\research-cxr-timika\dataset\timika-50k\preprocessed\manifest.csv")

OUT_ROOT = Path(r"E:\dataset\timika-50k\preprocessed_labels\disease")
OUT_MANIFEST = OUT_ROOT / "manifest.csv"
LOG_PATH = Path(r"C:\research\research-cxr-timika\logs\timika50k_geometric_labels.log")

CONFIRMED_NEGATIVE_METHODS = {
    "real_confirmed_negative_rle",
    "confirmed_negative_whole_image",
    "annotated_negative_for_class",
}
REAL_MASK_PREFIX = "real_mask"
IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def done_keys() -> set[tuple[str, str]]:
    if not OUT_MANIFEST.exists():
        return set()
    with OUT_MANIFEST.open(encoding="utf-8", newline="") as f:
        return {(r["id"], r["class"]) for r in csv.DictReader(f)}


def append_rows(rows: list[dict], fieldnames: list[str]) -> None:
    write_header = not OUT_MANIFEST.exists()
    with OUT_MANIFEST.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    log = setup_logging()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    with PREPROCESSED_MANIFEST.open(encoding="utf-8", newline="") as f:
        # "source/relative-within-source", the same key both manifests
        # resolve to: image_relative_path is "data/{source}/{rel}", and
        # original_relative_path is "{rel}" relative to that source's own
        # data/, so source + "/" + original_relative_path matches exactly.
        by_key = {
            f"{r['source']}/{r['original_relative_path']}".replace("\\", "/"): r
            for r in csv.DictReader(f)
        }

    with DISEASE_MANIFEST.open(encoding="utf-8", newline="") as f:
        disease_rows = list(csv.DictReader(f))

    relevant = [
        r for r in disease_rows
        if r["method"].startswith(REAL_MASK_PREFIX) or r["method"] in CONFIRMED_NEGATIVE_METHODS
    ]
    log.info(f"{len(disease_rows)} disease rows total, {len(relevant)} geometric-transform-eligible")

    skip = done_keys()
    fieldnames = [
        "id", "class", "source", "confidence", "positive", "method",
        "preprocessed_image_relative_path", "label_relative_path",
    ]
    pending = []
    unmatched = 0
    for r in relevant:
        key = r["image_relative_path"].split("data/", 1)[-1]
        pre_row = by_key.get(key)
        if pre_row is None:
            unmatched += 1
            continue
        out_key = (pre_row["id"], r["class"])
        if out_key in skip:
            continue
        pending.append((r, pre_row))
    log.info(f"{unmatched} disease rows had no matching preprocessed image, {len(pending)} pending, {len(skip)} already done")

    written = 0
    out_rows: list[dict] = []
    for disease_row, pre_row in tqdm(pending, desc="geometric labels"):
        h, w = int(pre_row["orig_height"]), int(pre_row["orig_width"])
        y0, x0, crop = crop_box(h, w)
        rel = Path(disease_row["class"]) / f"{pre_row['id']}"

        if disease_row["method"] in CONFIRMED_NEGATIVE_METHODS:
            out = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        else:
            src_mask = np.array(Image.open(TIMIKA_ROOT / "labels" / "disease" / disease_row["label_relative_path"]))
            transformed = transform_mask(src_mask, y0, x0, crop, DEVICE)
            out = (transformed > 0).astype(np.uint8) * 255

        out_path = OUT_ROOT / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out, mode="L").save(out_path)

        out_rows.append({
            "id": pre_row["id"],
            "class": disease_row["class"],
            "source": disease_row["source"],
            "confidence": disease_row["confidence"],
            "positive": disease_row["positive"],
            "method": disease_row["method"],
            "preprocessed_image_relative_path": pre_row["preprocessed_relative_path"],
            "label_relative_path": str(rel.with_suffix(".png")).replace("\\", "/"),
        })
        written += 1
        if len(out_rows) >= 2000:
            append_rows(out_rows, fieldnames)
            out_rows = []

    if out_rows:
        append_rows(out_rows, fieldnames)
    log.info(f"geometric labels complete: {written} written this run")


if __name__ == "__main__":
    main()
