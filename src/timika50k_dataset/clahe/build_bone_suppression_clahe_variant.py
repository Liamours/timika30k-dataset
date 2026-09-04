"""Preprocessing variant 4 (bone suppression + CLAHE): applies CLAHE to
bone_suppression/build_variant.py's own output, same clip_limit/tile_grid_size
as variant 5 (clahe/apply.py). Written to
E:\\dataset\\timika-50k\\preprocessed_bone_suppression_clahe\\.

CPU-only (OpenCV), no GPU contention. Resumable: an id already in the
output manifest is skipped.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from timika50k_dataset.clahe.apply import apply_clahe

SOURCE_ROOT = Path(r"E:\dataset\timika-50k\preprocessed_bone_suppressed")
SOURCE_MANIFEST = SOURCE_ROOT / "manifest.csv"

OUT_ROOT = Path(r"E:\dataset\timika-50k\preprocessed_bone_suppression_clahe")
OUT_MANIFEST = OUT_ROOT / "manifest.csv"
LOG_PATH = Path(r"C:\research\research-cxr-timika\logs\timika50k_bone_suppression_clahe.log")


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def done_ids() -> set[str]:
    if not OUT_MANIFEST.exists():
        return set()
    with OUT_MANIFEST.open(encoding="utf-8", newline="") as f:
        return {r["id"] for r in csv.DictReader(f)}


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

    with SOURCE_MANIFEST.open(encoding="utf-8", newline="") as f:
        source_rows = list(csv.DictReader(f))
    fieldnames = list(source_rows[0].keys()) + ["bone_suppression_clahe_relative_path"]

    skip = done_ids()
    pending = [r for r in source_rows if r["id"] not in skip]
    log.info(f"{len(source_rows)} total, {len(skip)} already done, {len(pending)} pending")

    out_rows: list[dict] = []
    for r in tqdm(pending, desc="bone suppression + clahe"):
        gray = np.array(Image.open(SOURCE_ROOT / r["bone_suppressed_relative_path"]))
        out_img = apply_clahe(gray)

        rel = Path(r["bone_suppressed_relative_path"])
        out_path = OUT_ROOT / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out_img, mode="L").save(out_path)

        out_rows.append({**r, "bone_suppression_clahe_relative_path": str(rel)})
        if len(out_rows) >= 2000:
            append_rows(out_rows, fieldnames)
            out_rows = []

    if out_rows:
        append_rows(out_rows, fieldnames)
    log.info(f"bone suppression + clahe complete: {len(pending)} written this run, {len(skip) + len(pending)} total")


if __name__ == "__main__":
    main()
