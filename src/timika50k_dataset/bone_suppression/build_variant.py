"""Bone-suppression preprocessing variant: runs every image in
dataset/timika-50k/preprocessed/ (the existing 512x512 center-crop+resize
variant) through ResNet-BS, writing the result to
E:\\dataset\\timika-50k\\preprocessed_bone_suppressed\\images\\, same id
and manifest schema as the source, plus a bone_suppressed_relative_path
column. Written to E: rather than following preprocessed/'s own C:-only
convention: C: had under 12 GB free when this was built, E: had 135 GB.

No extra resize: the model is fully convolutional and runs directly on
the already-512x512, already-square input (see model.py's own docstring
for why this is deliberate, not just convenient), so the output aligns
pixel-for-pixel with the source preprocessed image, and this variant
needs no geometry of its own for any existing label to track.

Resumable: an id already in the output manifest is skipped.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from timika50k_dataset.bone_suppression.model import load_model

SOURCE_ROOT = Path(r"C:\research\research-cxr-timika\dataset\timika-50k\preprocessed")
SOURCE_MANIFEST = SOURCE_ROOT / "manifest.csv"
WEIGHTS_NPZ = Path(r"C:\research\research-cxr-timika\weights\resnet-bonesuppression-jsrt\resnet_bs_weights.npz")

# preprocessed_relative_path already reads "images/{id}.png", relative to
# SOURCE_ROOT/OUT_ROOT themselves, not to an images/ subfolder joined here.
OUT_ROOT = Path(r"E:\dataset\timika-50k\preprocessed_bone_suppressed")
OUT_MANIFEST = OUT_ROOT / "manifest.csv"
LOG_PATH = Path(r"C:\research\research-cxr-timika\logs\timika50k_bone_suppression.log")

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )


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
    setup_logging()
    log = logging.getLogger(__name__)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    with SOURCE_MANIFEST.open(encoding="utf-8", newline="") as f:
        source_rows = list(csv.DictReader(f))
    fieldnames = list(source_rows[0].keys()) + ["bone_suppressed_relative_path"]

    skip = done_ids()
    pending = [r for r in source_rows if r["id"] not in skip]
    log.info(f"{len(source_rows)} total, {len(skip)} already done, {len(pending)} pending, device={DEVICE}")

    model = load_model(WEIGHTS_NPZ, DEVICE)

    written = 0
    for i in tqdm(range(0, len(pending), BATCH_SIZE), desc="bone suppression"):
        batch = pending[i : i + BATCH_SIZE]
        arrays = [np.array(Image.open(SOURCE_ROOT / r["preprocessed_relative_path"])) for r in batch]
        x = torch.from_numpy(np.stack(arrays)).float().div(255.0).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            pred = model(x)

        pred_u8 = pred.squeeze(1).clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()

        out_rows = []
        for r, out_img in zip(batch, pred_u8):
            rel = Path(r["preprocessed_relative_path"])
            out_path = OUT_ROOT / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(out_img, mode="L").save(out_path)
            out_rows.append({**r, "bone_suppressed_relative_path": str(rel)})

        append_rows(out_rows, fieldnames)
        written += len(out_rows)

    log.info(f"bone suppression complete: {written} written this run, {len(skip) + written} total")


if __name__ == "__main__":
    main()
