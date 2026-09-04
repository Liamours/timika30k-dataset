"""Bone-suppression preprocessing variant: runs every image in
dataset/timika-30k/preprocessed/ (the existing 512x512 center-crop+resize
variant) through ResNet-BS, writing the result to
E:\\dataset\\timika-30k\\preprocessed_bone_suppressed\\images\\, same id
and manifest schema as the source, plus a bone_suppressed_relative_path
column. Written to E: rather than following preprocessed/'s own C:-only
convention: C: had under 12 GB free when this was built, E: had 135 GB.

No extra resize: the model is fully convolutional and runs directly on
the already-512x512, already-square input (see model.py's own docstring
for why this is deliberate, not just convenient), so the output aligns
pixel-for-pixel with the source preprocessed image, and this variant
needs no geometry of its own for any existing label to track.

A first version loaded, ran, and saved each batch fully sequentially,
leaving the GPU idle during every disk read and write (visibly a sawtooth
in Task Manager's GPU graph, not a smooth line). Fixed here two ways: a
DataLoader with background workers prefetches the next batch's images
while the GPU is still computing the current one, and each batch's 16
output images save in parallel on a thread pool instead of one at a time.
Manifest rows for a batch are still only appended after all of that
batch's saves finish (coding.md's write-before-mark-done ordering), so
the speedup is free of that correctness cost.

Resumable: an id already in the output manifest is skipped.
"""
from __future__ import annotations

import csv
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from timika30k_dataset.bone_suppression.model import load_model

SOURCE_ROOT = Path(r"C:\research\research-cxr-timika\dataset\timika-30k\preprocessed")
SOURCE_MANIFEST = SOURCE_ROOT / "manifest.csv"
WEIGHTS_NPZ = Path(r"C:\research\research-cxr-timika\weights\resnet-bonesuppression-jsrt\resnet_bs_weights.npz")

# preprocessed_relative_path already reads "images/{id}.png", relative to
# SOURCE_ROOT/OUT_ROOT themselves, not to an images/ subfolder joined here.
OUT_ROOT = Path(r"E:\dataset\timika-30k\preprocessed_bone_suppressed")
OUT_MANIFEST = OUT_ROOT / "manifest.csv"
LOG_PATH = Path(r"C:\research\research-cxr-timika\logs\timika30k_bone_suppression.log")

BATCH_SIZE = 16
LOAD_WORKERS = 4
SAVE_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PreprocessedImages(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        r = self.rows[idx]
        arr = np.array(Image.open(SOURCE_ROOT / r["preprocessed_relative_path"]))
        return torch.from_numpy(arr).float().div(255.0).unsqueeze(0), idx


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


def save_one(rel: Path, out_img: np.ndarray) -> None:
    out_path = OUT_ROOT / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out_img, mode="L").save(out_path)


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
    dataset = PreprocessedImages(pending)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=LOAD_WORKERS,
        pin_memory=(DEVICE.type == "cuda"), shuffle=False,
    )

    written = 0
    with ThreadPoolExecutor(max_workers=SAVE_WORKERS) as pool:
        for x, idxs in tqdm(loader, desc="bone suppression", total=len(loader)):
            x = x.to(DEVICE, non_blocking=True)

            with torch.no_grad(), torch.autocast(device_type=DEVICE.type):
                pred = model(x)

            pred_u8 = pred.squeeze(1).clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()

            out_rows = []
            futures = []
            for idx, out_img in zip(idxs.tolist(), pred_u8):
                r = pending[idx]
                rel = Path(r["preprocessed_relative_path"])
                futures.append(pool.submit(save_one, rel, out_img))
                out_rows.append({**r, "bone_suppressed_relative_path": str(rel)})

            for fut in futures:
                fut.result()
            append_rows(out_rows, fieldnames)
            written += len(out_rows)

    log.info(f"bone suppression complete: {written} written this run, {len(skip) + written} total")


if __name__ == "__main__":
    main()
