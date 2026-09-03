"""Text-free sample grid for the GitHub README: 2 examples per source (7
sources x 2 = 14 cells), each the preprocessed image with its organ-region
mask (cyan outline + low-opacity cyan fill, all 6 zones merged to one
color, since the grid draws no per-zone distinction) and, where a disease
label exists, the union of its disease masks (orange outline + low-opacity
orange fill, all classes merged to one color). No captions, labels, or
axes anywhere in the output image itself.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.morphology import erosion
from tqdm import tqdm

TIMIKA_ROOT = Path(r"E:\dataset\timika-50k")
DISEASE_MANIFEST = TIMIKA_ROOT / "labels" / "disease" / "manifest.csv"
OUT_PATH = Path(r"C:\research\research-cxr-timika\repo\timika50k_dataset\sample_grid.png")

SOURCES = ("shenzhen", "montgomery", "tbx11k", "chestxdet", "covidrad", "caaxr", "siimacr")
CYAN = np.array([0, 210, 210], dtype=np.float32)
ORANGE = np.array([255, 140, 0], dtype=np.float32)
FILL_ALPHA = 0.28
THUMB = 384
GAP = 6


def load_gray_rgb(path: Path) -> np.ndarray:
    # No PIL .convert("L") here: for a 16-bit source (CAAXR) it does a lossy
    # bit-truncation before any rescale, washing the image out (this
    # project's own documented CAAXR bug, see build_preprocessed.py). Load
    # the raw array and rescale via its own min/max first instead.
    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo) * 255.0
    gray = img.astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


def overlay(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    if not mask.any():
        return rgb
    eroded = erosion(mask, np.ones((3, 3), dtype=bool))
    edge = mask & ~eroded
    out = rgb.astype(np.float32)
    out[mask] = out[mask] * (1 - FILL_ALPHA) + color * FILL_ALPHA
    out[edge] = color
    return out.clip(0, 255).astype(np.uint8)


def pick_examples(source: str, positive_by_source: dict[str, list[dict]]) -> list[Path]:
    candidates = positive_by_source.get(source, [])
    chosen = [Path(r["image_relative_path"]) for r in candidates[:2]]
    if len(chosen) < 2:
        data_root = TIMIKA_ROOT / "data" / source
        for p in data_root.rglob("*"):
            if not p.is_file():
                continue
            rel = Path("data") / source / p.relative_to(data_root)
            if rel not in chosen:
                chosen.append(rel)
            if len(chosen) == 2:
                break
    return chosen[:2]


def as_png(rel: Path) -> Path:
    # Label masks are always .png regardless of the source image's own
    # extension (TBX11K/CAAXR use .webp); length-based stripping, not
    # with_suffix(), since some source filenames (SIIM-ACR) have internal
    # dots that with_suffix() would mangle.
    rel_str = str(rel)
    stem = rel_str[: -len(rel.suffix)] if rel.suffix else rel_str
    return Path(stem + ".png")


def disease_union_mask(image_rel: Path, shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=bool)
    source_rel = as_png(image_rel.relative_to(Path("data") / image_rel.parts[1]))
    for class_dir in (TIMIKA_ROOT / "labels" / "disease").iterdir():
        if not class_dir.is_dir():
            continue
        candidate = class_dir / image_rel.parts[1] / source_rel
        if candidate.exists():
            m = np.array(Image.open(candidate))
            if m.shape == shape:
                union |= m > 127
    return union


def build_cell(image_rel: Path) -> np.ndarray:
    rgb = load_gray_rgb(TIMIKA_ROOT / image_rel)
    h, w = rgb.shape[:2]

    source_rel = as_png(image_rel.relative_to(Path("data") / image_rel.parts[1]))
    organ_path = TIMIKA_ROOT / "labels" / "organ_region" / image_rel.parts[1] / source_rel
    if organ_path.exists():
        organ = np.array(Image.open(organ_path)) > 0
        if organ.shape == (h, w):
            rgb = overlay(rgb, organ, CYAN)

    disease = disease_union_mask(image_rel, (h, w))
    rgb = overlay(rgb, disease, ORANGE)

    return np.array(Image.fromarray(rgb).resize((THUMB, THUMB)))


def main() -> None:
    with DISEASE_MANIFEST.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    positive_by_source: dict[str, list[dict]] = {}
    for r in rows:
        if r["positive"] == "True" and r["method"] != "sam_box_mask_tb_no_class":
            positive_by_source.setdefault(r["source"], []).append(r)
    tb_by_image: dict[str, dict] = {}
    for r in rows:
        if r["source"] == "tbx11k" and r["positive"] == "True":
            tb_by_image.setdefault(r["image_relative_path"], r)
    positive_by_source["tbx11k"] = list(tb_by_image.values())

    rows_of_cells = []
    for source in tqdm(SOURCES, desc="sources"):
        examples = pick_examples(source, positive_by_source)
        cells = [build_cell(ex) for ex in examples]
        while len(cells) < 2:
            cells.append(np.zeros((THUMB, THUMB, 3), dtype=np.uint8))
        gap_col = np.zeros((THUMB, GAP, 3), dtype=np.uint8)
        rows_of_cells.append(np.concatenate([cells[0], gap_col, cells[1]], axis=1))

    gap_row = np.zeros((GAP, rows_of_cells[0].shape[1], 3), dtype=np.uint8)
    grid_parts = []
    for i, row in enumerate(rows_of_cells):
        if i:
            grid_parts.append(gap_row)
        grid_parts.append(row)
    grid = np.concatenate(grid_parts, axis=0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(OUT_PATH)
    print(f"saved {OUT_PATH}, shape={grid.shape}")


if __name__ == "__main__":
    main()
