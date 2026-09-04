"""Disease labels for classes whose real method is a SAM box-mask
(CAAXR's per-finding masks, TBX11K/Montgomery's tb_lesion): re-segments
against the bone-suppressed 512x512 pixels instead of geometrically
warping the mask SAM already produced from the original image. Bone
suppression's whole point is making a finding easier to see past the
overlying ribs and clavicles (Rajaraman et al. 2021); reusing the old,
bone-confounded mask would throw that away, and could let SAM's own box
prompt latch onto a now-fainter rib edge rather than the finding itself
if simply re-run blind. So: transform the box only (label_matching/transform.py,
the same crop+scale every other label here uses), then run SAM fresh
inside that box against the bone-suppressed image.

Requires bone_suppression/build_variant.py to have already produced
E:\\dataset\\timika-30k\\preprocessed_bone_suppressed\\images\\. Depends
on repo/timika30k_pseudolabels for box loading (caaxr_source.py,
build_tb_box_masks.py's load_tbx11k/load_montgomery) and the SAM wrapper
(sam_masks.py), reused rather than reimplemented, per this project's own
timika30k-pseudolabels path dependency already added for this.

Output: E:\\dataset\\timika-30k\\preprocessed_labels\\disease\\{class}\\{id}.png,
appended to the same manifest.csv build_geometric_labels.py writes, same
schema, method suffixed "_bone_suppressed_resegmented" so provenance
stays distinguishable from the geometric-transform rows. Resumable.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from timika30k_dataset.label_matching.transform import crop_box, transform_box
from timika30k_pseudolabels.build_tb_box_masks import load_montgomery, load_tbx11k, montgomery_reading_classes
from timika30k_pseudolabels.caaxr_source import build_stem_index, duplicate_stems, load_boxes
from timika30k_pseudolabels.sam_masks import load_predictor, mask_from_box

PREPROCESSED_MANIFEST = Path(r"C:\research\research-cxr-timika\dataset\timika-30k\preprocessed\manifest.csv")
BONE_SUPPRESSED_ROOT = Path(r"E:\dataset\timika-30k\preprocessed_bone_suppressed")

OUT_ROOT = Path(r"E:\dataset\timika-30k\preprocessed_labels\disease")
OUT_MANIFEST = OUT_ROOT / "manifest.csv"
LOG_PATH = Path(r"C:\research\research-cxr-timika\logs\timika30k_sam_labels.log")
QC_PATH = LOG_PATH.with_suffix(".qc.jsonl")

TB_LESION_CLASS = "tb_lesion"
IMG_SIZE = 512


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


def preprocessed_index() -> dict[str, dict]:
    with PREPROCESSED_MANIFEST.open(encoding="utf-8", newline="") as f:
        return {f"{r['source']}/{r['original_relative_path']}".replace("\\", "/"): r for r in csv.DictReader(f)}


def caaxr_jobs(by_key: dict[str, dict]) -> list[tuple[dict, str, list]]:
    dupes = duplicate_stems()
    boxes_by_stem = load_boxes(mappable_only=True)
    stem_to_rel = build_stem_index(Path(r"E:\dataset\timika-30k\data\caaxr"))
    jobs = []
    for stem, boxes in boxes_by_stem.items():
        if stem in dupes:
            continue
        rel = stem_to_rel.get(stem)
        if rel is None:
            continue
        key = f"caaxr/{str(rel).replace(chr(92), '/')}"
        pre_row = by_key.get(key)
        if pre_row is None:
            continue
        by_class: dict[str, list] = {}
        for b in boxes:
            by_class.setdefault(b.rshs_class, []).append((b.x1, b.y1, b.x2, b.y2))
        for rshs_class, box_list in by_class.items():
            jobs.append((pre_row, rshs_class, box_list))
    return jobs


def tb_jobs(by_key: dict[str, dict]) -> list[tuple[dict, str, list, list[str]]]:
    jobs = []
    for source, loader in (("tbx11k", load_tbx11k), ("montgomery", load_montgomery)):
        boxes_by_rel = loader()
        for rel, boxes in boxes_by_rel.items():
            key = f"{source}/{str(rel).replace(chr(92), '/')}"
            pre_row = by_key.get(key)
            if pre_row is None:
                continue
            box_list = [(b.x1, b.y1, b.x2, b.y2) for b in boxes]
            named = montgomery_reading_classes(rel.stem) if source == "montgomery" else []
            jobs.append((pre_row, TB_LESION_CLASS, box_list, named))
    return jobs


def run_job(predictor, pre_row: dict, rshs_class: str, boxes_orig: list, named_extra: list[str], qc) -> dict | None:
    h, w = int(pre_row["orig_height"]), int(pre_row["orig_width"])
    y0, x0, crop = crop_box(h, w)

    bone_img_path = BONE_SUPPRESSED_ROOT / pre_row["preprocessed_relative_path"]
    if not bone_img_path.exists():
        return None
    gray = np.array(Image.open(bone_img_path))
    rgb = np.stack([gray, gray, gray], axis=-1)
    predictor.set_image(rgb)

    union = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
    for x1, y1, x2, y2 in boxes_orig:
        tx1, ty1, tx2, ty2, area_kept = transform_box(x1, y1, x2, y2, y0, x0, crop)
        if area_kept < 0.1:
            qc.write(f'{{"id": "{pre_row["id"]}", "class": "{rshs_class}", "dropped_box_area_kept": {area_kept:.4f}}}\n')
            continue
        res = mask_from_box(predictor, (round(tx1), round(ty1), round(tx2), round(ty2)), (IMG_SIZE, IMG_SIZE))
        union |= res.mask
        qc.write(
            f'{{"id": "{pre_row["id"]}", "class": "{rshs_class}", "area_ratio": {res.area_ratio:.4f}, '
            f'"degenerate": {str(res.degenerate).lower()}, "area_kept_after_crop": {area_kept:.4f}}}\n'
        )
    qc.flush()

    rel = Path(rshs_class) / pre_row["id"]
    out_path = OUT_ROOT / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((union.astype(np.uint8) * 255), mode="L").save(out_path)

    return {
        "id": pre_row["id"],
        "class": rshs_class,
        "source": pre_row["source"],
        "confidence": "high",
        "positive": str(bool(union.any())),
        "method": "sam_box_mask_bone_suppressed_resegmented",
        "preprocessed_image_relative_path": pre_row["preprocessed_relative_path"],
        "label_relative_path": str(rel.with_suffix(".png")).replace("\\", "/"),
    }


def main() -> None:
    log = setup_logging()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    by_key = preprocessed_index()
    caaxr = caaxr_jobs(by_key)
    tb = tb_jobs(by_key)
    log.info(f"caaxr jobs: {len(caaxr)}, tb_lesion jobs: {len(tb)}")

    skip = done_keys()
    fieldnames = [
        "id", "class", "source", "confidence", "positive", "method",
        "preprocessed_image_relative_path", "label_relative_path",
    ]

    predictor = load_predictor()
    out_rows: list[dict] = []
    with open(QC_PATH, "a", encoding="utf-8") as qc:
        for pre_row, rshs_class, box_list in tqdm(caaxr, desc="caaxr sam labels"):
            key = (pre_row["id"], rshs_class)
            if key in skip:
                continue
            row = run_job(predictor, pre_row, rshs_class, box_list, [], qc)
            if row:
                out_rows.append(row)
            if len(out_rows) >= 200:
                append_rows(out_rows, fieldnames)
                out_rows = []

        for pre_row, rshs_class, box_list, named in tqdm(tb, desc="tb_lesion sam labels"):
            key = (pre_row["id"], rshs_class)
            targets = [(rshs_class, key)]
            if len(named) == 1:
                targets.append((named[0], (pre_row["id"], named[0])))
            for cls, k in targets:
                if k in skip:
                    continue
                row = run_job(predictor, pre_row, cls, box_list, [], qc)
                if row:
                    out_rows.append(row)
            if len(out_rows) >= 200:
                append_rows(out_rows, fieldnames)
                out_rows = []

    if out_rows:
        append_rows(out_rows, fieldnames)
    log.info("sam labels complete")


if __name__ == "__main__":
    main()
