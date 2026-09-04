"""Phase 2: assemble timika-30k's data/ from its 6 canonical sources.
COVID-19 Radiography Database (covidrad) was dropped 2026-09-04: no
disease-level box or mask for any of its 3 abnormal classes, only an
organ-level lung mask, and none of its own labels map onto this
project's taxonomy anywhere in the codebase. Its own canonical copy at
E:\\dataset\\covid19_radiography\\ is untouched.

Eligibility, per dataset/manifest.md's timika-30k row: a source image
qualifies only if it already carries some form of localization (box,
mask, or polygon), and isn't a known duplicate. This script encodes both
halves of that rule per source:

- a fixed subfolder exclusion, only TBX11K's own data/extra/ (bonus
  Montgomery/Shenzhen/DA/DB augmentation, not part of its core release)
- a duplicate exclusion, any file its own source's own
  labels/quality_flags.csv.bz2 marks status=="duplicate" (read the same
  way repo/timika30k_pseudolabels/src/timika30k_pseudolabels/build_ds_group.py
  already does; a source with no such file, e.g. Shenzhen or Montgomery,
  has an empty duplicate set, not an error)

What survives is copied into timika-30k/data/{short_name}/, same
relative path each file already has under its own source's data/. The
canonical_name -> short_name mapping (e.g. bimcv_caaxr -> caaxr) matches
timika-30k's own existing data/ layout, not the longer names each source
uses at E:\\dataset\\.

--dry-run reports per-source included counts with no file I/O (just
directory listing and duplicate-set reads), for checking this script's
own logic against dataset/manifest.md's documented per-source counts
(662, 138, 11490, 3577, 2539, 12088, total 30494) before spending
the time on a real copy. Resumable: an existing destination file
whose size matches the source file is treated as already copied.
"""
from __future__ import annotations

import argparse
import bz2
import csv
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

CANONICAL_ROOT = Path(r"E:\dataset")
DATA_ROOT = Path(r"E:\dataset\timika-30k\data")
LOG_PATH = Path(r"C:\research\research-cxr-timika\logs\timika30k_assembly.log")


@dataclass(frozen=True)
class SourceConfig:
    canonical_name: str
    short_name: str
    exclude_subpaths: tuple[str, ...] = ()


SOURCES = (
    SourceConfig("shenzhen", "shenzhen"),
    SourceConfig("montgomery", "montgomery"),
    SourceConfig("tbx11k", "tbx11k", exclude_subpaths=("extra",)),
    SourceConfig("chestxdet", "chestxdet"),
    SourceConfig("bimcv_caaxr", "caaxr"),
    SourceConfig("siim_acr_pneumothorax", "siimacr"),
)


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )


def duplicate_set(quality_flags_path: Path) -> set[str]:
    if not quality_flags_path.exists():
        return set()
    with bz2.open(quality_flags_path, "rt", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["filename"] for r in rows if r["status"] == "duplicate"}


def eligible_files(source: SourceConfig) -> list[Path]:
    data_root = CANONICAL_ROOT / source.canonical_name / "data"
    dupes = duplicate_set(CANONICAL_ROOT / source.canonical_name / "labels" / "quality_flags.csv.bz2")
    files = []
    for path in data_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(data_root)
        if relative.parts[0] in source.exclude_subpaths:
            continue
        if relative.as_posix() in dupes:
            continue
        files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report counts only, no file copy")
    args = parser.parse_args()

    per_source = {source: eligible_files(source) for source in SOURCES}
    total = sum(len(files) for files in per_source.values())

    if args.dry_run:
        for source, files in per_source.items():
            print(f"{source.short_name:12} {len(files):6}")
        print(f"{'TOTAL':12} {total:6}")
        return

    setup_logging()
    log = logging.getLogger(__name__)
    log.info(f"assembling {total} eligible files across {len(SOURCES)} sources")

    jobs = [
        (source, path)
        for source, files in per_source.items()
        for path in files
    ]
    copied = skipped = 0
    for source, path in tqdm(jobs, desc="assembling data/"):
        data_root = CANONICAL_ROOT / source.canonical_name / "data"
        relative = path.relative_to(data_root)
        dest = DATA_ROOT / source.short_name / relative
        if dest.exists() and dest.stat().st_size == path.stat().st_size:
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        copied += 1

    log.info(f"assembly complete: {total} eligible, {copied} copied, {skipped} already present")
    for source, files in per_source.items():
        log.info(f"  {source.short_name}: {len(files)}")


if __name__ == "__main__":
    main()
