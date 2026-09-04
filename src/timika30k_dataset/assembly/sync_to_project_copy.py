"""Mirrors E:\\dataset\\timika-30k\\ (canonical) into dataset/timika-30k/
(this project's own, user-approved second copy, see dataset/manifest.md's
timika-30k row) so the C:-only phases (preprocessing, splits,
view_position, all of analyses/timika30k_preprocessing/) have something to
read. Mirrors data/ and labels/ only, same asymmetry documented in
dataset/manifest.md and this repo's own README: preprocessed/ is built
fresh on C: by build_preprocessed.py and never sourced from E:, so a
run of this script before build_preprocessed.py has run yet will not
create or touch preprocessed/ at all.

No dependency on any external copy tool: same resumable, size-checked
shutil.copy2 pattern as assembly/build_data.py. Self-contained since the
external tool previously used for this exact sync (per past
context/intent.md entries) is a separate project, out of this repo's own
scope, not guaranteed callable from wherever this script runs.

--dry-run reports counts (files already in sync vs. needing a copy) with
no file I/O.
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from tqdm import tqdm

SOURCE_ROOT = Path(r"E:\dataset\timika-30k")
DEST_ROOT = Path(r"C:\research\research-cxr-timika\dataset\timika-30k")
MIRRORED_SUBDIRS = ("data", "labels")
LOG_PATH = Path(r"C:\research\research-cxr-timika\logs\timika30k_sync_to_project_copy.log")


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )


def files_to_mirror() -> list[Path]:
    files = []
    for subdir in MIRRORED_SUBDIRS:
        root = SOURCE_ROOT / subdir
        if not root.exists():
            continue
        walked = tqdm(root.rglob("*"), desc=f"listing {subdir}", unit=" entries")
        files.extend(p for p in walked if p.is_file())
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report counts only, no file copy")
    args = parser.parse_args()

    files = files_to_mirror()
    pending = []
    for src in tqdm(files, desc="checking sync state"):
        dest = DEST_ROOT / src.relative_to(SOURCE_ROOT)
        if not (dest.exists() and dest.stat().st_size == src.stat().st_size):
            pending.append((src, dest))

    if args.dry_run:
        print(f"{'total':12} {len(files):6}")
        print(f"{'in sync':12} {len(files) - len(pending):6}")
        print(f"{'to copy':12} {len(pending):6}")
        return

    setup_logging()
    log = logging.getLogger(__name__)
    log.info(f"syncing {len(pending)} of {len(files)} files, E:\\dataset\\timika-30k -> {DEST_ROOT}")

    for src, dest in tqdm(pending, desc="syncing to project copy"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    log.info(f"sync complete: {len(pending)} copied, {len(files) - len(pending)} already in sync")


if __name__ == "__main__":
    main()
