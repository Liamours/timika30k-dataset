"""Runs timika-30k's full build pipeline end to end, in the dependency
order documented in dataset/manifest.md's timika-30k row and this repo's
own README: 11 phases, each a real script already built and already run
once by hand for its own part of the dataset, invoked here in sequence
instead of by memory.

Every phase is its own uv project (a separate pyproject.toml/dependency
set: repo/timika30k_dataset, repo/timika30k_pseudolabels,
analyses/timika30k_organ_labels, analyses/timika30k_preprocessing), so
this script shells out to `uv run` inside that phase's own directory
rather than importing across projects with incompatible dependencies.

Phases 1-3 (assembly, organ-region labels, disease labels) read and write
E:\\dataset\\timika-30k\\, the canonical copy. Phase 4 mirrors data/ and
labels/ from there into dataset/timika-30k/, this project's own
user-approved second copy (dataset/manifest.md's timika-30k row).
Phases 5-7 (preprocessing, splits, view_position) read only from that C:
copy and write only there too, per the 2026-09-02 decision to keep
preprocessed/ project-local; they are never mirrored back to E:.

Every phase script is independently resumable (an existing output is
treated as done and skipped, see each script's own docstring), so
re-running this orchestrator after a partial or complete prior run costs
only the time to re-check what is already there, not a full rebuild.

--dry-run prints the exact command sequence without running anything.
--phases restricts to a comma-separated subset of the PHASE names below,
default all.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(r"C:\research\research-cxr-timika")
LOG_PATH = PROJECT_ROOT / "logs" / "timika30k_orchestrate.log"


@dataclass(frozen=True)
class Phase:
    name: str
    cwd: Path
    command: tuple[str, ...]


PHASES = (
    Phase(
        "assembly",
        PROJECT_ROOT / "repo" / "timika30k_dataset",
        ("uv", "run", "python", "-m", "timika30k_dataset.assembly.build_data"),
    ),
    Phase(
        "organ_region_labels",
        PROJECT_ROOT / "analyses" / "timika30k_organ_labels",
        ("uv", "run", "python", "build_organ_region_labels.py"),
    ),
    Phase(
        "organ_region_manifest",
        PROJECT_ROOT / "analyses" / "timika30k_organ_labels",
        ("uv", "run", "python", "build_organ_manifest.py"),
    ),
    Phase(
        "disease_confirmed_negatives",
        PROJECT_ROOT / "repo" / "timika30k_pseudolabels",
        ("uv", "run", "python", "-m", "timika30k_pseudolabels.build_confirmed_negatives"),
    ),
    Phase(
        "disease_caaxr_box_masks",
        PROJECT_ROOT / "repo" / "timika30k_pseudolabels",
        ("uv", "run", "python", "-m", "timika30k_pseudolabels.build_caaxr_box_masks"),
    ),
    Phase(
        "disease_tb_box_masks",
        PROJECT_ROOT / "repo" / "timika30k_pseudolabels",
        ("uv", "run", "python", "-m", "timika30k_pseudolabels.build_tb_box_masks"),
    ),
    Phase(
        "disease_ds_group",
        PROJECT_ROOT / "repo" / "timika30k_pseudolabels",
        ("uv", "run", "python", "-m", "timika30k_pseudolabels.build_ds_group"),
    ),
    Phase(
        "disease_manifest",
        PROJECT_ROOT / "repo" / "timika30k_pseudolabels",
        ("uv", "run", "python", "-m", "timika30k_pseudolabels.build_label_manifest"),
    ),
    Phase(
        "sync_to_project_copy",
        PROJECT_ROOT / "repo" / "timika30k_dataset",
        ("uv", "run", "python", "-m", "timika30k_dataset.assembly.sync_to_project_copy"),
    ),
    Phase(
        "preprocessing",
        PROJECT_ROOT / "analyses" / "timika30k_preprocessing",
        ("uv", "run", "python", "build_preprocessed.py"),
    ),
    Phase(
        "splits",
        PROJECT_ROOT / "analyses" / "timika30k_preprocessing",
        ("uv", "run", "python", "build_splits.py"),
    ),
    Phase(
        "view_position",
        PROJECT_ROOT / "repo" / "timika30k_dataset",
        ("uv", "run", "python", "-m", "timika30k_dataset.metadata.add_view_position"),
    ),
)


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )


def select_phases(names: str | None) -> tuple[Phase, ...]:
    if not names:
        return PHASES
    wanted = set(names.split(","))
    known = {p.name for p in PHASES}
    unknown = wanted - known
    if unknown:
        raise SystemExit(f"unknown phase(s): {sorted(unknown)}, choose from {sorted(known)}")
    return tuple(p for p in PHASES if p.name in wanted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the command sequence, run nothing")
    parser.add_argument("--phases", type=str, default=None, help="comma-separated phase names, default all")
    args = parser.parse_args()

    selected = select_phases(args.phases)

    if args.dry_run:
        for phase in selected:
            print(f"{phase.name:28} cwd={phase.cwd}  {' '.join(phase.command)}")
        return

    setup_logging()
    log = logging.getLogger(__name__)
    log.info(f"running {len(selected)} phase(s): {[p.name for p in selected]}")

    for phase in selected:
        log.info(f"[{phase.name}] starting: {' '.join(phase.command)} (cwd={phase.cwd})")
        start = time.monotonic()
        subprocess.run(phase.command, cwd=phase.cwd, check=True)
        elapsed = time.monotonic() - start
        log.info(f"[{phase.name}] done in {elapsed / 60:.1f} min")

    log.info("pipeline complete")


if __name__ == "__main__":
    main()
