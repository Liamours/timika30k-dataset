"""Adds a `view_position` column to preprocessed/manifest.csv: ap, pa,
ll (lateral), or "-" where no real per-image or per-source information
exists. Only two sources have anything real, checked directly against
each source's own documentation and raw files before writing anything,
not assumed:

- caaxr: real per-image value, encoded directly in its own BIDS-style
  filename (`..._vp-{ap,pa,ll,rl}_...`). Not every CAAXR file carries the
  tag; untagged files get "-", not guessed.
- montgomery: its own datasheet states, unqualified, "138
  posterior-anterior chest X-rays", no exception noted anywhere checked
  (context/archive/tb_localization-260827/datasheets/montgomery.md) -> a
  real per-source constant, "pa" for all 138.

Every other source stays "-", including shenzhen, deliberately: its own
datasheet says "662 PA chest X-rays ... includes pediatric AP views", a
real, documented exception with no per-image breakdown available anywhere
in this store, so "pa" would be factually wrong for an unidentified
subset. Checked and confirmed absent, not just unchecked: chestxdet,
tbx11k, siimacr carry no per-image view-position field in any label file
this store has (2026-09-03 survey; covidrad was also checked and
confirmed absent the same way, moot now that it's dropped, 2026-09-04).

Rewrites manifest.csv in place (same row order, one new column appended);
the C: project copy is preprocessed/'s only copy, per the 2026-09-02
instruction that built it there and nowhere else.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

MANIFEST_PATH = Path(r"C:\research\research-cxr-timika\dataset\timika-30k\preprocessed\manifest.csv")
VIEW_POSITION_RE = re.compile(r"vp-(ap|pa|ll|rl)")


def view_position(source: str, original_relative_path: str) -> str:
    if source == "caaxr":
        m = VIEW_POSITION_RE.search(original_relative_path)
        return m.group(1) if m else "-"
    if source == "montgomery":
        return "pa"
    return "-"


def main() -> None:
    rows = list(csv.DictReader(open(MANIFEST_PATH, encoding="utf-8", newline="")))
    fields = list(rows[0].keys()) + ["view_position"]
    for r in rows:
        r["view_position"] = view_position(r["source"], r["original_relative_path"])

    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    by_source_vp = Counter((r["source"], r["view_position"]) for r in rows)
    print(f"wrote {len(rows)} rows to {MANIFEST_PATH}")
    for (source, vp), n in sorted(by_source_vp.items()):
        if vp != "-":
            print(f"  {source:12} {vp:4} {n}")
    print(f"  total non-'-': {sum(n for (s, vp), n in by_source_vp.items() if vp != '-')}")


if __name__ == "__main__":
    main()
