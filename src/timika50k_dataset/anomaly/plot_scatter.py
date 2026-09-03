"""Renders the UMAP scatter from reduce_and_detect.py's *_reduction.npz:
colored by source, flagged anomalies ringed in red, known-anomaly ids (if
any were passed to reduce_and_detect.py) marked with a black X regardless
of flagged status, so a miss is visible on the plot, not just in the log.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_OUT_ROOT = Path(r"C:\research\research-cxr-timika\output\timika50k_dataset\anomaly_detection")

parser = argparse.ArgumentParser()
parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
parser.add_argument("--in-prefix", default="")
parser.add_argument("--out", default=None)
parser.add_argument("--known-anomaly-ids", nargs="*", default=[])
args = parser.parse_args()

out_root = Path(args.out_root)
# allow_pickle=True is for the ids/sources string arrays np.savez stored as
# object dtype; the .npz is this same pipeline's own local output, written
# by reduce_and_detect.py moments earlier, never an externally-sourced file
data = np.load(out_root / f"{args.in_prefix}reduction.npz", allow_pickle=True)
ids, sources, scores, flagged, xy = data["ids"], data["sources"], data["scores"], data["flagged"], data["umap"]

fig, ax = plt.subplots(figsize=(11, 9), facecolor="#141519")
ax.set_facecolor("#141519")

uniq_sources = sorted(set(sources.tolist()))
cmap = plt.get_cmap("tab10")
colors = {s: cmap(i / max(len(uniq_sources) - 1, 1)) for i, s in enumerate(uniq_sources)}

for s in uniq_sources:
    m = sources == s
    ax.scatter(xy[m, 0], xy[m, 1], s=10, color=colors[s], label=s, alpha=0.65, linewidths=0)

flag_m = flagged.astype(bool)
ax.scatter(xy[flag_m, 0], xy[flag_m, 1], s=60, facecolors="none", edgecolors="#ff4d4d", linewidths=1.2, label="flagged anomaly")

known_set = set(args.known_anomaly_ids)
if known_set:
    known_m = np.array([i in known_set for i in ids])
    ax.scatter(xy[known_m, 0], xy[known_m, 1], s=140, marker="x", color="white", linewidths=2.2, label="known anomaly")

ax.set_title(f"timika-50k PSPNet embeddings, UMAP 2D  ({len(ids)} images, {int(flag_m.sum())} flagged)", color="#e8e6e1", fontsize=12)
ax.tick_params(colors="#9a9a94")
for spine in ax.spines.values():
    spine.set_color("#2a2d33")
legend = ax.legend(loc="upper right", fontsize=8, facecolor="#191b1f", edgecolor="#2a2d33", labelcolor="#e8e6e1")

out = Path(args.out) if args.out else out_root / f"{args.in_prefix}scatter.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
