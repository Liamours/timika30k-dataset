"""Two-color UMAP scatter: every point the same muted color (no supervised
labels exist for this task, so there is nothing else to legitimately color
by), except a small set of specific ids highlighted in one bold, distinct
color. Meant for showing whether a small number of known examples
separate from the unlabeled mass, not for showing cluster/source
structure, that's plot_scatter.py's job.
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
parser.add_argument("--highlight-ids", nargs="+", required=True)
parser.add_argument("--title", default=None)
args = parser.parse_args()

out_root = Path(args.out_root)
# allow_pickle=True: this .npz is this same pipeline's own local output,
# written by reduce_and_detect.py moments earlier, never externally sourced
data = np.load(out_root / f"{args.in_prefix}reduction.npz", allow_pickle=True)
ids, xy = data["ids"], data["umap"]

highlight = set(args.highlight_ids)
hl_mask = np.array([i in highlight for i in ids])
missing = highlight - set(ids[hl_mask].tolist())
if missing:
    print(f"warning: {len(missing)} highlight id(s) not found in this reduction: {missing}")

plt.rcParams["font.family"] = ["Times New Roman", "serif"]

INK = "#1a1a1a"
BG = "#ffffff"
ORANGE = "#e67e22"
BLUE = "#2c6fa8"

fig, ax = plt.subplots(figsize=(10, 8.5), facecolor=BG)
ax.set_facecolor(BG)

ax.scatter(xy[~hl_mask, 0], xy[~hl_mask, 1], s=10, color=ORANGE, alpha=0.55, linewidths=0, label=f"unlabeled ({int((~hl_mask).sum())})")
ax.scatter(xy[hl_mask, 0], xy[hl_mask, 1], s=170, color=BLUE, alpha=0.95, linewidths=1.4, edgecolors="white",
           label=f"known non-chest x-ray ({int(hl_mask.sum())})", zorder=5)

title = args.title or f"timika-50k PSPNet embeddings, UMAP 2D  ({len(ids)} images, no supervised labels)"
ax.set_title(title, color=INK, fontsize=13)
ax.tick_params(colors=INK)
for spine in ax.spines.values():
    spine.set_color("#888888")
legend = ax.legend(loc="upper right", fontsize=10, facecolor="#f5f5f5", edgecolor="#888888", labelcolor=INK)

out = Path(args.out) if args.out else out_root / f"{args.in_prefix}scatter_highlighted.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
