"""Clusters reduce_and_detect.py's UMAP 2D embedding with HDBSCAN (density-
based, no cluster count to guess upfront, and it labels sparse regions as
noise instead of forcing them into a nearest cluster, the right behavior
for a QC pass where "doesn't belong anywhere" is itself real information).
Clustering runs on the 2D UMAP coordinates deliberately, unlike the
anomaly score in reduce_and_detect.py: here the goal is recovering the
visually-coherent groups already apparent in the scatter plot, which is
exactly what UMAP's projection is built to preserve locally.

For each cluster (and noise, labeled -1), samples up to --n-per-cluster
real images spread across the cluster (evenly spaced after sorting by
distance from the cluster's own centroid, not random, so the grid shows
the cluster's actual spread, not N near-duplicates from one corner of it)
and renders one grid image, source-tagged per tile, matching the QC style
already used elsewhere in this project (dark ground, images read best
that way).

Writes clusters.csv (id, source, cluster) and cluster_samples.png.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import HDBSCAN

DEFAULT_OUT_ROOT = Path(r"C:\research\research-cxr-timika\output\timika50k_dataset\anomaly_detection")
DATA_ROOT = Path(r"E:\dataset\timika-50k\data")
SIXTEEN_BIT_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N"}
TILE = 220


def load_gray_thumb(path: Path, size: int = TILE) -> Image.Image:
    img = Image.open(path)
    if img.mode in SIXTEEN_BIT_MODES:
        a = np.array(img).astype(np.float32)
        lo, hi = a.min(), a.max()
        arr = np.zeros_like(a, dtype=np.uint8) if hi <= lo else ((a - lo) / (hi - lo) * 255).round().astype(np.uint8)
        img = Image.fromarray(arr)
    else:
        img = img.convert("L")
    w, h = img.size
    crop = min(w, h)
    img = img.crop(((w - crop) // 2, (h - crop) // 2, (w - crop) // 2 + crop, (h - crop) // 2 + crop))
    return img.resize((size, size)).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--in-prefix", default="")
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--n-per-cluster", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--only-cluster", type=int, default=None, help="skip re-clustering, render every member of this cluster id from an existing *clusters.csv")
    args = parser.parse_args()

    if args.only_cluster is not None:
        out_root = Path(args.out_root)
        rows = list(csv.DictReader(open(out_root / f"{args.in_prefix}clusters.csv", encoding="utf-8", newline="")))
        members = [r for r in rows if int(r["cluster"]) == args.only_cluster]
        print(f"cluster {args.only_cluster}: {len(members)} members, rendering all")
        tiles = []
        for r in members:
            rel = r["id"][len(r["source"]) + 1 :]
            thumb = load_gray_thumb(DATA_ROOT / r["source"] / rel)
            draw = ImageDraw.Draw(thumb)
            draw.rectangle([0, 0, TILE, 18], fill=(0, 0, 0))
            draw.text((3, 2), r["source"], fill=(255, 255, 255))
            tiles.append(thumb)
        cols = args.cols
        n_rows = (len(tiles) + cols - 1) // cols
        grid = Image.new("RGB", (cols * TILE, n_rows * TILE), (20, 20, 20))
        for i, t in enumerate(tiles):
            grid.paste(t, ((i % cols) * TILE, (i // cols) * TILE))
        out_path = out_root / f"{args.in_prefix}cluster{args.only_cluster}_full.png"
        grid.save(out_path)
        print(f"wrote {out_path} ({len(tiles)} tiles)")
        return

    out_root = Path(args.out_root)
    # allow_pickle=True: this .npz is this same pipeline's own local output,
    # written by reduce_and_detect.py moments earlier, never externally sourced
    data = np.load(out_root / f"{args.in_prefix}reduction.npz", allow_pickle=True)
    ids, sources, xy = data["ids"], data["sources"], data["umap"]

    clusterer = HDBSCAN(min_cluster_size=args.min_cluster_size)
    labels = clusterer.fit_predict(xy)

    with open(out_root / f"{args.in_prefix}clusters.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "source", "cluster", "umap_x", "umap_y"])
        for i, s, c, p in zip(ids, sources, labels, xy):
            writer.writerow([i, s, int(c), round(float(p[0]), 4), round(float(p[1]), 4)])

    uniq = sorted(set(labels.tolist()), key=lambda c: (c == -1, c))
    print(f"{len([c for c in uniq if c != -1])} clusters found, {int((labels == -1).sum())} noise points, of {len(ids)} total")
    for c in uniq:
        n = int((labels == c).sum())
        name = "noise" if c == -1 else f"cluster {c}"
        print(f"  {name}: {n} images, sources: {sorted(set(sources[labels == c].tolist()))}")

    plt.rcParams["font.family"] = ["Times New Roman", "serif"]
    INK, BG = "#1a1a1a", "#ffffff"
    real_clusters = [c for c in uniq if c != -1]
    cmap = plt.get_cmap("tab20" if len(real_clusters) > 10 else "tab10")
    colors = {c: cmap(i / max(len(real_clusters) - 1, 1)) for i, c in enumerate(real_clusters)}

    fig, ax = plt.subplots(figsize=(10, 8.5), facecolor=BG)
    ax.set_facecolor(BG)
    noise_m = labels == -1
    ax.scatter(xy[noise_m, 0], xy[noise_m, 1], s=8, color="#bbbbbb", alpha=0.5, linewidths=0, label=f"noise ({int(noise_m.sum())})")
    for c in real_clusters:
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=12, color=colors[c], alpha=0.75, linewidths=0, label=f"cluster {c} ({int(m.sum())})")
    ax.set_title(f"timika-50k PSPNet embeddings, UMAP 2D  ({len(ids)} images, HDBSCAN min_cluster_size={args.min_cluster_size})", color=INK, fontsize=12)
    ax.tick_params(colors=INK)
    for spine in ax.spines.values():
        spine.set_color("#888888")
    ax.legend(loc="upper right", fontsize=8, facecolor="#f5f5f5", edgecolor="#888888", labelcolor=INK, ncol=1)
    scatter_path = out_root / f"{args.in_prefix}cluster_scatter.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    print(f"wrote {scatter_path}")

    rows = []
    for c in uniq:
        mask = labels == c
        member_ids = ids[mask]
        member_xy = xy[mask]
        member_src = sources[mask]
        centroid = member_xy.mean(axis=0)
        order = np.argsort(np.linalg.norm(member_xy - centroid, axis=1))
        n_take = min(args.n_per_cluster, len(order))
        pick_idx = order[np.linspace(0, len(order) - 1, n_take, dtype=int)] if len(order) > 1 else order
        name = "noise" if c == -1 else f"cluster {c}"
        for idx in pick_idx:
            rows.append((name, member_src[idx], member_ids[idx]))

    tiles = []
    for name, source, id_ in rows:
        rel = id_[len(source) + 1 :]
        img_path = DATA_ROOT / source / rel
        thumb = load_gray_thumb(img_path)
        draw = ImageDraw.Draw(thumb)
        draw.rectangle([0, 0, TILE, 18], fill=(0, 0, 0))
        draw.text((3, 2), f"{name} / {source}", fill=(255, 255, 255))
        tiles.append(thumb)

    cols = args.cols
    n_rows = (len(tiles) + cols - 1) // cols
    grid = Image.new("RGB", (cols * TILE, n_rows * TILE), (20, 20, 20))
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % cols) * TILE, (i // cols) * TILE))
    out_path = out_root / f"{args.in_prefix}cluster_samples.png"
    grid.save(out_path)
    print(f"wrote {out_path} ({len(tiles)} tiles)")


if __name__ == "__main__":
    main()
