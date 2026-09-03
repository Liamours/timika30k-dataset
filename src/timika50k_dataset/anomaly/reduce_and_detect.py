"""Loads extract_features.py's embedding shards, reduces for anomaly
scoring and for visualization SEPARATELY, deliberately not the same step:
UMAP optimizes for local neighborhood structure, not for preserving global
distances, so an outlier score computed from UMAP coordinates would be
scoring an artifact of the projection, not the data. Isolation Forest runs
on PCA-denoised real feature space (50 components, keeps runtime down and
drops PCA's own noise floor); UMAP's 2D output is for looking at the
result, never for deciding it.

Outputs, all under --out-root (output/, not repo/, this is generated data):
- anomaly_scores.csv: id, source, iso_forest_score (lower = more anomalous,
  matching scikit-learn's own convention), umap_x, umap_y, flagged (bottom
  `--contamination` fraction by score).
- reduction.npz: same data as arrays, for plot_scatter.py.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import umap

DEFAULT_OUT_ROOT = Path(r"C:\research\research-cxr-timika\output\timika50k_dataset\anomaly_detection")
SEED = 42


def load_all(emb_dir: Path) -> tuple[list[dict], np.ndarray]:
    ids = list(csv.DictReader(open(emb_dir / "ids.csv", encoding="utf-8")))
    shards = sorted(emb_dir.glob("embeddings_*.npy"))
    arrays = [np.load(p) for p in shards]
    shard_offsets = {}
    offset = 0
    for i, arr in enumerate(arrays):
        shard_offsets[i] = offset
        offset += arr.shape[0]
    full = np.concatenate(arrays, axis=0)
    order = np.zeros(len(ids), dtype=np.int64)
    for i, row in enumerate(ids):
        order[i] = shard_offsets[int(row["shard"])] + int(row["row"])
    return ids, full[order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--known-anomaly-ids", nargs="*", default=[])
    parser.add_argument("--out-prefix", default="")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    ids, feats = load_all(out_root / "embeddings")
    print(f"loaded {feats.shape[0]} embeddings, dim={feats.shape[1]}")

    pca = PCA(n_components=min(50, feats.shape[0] - 1, feats.shape[1]), random_state=SEED)
    feats_pca = pca.fit_transform(feats)
    print(f"PCA: {feats_pca.shape[1]} components, explained variance {pca.explained_variance_ratio_.sum():.3f}")

    iso = IsolationForest(n_estimators=300, contamination=args.contamination, random_state=SEED)
    iso.fit(feats_pca)
    scores = iso.decision_function(feats_pca)  # lower = more anomalous
    flagged = iso.predict(feats_pca) == -1  # -1 = anomaly

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED)
    emb2d = reducer.fit_transform(feats_pca)

    known = set(args.known_anomaly_ids)
    out_csv = out_root / f"{args.out_prefix}anomaly_scores.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "source", "image_relative_path", "iso_forest_score", "umap_x", "umap_y", "flagged", "known_anomaly"])
        writer.writeheader()
        for row, score, is_flag, xy in zip(ids, scores, flagged, emb2d):
            writer.writerow({"id": row["id"], "source": row["source"], "image_relative_path": row["image_relative_path"],
                              "iso_forest_score": round(float(score), 5), "umap_x": round(float(xy[0]), 4), "umap_y": round(float(xy[1]), 4),
                              "flagged": bool(is_flag), "known_anomaly": row["id"] in known})
    print(f"wrote {out_csv}")

    n_flagged = int(flagged.sum())
    print(f"flagged {n_flagged}/{len(ids)} ({n_flagged/len(ids)*100:.2f}%) as anomalies")
    if known:
        found = [row["id"] for row, f in zip(ids, flagged) if row["id"] in known and f]
        missed = [row["id"] for row, f in zip(ids, flagged) if row["id"] in known and not f]
        print(f"known anomalies: {len(found)}/{len(known)} flagged")
        for m in missed:
            print(f"  MISSED: {m}")

    np.savez(out_root / f"{args.out_prefix}reduction.npz", ids=[r["id"] for r in ids], sources=[r["source"] for r in ids],
             scores=scores, flagged=flagged, umap=emb2d)


if __name__ == "__main__":
    main()
