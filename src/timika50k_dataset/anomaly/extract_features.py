"""Embedding extraction for timika-50k image-quality anomaly detection,
two interchangeable models (--model pspnet|densenet, see EXTRACTORS and
each class's own docstring for the tradeoff): PSPNet's segmentation-model
features are sensitive to how much recognizable lung structure is visible
(useful for catching organ_region-style failures, but also reacts to
off-center framing that isn't necessarily a real defect); the DenseNet
classifier's features are trained for disease presence, not localization,
so comparing the two tells apart real image-quality anomalies (both
models should agree those are unusual) from PSPNet-specific sensitivity
to framing (only PSPNet's clusters would show it). Same shared
preprocessing regardless of model (own-image min/max rescale,
xrv.datasets.normalize, center-crop to square) as
analyses/timika50k_organ_labels/build_organ_region_labels.py.

Chunked, resumable: writes embeddings_XXXXX.npy shards of CHUNK images
each plus one running ids.csv (id, source, image_relative_path, shard,
row_in_shard) under OUT_ROOT/<model>/ (output/, not repo/, this is
generated data, not source); an id already in ids.csv is skipped, so a
rerun after an interruption only processes what's missing, and the ids
CSV is the single source of truth for what row of what shard holds which
image.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)

DATASET_ROOT = Path(r"E:\dataset\timika-50k")
DATA_ROOT = DATASET_ROOT / "data"
DEFAULT_OUT_ROOT = Path(r"C:\research\research-cxr-timika\output\timika50k_dataset\anomaly_detection\embeddings")
LOG_PATH = Path(r"C:\research\research-cxr-timika\logs\timika50k_anomaly_features.log")

IMAGE_EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK = 500
FEATURE_DIM = 4096
FIELDS = ["id", "source", "image_relative_path", "shard", "row"]


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger("extract_features")


def load_grayscale(path: Path) -> np.ndarray:
    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min) * 255.0
    else:
        img = np.zeros_like(img)
    return img


def center_crop(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    crop = min(h, w)
    y0, x0 = h // 2 - crop // 2, w // 2 - crop // 2
    return image[y0 : y0 + crop, x0 : x0 + crop]


def resize_batch(crops: list[np.ndarray], target_size: int, device: str) -> torch.Tensor:
    """Each crop resized individually to (target_size, target_size) before
    batching, not padded to the batch's largest crop then resized together:
    the model's own internal resize (fix_resolution) would produce the same
    per-image result either way once it downsamples to its working
    resolution, but padding-then-batch-resize wastes memory/compute on a
    huge shared canvas for no benefit (PSPNet at 512 does this once per
    batch either way; a 224-resolution classifier would otherwise inherit a
    near-5000px canvas from a single large Montgomery image sharing its
    batch), and avoids a systematic black-border bias for smaller-than-max
    images in the same batch."""
    tensors = []
    for c in crops:
        t = torch.from_numpy(c).float()[None, None, ...].to(device)
        t = torch.nn.functional.interpolate(t, size=(target_size, target_size), mode="bilinear", align_corners=False)
        tensors.append(t)
    return torch.cat(tensors, dim=0)


class PSPNetFeatureExtractor:
    """Tap point: pyramid_pooling, PSPNet's own multi-scale context layer,
    see this module's own docstring for why. Segmentation-model features:
    sensitive to how much recognizable lung structure is visible, which is
    exactly right for catching organ_region-style failures, but means an
    off-center crop or unusual framing can dominate the embedding even on
    an image with no real quality defect."""

    NAME = "pspnet"
    TARGET_SIZE = 512
    FEATURE_DIM = 4096

    def __init__(self, device: str = DEVICE):
        self._device = device
        self._model = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
        self._captured: torch.Tensor | None = None
        self._model.model.pyramid_pooling.register_forward_hook(self._hook)

    def _hook(self, module, inp, out) -> None:
        self._captured = out

    @torch.no_grad()
    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        crops = [center_crop(xrv.datasets.normalize(img, 255)) for img in images]
        tensor = resize_batch(crops, self.TARGET_SIZE, self._device)
        self._model(tensor)
        feats = self._captured  # (B, 4096, 64, 64)
        pooled = feats.mean(dim=(2, 3))  # global average pool
        return pooled.cpu().numpy().astype(np.float32)


class DenseNetFeatureExtractor:
    """torchxrayvision's own multi-dataset classifier (densenet121-res224-all),
    features2's own global-average-pooled backbone output (1024-dim),
    trained for disease presence, not lung localization: unlike PSPNet, a
    correctly-exposed image with unusual framing shouldn't stand out here
    just because less of the lung is visible, so this is the natural
    check on whether PSPNet's own anomaly clusters (cluster 1 especially,
    dominated by off-center CAAXR crops) reflect real image-quality
    problems or PSPNet's own segmentation-specific sensitivity."""

    NAME = "densenet"
    TARGET_SIZE = 224
    FEATURE_DIM = 1024

    def __init__(self, device: str = DEVICE):
        self._device = device
        self._model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device).eval()

    @torch.no_grad()
    def embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        crops = [center_crop(xrv.datasets.normalize(img, 255)) for img in images]
        tensor = resize_batch(crops, self.TARGET_SIZE, self._device)
        feats = self._model.features2(tensor)
        return feats.cpu().numpy().astype(np.float32)


EXTRACTORS = {"pspnet": PSPNetFeatureExtractor, "densenet": DenseNetFeatureExtractor}


def iter_source_images() -> list[tuple[str, Path]]:
    out = []
    for source_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir()):
        for p in sorted(source_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                out.append((source_dir.name, p.relative_to(source_dir)))
    return out


def load_done_ids(ids_path: Path) -> set[str]:
    if not ids_path.exists():
        return set()
    with open(ids_path, "r", encoding="utf-8", newline="") as f:
        return {row["id"] for row in csv.DictReader(f)}


def next_shard_index(out_root: Path) -> int:
    existing = sorted(out_root.glob("embeddings_*.npy"))
    if not existing:
        return 0
    return int(existing[-1].stem.split("_")[1]) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(EXTRACTORS), default="pspnet")
    parser.add_argument("--out-root", default=None, help=f"default: {DEFAULT_OUT_ROOT}/<model>")
    parser.add_argument("--limit", type=int, default=None, help="max images (smoke test)")
    parser.add_argument("--sample-per-source", type=int, default=None, help="stratified sample, N per source")
    parser.add_argument("--include-ids", nargs="*", default=[], help="ids to force-include (e.g. known anomalies)")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # separate subfolder per model: the two are not comparable embeddings
    # and must never end up concatenated into one reduction by accident
    out_root = Path(args.out_root) if args.out_root else DEFAULT_OUT_ROOT.parent / args.model / DEFAULT_OUT_ROOT.name
    ids_path = out_root / "ids.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    log = setup_logging()
    log.info("model: %s | device: %s | out_root: %s", args.model, DEVICE, out_root)

    all_images = iter_source_images()
    log.info("%d images total under %s", len(all_images), DATA_ROOT)

    if args.sample_per_source is not None:
        rng = np.random.default_rng(SEED)
        by_source: dict[str, list[tuple[str, Path]]] = {}
        for s, p in all_images:
            by_source.setdefault(s, []).append((s, p))
        sampled = []
        for s, items in by_source.items():
            idx = rng.choice(len(items), size=min(args.sample_per_source, len(items)), replace=False)
            sampled.extend(items[i] for i in idx)
        wanted_ids = {f"{s}/{p}".replace("\\", "/") for s, p in sampled}
        for forced in args.include_ids:
            wanted_ids.add(forced)
        all_images = [(s, p) for s, p in all_images if f"{s}/{p}".replace('\\', '/') in wanted_ids]
        log.info("stratified sample: %d images (%d/source + %d forced)", len(all_images), args.sample_per_source, len(args.include_ids))

    if args.limit is not None:
        all_images = all_images[: args.limit]

    done = load_done_ids(ids_path)
    log.info("%d ids already embedded, resuming", len(done))
    todo = [(s, p) for s, p in all_images if f"{s}/{p}".replace("\\", "/") not in done]
    log.info("%d images to embed this run", len(todo))
    if not todo:
        return

    extractor = EXTRACTORS[args.model]()
    shard_idx = next_shard_index(out_root)
    write_header = not ids_path.exists()
    with open(ids_path, "a", encoding="utf-8", newline="") as idf:
        writer = csv.DictWriter(idf, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        buf_ids: list[tuple[str, str, str]] = []
        buf_imgs: list[np.ndarray] = []
        shard_feats: list[np.ndarray] = []
        errored = 0

        def flush_shard() -> None:
            nonlocal shard_idx, shard_feats
            if not shard_feats:
                return
            arr = np.concatenate(shard_feats, axis=0)
            np.save(out_root / f"embeddings_{shard_idx:05d}.npy", arr)
            shard_feats = []
            shard_idx += 1

        def flush_batch() -> None:
            nonlocal buf_ids, buf_imgs
            if not buf_imgs:
                return
            feats = extractor.embed_batch(buf_imgs)
            shard_feats.append(feats)
            base_row = sum(f.shape[0] for f in shard_feats[:-1])
            for i, (id_, source, rel) in enumerate(buf_ids):
                writer.writerow({"id": id_, "source": source, "image_relative_path": f"data/{source}/{rel}".replace("\\", "/"),
                                  "shard": shard_idx, "row": base_row + i})
            idf.flush()
            buf_ids, buf_imgs = [], []

        rows_in_shard = 0
        for source, relative in tqdm(todo, desc="embedding", unit="img"):
            id_ = f"{source}/{relative}".replace("\\", "/")
            try:
                img = load_grayscale(DATA_ROOT / source / relative)
            except Exception:
                log.exception("failed to load %s", id_)
                errored += 1
                continue
            buf_ids.append((id_, source, str(relative)))
            buf_imgs.append(img)
            if len(buf_imgs) >= args.batch_size:
                flush_batch()
                rows_in_shard += args.batch_size
                if rows_in_shard >= CHUNK:
                    flush_shard()
                    rows_in_shard = 0
        flush_batch()
        flush_shard()

    log.info("done: %d embedded, %d errored", len(todo) - errored, errored)


if __name__ == "__main__":
    main()
