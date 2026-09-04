"""The exact geometric transform analyses/timika30k_preprocessing/build_preprocessed.py's
own center_crop_resize() applies to images (center-crop to a square of
side min(h,w), then resize to IMG_SIZE), reimplemented here for labels:
a mask variant (nearest-neighbor, keeps exact 0/255 values) and a box
variant (crop-offset + scale, clipped, with an area-loss ratio the caller
uses to decide whether a box survived the crop or should be dropped).

Applying this to a label produces something aligned pixel-for-pixel with
preprocessed/images/'s own output for the same source image, and, since
bone_suppression/build_variant.py runs its model on that same 512x512
grid with no further resize, this same transformed label also aligns
with the bone-suppressed variant. One transform, two image variants.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

IMG_SIZE = 512


def crop_box(h: int, w: int) -> tuple[int, int, int]:
    crop = min(h, w)
    y0 = h // 2 - crop // 2
    x0 = w // 2 - crop // 2
    return y0, x0, crop


def transform_mask(mask: np.ndarray, y0: int, x0: int, crop: int) -> np.ndarray:
    # Nearest-neighbor on a mask is trivial work; CPU/PIL avoids the GPU
    # round-trip entirely, which matters here since this runs at high
    # frequency alongside bone_suppression/build_variant.py's own batched
    # GPU inference, and the two were found to contend for the GPU badly.
    cropped = mask[y0 : y0 + crop, x0 : x0 + crop]
    resized = Image.fromarray(cropped).resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
    return np.array(resized).astype(mask.dtype)


def transform_box(
    x1: float, y1: float, x2: float, y2: float, y0: int, x0: int, crop: int
) -> tuple[float, float, float, float, float]:
    """Returns (x1, y1, x2, y2) in the final IMG_SIZE grid, plus the
    fraction of the box's original area retained after the crop clips it
    (1.0 = fully inside the crop, 0.0 = entirely cropped away)."""
    orig_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    cx1, cy1 = x1 - x0, y1 - y0
    cx2, cy2 = x2 - x0, y2 - y0
    clipped_x1, clipped_y1 = max(0.0, cx1), max(0.0, cy1)
    clipped_x2, clipped_y2 = min(float(crop), cx2), min(float(crop), cy2)
    clipped_area = max(0.0, clipped_x2 - clipped_x1) * max(0.0, clipped_y2 - clipped_y1)
    area_kept = 0.0 if orig_area <= 0 else clipped_area / orig_area

    scale = IMG_SIZE / crop
    out = (clipped_x1 * scale, clipped_y1 * scale, clipped_x2 * scale, clipped_y2 * scale)
    return out[0], out[1], out[2], out[3], area_kept
