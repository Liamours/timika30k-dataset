"""CLAHE application shared by both CLAHE-based preprocessing variants
(4: bone suppression + CLAHE, 5: CLAHE only). clip_limit=2.0,
tile_grid_size=(8,8): no single primary source in this project's own
literature pins these exact numbers for chest X-rays specifically, but a
2026-09-04 search corroborated both from multiple directions: (8,8) as
the tile grid repeatedly named "the most commonly used configuration in
chest radiography studies", and a clip limit in the 2-4 range as the
standard, gentler-than-natural-photos setting used across the CXR/medical
CLAHE literature surveyed (values from 2.0 to 8.0 appear across
individual papers, with 2.0 specifically named for CXR classification).
Documented as literature-range-grounded, not a single pinned citation,
since a more precise source wasn't found within a reasonable search.
"""
from __future__ import annotations

import cv2
import numpy as np

CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)


def apply_clahe(gray_u8: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
    return clahe.apply(gray_u8)
