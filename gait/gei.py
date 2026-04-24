"""Gait Energy Image и сегментиране на цикли на походката."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import find_peaks

from .silhouette import normalize_silhouette


def compute_gei(
    silhouettes: Sequence[np.ndarray],
    target_height: int = 128,
    target_width: int = 88,
) -> np.ndarray | None:
    """Пресмята Gait Energy Image от последователност от силуети.

    GEI(x,y) = (1/N) * Σ S_t(x,y).  Преди осредняване всеки силует се
    нормализира (изрязва, мащабира, центрира). Връща изображение в степени
    на сивото (uint8) или None, ако не може да се изгради валиден GEI.
    """
    normalized = [
        normalize_silhouette(s, target_height, target_width) for s in silhouettes
    ]
    normalized = [n for n in normalized if n is not None]
    if not normalized:
        return None

    stack = np.stack(normalized, axis=0).astype(np.float32)
    gei = np.mean(stack, axis=0)
    return gei.astype(np.uint8)


def detect_gait_cycle(silhouettes: Sequence[np.ndarray], fps: float = 25.0) -> list[tuple[int, int]]:
    """Намира граници на отделни цикли на походката.

    Използва сигнала „ширина на ограничаващия правоъгълник“ – той е
    периодичен с локални максимуми при максимално отваряне на краката.
    Връща списък от (start_idx, end_idx) за всеки открит цикъл.
    """
    if len(silhouettes) < 10:
        return []

    widths = np.array([_bbox_width(s) for s in silhouettes], dtype=np.float32)
    if widths.max() <= 0:
        return []

    widths = widths - np.mean(widths)
    # При нормална скорост на ходене има около 2 стъпки в секунда,
    # което при 25 fps означава пик на всеки ~12-13 кадъра.
    min_distance = max(5, int(0.35 * fps))
    peaks, _ = find_peaks(widths, distance=min_distance, prominence=np.std(widths) * 0.5)

    if len(peaks) < 3:
        return []

    cycles: list[tuple[int, int]] = []
    for i in range(len(peaks) - 2):
        cycles.append((int(peaks[i]), int(peaks[i + 2])))
    return cycles


def _bbox_width(sil: np.ndarray) -> int:
    xs = np.where(sil.any(axis=0))[0]
    if xs.size == 0:
        return 0
    return int(xs.max() - xs.min() + 1)
