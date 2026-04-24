"""Извличане и нормализация на силуети от видео."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def extract_silhouettes(
    video_path: str | Path,
    history: int = 500,
    var_threshold: float = 16.0,
    min_area: int = 1500,
) -> list[np.ndarray]:
    """Извлича бинарни силуети на движещ се човек от видеофайл.

    Използва MOG2 за моделиране на фона. Всеки кадър се обработва
    независимо; връща се списък от маски, в които 255 означава преден план.
    Пропускат се кадри, в които няма достатъчно голям свързан компонент.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Не може да се отвори видеото: {video_path}")

    bg = cv2.createBackgroundSubtractorMOG2(
        history=history, varThreshold=var_threshold, detectShadows=False
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    silhouettes: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        largest = _largest_component(mask, min_area)
        if largest is not None:
            silhouettes.append(largest)

    cap.release()
    return silhouettes


def extract_silhouettes_from_frames(frames: Iterable[np.ndarray]) -> list[np.ndarray]:
    """Същото като extract_silhouettes, но приема списък от кадри.

    Полезно за тестване с генерирани кадри или когато вече имаме масив с кадри.
    """
    bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    silhouettes = []
    for frame in frames:
        mask = bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        largest = _largest_component(mask, min_area=500)
        if largest is not None:
            silhouettes.append(largest)
    return silhouettes


def _largest_component(mask: np.ndarray, min_area: int) -> np.ndarray | None:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    if stats[idx, cv2.CC_STAT_AREA] < min_area:
        return None
    out = np.zeros_like(mask)
    out[labels == idx] = 255
    return out


def normalize_silhouette(
    sil: np.ndarray, target_height: int = 128, target_width: int = 88
) -> np.ndarray | None:
    """Изрязва силуета по ограничаващия правоъгълник и го мащабира.

    Пази съотношението на страните (height e водещото измерение),
    центрира резултата хоризонтално в изображение с фиксиран размер.
    Връща None, ако силуетът е празен или по-широк от target_width.
    """
    ys, xs = np.where(sil > 0)
    if len(xs) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cropped = sil[y_min : y_max + 1, x_min : x_max + 1]

    h, w = cropped.shape
    scale = target_height / h
    new_w = max(1, int(round(w * scale)))
    if new_w > target_width:
        # Прекалено широк силует (вероятно чанта/разперени ръце) – мащабираме по ширина.
        scale = target_width / w
        new_w = target_width
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((target_height, target_width), dtype=np.uint8)
        y_off = (target_height - new_h) // 2
        canvas[y_off : y_off + new_h, :] = resized
        return canvas

    resized = cv2.resize(cropped, (new_w, target_height), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((target_height, target_width), dtype=np.uint8)
    x_off = (target_width - new_w) // 2
    canvas[:, x_off : x_off + new_w] = resized
    return canvas
