"""Зареждане на видео-набор от данни и построяване на GEI матрица за обучение."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .gei import compute_gei
from .silhouette import extract_silhouettes


def load_dataset(root: str | Path) -> dict[str, list[Path]]:
    """Очаквана структура: root/<subject_id>/*.mp4 (или .avi/.mov).

    Връща dict {subject_id: [пътища до видеа]}.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Няма такава директория: {root}")

    data: dict[str, list[Path]] = {}
    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir():
            continue
        videos = sorted(
            p for p in subject_dir.iterdir()
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        )
        if videos:
            data[subject_dir.name] = videos
    return data


def build_gei_dataset(
    root: str | Path,
    target_height: int = 128,
    target_width: int = 88,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str], list[Path]]:
    """Обхожда целия набор и връща (X, y, paths).

    X има форма (N, target_height, target_width); y е списък с ID на субектите.
    Видеата, от които не може да се построи валиден GEI, се пропускат.
    """
    data = load_dataset(root)
    X: list[np.ndarray] = []
    y: list[str] = []
    paths: list[Path] = []

    for subject, videos in data.items():
        for video in videos:
            if verbose:
                print(f"  [{subject}] {video.name}")
            silhouettes = extract_silhouettes(video)
            if len(silhouettes) < 10:
                if verbose:
                    print(f"    пропуснато: само {len(silhouettes)} силуета")
                continue
            gei = compute_gei(silhouettes, target_height, target_width)
            if gei is None:
                if verbose:
                    print("    пропуснато: неуспешен GEI")
                continue
            X.append(gei)
            y.append(subject)
            paths.append(video)

    if not X:
        raise RuntimeError("Не беше изграден нито един GEI. Проверете видео-файловете.")

    return np.stack(X, axis=0), y, paths
