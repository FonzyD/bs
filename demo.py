"""Демо end-to-end: синтетични силуети на няколко „субекта“, GEI, обучение, тест.

Този скрипт не изисква реални видео-файлове – полезен е за бърза проверка,
че целият пайплайн работи. При реална употреба, замени build_synthetic с
реална папка от видеа и използвай train.py.

    python demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from gait.classifier import GaitClassifier
from gait.gei import compute_gei
from gait.visualize import plot_confusion_matrix, plot_gei_grid


def synth_silhouette(
    height: int, width: int, phase: float, subject_params: dict
) -> np.ndarray:
    """Опростен „човек“: овално тяло + люлеещи се крайници."""
    img = np.zeros((height, width), dtype=np.uint8)

    cx = width // 2
    cy = height // 2

    # Глава
    head_r = subject_params["head_r"]
    _fill_circle(img, (cx, int(height * 0.12)), head_r, 255)

    # Тяло
    body_w = subject_params["body_w"]
    body_h = subject_params["body_h"]
    img[int(height * 0.18):int(height * 0.18 + body_h), cx - body_w // 2:cx + body_w // 2] = 255

    # Ръце
    arm_len = subject_params["arm_len"]
    arm_swing = subject_params["arm_swing"]
    arm_angle = np.sin(phase) * arm_swing
    shoulder_y = int(height * 0.22)
    _draw_limb(img, (cx - body_w // 2, shoulder_y), arm_len, np.pi / 2 + arm_angle)
    _draw_limb(img, (cx + body_w // 2, shoulder_y), arm_len, np.pi / 2 - arm_angle)

    # Крака
    leg_len = subject_params["leg_len"]
    leg_swing = subject_params["leg_swing"]
    leg_angle = np.sin(phase) * leg_swing
    hip_y = int(height * 0.18 + body_h)
    _draw_limb(img, (cx - body_w // 4, hip_y), leg_len, np.pi / 2 - leg_angle, thickness=3)
    _draw_limb(img, (cx + body_w // 4, hip_y), leg_len, np.pi / 2 + leg_angle, thickness=3)

    return img


def _fill_circle(img: np.ndarray, center: tuple[int, int], r: int, value: int) -> None:
    cx, cy = center
    h, w = img.shape
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    img[mask] = value


def _draw_limb(
    img: np.ndarray, start: tuple[int, int], length: int, angle: float, thickness: int = 2
) -> None:
    import cv2

    # Люлеене около вертикалата – ъгълът определя хоризонтална компонента.
    offset_x = int(np.sin(angle - np.pi / 2) * length * 0.5)
    end = (start[0] + offset_x, start[1] + length)
    cv2.line(img, start, end, 255, thickness)


def build_subject(subject_id: str, n_sequences: int, rng: np.random.Generator) -> tuple[list, list]:
    """Генерира n_sequences „записа“ за един субект – всеки е поредица от силуети."""
    # Уникални параметри, които определят „походката“ на субекта
    params = {
        "head_r": int(rng.integers(6, 10)),
        "body_w": int(rng.integers(18, 26)),
        "body_h": int(rng.integers(40, 52)),
        "arm_len": int(rng.integers(30, 42)),
        "arm_swing": float(rng.uniform(0.3, 0.7)),
        "leg_len": int(rng.integers(36, 48)),
        "leg_swing": float(rng.uniform(0.3, 0.7)),
    }

    geis, ids = [], []
    for _ in range(n_sequences):
        # 30 кадъра покриват ~2 цикъла на походката
        silhouettes = [
            synth_silhouette(128, 88, phase=2 * np.pi * t / 12, subject_params=params)
            for t in range(30)
        ]
        gei = compute_gei(silhouettes)
        if gei is not None:
            geis.append(gei)
            ids.append(subject_id)
    return geis, ids


def main() -> None:
    rng = np.random.default_rng(42)
    n_subjects = 8
    seqs_per_subject = 10

    X, y = [], []
    for sid in range(n_subjects):
        geis, ids = build_subject(f"subject_{sid:02d}", seqs_per_subject, rng)
        X.extend(geis)
        y.extend(ids)

    X = np.stack(X, axis=0)
    print(f"Синтезирани {len(X)} GEI за {n_subjects} субекта")

    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    plot_gei_grid(list(X[:16]), y[:16], report_dir / "demo_gei_grid.png", cols=4)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )

    for kind in ("knn", "svm"):
        clf = GaitClassifier(kind=kind)
        clf.fit(X_train, y_train)
        rep = clf.evaluate(X_test, y_test)
        print(f"\n=== {kind.upper()} ===")
        print(f"Точност: {rep.accuracy:.3f}")
        print(rep.per_class)
        plot_confusion_matrix(
            rep.confusion,
            rep.labels,
            report_dir / f"demo_confusion_{kind}.png",
            title=f"{kind.upper()} acc={rep.accuracy:.2f}",
        )

    print(f"\nРезултатите са записани в {report_dir}/")


if __name__ == "__main__":
    main()
