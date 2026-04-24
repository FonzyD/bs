"""Обучение на модел за разпознаване по походка върху GEI-набор от данни.

Употреба:
    python train.py --data data/videos --out models/gait.joblib --classifier knn
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from gait.classifier import GaitClassifier
from gait.dataset import build_gei_dataset
from gait.visualize import plot_confusion_matrix, plot_gei_grid


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="коренова папка с видеа (data/<subject>/*.mp4)")
    ap.add_argument("--out", type=Path, default=Path("models/gait.joblib"))
    ap.add_argument("--classifier", choices=["knn", "svm"], default="knn")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-dir", type=Path, default=Path("reports"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Зареждане на набор от {args.data}")
    X, y, paths = build_gei_dataset(args.data)
    print(f"Получени {len(X)} GEI за {len(set(y))} субекта")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    plot_gei_grid(
        [X[i] for i in range(min(len(X), 20))],
        [y[i] for i in range(min(len(y), 20))],
        args.report_dir / "gei_grid.png",
    )

    # stratify гарантира, че всеки субект присъства и в train, и в test.
    stratify = y if min(np.bincount(np.unique(y, return_inverse=True)[1])) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=stratify
    )

    clf = GaitClassifier(kind=args.classifier, k=args.k)
    clf.fit(X_train, y_train)

    report = clf.evaluate(X_test, y_test)
    print(f"\nТочност върху тестовата извадка: {report.accuracy:.3f}")
    print("\n" + report.per_class)

    plot_confusion_matrix(
        report.confusion,
        report.labels,
        args.report_dir / "confusion.png",
        title=f"{args.classifier.upper()}  acc={report.accuracy:.2f}",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    clf.save(args.out)
    print(f"Моделът е записан в: {args.out}")
    print(f"Отчетите са в: {args.report_dir}")


if __name__ == "__main__":
    main()
