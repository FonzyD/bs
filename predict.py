"""Разпознаване на субект от видео с вече обучен модел.

Употреба:
    python predict.py --model models/gait.joblib --video test.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gait.classifier import GaitClassifier
from gait.gei import compute_gei
from gait.silhouette import extract_silhouettes
from gait.visualize import save_gei


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--save-gei", type=Path, default=None)
    ap.add_argument("--top-k", type=int, default=3)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Извличане на силуети от {args.video}")
    silhouettes = extract_silhouettes(args.video)
    print(f"  открити {len(silhouettes)} силуета")

    if len(silhouettes) < 10:
        raise RuntimeError("Твърде малко силуети – проверете видеото/фона.")

    gei = compute_gei(silhouettes)
    if gei is None:
        raise RuntimeError("Неуспешно построяване на GEI.")

    if args.save_gei:
        save_gei(gei, args.save_gei)
        print(f"  GEI записано в {args.save_gei}")

    clf = GaitClassifier.load(args.model)

    proba, classes = clf.predict_proba(gei[None, ...])
    probs = proba[0]
    order = probs.argsort()[::-1][: args.top_k]

    print("\nНай-вероятни субекти:")
    for rank, idx in enumerate(order, 1):
        print(f"  {rank}. {classes[idx]}  ({probs[idx] * 100:.1f}%)")


if __name__ == "__main__":
    main()
