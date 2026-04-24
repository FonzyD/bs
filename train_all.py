"""Обучава k-NN и SVM върху всички набори в data/datasets и записва в runs.json.

Използва се, за да се напълни разделът „История на експериментите“.

    python train_all.py                   # всички налични набори
    python train_all.py --only small xlarge hard
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from gait.classifier import GaitClassifier
from gait.dataset import build_gei_dataset
from gait.visualize import plot_confusion_matrix, plot_gei_grid

APP_ROOT = Path(__file__).parent
DATA_DIR = APP_ROOT / "data" / "datasets"
MODEL_DIR = APP_ROOT / "models"
REPORT_DIR = APP_ROOT / "reports"
RUNS_FILE = MODEL_DIR / "runs.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_runs() -> list[dict]:
    if not RUNS_FILE.exists():
        return []
    return json.loads(RUNS_FILE.read_text())


def save_runs(runs: list[dict]) -> None:
    RUNS_FILE.write_text(json.dumps(runs, indent=2, ensure_ascii=False))


def run_experiment(
    dataset_path: Path,
    classifier: str,
    k: int | None,
    seed: int,
    test_size: float,
) -> dict | None:
    """Обучава един модел и връща meta-запис за runs.json."""
    name = f"{classifier}_{dataset_path.name.replace('synthetic_', '')}_{datetime.now().strftime('%H%M%S')}"
    print(f"\n▶ {name}")

    t0 = time.time()
    X, y, _ = build_gei_dataset(dataset_path, verbose=False)
    print(f"  GEI: {len(X)}, субекти: {len(set(y))}")

    y_arr = np.array(y)
    _, counts = np.unique(y_arr, return_counts=True)
    stratify = y if counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    clf_kwargs = {"kind": classifier}
    if classifier == "knn" and k is not None:
        clf_kwargs["k"] = k
    clf = GaitClassifier(**clf_kwargs)
    clf.fit(X_train, y_train)
    report = clf.evaluate(X_test, y_test)
    elapsed = time.time() - t0

    model_path = MODEL_DIR / f"{name}.joblib"
    clf.save(model_path)

    report_subdir = REPORT_DIR / name
    report_subdir.mkdir(parents=True, exist_ok=True)
    plot_gei_grid(
        [X[i] for i in range(min(len(X), 20))],
        [y[i] for i in range(min(len(y), 20))],
        report_subdir / "gei_grid.png",
    )
    plot_confusion_matrix(
        report.confusion, report.labels, report_subdir / "confusion.png",
        title=f"{classifier.upper()} на {dataset_path.name}  acc={report.accuracy:.2f}",
    )

    print(f"  точност: {report.accuracy * 100:.2f}%,  време: {elapsed:.1f}s")

    return {
        "name": name,
        "dataset": dataset_path.name,
        "classifier": classifier,
        "k": k if classifier == "knn" else None,
        "test_size": test_size,
        "seed": seed,
        "n_samples": len(X),
        "n_subjects": len(set(y)),
        "accuracy": float(report.accuracy),
        "elapsed_sec": elapsed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(model_path.relative_to(APP_ROOT)),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None,
                    help="обработвай само избрани набори (напр. small hard)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.3)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    datasets = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
    if args.only:
        wanted = {f"synthetic_{n}" if not n.startswith("synthetic_") else n for n in args.only}
        datasets = [d for d in datasets if d.name in wanted]

    if not datasets:
        print("Няма намерени набори.")
        return

    print(f"Ще бъдат обучени {len(datasets) * 2} модела върху {len(datasets)} набора.")
    runs = load_runs()

    for ds in datasets:
        for classifier, k in (("knn", 3), ("svm", None)):
            try:
                meta = run_experiment(
                    dataset_path=ds,
                    classifier=classifier,
                    k=k,
                    seed=args.seed,
                    test_size=args.test_size,
                )
                if meta is not None:
                    runs.append(meta)
                    save_runs(runs)
            except Exception as exc:
                print(f"  ✗ грешка: {exc}")

    print(f"\nГотово. Всичко записано в {RUNS_FILE}")


if __name__ == "__main__":
    main()
