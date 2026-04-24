"""Помощни функции за визуализация на резултатите."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_gei(gei: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), gei)


def plot_gei_grid(
    geis: list[np.ndarray],
    labels: list[str],
    path: str | Path,
    cols: int = 5,
) -> None:
    """Показва мрежа от GEI изображения, групирани по етикет."""
    n = len(geis)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 3))
    axes = np.atleast_2d(axes)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i < n:
            ax.imshow(geis[i], cmap="gray")
            ax.set_title(labels[i], fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray, labels: list[str], path: str | Path, title: str = "Confusion matrix"
) -> None:
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.6), max(4, len(labels) * 0.6)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Предсказан")
    ax.set_ylabel("Реален")
    ax.set_title(title)

    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pose_signal(signal: np.ndarray, path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(signal, linewidth=1.2)
    ax.set_xlabel("Кадър")
    ax.set_ylabel("Ъгъл (градуси)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
