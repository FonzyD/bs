"""Пакет за засичане и разпознаване на хора по походка."""

from .silhouette import extract_silhouettes, normalize_silhouette
from .gei import compute_gei, detect_gait_cycle
from .pose import PoseFeatureExtractor
from .classifier import GaitClassifier
from .dataset import load_dataset, build_gei_dataset

__all__ = [
    "extract_silhouettes",
    "normalize_silhouette",
    "compute_gei",
    "detect_gait_cycle",
    "PoseFeatureExtractor",
    "GaitClassifier",
    "load_dataset",
    "build_gei_dataset",
]
