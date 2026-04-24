"""Пакет за засичане и разпознаване на хора по походка."""

from .silhouette import extract_silhouettes, normalize_silhouette
from .gei import compute_gei, detect_gait_cycle
from .classifier import GaitClassifier
from .dataset import load_dataset, build_gei_dataset

# Pose модулът изисква mediapipe, което не винаги е налично
# (напр. на Streamlit Cloud с по-нов Python). Зареждаме го „лениво“,
# за да не падне целият пакет.
try:
    from .pose import PoseFeatureExtractor
except ImportError:
    PoseFeatureExtractor = None  # type: ignore[assignment]

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
