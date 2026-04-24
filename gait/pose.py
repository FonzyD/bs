"""Извличане на динамични характеристики на походката чрез MediaPipe Pose."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # MediaPipe е опционален – ако липсва, pose-модулът не работи
    mp = None


# Индекси на ключови точки в BlazePose (MediaPipe Pose)
LM = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}


@dataclass
class PoseSequence:
    """Последователност от ъглови характеристики по кадри."""
    left_knee: np.ndarray
    right_knee: np.ndarray
    left_hip: np.ndarray
    right_hip: np.ndarray

    def as_matrix(self) -> np.ndarray:
        return np.stack(
            [self.left_knee, self.right_knee, self.left_hip, self.right_hip], axis=1
        )


class PoseFeatureExtractor:
    """Извлича ъгли в коленните и тазобедрените стави за всеки кадър."""

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5):
        if mp is None:
            raise RuntimeError(
                "Необходим е mediapipe: pip install mediapipe"
            )
        self._pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def close(self):
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def process_video(self, video_path: str | Path) -> PoseSequence:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Не може да се отвори видеото: {video_path}")

        lk, rk, lh, rh = [], [], [], []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._pose.process(rgb)
            if not result.pose_landmarks:
                continue
            angles = _compute_angles(result.pose_landmarks.landmark)
            lk.append(angles["left_knee"])
            rk.append(angles["right_knee"])
            lh.append(angles["left_hip"])
            rh.append(angles["right_hip"])
        cap.release()

        return PoseSequence(
            left_knee=np.array(lk, dtype=np.float32),
            right_knee=np.array(rk, dtype=np.float32),
            left_hip=np.array(lh, dtype=np.float32),
            right_hip=np.array(rh, dtype=np.float32),
        )

    @staticmethod
    def summarize(seq: PoseSequence) -> np.ndarray:
        """Превръща последователност в компактен числов вектор.

        За всяка ставна ос се извличат: средна стойност, стандартно
        отклонение, размах, доминираща честота (грубо – чрез FFT).
        """
        parts = []
        for signal in (seq.left_knee, seq.right_knee, seq.left_hip, seq.right_hip):
            if signal.size == 0:
                parts.extend([0.0, 0.0, 0.0, 0.0])
                continue
            parts.extend(
                [
                    float(np.mean(signal)),
                    float(np.std(signal)),
                    float(np.ptp(signal)),
                    _dominant_frequency(signal),
                ]
            )
        return np.array(parts, dtype=np.float32)


def _compute_angles(landmarks) -> dict:
    def pt(i):
        lm = landmarks[i]
        return np.array([lm.x, lm.y], dtype=np.float32)

    return {
        "left_knee": _angle(pt(LM["left_hip"]), pt(LM["left_knee"]), pt(LM["left_ankle"])),
        "right_knee": _angle(pt(LM["right_hip"]), pt(LM["right_knee"]), pt(LM["right_ankle"])),
        "left_hip": _angle(pt(LM["left_shoulder"]), pt(LM["left_hip"]), pt(LM["left_knee"])),
        "right_hip": _angle(
            pt(LM["right_shoulder"]), pt(LM["right_hip"]), pt(LM["right_knee"])
        ),
    }


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ъгъл в b, образуван от отсечките ba и bc, в градуси."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return 0.0
    cos_a = float(np.dot(ba, bc) / denom)
    cos_a = max(-1.0, min(1.0, cos_a))
    return float(np.degrees(np.arccos(cos_a)))


def _dominant_frequency(signal: np.ndarray) -> float:
    if signal.size < 4:
        return 0.0
    x = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(x))
    if spectrum.size <= 1:
        return 0.0
    idx = int(np.argmax(spectrum[1:])) + 1
    return float(idx) / len(signal)
