"""Класификатор за разпознаване по GEI или pose-вектори."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


@dataclass
class TrainReport:
    accuracy: float
    per_class: str
    confusion: np.ndarray
    labels: list[str]


class GaitClassifier:
    """Опаковка около sklearn-класификатор с нормализация и encode на етикети."""

    def __init__(
        self,
        kind: Literal["knn", "svm"] = "knn",
        k: int = 3,
        svm_C: float = 10.0,
        svm_gamma: str | float = "scale",
    ):
        self.kind = kind
        if kind == "knn":
            self.model = KNeighborsClassifier(n_neighbors=k, metric="euclidean", weights="distance")
        elif kind == "svm":
            self.model = SVC(C=svm_C, gamma=svm_gamma, kernel="rbf", probability=True)
        else:
            raise ValueError(f"Непознат класификатор: {kind}")

        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self._fitted = False

    def fit(self, X: np.ndarray, y: list[str] | np.ndarray) -> None:
        X = _flatten(X)
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        y_enc = self.encoder.fit_transform(y)
        self.model.fit(Xs, y_enc)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = _flatten(X)
        Xs = self.scaler.transform(X)
        y_pred = self.model.predict(Xs)
        return self.encoder.inverse_transform(y_pred)

    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, list[str]]:
        self._check_fitted()
        X = _flatten(X)
        Xs = self.scaler.transform(X)
        proba = self.model.predict_proba(Xs)
        return proba, list(self.encoder.classes_)

    def evaluate(self, X: np.ndarray, y_true: list[str] | np.ndarray) -> TrainReport:
        y_pred = self.predict(X)
        acc = accuracy_score(y_true, y_pred)
        rep = classification_report(y_true, y_pred, zero_division=0)
        labels = sorted(set(list(y_true) + list(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return TrainReport(accuracy=acc, per_class=rep, confusion=cm, labels=labels)

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "kind": self.kind,
                "model": self.model,
                "scaler": self.scaler,
                "encoder": self.encoder,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "GaitClassifier":
        data = joblib.load(path)
        obj = cls(kind=data["kind"])
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.encoder = data["encoder"]
        obj._fitted = True
        return obj

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Класификаторът не е обучен. Извикайте fit() първо.")


def _flatten(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(1, -1)
    if X.ndim == 2:
        return X
    return X.reshape(X.shape[0], -1)
