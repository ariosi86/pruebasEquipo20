#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clase orientada a objetos para el modelo de P2 (Clasificación de Riesgo de Desbalance).
Utiliza sklearn.pipeline.Pipeline con StandardScaler y LogisticRegression balanceada,
permitiendo calibrar el umbral de decisión operativo.
"""

from pathlib import Path
from typing import List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from proyecto_mibici.config import RANDOM_SEED


# ==========================================
# MODELO P2: RIESGO DE DESBALANCE (CLASIFICACIÓN)
# ==========================================
class RiesgoLogisticoP2:
    """Pipeline clasificador logístico ponderado para predecir riesgo de desbalance."""

    def __init__(
        self,
        class_weight: str = "balanced",
        random_state: int = RANDOM_SEED,
        max_iter: int = 1000,
        umbral_decision: float = 0.5,
    ):
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.umbral_decision = umbral_decision
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight=class_weight,
                random_state=random_state,
                max_iter=max_iter,
            )),
        ])
        self.features_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        self.features_ = list(X.columns)
        self.pipeline.fit(X[self.features_], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.features_ is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir.")
        return self.pipeline.predict_proba(X[self.features_])[:, 1]

    def predict(self, X: pd.DataFrame, umbral: Optional[float] = None) -> np.ndarray:
        corte = self.umbral_decision if umbral is None else umbral
        return self.predict_proba(X) >= corte

    @property
    def estimator(self):
        """Acceso al estimador final por si se requiere registrar en MLflow."""
        return self.pipeline.named_steps["classifier"]

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "RiesgoLogisticoP2":
        return joblib.load(filepath)