#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clase orientada a objetos para el modelo de P1 (Regresión de Demanda Horaria).
Utiliza sklearn.pipeline.Pipeline con StandardScaler y LinearRegression,
garantizando cota inferior en 0.
"""

from pathlib import Path
from typing import List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DemandaLinealP1:
    """Pipeline de regresión lineal para predecir viajes horarios garantizando no-negatividad."""

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression(fit_intercept=fit_intercept)),
        ])
        self.features_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        self.features_ = list(X.columns)
        self.pipeline.fit(X[self.features_], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.features_ is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir.")
        y_pred = self.pipeline.predict(X[self.features_])
        return np.clip(y_pred, 0, None)

    @property
    def estimator(self):
        """Acceso al estimador final por si se requiere registrar en MLflow."""
        return self.pipeline.named_steps["regressor"]

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "DemandaLinealP1":
        return joblib.load(filepath)