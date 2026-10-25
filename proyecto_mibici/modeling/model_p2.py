#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clases orientadas a objetos para los modelos de P2 (Riesgo de Desbalance).
Fieles al notebook: Regresión Logística directa (sin scaler) y persistencia ingenua.
"""

from pathlib import Path
from typing import List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class BaselinePersistenciaP2(BaseEstimator, ClassifierMixin):
    """
    Línea base ingenua para P2 basada en inercia operativa:
    Asume que el estado de la estación en t+1 será idéntico al observado en t.
    """

    def __init__(self, col_persistencia: str = "en_riesgo"):
        self.col_persistencia = col_persistencia

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.col_persistencia not in X.columns:
            raise KeyError(
                f"La columna '{self.col_persistencia}' no existe en el conjunto de entrada."
            )
        return X[self.col_persistencia].astype(bool).to_numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve probabilidades degeneradas (0.0 o 1.0) para compatibilidad con curvas ROC/PR."""
        preds = self.predict(X).astype(float)
        return np.vstack([1.0 - preds, preds]).T


class RiesgoLogisticoP2(BaseEstimator, ClassifierMixin):
    """
    Regresión Logística para clasificar el riesgo de desbalance,
    idéntica al notebook (sin StandardScaler).

    C es la inversa de la fuerza de regularización. El valor por defecto (1.0)
    es el de scikit-learn, así que sin indicarlo el modelo es el mismo de siempre.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        class_weight: str = "balanced",
        random_state: int = 42,
        max_iter: int = 1000,
        umbral: float = 0.5,
        C: float = 1.0,
    ):
        self.features = features
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.umbral = umbral
        self.C = C

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        self.features_ = (
            list(self.features) if self.features is not None else list(X.columns)
        )
        self.estimator_ = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        self.estimator_.fit(X[self.features_], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise ValueError("El modelo debe ser entrenado antes de ejecutar predicciones.")
        return self.estimator_.predict_proba(X[self.features_])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prob = self.predict_proba(X)[:, 1]
        return (prob >= self.umbral).astype(bool)

    @property
    def classes_(self):
        return self.estimator_.classes_

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "RiesgoLogisticoP2":
        return joblib.load(Path(filepath))#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clases orientadas a objetos para los modelos de P2 (Riesgo de Desbalance).
Fieles al notebook: Regresión Logística directa (sin scaler) y persistencia ingenua.
"""

from pathlib import Path
from typing import List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class BaselinePersistenciaP2(BaseEstimator, ClassifierMixin):
    """
    Línea base ingenua para P2 basada en inercia operativa:
    Asume que el estado de la estación en t+1 será idéntico al observado en t.
    """

    def __init__(self, col_persistencia: str = "en_riesgo"):
        self.col_persistencia = col_persistencia

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.col_persistencia not in X.columns:
            raise KeyError(
                f"La columna '{self.col_persistencia}' no existe en el conjunto de entrada."
            )
        return X[self.col_persistencia].astype(bool).to_numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve probabilidades degeneradas (0.0 o 1.0) para compatibilidad con curvas ROC/PR."""
        preds = self.predict(X).astype(float)
        return np.vstack([1.0 - preds, preds]).T


class RiesgoLogisticoP2(BaseEstimator, ClassifierMixin):
    """
    Regresión Logística para clasificar el riesgo de desbalance,
    idéntica al notebook (sin StandardScaler).

    C es la inversa de la fuerza de regularización. El valor por defecto (1.0)
    es el de scikit-learn, así que sin indicarlo el modelo es el mismo de siempre.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        class_weight: str = "balanced",
        random_state: int = 42,
        max_iter: int = 1000,
        umbral: float = 0.5,
        C: float = 1.0,
    ):
        self.features = features
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_iter = max_iter
        self.umbral = umbral
        self.C = C

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        self.features_ = (
            list(self.features) if self.features is not None else list(X.columns)
        )
        self.estimator_ = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        self.estimator_.fit(X[self.features_], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise ValueError("El modelo debe ser entrenado antes de ejecutar predicciones.")
        return self.estimator_.predict_proba(X[self.features_])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prob = self.predict_proba(X)[:, 1]
        return (prob >= self.umbral).astype(bool)

    @property
    def classes_(self):
        return self.estimator_.classes_

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "RiesgoLogisticoP2":
        return joblib.load(Path(filepath))