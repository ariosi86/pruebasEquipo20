#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clase orientada a objetos para el modelo de P3 (Clustering de Estaciones).
Utiliza sklearn.pipeline.Pipeline integrando StandardScaler y KMeans.
"""

from pathlib import Path
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from proyecto_mibici.config import RANDOM_SEED, P3_K_ELEGIDO


# ==========================================
# MODELO P3: CLUSTERING DE ESTACIONES (USO)
# ==========================================
class ClusterEstacionesP3:
    """Pipeline que integra el escalamiento estándar y el agrupamiento KMeans."""

    def __init__(self, n_clusters: int = P3_K_ELEGIDO, random_state: int = RANDOM_SEED):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)),
        ])
        self.features_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame):
        self.features_ = list(X.columns)
        self.pipeline.fit(X[self.features_])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna las variables escaladas con los parámetros calculados en el fit."""
        if self.features_ is None:
            raise ValueError("El modelo debe ser entrenado antes de transformar datos.")
        return self.pipeline.named_steps["scaler"].transform(X[self.features_])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Asigna etiquetas de cluster a nuevas observaciones transformándolas primero."""
        if self.features_ is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir.")
        return self.pipeline.predict(X[self.features_])

    @property
    def labels_(self) -> np.ndarray:
        """Etiquetas asignadas al conjunto de entrenamiento."""
        return self.pipeline.named_steps["kmeans"].labels_

    @property
    def estimator(self):
        """Acceso al estimador KMeans por si se requiere registrar en MLflow."""
        return self.pipeline.named_steps["kmeans"]

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "ClusterEstacionesP3":
        return joblib.load(filepath)