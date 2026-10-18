#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de entrenamiento, evaluación y registro en MLflow para P2 (Riesgo de Desbalance).
Ejecución: python -m proyecto_mibici.modeling.train_p2
"""

import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import auc, classification_report, precision_recall_curve

from proyecto_mibici.config import (
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR,
    P2_OBJETIVO,
    P2_UMBRAL,
    P2_VARIABLES,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.features import construir_dataset_p2
from proyecto_mibici.modeling.model_p2 import RiesgoLogisticoP2
from proyecto_mibici.modeling.tracking import MLFlowTracker
from proyecto_mibici.plots import figuras_p2

app = typer.Typer()


def extraer_metricas_clasificacion(y_real, y_pred, y_prob, conjunto_nombre: str) -> dict:
    """Calcula precision, recall, f1-score y PR-AUC para la clase de riesgo (positiva)."""
    rep = classification_report(y_real, y_pred, output_dict=True, zero_division=0)
    pos_key = True if True in rep else "True"
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_real, y_prob)
    pr_auc_val = float(auc(recall_curve, precision_curve))
    
    return {
        "conjunto": conjunto_nombre,
        "precision": float(rep[pos_key]["precision"]),
        "recall": float(rep[pos_key]["recall"]),
        "f1_score": float(rep[pos_key]["f1-score"]),
        "pr_auc": pr_auc_val,
    }


@app.command()
def main():
    # Inicializar la clase orientada a objetos de tracking
    tracker = MLFlowTracker(
        tracking_uri=URI_MLFLOW_TRACKING,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )

    logger.info("Cargando y preparando dataset P2...")
    df_p2 = construir_dataset_p2()
    df_train = df_p2[df_p2["conjunto"] == "entrenamiento"].copy()
    df_val = df_p2[df_p2["conjunto"] == "validacion"].copy()
    df_test = df_p2[(df_p2["conjunto"] == "prueba") & (~df_p2["estacion_nueva"])].copy()
    splits = {"Train": df_train, "Validación": df_val, "Test": df_test}

    # 1. Instanciar y entrenar la clase POO / Pipeline
    logger.info("Entrenando clase RiesgoLogisticoP2 (StandardScaler + LogisticRegression)...")
    modelo = RiesgoLogisticoP2()
    modelo.fit(df_train[P2_VARIABLES], df_train[P2_OBJETIVO])

    # 2. Persistir modelo localmente
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    modelo.save(MODELS_DIR / "p2_logistic_model.pkl")
    logger.info("Pipeline P2 guardado exitosamente en models/p2_logistic_model.pkl")

    # 3. Inferencia y evaluación en Train y Test
    y_train_real = df_train[P2_OBJETIVO]
    y_train_pred = modelo.predict(df_train)
    y_train_prob = modelo.predict_proba(df_train)
    metricas_train = extraer_metricas_clasificacion(y_train_real, y_train_pred, y_train_prob, "Train")

    y_test_real = df_test[P2_OBJETIVO]
    y_test_pred = modelo.predict(df_test)
    y_test_prob = modelo.predict_proba(df_test)
    metricas_test = extraer_metricas_clasificacion(y_test_real, y_test_pred, y_test_prob, "Test")

    tabla = pd.DataFrame([metricas_train, metricas_test])
    tabla_est_vacia = pd.DataFrame(columns=["origen_id", "nombre_estacion", "MAE"])
    figuras = figuras_p2(y_test_real, y_test_pred, y_test_prob, "v1_logistic")

    # 4. Registro formal en MLflow Tracking usando la clase MLFlowTracker
    params_base, tags_base = tracker.armar_params_tags(
        "P2",
        splits,
        extra_params={"modelo": "RiesgoLogisticoP2", "umbral": P2_UMBRAL, "variables": ", ".join(P2_VARIABLES)},
        col_tiempo="hora",
    )
    tracker.registrar_run(
        run_name="P2_v1_logistic_regression",
        params=params_base,
        tags={**tags_base, "version": "v1_logistic"},
        tabla=tabla,
        tabla_est=tabla_est_vacia,
        modelo=modelo.pipeline,  # Pipeline completo de scikit-learn
        X_ejemplo=df_train[P2_VARIABLES].head(5),
        figuras=figuras,
    )
    logger.info(
        f"P2 completado con éxito. Test Precision: {metricas_test['precision']:.3f} | "
        f"Recall: {metricas_test['recall']:.3f} | F1: {metricas_test['f1_score']:.3f} | "
        f"PR-AUC: {metricas_test['pr_auc']:.3f}"
    )


if __name__ == "__main__":
    app()