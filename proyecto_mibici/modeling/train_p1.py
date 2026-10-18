#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de entrenamiento, evaluación y registro en MLflow para P1 (Demanda Horaria).
Ejecución: python -m proyecto_mibici.modeling.train_p1
"""

import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from proyecto_mibici.config import (
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR,
    P1_FEATURES_V2,
    P1_TARGET,
    P1_TEST_FIN,
    P1_TEST_INICIO,
    P1_VAL_INICIO,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.features import construir_dataset_p1
from proyecto_mibici.modeling.model_p1 import DemandaLinealP1
from proyecto_mibici.modeling.tracking import MLFlowTracker
from proyecto_mibici.plots import figuras_p1

app = typer.Typer()


@app.command()
def main():
    # Inicializar la clase orientada a objetos de tracking
    tracker = MLFlowTracker(
        tracking_uri=URI_MLFLOW_TRACKING,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )

    logger.info("Cargando y preparando dataset P1...")
    df_p1, df_stations = construir_dataset_p1()

    t = df_p1["target_hour"]
    df_train = df_p1.loc[t < pd.Timestamp(P1_VAL_INICIO)].copy()
    df_val = df_p1.loc[(t >= pd.Timestamp(P1_VAL_INICIO)) & (t < pd.Timestamp(P1_TEST_INICIO))].copy()
    df_test = df_p1.loc[(t >= pd.Timestamp(P1_TEST_INICIO)) & (t < pd.Timestamp(P1_TEST_FIN))].copy()
    splits = {"Train": df_train, "Validación": df_val, "Test": df_test}

    # 1. Instanciar y entrenar la clase POO / Pipeline
    logger.info("Entrenando DemandaLinealP1 (StandardScaler + LinearRegression)...")
    modelo = DemandaLinealP1()
    modelo.fit(df_train[P1_FEATURES_V2], df_train[P1_TARGET])

    # 2. Persistir modelo localmente
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    modelo.save(MODELS_DIR / "p1_linear_model.pkl")
    logger.info("Pipeline P1 guardado exitosamente en models/p1_linear_model.pkl")

    # 3. Inferencia y cálculo de métricas para Train y Test
    y_train_pred = modelo.predict(df_train)
    y_train_real = df_train[P1_TARGET]
    mae_train = mean_absolute_error(y_train_real, y_train_pred)
    rmse_train = root_mean_squared_error(y_train_real, y_train_pred)

    y_test_pred = modelo.predict(df_test)
    y_test_real = df_test[P1_TARGET]
    mae_test = mean_absolute_error(y_test_real, y_test_pred)
    rmse_test = root_mean_squared_error(y_test_real, y_test_pred)

    tabla = pd.DataFrame([
        {"conjunto": "Train", "MAE": mae_train, "RMSE": rmse_train},
        {"conjunto": "Test", "MAE": mae_test, "RMSE": rmse_test},
    ])
    tabla_est_vacia = pd.DataFrame(columns=["origen_id", "nombre_estacion", "MAE"])
    figuras = figuras_p1(df_test, y_test_pred, P1_TARGET, "v2_lineal_rolling")

    # 4. Registro formal en MLflow Tracking usando la clase MLFlowTracker
    params_base, tags_base = tracker.armar_params_tags(
        "P1", splits, extra_params={"modelo": "DemandaLinealP1", "features": ", ".join(P1_FEATURES_V2)}
    )
    tracker.registrar_run(
        run_name="P1_v2_lineal_rolling",
        params=params_base,
        tags={**tags_base, "version": "v2_lineal_rolling"},
        tabla=tabla,
        tabla_est=tabla_est_vacia,
        modelo=modelo.pipeline,  # Pipeline completo serializado en MLflow
        X_ejemplo=df_train[P1_FEATURES_V2].head(5),
        figuras=figuras,
    )
    logger.info(f"P1 completado. Train MAE: {mae_train:.3f} | Test MAE: {mae_test:.3f}, RMSE: {rmse_test:.3f}")


if __name__ == "__main__":
    app()