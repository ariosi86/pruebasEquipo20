#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de entrenamiento, evaluación y registro en MLflow para P1 (Demanda Horaria).
Integra, evalúa y registra en MLflow los 6 modelos:
  - 4 baselines ingenuos (persistencia, lag 24h, lag 168h, promedio 24h/168h)
  - v1 Regresión Lineal (lags + componentes temporales)
  - v2 Regresión Lineal (lags + calendario + medias móviles rolling_3h y rolling_168h)
Exporta métricas en reports/p1_comparacion.parquet y predicciones en reports/p1_predicciones_modelos.parquet.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from proyecto_mibici.config import (
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR,
    P1_FEATURES_V1,
    P1_FEATURES_V2,
    P1_TARGET,
    P1_TEST_FIN,
    P1_TEST_INICIO,
    P1_VAL_INICIO,
    REPORTS_DIR,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.features import construir_dataset_p1
from proyecto_mibici.modeling.model_p1 import BaselineIngenuoP1, DemandaLinealP1
from proyecto_mibici.modeling.tracking import MLFlowTracker
from proyecto_mibici.plots import figuras_p1

app = typer.Typer()


def evaluar_metricas_regresion(y_real, y_pred, conjunto_nombre: str) -> dict:
    """Calcula MAE, RMSE, WAPE y Sesgo para series de tiempo."""
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    suma_real = y_real.sum()
    wape = float(np.abs(y_real - y_pred).sum() / suma_real) if suma_real > 0 else 0.0

    return {
        "conjunto": conjunto_nombre,
        "MAE": float(mean_absolute_error(y_real, y_pred)),
        "RMSE": float(root_mean_squared_error(y_real, y_pred)),
        "WAPE": wape,
        "Sesgo": float((y_pred - y_real).mean()),
    }


def calcular_metricas_por_estacion(
    df: pd.DataFrame, pred: np.ndarray, target: str, df_stations: pd.DataFrame = None
) -> pd.DataFrame:
    """Calcula métricas desglosadas por estación para el artefacto en MLflow."""
    res = df[["origen_id", target]].copy()
    res["pred"] = np.asarray(pred)
    filas = []
    for est, g in res.groupby("origen_id"):
        y_r = g[target].to_numpy()
        y_p = g["pred"].to_numpy()
        suma_r = y_r.sum()
        wape = float(np.abs(y_r - y_p).sum() / suma_r) if suma_r > 0 else 0.0
        filas.append({
            "origen_id": est,
            "horas_evaluadas": len(g),
            "viajes_reales_total": float(suma_r),
            "viajes_predichos_total": float(y_p.sum()),
            "MAE": float(mean_absolute_error(y_r, y_p)),
            "RMSE": float(root_mean_squared_error(y_r, y_p)),
            "WAPE": wape,
            "Sesgo": float((y_p - y_r).mean()),
        })
    out = pd.DataFrame(filas)
    if df_stations is not None and "id" in df_stations.columns and "name" in df_stations.columns:
        out = (
            out.merge(df_stations[["id", "name"]], left_on="origen_id", right_on="id", how="left")
            .drop(columns="id")
            .rename(columns={"name": "nombre_estacion"})
        )
    out["error_abs_total"] = out["MAE"] * out["horas_evaluadas"]
    return out


def imprimir_metricas_modelo(nombre: str, m_train: dict, m_val: dict, m_test: dict):
    """Muestra en terminal las métricas del modelo evaluado en los 3 splits."""
    logger.info(
        f"[{nombre}] "
        f"Train -> MAE: {m_train['MAE']:.3f} | WAPE: {m_train['WAPE']:.3f} || "
        f"Val -> MAE: {m_val['MAE']:.3f} | WAPE: {m_val['WAPE']:.3f} || "
        f"Test -> MAE: {m_test['MAE']:.3f} | RMSE: {m_test['RMSE']:.3f} | WAPE: {m_test['WAPE']:.3f} | Sesgo: {m_test['Sesgo']:.3f}"
    )


@app.command()
def main():
    tracker = MLFlowTracker(
        tracking_uri=URI_MLFLOW_TRACKING,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )

    logger.info("Cargando y preparando dataset P1...")
    res = construir_dataset_p1()

    if isinstance(res, tuple):
        df_p1 = res[0].copy()
        df_stations = res[1].copy() if len(res) > 1 else None
    else:
        df_p1 = res.copy()
        df_stations = None

    # Identificar columna objetivo
    posibles_targets = [P1_TARGET, "target_next_hour", "demand_count", "demanda", "viajes"]
    target_col = next((col for col in posibles_targets if col in df_p1.columns), None)
    if target_col is None:
        raise KeyError(
            f"No se encontró la columna objetivo en el dataset. Columnas disponibles: {list(df_p1.columns)}"
        )
    logger.info(f"Columna objetivo detectada: '{target_col}'")

    col_tiempo = "target_hour" if "target_hour" in df_p1.columns else "hour_window"
    logger.info(f"Columna temporal usada para partición: '{col_tiempo}'")

    # Partición temporal idéntica al notebook (< P1_TEST_FIN)
    df_con_target = df_p1.dropna(subset=[target_col]).copy()

    train_mask = df_con_target[col_tiempo] < pd.Timestamp(P1_VAL_INICIO)
    val_mask = (
        (df_con_target[col_tiempo] >= pd.Timestamp(P1_VAL_INICIO))
        & (df_con_target[col_tiempo] < pd.Timestamp(P1_TEST_INICIO))
    )
    test_mask = (
        (df_con_target[col_tiempo] >= pd.Timestamp(P1_TEST_INICIO))
        & (df_con_target[col_tiempo] < pd.Timestamp(P1_TEST_FIN))
    )

    df_train = df_con_target.loc[train_mask].copy()
    df_val = df_con_target.loc[val_mask].copy()
    df_test = df_con_target.loc[test_mask].copy()

    if len(df_test) == 0:
        logger.warning("Filtro de fechas de test no arrojó registros. Usando el 20% final para Test.")
        corte = int(len(df_con_target) * 0.8)
        df_train = df_con_target.iloc[:corte].copy()
        df_test = df_con_target.iloc[corte:].copy()
        df_val = df_train.tail(int(len(df_train) * 0.1)).copy()

    if "target_hour" not in df_test.columns:
        df_test["target_hour"] = df_test[col_tiempo]

    splits = {
        "Train": df_train,
        "Validación": df_val,
        "Test": df_test,
    }

    y_train = df_train[target_col]
    y_val = df_val[target_col]
    y_test = df_test[target_col]

    logger.info(
        f"Registros listos -> Train: {len(df_train):,} | Validación: {len(df_val):,} | Test: {len(df_test):,}"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    params_base, tags_base = tracker.armar_params_tags(
        "P1",
        splits,
        extra_params={"variable_objetivo": target_col, "frecuencia": "horaria"},
        col_tiempo=col_tiempo,
    )

    filas_comparacion = []
    cols_identificadores = [c for c in ["origen_id", col_tiempo] if c in df_test.columns]
    df_predicciones = df_test[cols_identificadores].copy()
    df_predicciones["real"] = y_test.values

    # -------------------------------------------------------------
    # 1. EVALUACIÓN Y REGISTRO DE 4 MODELOS INGENUOS (BASELINES)
    # -------------------------------------------------------------
    logger.info("--- Evaluando Modelos Ingenuos (Baselines) ---")
    estrategias_ingenuas = [
        ("naive_persistencia", "persistencia"),
        ("naive_lag_24h", "lag_24h"),
        ("naive_lag_168h", "lag_168h"),
        ("naive_promedio_24h_168h", "promedio_24h_168h"),
    ]

    for nombre_run, estrategia in estrategias_ingenuas:
        modelo_ingenuo = BaselineIngenuoP1(estrategia=estrategia)
        pred_train = modelo_ingenuo.predict(df_train)
        pred_val = modelo_ingenuo.predict(df_val)
        pred_test = modelo_ingenuo.predict(df_test)

        df_predicciones[nombre_run] = np.round(pred_test, 2)

        m_train = evaluar_metricas_regresion(y_train, pred_train, "Train")
        m_val = evaluar_metricas_regresion(y_val, pred_val, "Validación")
        m_test = evaluar_metricas_regresion(y_test, pred_test, "Test")

        imprimir_metricas_modelo(nombre_run, m_train, m_val, m_test)

        tabla_metricas = pd.DataFrame([m_train, m_val, m_test])
        tabla_est = calcular_metricas_por_estacion(df_test, pred_test, target_col, df_stations)

        tracker.registrar_run(
            run_name=f"P1_{nombre_run}",
            params={
                **params_base,
                "modelo": nombre_run,
                "variables_predictoras": "ninguna (regla fija)",
            },
            tags={**tags_base, "version": nombre_run},
            tabla=tabla_metricas,
            tabla_est=tabla_est,
        )

        filas_comparacion.append({
            "modelo": nombre_run,
            "train_mae": m_train["MAE"],
            "train_wape": m_train["WAPE"],
            "val_wape": m_val["WAPE"],
            "test_mae": m_test["MAE"],
            "test_rmse": m_test["RMSE"],
            "test_wape": m_test["WAPE"],
            "test_sesgo": m_test["Sesgo"],
        })

    # -------------------------------------------------------------
    # 2. MODELO V1: REGRESIÓN LINEAL (SIN MEDIAS MÓVILES)
    # -------------------------------------------------------------
    logger.info("--- Entrenando Modelo v1: Regresión Lineal Base ---")
    modelo_v1 = DemandaLinealP1(features=P1_FEATURES_V1)
    modelo_v1.fit(df_train[P1_FEATURES_V1], y_train)

    ruta_modelo_v1 = MODELS_DIR / "p1_v1_model.pkl"
    modelo_v1.save(ruta_modelo_v1)
    logger.info(f"Modelo v1 guardado en {ruta_modelo_v1}")

    pred_v1_train = modelo_v1.predict(df_train[P1_FEATURES_V1])
    pred_v1_val = modelo_v1.predict(df_val[P1_FEATURES_V1])
    pred_v1_test = modelo_v1.predict(df_test[P1_FEATURES_V1])
    df_predicciones["v1_lineal"] = np.round(pred_v1_test, 2)

    m_v1_train = evaluar_metricas_regresion(y_train, pred_v1_train, "Train")
    m_v1_val = evaluar_metricas_regresion(y_val, pred_v1_val, "Validación")
    m_v1_test = evaluar_metricas_regresion(y_test, pred_v1_test, "Test")

    imprimir_metricas_modelo("v1_lineal", m_v1_train, m_v1_val, m_v1_test)

    tabla_est_v1 = calcular_metricas_por_estacion(df_test, pred_v1_test, target_col, df_stations)
    figuras_v1 = figuras_p1(df_test, pred_v1_test, target_col, "v1")

    tracker.registrar_run(
        run_name="P1_v1_linear",
        params={
            **params_base,
            "modelo": "LinearRegression",
            "variables_predictoras": ", ".join(P1_FEATURES_V1),
            "postprocesamiento": "np.clip(predicciones, 0, None)",
        },
        tags={**tags_base, "version": "v1"},
        tabla=pd.DataFrame([m_v1_train, m_v1_val, m_v1_test]),
        tabla_est=tabla_est_v1,
        modelo=modelo_v1.estimator_,
        X_ejemplo=df_train[P1_FEATURES_V1].head(5),
        figuras=figuras_v1,
    )

    filas_comparacion.append({
        "modelo": "v1_lineal",
        "train_mae": m_v1_train["MAE"],
        "train_wape": m_v1_train["WAPE"],
        "val_wape": m_v1_val["WAPE"],
        "test_mae": m_v1_test["MAE"],
        "test_rmse": m_v1_test["RMSE"],
        "test_wape": m_v1_test["WAPE"],
        "test_sesgo": m_v1_test["Sesgo"],
    })

    # -------------------------------------------------------------
    # 3. MODELO V2: REGRESIÓN LINEAL CON ROLLING
    # -------------------------------------------------------------
    logger.info("--- Entrenando Modelo v2: Regresión Lineal + Medias Móviles ---")
    modelo_v2 = DemandaLinealP1(features=P1_FEATURES_V2)
    modelo_v2.fit(df_train[P1_FEATURES_V2], y_train)

    ruta_modelo_v2 = MODELS_DIR / "p1_v2_model.pkl"
    modelo_v2.save(ruta_modelo_v2)
    modelo_v2.save(MODELS_DIR / "p1_linear_model.pkl")
    logger.info(f"Modelo v2 guardado en {ruta_modelo_v2}")

    pred_v2_train = modelo_v2.predict(df_train[P1_FEATURES_V2])
    pred_v2_val = modelo_v2.predict(df_val[P1_FEATURES_V2])
    pred_v2_test = modelo_v2.predict(df_test[P1_FEATURES_V2])
    df_predicciones["v2_lineal_rolling"] = np.round(pred_v2_test, 2)

    m_v2_train = evaluar_metricas_regresion(y_train, pred_v2_train, "Train")
    m_v2_val = evaluar_metricas_regresion(y_val, pred_v2_val, "Validación")
    m_v2_test = evaluar_metricas_regresion(y_test, pred_v2_test, "Test")

    imprimir_metricas_modelo("v2_lineal_rolling", m_v2_train, m_v2_val, m_v2_test)

    tabla_est_v2 = calcular_metricas_por_estacion(df_test, pred_v2_test, target_col, df_stations)
    figuras_v2 = figuras_p1(df_test, pred_v2_test, target_col, "v2")

    tracker.registrar_run(
        run_name="P1_v2_rolling_mean",
        params={
            **params_base,
            "modelo": "LinearRegression",
            "variables_predictoras": ", ".join(P1_FEATURES_V2),
            "postprocesamiento": "np.clip(predicciones, 0, None)",
        },
        tags={**tags_base, "version": "v2"},
        tabla=pd.DataFrame([m_v2_train, m_v2_val, m_v2_test]),
        tabla_est=tabla_est_v2,
        modelo=modelo_v2.estimator_,
        X_ejemplo=df_train[P1_FEATURES_V2].head(5),
        figuras=figuras_v2,
    )

    filas_comparacion.append({
        "modelo": "v2_lineal_rolling",
        "train_mae": m_v2_train["MAE"],
        "train_wape": m_v2_train["WAPE"],
        "val_wape": m_v2_val["WAPE"],
        "test_mae": m_v2_test["MAE"],
        "test_rmse": m_v2_test["RMSE"],
        "test_wape": m_v2_test["WAPE"],
        "test_sesgo": m_v2_test["Sesgo"],
    })

    # -------------------------------------------------------------
    # 4. GUARDAR COMPARATIVA Y PREDICCIONES (SOLO PARQUET)
    # -------------------------------------------------------------
    df_p1_comp = pd.DataFrame(filas_comparacion).set_index("modelo")
    p1_comp_path = REPORTS_DIR / "p1_comparacion.parquet"
    df_p1_comp.round(3).reset_index().to_parquet(p1_comp_path, index=False)

    logger.info(f"Reporte de entrenamiento P1 guardado en Parquet: {p1_comp_path}")
    logger.info("\n" + df_p1_comp.round(3).to_string())

    preds_out_path = REPORTS_DIR / "p1_predicciones_modelos.parquet"
    df_predicciones.to_parquet(preds_out_path, index=False)
    logger.success(f"Predicciones de los 6 modelos guardadas en {preds_out_path}")


if __name__ == "__main__":
    app()