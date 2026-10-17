import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typer
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from proyecto_mibici.config import (
    MLFLOW_EXPERIMENT_NAME, P1_FEATURES_V1, P1_FEATURES_V2, P1_TARGET,
    P1_TEST_FIN, P1_TEST_INICIO, P1_VAL_INICIO, REPORTS_DIR, URI_MLFLOW_TRACKING,
)
from proyecto_mibici.features import construir_dataset_p1
from proyecto_mibici.modeling.predict import predecir
from proyecto_mibici.modeling.tracking import armar_params_tags, registrar_run
from proyecto_mibici.plots import figuras_p1

app = typer.Typer()

# ==========================================
# P1: DEMANDA HORARIA
# ==========================================

# Reglas fijas: no entrenan nada; usan los rezagos como predicción
REGLAS_INGENUAS = {
    "naive_persistencia":      lambda d: d["lag_1h"],
    "naive_lag_24h":           lambda d: d["lag_24h"],
    "naive_lag_168h":          lambda d: d["lag_168h"],
    "naive_promedio_24h_168h": lambda d: (d["lag_24h"] + d["lag_168h"]) / 2,
}


def temporal_train_val_test_split(df):
    """Divide según la hora que se quiere predecir (target_hour)."""
    t = df["target_hour"]
    df_train = df.loc[t < pd.Timestamp(P1_VAL_INICIO)].copy()
    df_val = df.loc[(t >= pd.Timestamp(P1_VAL_INICIO)) & (t < pd.Timestamp(P1_TEST_INICIO))].copy()
    df_test = df.loc[(t >= pd.Timestamp(P1_TEST_INICIO)) & (t < pd.Timestamp(P1_TEST_FIN))].copy()
    return df_train, df_val, df_test


def evaluar(y_real, y_pred):
    """MAE, RMSE, WAPE y sesgo (predicho - real)."""
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    suma_real = y_real.sum()
    # Si la demanda real suma 0, WAPE no está definido; se devuelve 0.0
    wape = np.abs(y_real - y_pred).sum() / suma_real if suma_real > 0 else 0.0
    return {
        "MAE": mean_absolute_error(y_real, y_pred),
        "RMSE": root_mean_squared_error(y_real, y_pred),
        "WAPE": wape,
        "Sesgo": (y_pred - y_real).mean(),
    }


def tabla_metricas(preds, target):
    """preds = {'Train': (df, pred), 'Validación': (df, pred), 'Test': (df, pred)}"""
    return pd.DataFrame(
        [{"conjunto": n, **evaluar(d[target], p)} for n, (d, p) in preds.items()]
    )


def metricas_por_estacion(df, pred, target, df_stations):
    """Métricas del conjunto dado, desglosadas por estación de origen."""
    res = df[["origen_id", target]].copy()
    res["pred"] = np.asarray(pred)
    filas = []
    for est, g in res.groupby("origen_id"):
        filas.append({
            "origen_id": est,
            "horas_evaluadas": len(g),
            "viajes_reales_total": g[target].sum(),
            "viajes_predichos_total": g["pred"].sum(),
            **evaluar(g[target], g["pred"]),
        })
    out = (pd.DataFrame(filas)
           .merge(df_stations[["id", "name"]], left_on="origen_id", right_on="id", how="left")
           .drop(columns="id").rename(columns={"name": "nombre_estacion"}))
    out["error_abs_total"] = out["MAE"] * out["horas_evaluadas"]
    return out


def entrenar_modelo(features, df_train, target=P1_TARGET):
    """Regresión lineal."""
    modelo = LinearRegression()
    modelo.fit(df_train[features], df_train[target])
    return modelo


@app.command()
def main():
    mlflow.set_tracking_uri(URI_MLFLOW_TRACKING)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info("Construyendo dataset de P1...")
    df_p1, df_stations = construir_dataset_p1()
    df_train, df_val, df_test = temporal_train_val_test_split(df_p1)
    splits = {"Train": df_train, "Validación": df_val, "Test": df_test}
    logger.info(f"Filas: train={len(df_train):,} val={len(df_val):,} test={len(df_test):,}")

    params_base, tags_base = armar_params_tags(
        "P1", splits,
        extra_params={"variable_objetivo": P1_TARGET,
                      "horizonte_prediccion": "1 hora", "frecuencia_datos": "horaria"},
    )
    resumen = []

    # Referencias ingenuas
    for nombre, regla in REGLAS_INGENUAS.items():
        preds = {n: (d, regla(d).to_numpy()) for n, d in splits.items()}
        tabla = tabla_metricas(preds, P1_TARGET)
        tabla_est = metricas_por_estacion(df_test, preds["Test"][1], P1_TARGET, df_stations)
        registrar_run(
            f"P1_{nombre}",
            {**params_base, "modelo": nombre, "variables_predictoras": "ninguna (regla fija)"},
            {**tags_base, "version": nombre}, tabla, tabla_est,
        )
        resumen.append({"modelo": nombre, **tabla.set_index("conjunto").loc["Test"].to_dict()})

    # Modelos lineales: v1 (rezagos + calendario) y v2 (+ medias móviles, baseline del proyecto)
    modelos = [
        ("P1_v1_linear", P1_FEATURES_V1, "v1_lineal", {}),
        ("P1_v2_rolling_mean", P1_FEATURES_V2, "v2_lineal_rolling", {"baseline": "true"}),
    ]
    for run_name, feats, version, tags_extra in modelos:
        modelo = entrenar_modelo(feats, df_train)
        preds = {n: (d, predecir(modelo, feats, d)) for n, d in splits.items()}
        tabla = tabla_metricas(preds, P1_TARGET)
        tabla_est = metricas_por_estacion(df_test, preds["Test"][1], P1_TARGET, df_stations)
        figuras = figuras_p1(df_test, preds["Test"][1], P1_TARGET, version)

        run_id = registrar_run(
            run_name,
            {**params_base, "modelo": "LinearRegression",
             "variables_predictoras": ", ".join(feats),
             "postprocesamiento": "np.clip(predicciones, 0, None)"},
            {**tags_base, "version": version, **tags_extra}, tabla, tabla_est,
            modelo=modelo, X_ejemplo=df_train[feats].head(5), figuras=figuras,
        )
        for fig in figuras.values():
            plt.close(fig)
        logger.info(f"{run_name} registrado: {run_id}")
        resumen.append({"modelo": version, **tabla.set_index("conjunto").loc["Test"].to_dict()})

    comparacion = pd.DataFrame(resumen).set_index("modelo")
    comparacion.to_csv(REPORTS_DIR / "p1_comparacion.csv")
    logger.info("\n" + comparacion.to_string())


if __name__ == "__main__":
    app()