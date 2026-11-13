#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A/B testing offline de P1 (Entregable 8): champion vs challenger sobre octubre 2026.
Ninguna de las dos versiones vio octubre. No modifica el Registry salvo con --promover.
"""

from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import typer
from loguru import logger
from mlflow import MlflowClient
from sklearn.metrics import mean_tweedie_deviance

from proyecto_mibici.config import (
    P1_ACTUAL_FIN,
    P1_FEATURES_V2,
    P1_TARGET,
    P1_TEST_FIN,
    RANDOM_SEED,
    REPORTS_DIR,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.features import construir_dataset_p1
from proyecto_mibici.modeling.train_p1 import (
    calcular_metricas_por_estacion,
    evaluar_metricas_regresion,
)

MODEL_NAME = "mibici-equipo20-p1-demanda"
TWEEDIE_POWER = 1.8          # el del objetivo del champion
MIN_MEJORA_MAE_PCT = 1.0     # DEFINIR CON EL EQUIPO ANTES DE VER RESULTADOS
N_BOOTSTRAP = 5000
METRICAS = ["MAE", "RMSE", "WAPE", "Sesgo", "Deviance_Tweedie"]

app = typer.Typer(help="A/B testing offline champion vs challenger (P1, octubre 2026).")


def resolver_version(client: MlflowClient, version: Optional[int], alias: str):
    if version is None:
        return client.get_model_version_by_alias(MODEL_NAME, alias)
    return client.get_model_version(MODEL_NAME, str(version))


def predecir(version: str, X: pd.DataFrame) -> np.ndarray:
    modelo = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")
    return np.clip(np.asarray(modelo.predict(X), dtype=float).ravel(), 0, None)


def calcular_metricas(y: np.ndarray, pred: np.ndarray, nombre: str) -> dict:
    m = evaluar_metricas_regresion(y, pred, nombre)
    m["Deviance_Tweedie"] = float(
        mean_tweedie_deviance(y, np.clip(pred, 1e-6, None), power=TWEEDIE_POWER)
    )
    return m


def comparar_por_dia(df: pd.DataFrame, y, pred_a, pred_b) -> pd.DataFrame:
    d = pd.DataFrame({
        "dia": df["target_hour"].dt.normalize().to_numpy(),
        "mae_champion": np.abs(y - pred_a),
        "mae_challenger": np.abs(y - pred_b),
    })
    diario = d.groupby("dia").mean()
    diario["mejora"] = diario["mae_champion"] - diario["mae_challenger"]  # >0: gana challenger
    return diario


def bootstrap_ic95(valores: np.ndarray) -> tuple:
    rng = np.random.default_rng(RANDOM_SEED)
    medias = rng.choice(valores, size=(N_BOOTSTRAP, len(valores)), replace=True).mean(axis=1)
    return tuple(np.percentile(medias, [2.5, 97.5]))


@app.command()
def main(
    champion_version: Optional[int] = typer.Option(
        None, "--champion-version", help="Por defecto, la versión del alias 'champion'."),
    challenger_version: Optional[int] = typer.Option(
        None, "--challenger-version", help="Por defecto, la versión del alias 'challenger'."),
    experimento: str = typer.Option(
        "mibici-equipo20-p1-monitoreo", "--experimento", help="Experimento de MLflow del A/B."),
    min_mejora_mae: float = typer.Option(
        MIN_MEJORA_MAE_PCT, "--min-mejora-mae", help="Mejora mínima de MAE (%) para promover."),
    promover: bool = typer.Option(
        False, "--promover", help="Mueve el alias 'champion' al challenger si cumple los criterios."),
):
    mlflow.set_tracking_uri(URI_MLFLOW_TRACKING)
    client = MlflowClient(tracking_uri=URI_MLFLOW_TRACKING)

    va = resolver_version(client, champion_version, "champion")
    vb = resolver_version(client, challenger_version, "challenger")
    if va.version == vb.version:
        raise ValueError("Champion y challenger son la misma versión.")
    etiqueta_a, etiqueta_b = f"v{va.version} (champion)", f"v{vb.version} (challenger)"
    logger.info(f"A/B: {etiqueta_a} vs {etiqueta_b}")

    # 1. Conjunto de evaluación: octubre 2026, el mismo para ambas versiones
    df_p1, df_stations = construir_dataset_p1()
    t = df_p1["target_hour"]
    df = df_p1[(t >= pd.Timestamp(P1_TEST_FIN)) & (t < pd.Timestamp(P1_ACTUAL_FIN))]
    df = df.dropna(subset=P1_FEATURES_V2 + [P1_TARGET]).copy()
    if df.empty:
        raise ValueError("No hay filas de octubre en la tabla de modelado.")
    n_dias = df["target_hour"].dt.normalize().nunique()
    logger.info(f"Evaluación: {len(df):,} filas, {n_dias} días "
                f"({df['target_hour'].min()} -> {df['target_hour'].max()})")

    X, y = df[P1_FEATURES_V2], df[P1_TARGET].to_numpy(dtype=float)
    pred_a, pred_b = predecir(va.version, X), predecir(vb.version, X)

    # 2. Métricas globales
    m_a = calcular_metricas(y, pred_a, etiqueta_a)
    m_b = calcular_metricas(y, pred_b, etiqueta_b)
    tabla = pd.DataFrame([m_a, m_b]).set_index("conjunto")[METRICAS]
    cambio = (tabla.iloc[1] - tabla.iloc[0]) / tabla.iloc[0].abs() * 100
    cambio["Sesgo"] = np.nan  # un cambio relativo del sesgo no es interpretable
    tabla.loc["cambio_relativo_%"] = cambio

    # 3. Día a día, con IC95% bootstrap (remuestreando días)
    diario = comparar_por_dia(df, y, pred_a, pred_b)
    ic_inf, ic_sup = bootstrap_ic95(diario["mejora"].to_numpy())
    dias_gana_b = int((diario["mejora"] > 0).sum())

    # 4. Por estación
    est_a = calcular_metricas_por_estacion(df, pred_a, P1_TARGET, df_stations)
    est_b = calcular_metricas_por_estacion(df, pred_b, P1_TARGET, df_stations)
    cols = [c for c in ["origen_id", "nombre_estacion", "horas_evaluadas"] if c in est_a.columns]
    est = est_a[cols + ["MAE"]].rename(columns={"MAE": "MAE_champion"}).merge(
        est_b[["origen_id", "MAE"]].rename(columns={"MAE": "MAE_challenger"}), on="origen_id")
    est["mejora_mae"] = est["MAE_champion"] - est["MAE_challenger"]
    pct_est_gana_b = float((est["mejora_mae"] > 0).mean() * 100)

    # 5. Veredicto con criterios definidos de antemano
    mejora_mae = (m_a["MAE"] - m_b["MAE"]) / m_a["MAE"] * 100
    criterios = {
        f"MAE mejora >= {min_mejora_mae}%": mejora_mae >= min_mejora_mae,
        "RMSE no empeora": m_b["RMSE"] <= m_a["RMSE"],
        "IC95% de la mejora diaria > 0": ic_inf > 0,
        "Deviance de Tweedie no empeora": m_b["Deviance_Tweedie"] <= m_a["Deviance_Tweedie"]
    }
    cumple = all(criterios.values())

    logger.info("\n" + tabla.round(4).to_string())
    logger.info(f"Días en que gana el challenger: {dias_gana_b}/{len(diario)} | "
                f"IC95% mejora diaria de MAE: [{ic_inf:.4f}, {ic_sup:.4f}]")
    logger.info(f"Estaciones donde gana el challenger: {pct_est_gana_b:.1f}%")
    for nombre, ok in criterios.items():
        logger.info(f"  [{'OK' if ok else 'NO'}] {nombre}")
    logger.success("VEREDICTO: " + (
        f"reemplazar con {etiqueta_b}" if cumple else f"mantener {etiqueta_a}"))

    # 6. Salidas locales
    out_dir = REPORTS_DIR / "ab_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    tabla.round(5).to_csv(out_dir / "ab_metricas_globales.csv")
    diario.to_csv(out_dir / "ab_metricas_diarias.csv")
    est.sort_values("mejora_mae").to_csv(out_dir / "ab_metricas_por_estacion.csv", index=False)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(diario.index, diario["mae_champion"], label=etiqueta_a)
    ax[0].plot(diario.index, diario["mae_challenger"], label=etiqueta_b)
    ax[0].set_ylabel("MAE diario")
    ax[0].set_title("A/B testing P1: octubre 2026")
    ax[0].legend()
    ax[1].bar(diario.index, diario["mejora"],
              color=np.where(diario["mejora"] > 0, "tab:green", "tab:red"))
    ax[1].axhline(0, color="black", linewidth=0.8)
    ax[1].set_ylabel("MAE champion - MAE challenger")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "ab_mae_diario.png", dpi=150, bbox_inches="tight")

    # 7. Registrar el A/B en MLflow
    mlflow.set_experiment(experimento)
    with mlflow.start_run(run_name="P1_ab_test_oct2026"):
        mlflow.log_params({
            "champion_version": va.version, "challenger_version": vb.version,
            "champion_run_id": va.run_id, "challenger_run_id": vb.run_id,
            "eval_inicio": str(df["target_hour"].min()), "eval_fin": str(df["target_hour"].max()),
            "n_filas_eval": len(df), "n_dias_eval": n_dias,
            "min_mejora_mae_pct": min_mejora_mae, "tweedie_power": TWEEDIE_POWER,
        })
        for k in METRICAS:
            mlflow.log_metric(f"champion_{k}", float(tabla.loc[etiqueta_a, k]))
            mlflow.log_metric(f"challenger_{k}", float(tabla.loc[etiqueta_b, k]))
        mlflow.log_metrics({
            "mejora_mae_pct": float(mejora_mae),
            "dias_challenger_mejor": dias_gana_b,
            "ic95_inf_mejora_diaria": float(ic_inf), "ic95_sup_mejora_diaria": float(ic_sup),
            "estaciones_challenger_mejor_pct": pct_est_gana_b,
        })
        mlflow.set_tags({
            "stage_of_work": "ab_test",
            "veredicto": "reemplazar_champion" if cumple else "mantener_champion",
        })
        mlflow.log_artifacts(str(out_dir), artifact_path="ab_test")
    plt.close(fig)

    # 8. Promoción: solo si se pide y se cumplen los criterios
    if promover:
        if not cumple:
            logger.warning("--promover ignorado: el challenger no cumple todos los criterios.")
        else:
            client.set_registered_model_alias(MODEL_NAME, "previous_champion", va.version)
            client.set_registered_model_alias(MODEL_NAME, "champion", vb.version)
            logger.success(f"Alias 'champion' -> v{vb.version} (anterior: v{va.version}).")


if __name__ == "__main__":
    app()