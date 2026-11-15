#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monitoreo de drift de P2 con Evidently (Entregable 8).

Referencia: entrenamiento del champion (2025).
Actual: julio a septiembre de 2026 (ver datos_p2.py).
Misma metodología que drift_report.py (P1): muestra de filas,
Wasserstein normalizada para numéricas y Jensen-Shannon para
categóricas. No modifica modelos ni el Registry.
"""

import mlflow
import numpy as np
import pandas as pd
import typer
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from proyecto_mibici.config import (
    P2_OBJETIVO,
    P2_VARIABLES,
    RANDOM_SEED,
    REPORTS_DIR,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.monitoring.datos_p2 import SALIDA as DATOS_MONITOREO

EXPERIMENTO = "mibici-equipo20-p2-monitoreo"
N_MUESTRA = 100_000
UMBRAL_NUM = 0.1  # Wasserstein normalizada (distancia / std de la referencia)
UMBRAL_CAT = 0.1  # Jensen-Shannon
COLUMNAS = list(P2_VARIABLES) + [P2_OBJETIVO]
CAT = [
    "hora_del_dia",
    "dia_semana",
    "fin_de_semana",
    "riesgo_vacia",
    "riesgo_llena",
    P2_OBJETIVO,
]
NUM = [c for c in COLUMNAS if c not in CAT]
OUT_DIR = REPORTS_DIR / "drift" / "p2"

app = typer.Typer(help="Reportes de drift de P2 con Evidently.")


def cargar():
    ref = pd.read_parquet(DATOS_MONITOREO / "p2_referencia.parquet")
    act = pd.read_parquet(DATOS_MONITOREO / "p2_actual.parquet")
    for df in (ref, act):
        df[CAT] = df[CAT].astype(int)
    return ref, act


def muestrear(df: pd.DataFrame, n: int) -> pd.DataFrame:
    n = min(n, len(df))
    return df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)


def js_categorica(ref: pd.Series, cur: pd.Series) -> float:
    p = ref.value_counts(normalize=True)
    q = cur.value_counts(normalize=True)
    p, q = p.align(q, fill_value=0)
    return float(jensenshannon(p, q))


def tabla_scores(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    """Tabla complementaria, calculada aparte de Evidently, para el reporte."""
    filas = []
    for c in NUM + CAT:
        fila = {
            "variable": c,
            "tipo": "numerica" if c in NUM else "categorica",
            "media_ref": ref[c].mean(),
            "media_actual": cur[c].mean(),
        }
        if c in NUM:
            sd = ref[c].std()
            dist = wasserstein_distance(ref[c], cur[c])
            fila["metodo"] = "wasserstein_norm"
            fila["score"] = dist / sd if sd > 0 else np.nan
            fila["drift"] = bool(fila["score"] > UMBRAL_NUM)
        else:
            fila["metodo"] = "jensenshannon"
            fila["score"] = js_categorica(ref[c], cur[c])
            fila["drift"] = bool(fila["score"] > UMBRAL_CAT)
        filas.append(fila)
    return pd.DataFrame(filas)


def reporte_evidently(nombre, ref, cur) -> None:
    definicion = DataDefinition(numerical_columns=NUM, categorical_columns=CAT)
    ds_ref = Dataset.from_pandas(ref[COLUMNAS], data_definition=definicion)
    ds_cur = Dataset.from_pandas(cur[COLUMNAS], data_definition=definicion)
    report = Report([
        DataDriftPreset(
            num_method="wasserstein",
            num_threshold=UMBRAL_NUM,
            cat_method="jensenshannon",
            cat_threshold=UMBRAL_CAT,
        ),
        DataSummaryPreset(),
    ])
    snapshot = report.run(current_data=ds_cur, reference_data=ds_ref)
    snapshot.save_html(str(OUT_DIR / f"drift_{nombre}.html"))


@app.command()
def main(
    n_muestra: int = typer.Option(N_MUESTRA, "--n-muestra", help="Filas por conjunto."),
    registrar: bool = typer.Option(True, "--mlflow/--sin-mlflow", help="Registrar en MLflow."),
):
    ref, actual = cargar()
    meses = ref["hora_objetivo"].dt.month
    comparaciones = {
        # Lo que vio el champion en entrenamiento contra julio a septiembre de 2026
        "ref2025_vs_jul_sep2026": ref,
        # Mismos meses, otro año: separa estacionalidad de cambio estructural
        "jul_sep2025_vs_jul_sep2026": ref[meses.between(7, 9)],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resumen = {}
    for nombre, r in comparaciones.items():
        logger.info(f"=== {nombre}: ref {len(r):,} | actual {len(actual):,} ===")
        ref_m = muestrear(r, n_muestra)
        cur_m = muestrear(actual, n_muestra)

        scores = tabla_scores(ref_m, cur_m)
        scores.to_csv(OUT_DIR / f"drift_{nombre}_scores.csv", index=False)
        logger.info("\n" + scores.round(4).to_string(index=False))
        resumen[nombre] = int(scores["drift"].sum())

        reporte_evidently(nombre, ref_m, cur_m)
        logger.success(f"Reporte HTML -> {OUT_DIR / f'drift_{nombre}.html'}")

    if registrar:
        mlflow.set_tracking_uri(URI_MLFLOW_TRACKING)
        mlflow.set_experiment(EXPERIMENTO)
        with mlflow.start_run(run_name="P2_drift_jul_sep2026"):
            mlflow.set_tags({"question": "p2", "stage_of_work": "monitoring"})
            mlflow.log_params({
                "n_muestra": n_muestra,
                "umbral_wasserstein_norm": UMBRAL_NUM,
                "umbral_jensenshannon": UMBRAL_CAT,
                "actual_inicio": str(actual["hora_objetivo"].min()),
                "actual_fin": str(actual["hora_objetivo"].max()),
            })
            metricas = {f"vars_con_drift_{k}": v for k, v in resumen.items()}
            mlflow.log_metrics(metricas)
            mlflow.log_artifacts(str(OUT_DIR), artifact_path="drift_p2")


if __name__ == "__main__":
    app()