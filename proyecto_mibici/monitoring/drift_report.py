#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monitoreo de drift de P1 con Evidently (Entregable 8).
Referencia vs octubre 2026. No modifica modelos ni el Registry.
"""

import mlflow
import numpy as np
import pandas as pd
import typer
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger
from scipy.stats import wasserstein_distance

from proyecto_mibici.config import (
    MLFLOW_EXPERIMENT_MONITOREO,
    P1_ACTUAL_FIN,
    P1_FEATURES_V2,
    P1_TARGET,
    P1_TEST_FIN,
    P1_VAL_INICIO,
    RANDOM_SEED,
    REPORTS_DIR,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.features import construir_dataset_p1

N_MUESTRA = 100_000
UMBRAL_NUM = 0.1   # Wasserstein normalizada (distancia / std de la referencia)
UMBRAL_CAT = 0.1   # Jensen-Shannon, para la variable binaria
COLUMNAS = list(P1_FEATURES_V2) + [P1_TARGET]
CAT = [c for c in COLUMNAS if c == "target_is_weekend"]
NUM = [c for c in COLUMNAS if c not in CAT]

app = typer.Typer(help="Reportes de drift de P1 con Evidently.")


def muestrear(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.sample(n=min(n, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)


def tabla_scores(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    """Tabla complementaria, calculada aparte de Evidently, para el PDF."""
    filas = []
    for c in NUM + CAT:
        fila = {"variable": c, "media_ref": ref[c].mean(), "media_actual": cur[c].mean(),
                "std_ref": ref[c].std(), "std_actual": cur[c].std()}
        if c in NUM:
            sd = ref[c].std()
            fila["wasserstein_norm"] = (
                wasserstein_distance(ref[c], cur[c]) / sd if sd > 0 else np.nan)
            fila["drift"] = bool(fila["wasserstein_norm"] > UMBRAL_NUM)
        filas.append(fila)
    return pd.DataFrame(filas)


def reporte_evidently(nombre, ref, cur, out_dir) -> None:
    definicion = DataDefinition(numerical_columns=NUM, categorical_columns=CAT)
    ds_ref = Dataset.from_pandas(ref[COLUMNAS], data_definition=definicion)
    ds_cur = Dataset.from_pandas(cur[COLUMNAS], data_definition=definicion)
    report = Report([
        DataDriftPreset(num_method="wasserstein", num_threshold=UMBRAL_NUM,
                        cat_method="jensenshannon", cat_threshold=UMBRAL_CAT),
        DataSummaryPreset(),
    ])
    snapshot = report.run(current_data=ds_cur, reference_data=ds_ref)
    snapshot.save_html(str(out_dir / f"drift_{nombre}.html"))


@app.command()
def main(n_muestra: int = typer.Option(N_MUESTRA, "--n-muestra", help="Filas por conjunto.")):
    df_p1, _ = construir_dataset_p1()
    df_p1 = df_p1.dropna(subset=COLUMNAS)
    t = df_p1["target_hour"]

    actual = df_p1[(t >= pd.Timestamp(P1_TEST_FIN)) & (t < pd.Timestamp(P1_ACTUAL_FIN))]
    comparaciones = {
        # Lo que vio v1 en entrenamiento contra octubre 2026
        "ref2025_vs_oct2026": df_p1[t < pd.Timestamp(P1_VAL_INICIO)],
        # Mismo mes, otro año: separa estacionalidad de cambio estructural
        "oct2025_vs_oct2026": df_p1[(t >= pd.Timestamp("2025-10-01"))
                                    & (t < pd.Timestamp("2025-11-01"))],
    }
    if actual.empty:
        raise ValueError("No hay filas de octubre 2026 en la tabla de modelado.")

    out_dir = REPORTS_DIR / "drift"
    out_dir.mkdir(parents=True, exist_ok=True)
    resumen = {}

    for nombre, ref in comparaciones.items():
        logger.info(f"=== {nombre}: ref {len(ref):,} filas | actual {len(actual):,} filas ===")
        est_ref, est_cur = set(ref["origen_id"]), set(actual["origen_id"])
        logger.info(f"Estaciones: ref={len(est_ref)} | actual={len(est_cur)} | "
                    f"solo en actual={len(est_cur - est_ref)}")

        ref_m, cur_m = muestrear(ref, n_muestra), muestrear(actual, n_muestra)
        scores = tabla_scores(ref_m, cur_m)
        scores.to_csv(out_dir / f"drift_{nombre}_scores.csv", index=False)
        logger.info("\n" + scores.round(4).to_string(index=False))
        resumen[nombre] = int(scores["drift"].fillna(False).sum())

        reporte_evidently(nombre, ref_m, cur_m, out_dir)
        logger.success(f"Reporte HTML -> {out_dir / f'drift_{nombre}.html'}")

    mlflow.set_tracking_uri(URI_MLFLOW_TRACKING)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_MONITOREO)
    with mlflow.start_run(run_name="P1_drift_oct2026"):
        mlflow.log_params({"n_muestra": n_muestra, "umbral_wasserstein_norm": UMBRAL_NUM,
                           "actual_inicio": str(actual["target_hour"].min()),
                           "actual_fin": str(actual["target_hour"].max())})
        mlflow.log_metrics({f"vars_con_drift_{k}": v for k, v in resumen.items()})
        mlflow.log_artifacts(str(out_dir), artifact_path="drift")


if __name__ == "__main__":
    app()