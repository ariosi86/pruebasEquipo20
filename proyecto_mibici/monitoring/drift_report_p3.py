#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reporte de drift de P3 con Evidently (Entregable 8).

Compara el perfil de las estaciones del periodo actual (octubre 2026) contra:
  1. la referencia de entrenamiento del champion (2025 completo), y
  2. octubre 2025, para separar el efecto de la temporada.

Usa la clase EvidentlyMonitor con la misma metodología de P1 y P2:
Wasserstein normalizada con umbral 0.1 para las ocho variables (todas numéricas).
Los HTML y CSV quedan en reports/drift/p3/ y se registran en MLflow.
"""

from pathlib import Path

from loguru import logger
import mlflow
import pandas as pd
import typer

from proyecto_mibici.config import (
    MLFLOW_EXPERIMENT_MONITOREO_P3,
    P3_VARIABLES,
    REPORTS_DIR,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.monitoring.datos_p3 import PERIODOS_P3, cargar_conjunto
from proyecto_mibici.monitoring.evidently_monitor import EvidentlyMonitor

SALIDA = REPORTS_DIR / "drift" / "p3"
UMBRAL = 0.1

COMPARACIONES = {
    "ref2025_vs_oct2026": "referencia",
    "oct2025_vs_oct2026": "referencia_octubre",
}

app = typer.Typer(help="Reporte de drift de P3 con Evidently.")


def resumen_estaciones(referencia: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """Cuántas estaciones comparten ambos conjuntos y cuántas entran o salen."""
    ref, act = set(referencia.index), set(actual.index)
    return {
        "estaciones_referencia": len(ref),
        "estaciones_actual": len(act),
        "estaciones_comunes": len(ref & act),
        "estaciones_nuevas": len(act - ref),
        "estaciones_ausentes": len(ref - act),
    }


def tabla_drift(scores: pd.DataFrame) -> pd.DataFrame:
    """Agrega el cambio relativo de la media para leer la dirección del drift."""
    tabla = scores[["variable", "media_ref", "media_actual", "std_ref", "std_actual",
                    "wasserstein_norm", "drift"]].copy()
    tabla["cambio_media_pct"] = 100 * (tabla["media_actual"] - tabla["media_ref"]) / tabla["media_ref"].abs()
    return tabla.sort_values("wasserstein_norm", ascending=False).reset_index(drop=True)


@app.command()
def main(
    log_mlflow: bool = typer.Option(True, "--log-mlflow/--sin-mlflow", help="Registrar la corrida en MLflow."),
    usar_api_evidently: bool = typer.Option(
        False, "--api-evidently/--sin-api-evidently",
        help="Subir los reportes a Evidently UI (requiere `evidently ui` en el puerto 8000).",
    ),
    salida: Path = typer.Option(SALIDA, "--salida", help="Carpeta de salida de HTML y CSV."),
):
    actual = cargar_conjunto("actual")
    salida.mkdir(parents=True, exist_ok=True)

    monitor = EvidentlyMonitor(
        numerical_columns=P3_VARIABLES,
        categorical_columns=[],
        num_threshold=UMBRAL,
        out_dir=salida,
        usar_api_evidently=usar_api_evidently,
    )

    resultados = {}
    for nombre, conjunto_ref in COMPARACIONES.items():
        referencia = cargar_conjunto(conjunto_ref)
        resumen = monitor.comparar(
            referencia,
            actual,
            nombre=f"p3_{nombre}",
            log_mlflow=False,
            registrar_proyecto_nombre="mibici-equipo20-p3" if usar_api_evidently else None,
        )
        tabla = tabla_drift(pd.read_csv(resumen["scores_csv"]))
        tabla.to_csv(salida / f"p3_{nombre}_tabla.csv", index=False)
        estaciones = resumen_estaciones(referencia, actual)
        resultados[nombre] = {"resumen": resumen, "tabla": tabla, "estaciones": estaciones}

        logger.info(f"--- {nombre}: {conjunto_ref} {PERIODOS_P3[conjunto_ref]} vs actual {PERIODOS_P3['actual']}")
        logger.info(f"Estaciones: {estaciones}")
        logger.info("\n" + tabla.round(4).to_string(index=False))
        logger.info(f"Variables con drift (umbral {UMBRAL}): {resumen['vars_con_drift']} de {len(P3_VARIABLES)}")

    if not log_mlflow:
        return

    mlflow.set_tracking_uri(URI_MLFLOW_TRACKING)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_MONITOREO_P3)
    with mlflow.start_run(run_name="p3_drift_oct2026"):
        mlflow.set_tags({"question": "P3", "stage_of_work": "monitoring"})
        mlflow.log_params({
            "num_method": monitor.num_method,
            "num_threshold": UMBRAL,
            "variables": ", ".join(P3_VARIABLES),
            **{f"periodo_{k}": f"{v[0]} a {v[1]}" for k, v in PERIODOS_P3.items()},
        })
        for nombre, r in resultados.items():
            mlflow.log_metric(f"{nombre}_vars_con_drift", float(r["resumen"]["vars_con_drift"]))
            mlflow.log_metrics({f"{nombre}_{k}": float(v) for k, v in r["estaciones"].items()})
            for _, fila in r["tabla"].iterrows():
                mlflow.log_metric(f"{nombre}_w_{fila['variable']}", float(fila["wasserstein_norm"]))
        mlflow.log_artifacts(str(salida), artifact_path="drift_p3")
    logger.success("Corrida p3_drift_oct2026 registrada en MLflow.")


if __name__ == "__main__":
    app()
