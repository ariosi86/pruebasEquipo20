#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Umbral del A/B de P1 a partir de la variación mensual de v1.

Calcula el WAPE de la versión vigente (v1) mes a mes, de mayo a octubre de 2026
(meses que v1 no usó para entrenar), y toma la variación relativa entre meses
(desviación estándar / media) como umbral de mejora: una mejora menor que lo que
el error de v1 varía normalmente de un mes a otro no se distingue de ese vaivén.

Como el A/B se evalúa sobre el mismo conjunto, el cambio relativo del WAPE es
igual al del MAE, así que el umbral se usa directo en ab_test_p1.py:

    python -m proyecto_mibici.monitoring.ab_test_p1 --min-mejora-mae <UMBRAL>

y se agrega la deviance como criterio (en ab_test_p1.py, dentro del dict
`criterios`):

        "Deviance de Tweedie no empeora": m_b["Deviance_Tweedie"] <= m_a["Deviance_Tweedie"],

Uso:
    python -m proyecto_mibici.monitoring.umbral_ab_p1
"""

import math
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import typer
from loguru import logger
from mlflow import MlflowClient

from proyecto_mibici.config import (
    P1_ACTUAL_FIN,
    P1_FEATURES_V2,
    P1_TARGET,
    P1_TEST_INICIO,
    REPORTS_DIR,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.features import construir_dataset_p1
from proyecto_mibici.monitoring.ab_test_p1 import predecir, resolver_version

app = typer.Typer(help="Umbral del A/B a partir de la variación mensual del WAPE de v1.")


@app.command()
def main(
    champion_version: Optional[int] = typer.Option(
        None, "--champion-version", help="Por defecto, la versión del alias 'champion'."),
):
    mlflow.set_tracking_uri(URI_MLFLOW_TRACKING)
    client = MlflowClient(tracking_uri=URI_MLFLOW_TRACKING)
    va = resolver_version(client, champion_version, "champion")
    logger.info(f"Versión vigente: v{va.version}")

    df_p1, _ = construir_dataset_p1()
    t = df_p1["target_hour"]
    df = df_p1[(t >= pd.Timestamp(P1_TEST_INICIO)) & (t < pd.Timestamp(P1_ACTUAL_FIN))]
    df = df.dropna(subset=P1_FEATURES_V2 + [P1_TARGET]).copy()
    df["pred"] = predecir(va.version, df[P1_FEATURES_V2])
    df["mes"] = df["target_hour"].dt.to_period("M")

    filas = []
    for mes, g in df.groupby("mes"):
        y = g[P1_TARGET].to_numpy(dtype=float)
        p = g["pred"].to_numpy(dtype=float)
        filas.append({
            "mes": str(mes),
            "filas": len(g),
            "demanda_media": float(y.mean()),
            "MAE": float(np.abs(y - p).mean()),
            "WAPE": float(np.abs(y - p).sum() / y.sum()),
        })
    tabla = pd.DataFrame(filas)

    media = float(tabla["WAPE"].mean())
    desv = float(tabla["WAPE"].std(ddof=1))
    variacion_pct = desv / media * 100
    umbral = math.ceil(variacion_pct * 10) / 10  # redondeo hacia arriba a 0.1%

    logger.info("\n" + tabla.round(4).to_string(index=False))
    logger.info(f"WAPE medio mensual: {media:.4f} | desviación estándar: {desv:.4f}")
    logger.success(
        f"Variación relativa mes a mes: {variacion_pct:.2f}% -> UMBRAL propuesto: {umbral}% "
        f"(usar --min-mejora-mae {umbral})"
    )

    out_dir = REPORTS_DIR / "ab_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(out_dir / "umbral_wape_mensual.csv", index=False)
    logger.info(f"Tabla guardada en {out_dir / 'umbral_wape_mensual.csv'}")


if __name__ == "__main__":
    app()