#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo de inferencia para P2: Predicción del riesgo de desbalance en estaciones.
Carga el Pipeline serializado (PipelineRiesgoP2) que encapsula el StandardScaler 
y la LogisticRegression balanceada.
"""

from pathlib import Path
from typing import Optional
import typer
from loguru import logger
import pandas as pd
import numpy as np

from proyecto_mibici.config import MODELS_DIR
from proyecto_mibici.modeling.model_p2 import RiesgoLogisticoP2

app = typer.Typer(help="CLI para generar inferencia sobre el riesgo de desbalance (P2).")


def cargar_pipeline_p2(ruta_modelo: Optional[Path] = None) -> RiesgoLogisticoP2:
    """Carga la instancia completa del PipelineRiesgoP2 guardada en models/."""
    ruta = ruta_modelo or (MODELS_DIR / "p2_logistic_model.pkl")
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {ruta}. Ejecuta primero train_p2.py."
        )
    return RiesgoLogisticoP2.load(ruta)


def predecir_probabilidades(
    df: pd.DataFrame, ruta_modelo: Optional[Path] = None
) -> np.ndarray:
    """
    Calcula la probabilidad continua de que cada estación entre en estado de riesgo (t+1).
    """
    pipeline = cargar_pipeline_p2(ruta_modelo)
    return pipeline.predict_proba(df)


def predecir_riesgo(
    df: pd.DataFrame,
    umbral_decision: float = 0.5,
    ruta_modelo: Optional[Path] = None,
) -> np.ndarray:
    """
    Retorna un vector booleano indicando si la estación estará en riesgo en t+1,
    evaluado contra un umbral de corte calibrable.
    """
    pipeline = cargar_pipeline_p2(ruta_modelo)
    return pipeline.predict(df, umbral=umbral_decision)


def generar_reporte_inferencia(
    df: pd.DataFrame,
    umbral_decision: float = 0.5,
    ruta_modelo: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Retorna el DataFrame original anexando las columnas 'prob_riesgo' y 'pred_riesgo'.
    """
    df_out = df.copy()
    probs = predecir_probabilidades(df, ruta_modelo=ruta_modelo)
    df_out["prob_riesgo"] = probs
    df_out["pred_riesgo"] = probs >= umbral_decision
    return df_out


@app.command()
def main(
    input_path: Path = typer.Option(
        ..., "--input", "-i", help="Ruta al archivo Parquet o CSV con las features de P2."
    ),
    output_path: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Ruta para guardar el archivo con predicciones."
    ),
    umbral: float = typer.Option(
        0.5, "--umbral", "-u", help="Umbral de decisión para clasificar riesgo (0.0 a 1.0)."
    ),
):
    """Ejecuta inferencia de P2 desde terminal."""
    logger.info(f"Cargando datos para inferencia desde {input_path}...")
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    logger.info(f"Generando predicciones con umbral = {umbral}...")
    resultado = generar_reporte_inferencia(df, umbral_decision=umbral)

    total_riesgo = int(resultado["pred_riesgo"].sum())
    pct_riesgo = (total_riesgo / len(resultado)) * 100
    logger.info(f"Estaciones-hora clasificadas en riesgo: {total_riesgo:,} ({pct_riesgo:.2f}%)")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".parquet":
            resultado.to_parquet(output_path, index=False)
        else:
            resultado.to_csv(output_path, index=False)
        logger.info(f"Predicciones guardadas exitosamente en {output_path}")
    else:
        logger.info("\n" + resultado[["estacion_id", "hora", "prob_riesgo", "pred_riesgo"]].head(10).to_string())


if __name__ == "__main__":
    app()