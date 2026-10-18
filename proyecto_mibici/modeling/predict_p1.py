#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo de inferencia para P1: Predicción de Demanda Horaria de Viajes.
Carga el Pipeline serializado (DemandaLinealP1) que encapsula el StandardScaler 
y la LinearRegression con cota no-negativa.
"""

from pathlib import Path
from typing import Optional
import typer
from loguru import logger
import pandas as pd
import numpy as np

from proyecto_mibici.config import MODELS_DIR
from proyecto_mibici.modeling.model_p1 import DemandaLinealP1

app = typer.Typer(help="CLI para generar inferencia sobre demanda horaria (P1).")


def cargar_pipeline_p1(ruta_modelo: Optional[Path] = None) -> DemandaLinealP1:
    """Carga la instancia serializada de DemandaLinealP1 desde models/."""
    ruta = ruta_modelo or (MODELS_DIR / "p1_linear_model.pkl")
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {ruta}. Ejecuta primero train_p1.py."
        )
    return DemandaLinealP1.load(ruta)


def predecir(df: pd.DataFrame, ruta_modelo: Optional[Path] = None) -> np.ndarray:
    """Retorna las predicciones de demanda horaria con cota inferior en 0."""
    pipeline = cargar_pipeline_p1(ruta_modelo)
    return pipeline.predict(df)


def generar_reporte_inferencia(
    df: pd.DataFrame, ruta_modelo: Optional[Path] = None
) -> pd.DataFrame:
    """Retorna una copia del DataFrame anexando la columna 'demanda_predicha'."""
    df_out = df.copy()
    df_out["demanda_predicha"] = predecir(df, ruta_modelo=ruta_modelo)
    return df_out


@app.command()
def main(
    input_path: Path = typer.Option(
        ..., "--input", "-i", help="Ruta al archivo Parquet o CSV con las features de P1."
    ),
    output_path: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Ruta para guardar el archivo con predicciones."
    ),
):
    """Ejecuta inferencia de demanda horaria (P1) desde la terminal."""
    logger.info(f"Cargando datos para inferencia desde {input_path}...")
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    logger.info("Generando predicciones de demanda...")
    resultado = generar_reporte_inferencia(df)

    logger.info(f"Total de registros procesados: {len(resultado):,}")
    logger.info(f"Demanda promedio estimada: {resultado['demanda_predicha'].mean():.2f} viajes/hora")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".parquet":
            resultado.to_parquet(output_path, index=False)
        else:
            resultado.to_csv(output_path, index=False)
        logger.info(f"Predicciones guardadas exitosamente en {output_path}")
    else:
        cols_mostrar = [c for c in ["origen_id", "target_hour", "demanda_predicha"] if c in resultado.columns]
        logger.info("\n" + resultado[cols_mostrar].head(10).to_string())


if __name__ == "__main__":
    app()