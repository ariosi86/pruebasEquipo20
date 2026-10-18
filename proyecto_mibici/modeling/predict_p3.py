#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo de inferencia para P3: Asignación de grupos funcionales (Clustering).
Carga el Pipeline serializado (ClusterEstacionesP3) que encapsula el StandardScaler 
y el estimador KMeans.
"""

from pathlib import Path
from typing import Optional
import typer
from loguru import logger
import pandas as pd
import numpy as np

from proyecto_mibici.config import MODELS_DIR
from proyecto_mibici.modeling.model_p3 import ClusterEstacionesP3

app = typer.Typer(help="CLI para inferencia de clusters funcionales en estaciones (P3).")


def cargar_pipeline_p3(ruta_modelo: Optional[Path] = None) -> ClusterEstacionesP3:
    """Carga la instancia serializada de ClusterEstacionesP3 desde models/."""
    ruta = ruta_modelo or (MODELS_DIR / "p3_kmeans_model.pkl")
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {ruta}. Ejecuta primero train_p3.py."
        )
    return ClusterEstacionesP3.load(ruta)


def predecir_clusters(df: pd.DataFrame, ruta_modelo: Optional[Path] = None) -> np.ndarray:
    """
    Retorna el vector con las etiquetas de grupo asignadas a cada estación
    después de transformar las variables con el StandardScaler interno.
    """
    pipeline = cargar_pipeline_p3(ruta_modelo)
    return pipeline.predict(df)


def generar_reporte_inferencia(
    df: pd.DataFrame, ruta_modelo: Optional[Path] = None
) -> pd.DataFrame:
    """Retorna una copia del DataFrame de perfiles anexando la columna 'grupo'."""
    df_out = df.copy()
    df_out["grupo"] = predecir_clusters(df, ruta_modelo=ruta_modelo)
    return df_out


@app.command()
def main(
    input_path: Path = typer.Option(
        ..., "--input", "-i", help="Ruta al archivo Parquet o CSV con los perfiles de P3."
    ),
    output_path: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Ruta para guardar el archivo con la asignación de grupos."
    ),
):
    """Ejecuta inferencia de agrupamiento funcional (P3) desde terminal."""
    logger.info(f"Cargando perfiles para clustering desde {input_path}...")
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    logger.info("Asignando clusters a estaciones...")
    resultado = generar_reporte_inferencia(df)

    distribucion = resultado["grupo"].value_counts().sort_index()
    logger.info("Distribución de estaciones por grupo:\n" + distribucion.to_string())

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".parquet":
            resultado.to_parquet(output_path, index=True)
        else:
            resultado.to_csv(output_path, index=True)
        logger.info(f"Asignación de clusters guardada exitosamente en {output_path}")
    else:
        logger.info("\n" + resultado[["grupo"]].head(10).to_string())


if __name__ == "__main__":
    app()