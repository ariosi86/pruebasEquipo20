#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de entrenamiento, evaluación y registro en MLflow para P3 (Clustering de Estaciones).
Compara la agrupación funcional KMeans contra la línea base geográfica municipal.
Ejecución: python -m proyecto_mibici.modeling.train_p3
"""

import matplotlib.pyplot as plt
import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import davies_bouldin_score, silhouette_score

from proyecto_mibici.config import (
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR,
    P3_DURACION_MAX_MIN,
    P3_K_ELEGIDO,
    P3_MIN_VIAJES_ESTACION,
    P3_TEST_FIN,
    P3_TEST_INICIO,
    P3_TRAIN_FIN,
    P3_TRAIN_INICIO,
    P3_VARIABLES,
    RANDOM_SEED,
    REPORTS_DIR,
    STATIONS_PROCESSED_PATH,
    TRIPS_PROCESSED_PATH,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.features import construir_perfil_estaciones, limpiar_viajes_p3
from proyecto_mibici.modeling.model_p3 import ClusterEstacionesP3
from proyecto_mibici.modeling.tracking import MLFlowTracker
from proyecto_mibici.plots import graficar_mapa_grupos

app = typer.Typer()


def evaluar_clusters(X, etiquetas):
    """Calcula silhouette (mayor es mejor) y Davies-Bouldin (menor es mejor)."""
    return {
        "silhouette": silhouette_score(X, etiquetas),
        "davies_bouldin": davies_bouldin_score(X, etiquetas),
    }


def construir_tabla_metricas(metrics_train, metrics_test):
    """Formatea las métricas con la columna 'conjunto' compatible con tracking."""
    return pd.DataFrame([
        {"conjunto": "Train", **metrics_train},
        {"conjunto": "Test", **metrics_test},
    ])


@app.command()
def main():
    # Inicializar la clase orientada a objetos de tracking
    tracker = MLFlowTracker(
        tracking_uri=URI_MLFLOW_TRACKING,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )

    logger.info("Cargando y preparando datos de viajes y estaciones para P3...")
    cols = ["viaje_id", "inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"]
    df_trips = pd.read_parquet(TRIPS_PROCESSED_PATH, columns=cols)
    df_stations = pd.read_parquet(STATIONS_PROCESSED_PATH)

    logger.info("Limpiando viajes y filtrando periodos temporales...")
    df_clean = limpiar_viajes_p3(df_trips, df_stations, duracion_max_min=P3_DURACION_MAX_MIN)
    df_train = df_clean[(df_clean["inicio_del_viaje"] >= P3_TRAIN_INICIO) & (df_clean["inicio_del_viaje"] < P3_TRAIN_FIN)]
    df_test = df_clean[(df_clean["inicio_del_viaje"] >= P3_TEST_INICIO) & (df_clean["inicio_del_viaje"] < P3_TEST_FIN)]

    logger.info("Construyendo perfiles funcionales de estaciones...")
    perfil_train = construir_perfil_estaciones(df_train, min_viajes=P3_MIN_VIAJES_ESTACION)
    perfil_test = construir_perfil_estaciones(df_test, min_viajes=P3_MIN_VIAJES_ESTACION)

    # 1. Instanciar y ajustar Pipeline POO (StandardScaler + KMeans)
    logger.info(f"Entrenando Pipeline ClusterEstacionesP3 (K={P3_K_ELEGIDO})...")
    modelo = ClusterEstacionesP3(n_clusters=P3_K_ELEGIDO, random_state=RANDOM_SEED)
    modelo.fit(perfil_train[P3_VARIABLES])

    # Guardar objeto completo en models/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    modelo.save(MODELS_DIR / "p3_kmeans_model.pkl")
    logger.info("Pipeline P3 guardado exitosamente en models/p3_kmeans_model.pkl")

    # Inferencia y espacio transformado para evaluación
    X_train_scaled = modelo.transform(perfil_train[P3_VARIABLES])
    X_test_scaled = modelo.transform(perfil_test[P3_VARIABLES])
    etiquetas_train = modelo.labels_
    etiquetas_test = modelo.predict(perfil_test[P3_VARIABLES])

    tabla_metricas_kmeans = construir_tabla_metricas(
        evaluar_clusters(X_train_scaled, etiquetas_train),
        evaluar_clusters(X_test_scaled, etiquetas_test),
    )

    fig_mapa_kmeans = graficar_mapa_grupos(
        perfil_train, etiquetas_train, df_stations, f"Grupos de estaciones MiBici (K={P3_K_ELEGIDO})"
    )

    # 2. Evaluación de la Línea Base por Municipio
    logger.info("Evaluando Línea Base Municipal...")
    municipio_map = df_stations.set_index("id")["municipality_code"]
    etiquetas_mun_train = municipio_map.loc[perfil_train.index]
    etiquetas_mun_test = municipio_map.loc[perfil_test.index]

    tabla_metricas_base = construir_tabla_metricas(
        evaluar_clusters(X_train_scaled, etiquetas_mun_train),
        evaluar_clusters(X_test_scaled, etiquetas_mun_test),
    )

    fig_mapa_base = graficar_mapa_grupos(
        perfil_train,
        etiquetas_mun_train.astype("category").cat.codes,
        df_stations,
        "Agrupación por Municipio (Línea Base)",
    )

    # 3. Registro en MLflow Tracking usando la clase MLFlowTracker
    splits_ficticios = {"Train": df_train, "Validación": df_test, "Test": df_test}
    params_base, tags_base = tracker.armar_params_tags(
        "P3",
        splits_ficticios,
        extra_params={
            "semilla": RANDOM_SEED,
            "variables": ", ".join(P3_VARIABLES),
            "min_viajes_estacion": P3_MIN_VIAJES_ESTACION,
            "duracion_max_min": P3_DURACION_MAX_MIN,
        },
        col_tiempo="inicio_del_viaje",
    )
    tabla_est_vacia = pd.DataFrame(columns=["origen_id", "nombre_estacion", "MAE"])

    # Run Baseline
    logger.info("Registrando Run de Línea Base en MLflow...")
    tracker.registrar_run(
        run_name="p3-baseline-municipio",
        params={**params_base, "modelo": "AgrupacionMunicipio"},
        tags={**tags_base, "version": "baseline_municipio", "baseline": "true"},
        tabla=tabla_metricas_base,
        tabla_est=tabla_est_vacia,
        figuras={"mapa_grupos_baseline": fig_mapa_base},
    )
    plt.close(fig_mapa_base)

    # Run KMeans
    logger.info(f"Registrando Run de KMeans (K={P3_K_ELEGIDO}) en MLflow...")
    tracker.registrar_run(
        run_name=f"p3-kmeans-k{P3_K_ELEGIDO}",
        params={**params_base, "modelo": "ClusterEstacionesP3", "n_clusters": P3_K_ELEGIDO, "n_init": 10},
        tags={**tags_base, "version": f"kmeans_k{P3_K_ELEGIDO}", "baseline": "false"},
        tabla=tabla_metricas_kmeans,
        tabla_est=tabla_est_vacia,
        modelo=modelo.pipeline,
        X_ejemplo=perfil_train[P3_VARIABLES].head(5),
        figuras={"mapa_grupos_kmeans": fig_mapa_kmeans},
    )
    plt.close(fig_mapa_kmeans)

    # 4. Tabla comparativa final y guardado de reporte
    comparacion = pd.concat(
        {
            "baseline_municipio": tabla_metricas_base.set_index("conjunto"),
            f"kmeans_k{P3_K_ELEGIDO}": tabla_metricas_kmeans.set_index("conjunto"),
        },
        axis=1,
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparacion.to_csv(REPORTS_DIR / "p3_comparacion.csv")
    
    logger.info("Reporte comparativo guardado en reports/p3_comparacion.csv:")
    logger.info("\n" + comparacion.round(3).to_string())
    logger.info("P3 completado con éxito.")


if __name__ == "__main__":
    app()