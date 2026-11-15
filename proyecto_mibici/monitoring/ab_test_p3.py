#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A/B testing de P3 (Entregable 8): champion contra challenger en octubre 2026.

Diseño pareado: las dos versiones se cargan del Model Registry por su alias y
asignan grupo a exactamente las mismas estaciones del periodo actual. Como el
clustering no tiene etiqueta real, la calidad se mide con métricas internas
calculadas en un mismo espacio para ambas versiones (las ocho variables
estandarizadas con la media y desviación de octubre 2026, ajenas a los dos
modelos):

  silhouette (mayor es mejor, métrica de decisión, la misma de la selección de k)
  Davies-Bouldin (menor es mejor)
  Calinski-Harabasz (mayor es mejor)

La diferencia B - A se acompaña de un bootstrap por estación (se remuestrean
estaciones y se recalculan las métricas de ambas versiones sobre la misma
muestra). Además se mide la estabilidad: qué parte de las estaciones cambia de
grupo entre el periodo con el que se entrenó cada versión y octubre 2026.

Regla de decisión, fijada antes de ver resultados:
  promover el challenger si el IC 95 % de la diferencia de silhouette (B - A)
  queda por encima de 0, su Davies-Bouldin no empeora y ningún grupo queda con
  menos del 5 % de las estaciones; en cualquier otro caso se mantiene el champion.
"""

import os
import tempfile

import joblib
from loguru import logger
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
import typer

from proyecto_mibici.config import (
    MLFLOW_EXPERIMENT_MONITOREO_P3,
    P3_VARIABLES,
    RANDOM_SEED,
    REPORTS_DIR,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.monitoring.datos_p3 import PERIODOS_P3, cargar_conjunto

NOMBRE_MODELO = "mibici-equipo20-p3-grupos"
SALIDA = REPORTS_DIR / "ab_test" / "p3"
MIN_PROPORCION_GRUPO = 0.05

app = typer.Typer(help="A/B testing de P3: champion vs challenger.")


def resolver_version(client: MlflowClient, version: int | None, alias: str):
    if version is None:
        return client.get_model_version_by_alias(NOMBRE_MODELO, alias)
    return client.get_model_version(NOMBRE_MODELO, str(version))


def cargar_pipeline(mv):
    """Descarga el pipeline sklearn guardado por el envoltorio pyfunc (artifacts/modelo.pkl)."""
    destino = tempfile.mkdtemp(prefix=f"p3_v{mv.version}_")
    ruta = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{mv.run_id}/model", dst_path=destino)
    archivo = os.path.join(ruta, "artifacts", "modelo.pkl")
    if not os.path.isfile(archivo):
        raise FileNotFoundError(f"No se encontró artifacts/modelo.pkl en la versión {mv.version}")
    return joblib.load(archivo)


def metricas_internas(X: np.ndarray, etiquetas: np.ndarray) -> dict:
    if len(np.unique(etiquetas)) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    return {
        "silhouette": float(silhouette_score(X, etiquetas)),
        "davies_bouldin": float(davies_bouldin_score(X, etiquetas)),
        "calinski_harabasz": float(calinski_harabasz_score(X, etiquetas)),
    }


def bootstrap_diferencias(X, etiquetas_a, etiquetas_b, n_boot: int, semilla: int) -> pd.DataFrame:
    """Remuestrea estaciones y calcula B - A en cada métrica sobre la misma muestra."""
    rng = np.random.default_rng(semilla)
    n = len(X)
    filas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m_a = metricas_internas(X[idx], etiquetas_a[idx])
        m_b = metricas_internas(X[idx], etiquetas_b[idx])
        filas.append({k: m_b[k] - m_a[k] for k in m_a})
    return pd.DataFrame(filas).dropna()


def emparejar_grupos(centros_a: np.ndarray, centros_b: np.ndarray) -> dict:
    """Mapa grupo_b -> grupo_a que minimiza la distancia entre centroides (Hungarian)."""
    costo = ((centros_a[:, None, :] - centros_b[None, :, :]) ** 2).sum(axis=2)
    filas, columnas = linear_sum_assignment(costo)
    return {int(b): int(a) for a, b in zip(filas, columnas)}


def proporcion_cambios(pipeline, perfil_entrenamiento: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """Estaciones comunes que el mismo modelo asigna a otro grupo en el periodo actual."""
    comunes = perfil_entrenamiento.index.intersection(actual.index)
    antes = pipeline.predict(perfil_entrenamiento.loc[comunes, P3_VARIABLES])
    despues = pipeline.predict(actual.loc[comunes, P3_VARIABLES])
    return {
        "estaciones_comunes": len(comunes),
        "cambian_de_grupo": int((antes != despues).sum()),
        "pct_cambian": float(100 * (antes != despues).mean()),
    }


def decidir(ic_silhouette: tuple, delta_db: float, min_prop_b: float) -> tuple:
    if ic_silhouette[0] > 0 and delta_db <= 0 and min_prop_b >= MIN_PROPORCION_GRUPO:
        return "PROMOVER CHALLENGER", True
    if ic_silhouette[1] < 0:
        return "MANTENER CHAMPION (el challenger es peor)", False
    return "MANTENER CHAMPION (sin diferencia clara a favor del challenger)", False


@app.command()
def main(
    champion_version: int | None = typer.Option(None, "--champion-version", help="Por defecto, alias 'champion'."),
    challenger_version: int | None = typer.Option(None, "--challenger-version", help="Por defecto, alias 'challenger'."),
    n_boot: int = typer.Option(1000, "--n-boot", help="Repeticiones del bootstrap por estación."),
    promover: bool = typer.Option(False, "--promover", help="Mueve el alias 'champion' si la regla lo indica."),
):
    mlflow.set_tracking_uri(URI_MLFLOW_TRACKING)
    client = MlflowClient()
    va = resolver_version(client, champion_version, "champion")
    vb = resolver_version(client, challenger_version, "challenger")
    etiqueta_a, etiqueta_b = f"A: champion (v{va.version})", f"B: challenger (v{vb.version})"
    logger.info(f"{etiqueta_a} run {va.run_id} | {etiqueta_b} run {vb.run_id}")

    modelo_a, modelo_b = cargar_pipeline(va), cargar_pipeline(vb)

    # 1. Ambas versiones asignan grupo a las mismas estaciones de octubre 2026
    actual = cargar_conjunto("actual")
    X_actual = actual[P3_VARIABLES]
    grupos_a = modelo_a.predict(X_actual)
    grupos_b_crudos = modelo_b.predict(X_actual)

    # KMeans numera sus grupos de forma arbitraria, así que cada grupo de B toma el número del grupo de A más parecido
    espacio = StandardScaler().fit(X_actual)
    X = espacio.transform(X_actual)
    centros_a = espacio.transform(pd.DataFrame(
        modelo_a.named_steps["scaler"].inverse_transform(modelo_a.named_steps["clusterer"].cluster_centers_),
        columns=P3_VARIABLES))
    centros_b = espacio.transform(pd.DataFrame(
        modelo_b.named_steps["scaler"].inverse_transform(modelo_b.named_steps["clusterer"].cluster_centers_),
        columns=P3_VARIABLES))
    mapa = emparejar_grupos(centros_a, centros_b)
    grupos_b = np.vectorize(mapa.get)(grupos_b_crudos)

    # 2. Métricas internas en el mismo espacio
    m_a, m_b = metricas_internas(X, grupos_a), metricas_internas(X, grupos_b)
    tamanos = pd.DataFrame({
        etiqueta_a: pd.Series(grupos_a).value_counts().sort_index(),
        etiqueta_b: pd.Series(grupos_b).value_counts().sort_index(),
    }).fillna(0).astype(int).rename_axis("grupo")
    min_prop = (tamanos.min() / len(actual)).to_dict()

    tabla = pd.DataFrame([m_a, m_b], index=[etiqueta_a, etiqueta_b])
    tabla["min_pct_estaciones_grupo"] = [100 * min_prop[etiqueta_a], 100 * min_prop[etiqueta_b]]

    # 3. Bootstrap por estación
    boot = bootstrap_diferencias(X, grupos_a, grupos_b, n_boot, RANDOM_SEED)
    ic = {k: (float(boot[k].quantile(0.025)), float(boot[k].quantile(0.975))) for k in boot.columns}
    gana_b = float((boot["silhouette"] > 0).mean())

    # 4. Estabilidad frente al periodo de entrenamiento de cada versión
    cambios = pd.DataFrame({
        etiqueta_a: proporcion_cambios(modelo_a, cargar_conjunto("referencia"), actual),
        etiqueta_b: proporcion_cambios(modelo_b, cargar_conjunto("reciente"), actual),
    }).T
    nuevas = actual.index.difference(cargar_conjunto("referencia").index)
    acuerdo = float(adjusted_rand_score(grupos_a, grupos_b))

    # 5. Perfil medio de cada grupo en octubre 2026 (unidades originales)
    perfiles = pd.concat({
        "A": X_actual.assign(grupo=grupos_a).groupby("grupo").mean(),
        "B": X_actual.assign(grupo=grupos_b).groupby("grupo").mean(),
    }, names=["version", "grupo"]).round(3)

    decision, promueve = decidir(ic["silhouette"], m_b["davies_bouldin"] - m_a["davies_bouldin"],
                                 min_prop[etiqueta_b])

    # 6. Salida
    pd.set_option("display.width", 160)
    logger.info(f"Estaciones evaluadas: {len(actual)} (nuevas respecto a 2025: {len(nuevas)})")
    logger.info("\n" + tabla.round(4).to_string())
    logger.info("Diferencia B - A (bootstrap por estación, IC 95 %):")
    for k, (lo, hi) in ic.items():
        logger.info(f"  {k:<18} {m_b[k] - m_a[k]:+.4f}  IC [{lo:+.4f}, {hi:+.4f}]")
    logger.info(f"B mejor en silhouette en {100 * gana_b:.1f} % de las repeticiones")
    logger.info(f"Acuerdo entre versiones (ARI): {acuerdo:.3f}")
    logger.info("Tamaño de los grupos:\n" + tamanos.to_string())
    logger.info("Estabilidad:\n" + cambios.to_string())
    logger.info("Perfil medio por grupo (oct 2026):\n" + perfiles.to_string())
    logger.success(f"Decisión: {decision}")

    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.round(5).to_csv(SALIDA / "ab_metricas_globales.csv")
    pd.DataFrame(
        [{"metrica": k, "diferencia_b_menos_a": m_b[k] - m_a[k], "ic95_inf": lo, "ic95_sup": hi}
         for k, (lo, hi) in ic.items()]
    ).to_csv(SALIDA / "ab_bootstrap.csv", index=False)
    tamanos.to_csv(SALIDA / "ab_tamano_grupos.csv")
    cambios.to_csv(SALIDA / "ab_estabilidad.csv")
    perfiles.to_csv(SALIDA / "ab_perfil_grupos.csv")
    pd.DataFrame({"estacion_id": actual.index, "grupo_a": grupos_a, "grupo_b": grupos_b}).to_csv(
        SALIDA / "ab_asignaciones.csv", index=False)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_MONITOREO_P3)
    with mlflow.start_run(run_name="p3_ab_test_oct2026"):
        mlflow.set_tags({"question": "P3", "stage_of_work": "ab_test", "decision": decision})
        mlflow.log_params({
            "champion_version": va.version,
            "challenger_version": vb.version,
            "periodo_evaluacion": f"{PERIODOS_P3['actual'][0]} a {PERIODOS_P3['actual'][1]}",
            "estaciones_evaluadas": len(actual),
            "n_boot": n_boot,
            "min_proporcion_grupo": MIN_PROPORCION_GRUPO,
        })
        for k in m_a:
            mlflow.log_metrics({f"champion_{k}": m_a[k], f"challenger_{k}": m_b[k],
                                f"delta_{k}": m_b[k] - m_a[k],
                                f"delta_{k}_ic95_inf": ic[k][0], f"delta_{k}_ic95_sup": ic[k][1]})
        mlflow.log_metrics({
            "ari_champion_challenger": acuerdo,
            "pct_boot_challenger_mejor": 100 * gana_b,
            "champion_pct_cambian_grupo": float(cambios.loc[etiqueta_a, "pct_cambian"]),
            "challenger_pct_cambian_grupo": float(cambios.loc[etiqueta_b, "pct_cambian"]),
        })
        mlflow.log_artifacts(str(SALIDA), artifact_path="ab_test_p3")

    if promover and promueve:
        client.set_registered_model_alias(NOMBRE_MODELO, "previous_champion", va.version)
        client.set_registered_model_alias(NOMBRE_MODELO, "champion", vb.version)
        logger.success(f"Alias 'champion' movido a la versión {vb.version}.")
    elif promover:
        logger.info("No se mueve el alias: la regla de decisión no lo indica.")


if __name__ == "__main__":
    app()
