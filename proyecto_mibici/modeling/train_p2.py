#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de entrenamiento, evaluación y registro en MLflow para P2 (Riesgo de Desbalance).
Integra y evalúa en Train, Validación y Test:
  - Baseline de Persistencia (inercia operativa)
  - Candidato: Regresión Logística (variantes de peso de clases y umbral)
  - Búsqueda de hiperparámetros de la logística con nested runs (logistic_search),
    seguida del candidato final con los mejores valores (logistic_search_tuned)
  - Candidato: Gradient Boosting (HistGradientBoostingClassifier), como en la semana 5
  - Búsqueda de hiperparámetros del gradient boosting con nested runs (gbm_search),
    seguida del candidato final con los mejores valores (gbm_tuned)
Exporta métricas a reports/p2_comparacion.parquet y predicciones a reports/p2_predicciones_modelos.parquet.

Nota sobre nombres de parámetros: tracking.py descarta los parámetros llamados
modelo, umbral, class_weight o max_iter (entre otros), porque `mlflow run` ya los
registra desde -P. Por eso los hiperparámetros de los candidatos se registran con
prefijo (logistic_*, gbm_*) y el algoritmo como `modelo_clase`.
"""

import itertools
import time

import mlflow
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report

from proyecto_mibici.config import (
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR,
    P2_GBM_GRID,
    P2_GBM_PARAMS,
    P2_GRID_LOGISTIC,
    P2_OBJETIVO,
    P2_UMBRAL,
    P2_UMBRAL_DECISION,
    P2_VARIABLES,
    RANDOM_SEED,
    REPORTS_DIR,
    TRIPS_TRAIN_PATH,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.features import construir_dataset_p2
from proyecto_mibici.modeling.model_p2 import BaselinePersistenciaP2, RiesgoLogisticoP2
from proyecto_mibici.modeling.tracking import MLFlowTracker
from proyecto_mibici.plots import figuras_p2

app = typer.Typer()

MODELOS_VALIDOS = (
    "baseline", "naive", "logistic", "logistica", "ambos",
    "logistic_search", "gbm", "gbm_search",
)


def extraer_metricas_clasificacion(y_real, y_pred, y_prob, conjunto_nombre: str) -> dict:
    """Calcula precision, recall, f1-score y PR-AUC para la clase de riesgo (positiva).

    PR-AUC se calcula como average precision (average_precision_score), igual que en
    la semana 5. No usa interpolación lineal, que sobreestima el área en modelos con
    pocas probabilidades distintas, como la persistencia.
    """
    y_real = np.asarray(y_real, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    rep = classification_report(y_real, y_pred, output_dict=True, zero_division=0)
    pos_key = True if True in rep else "True"

    return {
        "conjunto": conjunto_nombre,
        "precision": float(rep[pos_key]["precision"]),
        "recall": float(rep[pos_key]["recall"]),
        "f1_score": float(rep[pos_key]["f1-score"]),
        "pr_auc": float(average_precision_score(y_real, y_prob)),
    }


def imprimir_metricas_modelo(nombre: str, m_train: dict, m_val: dict, m_test: dict):
    """Despliega en terminal las métricas del modelo evaluado en los 3 splits."""
    logger.info(
        f"[{nombre}] "
        f"Train -> F1: {m_train['f1_score']:.3f} | PR-AUC: {m_train['pr_auc']:.3f} || "
        f"Val -> F1: {m_val['f1_score']:.3f} | PR-AUC: {m_val['pr_auc']:.3f} || "
        f"Test -> Precision: {m_test['precision']:.3f} | Recall: {m_test['recall']:.3f} | F1: {m_test['f1_score']:.3f} | PR-AUC: {m_test['pr_auc']:.3f}"
    )


def fila_comparacion(nombre: str, m_train: dict, m_val: dict, m_test: dict) -> dict:
    """Fila de la tabla comparativa de P2."""
    return {
        "modelo": nombre,
        "train_f1": m_train["f1_score"],
        "val_f1": m_val["f1_score"],
        "test_precision": m_test["precision"],
        "test_recall": m_test["recall"],
        "test_f1": m_test["f1_score"],
        "test_pr_auc": m_test["pr_auc"],
    }


@app.command()
def main(
    modelo: str = typer.Option(
        "ambos", "--modelo",
        help="Modelo a entrenar: 'baseline', 'logistic', 'ambos', 'logistic_search', 'gbm' o 'gbm_search'",
    ),
    umbral: float = typer.Option(
        P2_UMBRAL_DECISION, "--umbral",
        help="Umbral de decisión: probabilidad a partir de la cual se predice riesgo (P2)",
    ),
    class_weight: str = typer.Option("balanced", "--class-weight", help="Peso de clases para Regresión Logística"),
    max_iter: int = typer.Option(1000, "--max-iter", help="Iteraciones máximas para Regresión Logística"),
):
    modelo = modelo.lower()
    if modelo not in MODELOS_VALIDOS:
        raise typer.BadParameter(f"Modelo desconocido: '{modelo}'. Opciones: {', '.join(MODELOS_VALIDOS)}")

    tracker = MLFlowTracker(
        tracking_uri=URI_MLFLOW_TRACKING,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )

    logger.info("Cargando y preparando dataset P2...")
    df_p2 = construir_dataset_p2()

    # Partición temporal excluyendo estaciones nuevas del test
    df_train = df_p2[df_p2["conjunto"] == "entrenamiento"].copy()
    df_val = df_p2[df_p2["conjunto"] == "validacion"].copy()
    df_test = df_p2[(df_p2["conjunto"] == "prueba") & (~df_p2["estacion_nueva"])].copy()

    splits = {"Train": df_train, "Validación": df_val, "Test": df_test}
    y_train = df_train[P2_OBJETIVO].astype(bool)
    y_val = df_val[P2_OBJETIVO].astype(bool)
    y_test = df_test[P2_OBJETIVO].astype(bool)

    logger.info(
        f"Registros P2 -> Train: {len(df_train):,} | Validación: {len(df_val):,} | Test: {len(df_test):,}"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    params_base, tags_base = tracker.armar_params_tags(
        "P2",
        splits,
        extra_params={
            "variable_objetivo": P2_OBJETIVO,
            "umbral_capacidad": P2_UMBRAL,
            "umbral_decision": umbral,
        },
        col_tiempo="hora",
    )

    filas_comparacion = []
    cols_id = [c for c in ["estacion_id", "hora"] if c in df_test.columns]
    df_predicciones = df_test[cols_id].copy()
    df_predicciones["real"] = y_test.values.astype(int)

    # Matrices de variables, una sola vez y solo si se entrena un modelo
    # (la persistencia no las necesita y así se ahorra memoria)
    if modelo not in ("baseline", "naive"):
        X_train = df_train[P2_VARIABLES]
        X_val = df_val[P2_VARIABLES]
        X_test = df_test[P2_VARIABLES]

    # -------------------------------------------------------------
    # RECETAS DE ENTRENAMIENTO Y REGISTRO DE CANDIDATOS
    # -------------------------------------------------------------
    def entrenar_logistico_candidato(
        nombre: str,
        hp: dict,
        extra_tags: dict = None,
        guardar_pkl: bool = False,
        cols_pred: tuple = None,
    ):
        """Entrena RiesgoLogisticoP2 con hp (class_weight, max_iter, C), lo evalúa y lo registra."""
        logger.info(f"--- Entrenando Candidato: Regresión Logística ({nombre}) {hp} ---")
        t0 = time.time()
        modelo_log = RiesgoLogisticoP2(
            features=P2_VARIABLES,
            class_weight=hp["class_weight"],
            random_state=RANDOM_SEED,
            max_iter=hp["max_iter"],
            C=hp["C"],
            umbral=umbral,
        )
        modelo_log.fit(X_train, y_train)
        segundos = time.time() - t0

        if guardar_pkl:
            ruta_modelo = MODELS_DIR / "p2_logistic_model.pkl"
            modelo_log.save(ruta_modelo)
            logger.info(f"Modelo P2 guardado exitosamente en {ruta_modelo}")

        prob_train = modelo_log.predict_proba(X_train)[:, 1]
        prob_val = modelo_log.predict_proba(X_val)[:, 1]
        prob_test = modelo_log.predict_proba(X_test)[:, 1]
        pred_train = modelo_log.predict(X_train)
        pred_val = modelo_log.predict(X_val)
        pred_test = modelo_log.predict(X_test)

        m_train = extraer_metricas_clasificacion(y_train, pred_train, prob_train, "Train")
        m_val = extraer_metricas_clasificacion(y_val, pred_val, prob_val, "Validación")
        m_test = extraer_metricas_clasificacion(y_test, pred_test, prob_test, "Test")

        col_prob, col_pred = cols_pred or (f"prob_{nombre}", f"pred_{nombre}")
        df_predicciones[col_prob] = np.round(prob_test, 4)
        df_predicciones[col_pred] = pred_test.astype(int)

        imprimir_metricas_modelo(nombre, m_train, m_val, m_test)

        run_id = tracker.registrar_run(
            run_name=f"p2-{nombre.replace('_', '-')}",
            params={
                **params_base,
                "modelo_clase": "LogisticRegression",
                "variables_predictoras": ", ".join(P2_VARIABLES),
                **{f"logistic_{k}": v for k, v in hp.items()},
            },
            tags={
                **tags_base,
                "version": nombre,
                "question": "P2",
                "model_family": "logistic_regression",
                "stage_of_work": "candidate",
                **(extra_tags or {}),
            },
            tabla=pd.DataFrame([m_train, m_val, m_test]),
            tabla_est=pd.DataFrame(),
            modelo=getattr(modelo_log, "estimator_", modelo_log),
            X_ejemplo=X_train.head(5),
            figuras=figuras_p2(y_test, pred_test, prob_test, nombre),
            df_train=X_train,
            ruta_entrenamiento=str(TRIPS_TRAIN_PATH),
        )
        # Costo de entrenamiento, para comparar candidatos
        mlflow.MlflowClient().log_metric(run_id, "train_seconds", segundos)
        filas_comparacion.append(fila_comparacion(nombre, m_train, m_val, m_test))

    def entrenar_gbm_candidato(nombre: str, hp: dict, extra_tags: dict = None):
        """Entrena un HistGradientBoostingClassifier, lo evalúa en los 3 conjuntos y lo registra.

        El modelo se registra tal cual; su predict decide con 0.5, igual que P2_UMBRAL_DECISION.
        """
        logger.info(f"--- Entrenando Candidato: Gradient Boosting ({nombre}) {hp} ---")
        t0 = time.time()
        modelo_gbm = HistGradientBoostingClassifier(random_state=RANDOM_SEED, **hp)
        modelo_gbm.fit(X_train, y_train)
        segundos = time.time() - t0

        prob_train = modelo_gbm.predict_proba(X_train)[:, 1]
        prob_val = modelo_gbm.predict_proba(X_val)[:, 1]
        prob_test = modelo_gbm.predict_proba(X_test)[:, 1]
        pred_train = prob_train >= umbral
        pred_val = prob_val >= umbral
        pred_test = prob_test >= umbral

        m_train = extraer_metricas_clasificacion(y_train, pred_train, prob_train, "Train")
        m_val = extraer_metricas_clasificacion(y_val, pred_val, prob_val, "Validación")
        m_test = extraer_metricas_clasificacion(y_test, pred_test, prob_test, "Test")

        df_predicciones[f"prob_{nombre}"] = np.round(prob_test, 4)
        df_predicciones[f"pred_{nombre}"] = pred_test.astype(int)

        imprimir_metricas_modelo(nombre, m_train, m_val, m_test)
        logger.info(f"[{nombre}] Iteraciones usadas: {modelo_gbm.n_iter_} | Entrenamiento: {segundos:.1f} s")

        run_id = tracker.registrar_run(
            run_name=f"p2-{nombre.replace('_', '-')}",
            params={
                **params_base,
                "modelo_clase": "HistGradientBoostingClassifier",
                "variables_predictoras": ", ".join(P2_VARIABLES),
                "gbm_n_iter_real": modelo_gbm.n_iter_,
                **{f"gbm_{k}": v for k, v in hp.items()},
            },
            tags={
                **tags_base,
                "version": nombre,
                "question": "P2",
                "model_family": "hist_gradient_boosting",
                "stage_of_work": "candidate",
                **(extra_tags or {}),
            },
            tabla=pd.DataFrame([m_train, m_val, m_test]),
            tabla_est=pd.DataFrame(),
            modelo=modelo_gbm,
            X_ejemplo=X_train.head(5),
            figuras=figuras_p2(y_test, pred_test, prob_test, nombre),
            df_train=X_train,
            ruta_entrenamiento=str(TRIPS_TRAIN_PATH),
            serialization_format="cloudpickle",
        )
        # Costo de entrenamiento, para comparar candidatos
        mlflow.MlflowClient().log_metric(run_id, "train_seconds", segundos)
        filas_comparacion.append(fila_comparacion(nombre, m_train, m_val, m_test))

    def busqueda_anidada(nombre_busqueda: str, combinaciones: list, crear_modelo, familia: str, criterio: str):
        """Corrida padre con una corrida hija por combinación, evaluada en validación.

        Devuelve (mejores hiperparámetros, run_id de la corrida padre).
        criterio es la métrica de validación a maximizar: 'f1_score' o 'pr_auc'.
        """
        tags_busqueda = {
            **tags_base,
            "question": "P2",
            "model_family": familia,
            "stage_of_work": "search",
        }
        mejor_hp, mejor_valor = None, float("-inf")

        with mlflow.start_run(run_name=nombre_busqueda) as padre:
            mlflow.set_tags(tags_busqueda)
            mlflow.log_params({
                "n_combinaciones": len(combinaciones),
                "criterio": f"validation_{criterio}",
                "umbral_decision": umbral,
            })

            for i, hp in enumerate(combinaciones):
                with mlflow.start_run(run_name=f"search_{i:02d}", nested=True):
                    mlflow.set_tags(tags_busqueda)
                    mlflow.log_params(hp)
                    logger.info(f"search_{i:02d}: {hp}")
                    try:
                        t0 = time.time()
                        m = crear_modelo(hp)
                        m.fit(X_train, y_train)
                        segundos = time.time() - t0

                        prob_val = m.predict_proba(X_val)[:, 1]
                        m_val = extraer_metricas_clasificacion(y_val, prob_val >= umbral, prob_val, "Validación")
                        metricas = {**tracker.metricas_a_dict(pd.DataFrame([m_val])), "train_seconds": segundos}
                        if hasattr(m, "n_iter_") and np.isscalar(m.n_iter_):
                            metricas["n_iter_real"] = float(m.n_iter_)
                        mlflow.log_metrics(metricas)
                        logger.info(
                            f"search_{i:02d}: Val PR-AUC {m_val['pr_auc']:.4f} | "
                            f"Val F1 {m_val['f1_score']:.4f} | {segundos:.1f} s"
                        )
                        if m_val[criterio] > mejor_valor:
                            mejor_valor, mejor_hp = m_val[criterio], hp
                    except Exception as e:
                        mlflow.set_tag("fallo", "true")
                        mlflow.set_tag("error", str(e)[:250])
                        logger.warning(f"search_{i:02d} falló y se omite: {e}")

            if mejor_hp is None:
                raise RuntimeError("Ninguna combinación de la búsqueda terminó correctamente.")
            mlflow.log_metric(f"best_validation_{criterio}", mejor_valor)
            mlflow.log_params({f"best_{k}": v for k, v in mejor_hp.items()})
            logger.info(f"{nombre_busqueda}: mejor combinación {mejor_hp} (validation_{criterio} = {mejor_valor:.4f})")
            return mejor_hp, padre.info.run_id

    # -------------------------------------------------------------
    # 1. LÍNEA BASE (PERSISTENCIA INGENUA)
    # -------------------------------------------------------------
    if modelo in ("baseline", "naive", "ambos"):
        logger.info("--- Evaluando Línea Base Ingenua: Persistencia ---")
        modelo_naive = BaselinePersistenciaP2(col_persistencia="en_riesgo")

        pred_train_naive = modelo_naive.predict(df_train)
        prob_train_naive = modelo_naive.predict_proba(df_train)[:, 1]
        m_naive_train = extraer_metricas_clasificacion(y_train, pred_train_naive, prob_train_naive, "Train")

        pred_val_naive = modelo_naive.predict(df_val)
        prob_val_naive = modelo_naive.predict_proba(df_val)[:, 1]
        m_naive_val = extraer_metricas_clasificacion(y_val, pred_val_naive, prob_val_naive, "Validación")

        pred_test_naive = modelo_naive.predict(df_test)
        prob_test_naive = modelo_naive.predict_proba(df_test)[:, 1]
        m_naive_test = extraer_metricas_clasificacion(y_test, pred_test_naive, prob_test_naive, "Test")

        df_predicciones["naive_persistencia"] = pred_test_naive.astype(int)
        imprimir_metricas_modelo("naive_persistencia", m_naive_train, m_naive_val, m_naive_test)

        tracker.registrar_run(
            run_name="P2_naive_persistencia",
            params={
                **params_base,
                "modelo_clase": "BaselinePersistenciaP2",
                "regla": "en_riesgo_t",
            },
            tags={
                **tags_base,
                "version": "naive_persistencia",
                "question": "P2",
                "model_family": "naive",
                "stage_of_work": "baseline",
            },
            tabla=pd.DataFrame([m_naive_train, m_naive_val, m_naive_test]),
            tabla_est=pd.DataFrame(),
            df_train=df_train,
            ruta_entrenamiento=str(TRIPS_TRAIN_PATH),
        )
        filas_comparacion.append(fila_comparacion("naive_persistencia", m_naive_train, m_naive_val, m_naive_test))

    # -------------------------------------------------------------
    # 2. CANDIDATO: REGRESIÓN LOGÍSTICA
    # -------------------------------------------------------------
    if modelo in ("logistic", "logistica", "ambos"):
        cw = None if class_weight == "None" else class_weight
        entrenar_logistico_candidato(
            "logistic_regression",
            {"class_weight": cw, "max_iter": max_iter, "C": 1.0},
            guardar_pkl=True,
            cols_pred=("prob_riesgo", "pred_riesgo"),
        )

    # -------------------------------------------------------------
    # 3. BÚSQUEDA ANIDADA: REGRESIÓN LOGÍSTICA -> logistic_search_tuned
    # -------------------------------------------------------------
    if modelo == "logistic_search":
        combinaciones = [
            dict(zip(P2_GRID_LOGISTIC, valores)) for valores in itertools.product(*P2_GRID_LOGISTIC.values())
        ]
        mejor_hp, id_padre = busqueda_anidada(
            "P2_logistic_search",
            combinaciones,
            crear_modelo=lambda hp: LogisticRegression(random_state=RANDOM_SEED, **hp),
            familia="logistic_regression",
            criterio="f1_score",
        )
        # El modelo final usa TODOS los hiperparámetros ganadores (incluido C)
        entrenar_logistico_candidato(
            "logistic_search_tuned",
            {"class_weight": mejor_hp["class_weight"], "max_iter": mejor_hp["max_iter"], "C": mejor_hp["C"]},
            extra_tags={"search_run_id": id_padre},
        )

    # -------------------------------------------------------------
    # 4. CANDIDATO: GRADIENT BOOSTING (como en la semana 5)
    # -------------------------------------------------------------
    if modelo == "gbm":
        entrenar_gbm_candidato("gbm", P2_GBM_PARAMS)

    # -------------------------------------------------------------
    # 5. BÚSQUEDA ANIDADA: GRADIENT BOOSTING -> gbm_tuned
    # -------------------------------------------------------------
    if modelo == "gbm_search":
        combinaciones = [
            dict(zip(P2_GBM_GRID, valores)) for valores in itertools.product(*P2_GBM_GRID.values())
        ]
        mejor_hp, id_padre = busqueda_anidada(
            "P2_gbm_search",
            combinaciones,
            crear_modelo=lambda hp: HistGradientBoostingClassifier(random_state=RANDOM_SEED, **hp),
            familia="hist_gradient_boosting",
            criterio="pr_auc",
        )
        entrenar_gbm_candidato("gbm_tuned", mejor_hp, extra_tags={"search_run_id": id_padre})

    # -------------------------------------------------------------
    # 6. GUARDAR COMPARATIVA Y PREDICCIONES (SOLO PARQUET)
    # -------------------------------------------------------------
    df_p2_comp = pd.DataFrame(filas_comparacion).set_index("modelo")
    p2_comp_path = REPORTS_DIR / "p2_comparacion.parquet"
    df_p2_comp.round(3).reset_index().to_parquet(p2_comp_path, index=False)

    logger.info(f"Reporte de entrenamiento P2 guardado en Parquet: {p2_comp_path}")
    logger.info("\n" + df_p2_comp.round(3).to_string())

    preds_out_path = REPORTS_DIR / "p2_predicciones_modelos.parquet"
    df_predicciones.to_parquet(preds_out_path, index=False)
    logger.success(f"Predicciones de P2 guardadas en {preds_out_path}")


if __name__ == "__main__":
    app()