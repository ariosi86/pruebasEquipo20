#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A/B testing de P2 (Entregable 8): champion (v3) contra challenger (v4).

Diseño pareado: las dos versiones predicen exactamente las mismas filas
(horas objetivo de julio a septiembre de 2026), un periodo que ninguna vio.
- Métricas: las mismas de train_p2.py (precisión, recall, F1, PR-AUC).
- McNemar: compara las filas donde solo una de las dos versiones acierta.
- Bootstrap por estación: intervalo de confianza de la diferencia B - A en
  F1 y PR-AUC. Se remuestrean estaciones, no filas, porque las horas de una
  misma estación no son independientes.

Regla de decisión (fijada antes de ver resultados):
- IC95 de F1 y de PR-AUC por encima de 0 -> recomendar el challenger.
- IC95 de F1 y de PR-AUC por debajo de 0 -> mantener el champion.
- Otro caso -> sin diferencia clara: mantener el champion.
No cambia los alias del Registry: la promoción se decide después de revisar.
"""

import mlflow
import numpy as np
import pandas as pd
import typer
from loguru import logger
from scipy.stats import chi2
from sklearn.metrics import auc, f1_score, precision_recall_curve

from proyecto_mibici.config import (
    P2_OBJETIVO,
    P2_VARIABLES,
    RANDOM_SEED,
    REPORTS_DIR,
    URI_MLFLOW_TRACKING,
)
from proyecto_mibici.modeling.train_p2 import extraer_metricas_clasificacion
from proyecto_mibici.monitoring.datos_p2 import (
    FIN_ACTUAL,
    INICIO_ACTUAL,
    construir_serie,
)

NOMBRE_MODELO = "mibici-equipo20-p2-riesgo"
EXPERIMENTO = "mibici-equipo20-p2-monitoreo"
VERSIONES = {"A_champion": "champion", "B_challenger": "challenger"}
VARIABLES = list(P2_VARIABLES)
N_BOOT = 300
OUT_DIR = REPORTS_DIR / "ab_test" / "p2"

app = typer.Typer(help="A/B testing de P2: champion contra challenger.")


def cargar_modelos():
    cliente = mlflow.MlflowClient()
    modelos, numeros = {}, {}
    for etiqueta, alias in VERSIONES.items():
        uri = f"models:/{NOMBRE_MODELO}@{alias}"
        modelos[etiqueta] = mlflow.sklearn.load_model(uri)
        version = cliente.get_model_version_by_alias(NOMBRE_MODELO, alias)
        numeros[etiqueta] = version.version
    return modelos, numeros


def datos_evaluacion() -> pd.DataFrame:
    serie = construir_serie(fin_periodo=FIN_ACTUAL)
    inicio, fin = pd.Timestamp(INICIO_ACTUAL), pd.Timestamp(FIN_ACTUAL)
    en_actual = serie["hora_objetivo"].between(inicio, fin)
    return serie[en_actual].reset_index(drop=True)


def predecir(modelos, df):
    X = df[VARIABLES]
    resultados = {}
    for etiqueta, modelo in modelos.items():
        prob = modelo.predict_proba(X)[:, 1]
        pred = np.asarray(modelo.predict(X)).astype(bool)
        resultados[etiqueta] = (prob, pred)
    return resultados


def pr_auc(y, prob) -> float:
    precision, recall, _ = precision_recall_curve(y, prob)
    return float(auc(recall, precision))


def metricas(y, pred, prob, nombre) -> dict:
    m = extraer_metricas_clasificacion(y, pred, prob, nombre)
    m["tp"] = int((pred & y).sum())
    m["fp"] = int((pred & ~y).sum())
    m["fn"] = int((~pred & y).sum())
    return m


def mcnemar(y, pred_a, pred_b):
    ok_a, ok_b = pred_a == y, pred_b == y
    solo_a = int((ok_a & ~ok_b).sum())
    solo_b = int((~ok_a & ok_b).sum())
    total = solo_a + solo_b
    estadistico = (abs(solo_a - solo_b) - 1) ** 2 / total if total else 0.0
    return solo_a, solo_b, estadistico, float(chi2.sf(estadistico, df=1))


def por_mes(df, y, resultados) -> pd.DataFrame:
    meses = df["hora_objetivo"].dt.month.to_numpy()
    filas = []
    for mes in sorted(set(meses)):
        s = meses == mes
        for etiqueta, (prob, pred) in resultados.items():
            filas.append({
                "mes": int(mes),
                "version": etiqueta,
                "tasa_riesgo": float(y[s].mean()),
                "f1": f1_score(y[s], pred[s]),
                "pr_auc": pr_auc(y[s], prob[s]),
            })
    return pd.DataFrame(filas)


def bootstrap_estaciones(df, y, resultados, n_boot) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    grupos = list(df.groupby("estacion_id").indices.values())
    prob_a, pred_a = resultados["A_champion"]
    prob_b, pred_b = resultados["B_challenger"]
    difs = {"f1": [], "pr_auc": []}
    for i in range(n_boot):
        elegidas = rng.integers(0, len(grupos), len(grupos))
        idx = np.concatenate([grupos[j] for j in elegidas])
        yy = y[idx]
        difs["f1"].append(f1_score(yy, pred_b[idx]) - f1_score(yy, pred_a[idx]))
        difs["pr_auc"].append(pr_auc(yy, prob_b[idx]) - pr_auc(yy, prob_a[idx]))
        if (i + 1) % 50 == 0:
            logger.info(f"Bootstrap {i + 1}/{n_boot}")
    filas = []
    for metrica, valores in difs.items():
        v = np.array(valores)
        filas.append({
            "metrica": metrica,
            "dif_media_B_menos_A": v.mean(),
            "ic95_inf": np.percentile(v, 2.5),
            "ic95_sup": np.percentile(v, 97.5),
            "prob_B_mejor": (v > 0).mean(),
        })
    return pd.DataFrame(filas)


def decidir(boot: pd.DataFrame) -> str:
    inf = boot.set_index("metrica")["ic95_inf"]
    sup = boot.set_index("metrica")["ic95_sup"]
    if (inf > 0).all():
        return "challenger"
    if (sup < 0).all():
        return "champion"
    return "sin_diferencia_mantener_champion"


@app.command()
def main(
    n_boot: int = typer.Option(N_BOOT, "--n-boot", help="Repeticiones del bootstrap."),
    registrar: bool = typer.Option(True, "--mlflow/--sin-mlflow", help="Registrar en MLflow."),
):
    mlflow.set_tracking_uri(URI_MLFLOW_TRACKING)
    modelos, numeros = cargar_modelos()
    logger.info(f"Versiones comparadas: {numeros}")

    actual = datos_evaluacion()
    segmentos = {
        "conocidas": actual[~actual["estacion_nueva"]].reset_index(drop=True),
        "nuevas": actual[actual["estacion_nueva"]].reset_index(drop=True),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filas = []
    for segmento, df in segmentos.items():
        if df.empty:
            continue
        y = df[P2_OBJETIVO].astype(bool).to_numpy()
        resultados = predecir(modelos, df)
        for etiqueta, (prob, pred) in resultados.items():
            filas.append({
                "segmento": segmento,
                "version": etiqueta,
                "model_version": numeros[etiqueta],
                "filas": len(df),
                "estaciones": df["estacion_id"].nunique(),
                **metricas(y, pred, prob, segmento),
            })
        if segmento == "conocidas":
            principal = (df, y, resultados)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(OUT_DIR / "ab_metricas.csv", index=False)
    logger.info("Métricas por segmento:\n" + tabla.round(4).to_string(index=False))

    df, y, resultados = principal
    pred_a, pred_b = resultados["A_champion"][1], resultados["B_challenger"][1]
    solo_a, solo_b, estadistico, p_mcnemar = mcnemar(y, pred_a, pred_b)
    logger.info(
        f"McNemar: solo A acierta {solo_a:,} | solo B acierta {solo_b:,} | "
        f"chi2 {estadistico:.2f} | p {p_mcnemar:.3g}"
    )

    mensual = por_mes(df, y, resultados)
    mensual.to_csv(OUT_DIR / "ab_por_mes.csv", index=False)
    logger.info("Por mes:\n" + mensual.round(4).to_string(index=False))

    boot = bootstrap_estaciones(df, y, resultados, n_boot)
    boot.to_csv(OUT_DIR / "ab_bootstrap.csv", index=False)
    logger.info("Bootstrap (B - A):\n" + boot.round(4).to_string(index=False))

    decision = decidir(boot)
    logger.success(f"Recomendación según la regla: {decision}")

    if not registrar:
        return
    conocidas = tabla[tabla["segmento"] == "conocidas"].set_index("version")
    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name="P2_ab_test_jul_sep2026"):
        mlflow.set_tags({"question": "P2", "stage_of_work": "ab_test"})
        mlflow.log_params({
            "version_A_champion": numeros["A_champion"],
            "version_B_challenger": numeros["B_challenger"],
            "periodo_inicio": INICIO_ACTUAL,
            "periodo_fin": FIN_ACTUAL,
            "n_boot": n_boot,
            "unidad_bootstrap": "estacion",
            "recomendacion": decision,
        })
        for etiqueta, fila in conocidas.iterrows():
            for k in ["precision", "recall", "f1_score", "pr_auc"]:
                mlflow.log_metric(f"{etiqueta}_{k}", float(fila[k]))
        mlflow.log_metric("mcnemar_p", p_mcnemar)
        for _, fila in boot.iterrows():
            nombre = fila["metrica"]
            mlflow.log_metric(f"dif_{nombre}_media", float(fila["dif_media_B_menos_A"]))
            mlflow.log_metric(f"dif_{nombre}_ic95_inf", float(fila["ic95_inf"]))
            mlflow.log_metric(f"dif_{nombre}_ic95_sup", float(fila["ic95_sup"]))
        mlflow.log_artifacts(str(OUT_DIR), artifact_path="ab_test_p2")


if __name__ == "__main__":
    app()