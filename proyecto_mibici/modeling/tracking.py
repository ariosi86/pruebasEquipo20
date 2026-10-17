import hashlib
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from proyecto_mibici.config import MANIFEST_PATH, PROJ_ROOT, TRIPS_PROCESSED_PATH

NOMBRES_CONJUNTO = {"Train": "train", "Validación": "validation", "Test": "test"}


def hash_archivo(ruta, n=12):
    """Hash SHA256 corto del archivo, para identificar la versión de los datos."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()[:n]


def git_commit():
    """Commit actual del repo; agrega '-dirty' si hay cambios sin commitear."""
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        sucio = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJ_ROOT
        ).decode().strip() != ""
        return h[:10] + ("-dirty" if sucio else "")
    except Exception:
        return "no_disponible"


def metricas_a_dict(tabla):
    """Convierte la tabla (una fila por conjunto) a {'test_mae': ..., ...}."""
    return {
        f"{NOMBRES_CONJUNTO[fila['conjunto']]}_{m.lower()}": float(fila[m])
        for _, fila in tabla.iterrows()
        for m in tabla.columns
        if m != "conjunto"
    }


def armar_params_tags(pregunta, splits, extra_params=None,
                      ruta_datos=TRIPS_PROCESSED_PATH, col_tiempo="target_hour"):
    """Parámetros y etiquetas comunes a todos los runs de una pregunta."""
    version_datos = hash_archivo(ruta_datos)
    d_train, d_val, d_test = splits["Train"], splits["Validación"], splits["Test"]
    params = {
        "pregunta": pregunta,
        "train_inicio": d_train[col_tiempo].min(), "train_fin": d_train[col_tiempo].max(),
        "validation_inicio": d_val[col_tiempo].min(), "validation_fin": d_val[col_tiempo].max(),
        "test_inicio": d_test[col_tiempo].min(), "test_fin": d_test[col_tiempo].max(),
        "datos_version": version_datos,
        **(extra_params or {}),
    }
    tags = {"equipo": "20", "pregunta": pregunta,
            "git_commit": git_commit(), "datos_version": version_datos}
    return params, tags


def registrar_run(run_name, params, tags, tabla, tabla_est,
                  modelo=None, X_ejemplo=None, figuras=None):
    """Crea un run con parámetros, métricas, modelo, figuras y tabla por estación."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics(metricas_a_dict(tabla))
        mlflow.set_tags(tags)

        if modelo is not None:
            mlflow.sklearn.log_model(
                sk_model=modelo, name=run_name,
                signature=infer_signature(X_ejemplo, modelo.predict(X_ejemplo)),
                input_example=X_ejemplo,
            )
        for nombre, fig in (figuras or {}).items():
            mlflow.log_figure(fig, f"graficas/{nombre}.png")

        mlflow.log_text(tabla_est.to_csv(index=False), "metricas_por_estacion.csv")
        if Path(MANIFEST_PATH).exists():
            mlflow.log_artifact(str(MANIFEST_PATH))
        return run.info.run_id