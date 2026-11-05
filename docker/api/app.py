import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from model_loader import load_model_and_params
from schemas import (
    P1Input, P1Output,
    P2Input, P2Output,
    P3Input, P3Output,
    ValidationError,
)

load_dotenv()

app = Flask(__name__)
CORS(app)

PREGUNTA = os.environ.get("PREGUNTA", "p1")
MODELO_REGISTRY = os.environ.get("MODELO_REGISTRY")

def obtener_modelo_desde_registry():
    """Carga el modelo desde MLflow en cada petición."""
    try:
        return load_model_and_params()
    except Exception as exc:  # noqa: BLE001
        print(f"No se pudo cargar el modelo {MODELO_REGISTRY}: {exc}")
        traceback.print_exc()
        return None


@app.route("/health", methods=["GET"])
def health():
    ctx = obtener_modelo_desde_registry()
    if not ctx:
        return jsonify({
            "status": "error",
            "pregunta": PREGUNTA,
            "modelo_cargado": False,
            "error": "Modelo no disponible",
        }), 503

    return jsonify({
        "status": "ok",
        "pregunta": PREGUNTA,
        "modelo_cargado": True,
        "modelo": ctx["model_name"],
        "version": ctx["version"],
        "run_id": ctx["run_id"],
        "alias": ctx["alias"],
        "model_uri": ctx["model_uri"],
        "parametros_entrenamiento": ctx["params"],
        "tags": ctx["tags"],
    })


def _predecir_p1(modelo, datos):
    import pandas as pd
    entrada = P1Input(**datos)
    df = entrada.to_dataframe()
    pred = modelo.predict(df)
    return P1Output(
        modelo=modelo_name_cache["model_name"],
        version=modelo_name_cache["version"],
        run_id=modelo_name_cache["run_id"],
        demanda_predicha=float(pred[0]),
    )


def _predecir_p2(modelo, datos):
    import pandas as pd
    entrada = P2Input(**datos)
    df = entrada.to_dataframe()
    pred = modelo.predict(df)
    try:
        proba = modelo.predict_proba(df)[:, 1]
        prob_riesgo = float(proba[0])
    except Exception:
        prob_riesgo = float(pred[0])
    return P2Output(
        modelo=modelo_name_cache["model_name"],
        version=modelo_name_cache["version"],
        run_id=modelo_name_cache["run_id"],
        prob_riesgo=prob_riesgo,
        pred_riesgo=int(pred[0]),
    )


def _predecir_p3(modelo, datos):
    import pandas as pd
    entrada = P3Input(**datos)
    df = entrada.to_dataframe()
    pred = modelo.predict(df)
    return P3Output(
        modelo=modelo_name_cache["model_name"],
        version=modelo_name_cache["version"],
        run_id=modelo_name_cache["run_id"],
        grupo_kmeans=int(pred[0]),
    )


@app.route("/predict", methods=["POST"])
def predict():
    ctx = obtener_modelo_desde_registry()
    if not ctx:
        return jsonify({
            "error": "Modelo no cargado. Verifica MLFLOW_TRACKING_URI y MODELO_REGISTRY.",
        }), 503

    datos = request.get_json(silent=True)
    if datos is None:
        return jsonify({"error": "Se espera un cuerpo JSON válido."}), 400

    try:
        modelo_obj = ctx["modelo"]
        modelo_name_cache.update(ctx)

        if PREGUNTA == "p1":
            resultado = _predecir_p1(modelo_obj, datos)
            return jsonify(resultado.model_dump()), 200
        elif PREGUNTA == "p2":
            resultado = _predecir_p2(modelo_obj, datos)
            return jsonify(resultado.model_dump()), 200
        elif PREGUNTA == "p3":
            resultado = _predecir_p3(modelo_obj, datos)
            return jsonify(resultado.model_dump()), 200
        else:
            return jsonify({"error": f"Pregunta desconocida: {PREGUNTA}"}), 400

    except ValidationError as exc:
        return jsonify({"error": "Datos de entrada inválidos", "detalle": exc.errors()}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


modelo_name_cache = {}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
