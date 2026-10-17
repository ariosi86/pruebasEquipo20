import numpy as np

def predecir(modelo, features, df):
    """Predicciones acotadas sin valores negativos (recorte en 0)."""
    return np.clip(modelo.predict(df[features]), 0, None)