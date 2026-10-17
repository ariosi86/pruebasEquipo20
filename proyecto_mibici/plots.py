import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import pandas as pd


def figuras_p1(df_test, pred, target, etiqueta, estacion_id=51):
    """Penitenciaría (primera semana del test) y dispersión global real vs. predicho."""
    res = df_test[["origen_id", "target_hour", target]].copy()
    res["pred"] = pred

    # Penitenciaría: primera semana del test
    g = res[res["origen_id"] == estacion_id].sort_values("target_hour")
    g = g[g["target_hour"] < g["target_hour"].min() + pd.Timedelta(days=7)]
    fig1, ax = plt.subplots(figsize=(14, 5))
    ax.plot(g["target_hour"], g[target], label="Viajes reales")
    ax.plot(g["target_hour"], g["pred"], label="Viajes predichos")
    ax.set_title(f"Demanda real vs. predicha, López Cotilla / Penitenciaría ({etiqueta})")
    ax.set_xlabel("Hora"); ax.set_ylabel("Número de viajes"); ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig1.tight_layout()

    # Dispersión global: real vs. predicho (test)
    fig2, ax = plt.subplots(figsize=(7, 7))
    y_real = df_test[target].to_numpy()
    ax.scatter(y_real, pred, alpha=0.15, s=8)
    lim = max(y_real.max(), pred.max())
    ax.plot([0, lim], [0, lim], linestyle="--", label="Predicción perfecta")
    ax.set_title(f"Viajes reales vs. predichos, Test ({etiqueta})")
    ax.set_xlabel("Viajes reales"); ax.set_ylabel("Viajes predichos"); ax.legend()
    fig2.tight_layout()

    return {f"penitenciaria_real_vs_pred_{etiqueta}": fig1,
            f"scatter_test_{etiqueta}": fig2}