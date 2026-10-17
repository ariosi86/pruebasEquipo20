import numpy as np
import pandas as pd

from proyecto_mibici.config import (
    P1_MAX_PCT_VACIAS,
    STATIONS_PROCESSED_PATH,
    TRIPS_PROCESSED_PATH,
)

# ==========================================
# P1: DEMANDA HORARIA
# ==========================================

def limpiar_viajes(df_trips):
    """Elimina duplicados exactos (verificado en el EDA: no hay conflictos)."""
    df_clean = df_trips.drop_duplicates().copy()
    assert df_clean["viaje_id"].is_unique, "Hay viaje_id repetidos con datos distintos."
    return df_clean


def filtrar_estaciones_activas(df_trips_clean, df_stations, max_pct_vacias=P1_MAX_PCT_VACIAS):
    """Quita estaciones fuera del catálogo y las que pasan casi todo el periodo sin viajes.
    Limitante documentada: el porcentaje se calcula con todo el periodo, incluido el test.
    """
    df = df_trips_clean.assign(
        hour_window=df_trips_clean["inicio_del_viaje"].dt.floor("h")
    )
    cov = df.groupby("origen_id").agg(
        first_hour=("hour_window", "min"),
        last_hour=("hour_window", "max"),
        observed_hours=("hour_window", "nunique"),
    )
    expected = (
        (cov["last_hour"] - cov["first_hour"]) / pd.Timedelta(hours=1)
    ).astype(int) + 1
    pct_vacias = (expected - cov["observed_hours"]) / expected * 100

    validas = pct_vacias[pct_vacias <= max_pct_vacias].index
    validas = validas.intersection(df_stations["id"])  # quita 405 y 407
    return df_trips_clean[df_trips_clean["origen_id"].isin(validas)]


def build_hourly_demand_with_lags(df_trips):
    """Construye la malla horaria por estación, rellena ceros y calcula
    rezagos, medias móviles y el objetivo (demanda de la hora siguiente).
    Cada fila representa la hora t; el objetivo es la demanda de t+1.
    """
    # 1. Asignar cada viaje a su ventana horaria (14:34 -> 14:00)
    df = df_trips.assign(
        hour_window=pd.to_datetime(df_trips["inicio_del_viaje"]).dt.floor("h")
    )

    # 2. Contar viajes por estación y hora
    df_demand = (
        df.groupby(["origen_id", "hour_window"])
        .size()
        .rename("demand_count")
        .reset_index()
    )

    # 3. Malla continua: todas las horas entre el primer y el último viaje
    #    de cada estación, para no perder las horas sin viajes
    df_bounds = df_demand.groupby("origen_id")["hour_window"].agg(["min", "max"])
    grid = pd.concat([
        pd.DataFrame({
            "origen_id": sid,
            "hour_window": pd.date_range(r["min"], r["max"], freq="h"),
        })
        for sid, r in df_bounds.iterrows()
    ])

    # 4. Rellenar con 0 las horas sin viajes (se asume demanda nula)
    df_out = grid.merge(df_demand, on=["origen_id", "hour_window"], how="left")
    df_out["demand_count"] = df_out["demand_count"].fillna(0).astype(int)
    df_out = df_out.sort_values(["origen_id", "hour_window"]).reset_index(drop=True)

    # 5. Rezagos, medidos respecto a la hora objetivo t+1.
    #    Por eso shift(23) y shift(167) desde la fila t:
    #    t-23 = (t+1)-24 h  y  t-167 = (t+1)-168 h.
    por_estacion = df_out.groupby("origen_id")["demand_count"]
    df_out["lag_1h"] = df_out["demand_count"]      # hora t, 1 h antes del objetivo
    df_out["lag_24h"] = por_estacion.shift(23)     # misma hora de ayer
    df_out["lag_168h"] = por_estacion.shift(167)   # misma hora de la semana pasada

    # 6. Medias móviles: solo usan horas <= t, no hay fuga de información
    df_out["rolling_3h"] = (
        por_estacion.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df_out["rolling_168h"] = (
        por_estacion.rolling(168, min_periods=24).mean().reset_index(level=0, drop=True)
    )

    # 7. Objetivo: demanda de la hora siguiente (t+1)
    df_out["target_next_hour"] = por_estacion.shift(-1)

    # 8. El objetivo solo es válido si la fila siguiente es exactamente
    #    una hora después dentro de la misma estación (sin huecos)
    next_hour = df_out.groupby("origen_id")["hour_window"].shift(-1)
    valid_next_hour = next_hour == df_out["hour_window"] + pd.Timedelta(hours=1)
    df_out.loc[~valid_next_hour, "target_next_hour"] = np.nan

    # 9. Descartar filas sin historial suficiente o sin objetivo válido
    required_columns = ["lag_1h", "lag_24h", "lag_168h", "target_next_hour"]
    return df_out.dropna(subset=required_columns).reset_index(drop=True)


def add_calendar_features(df):
    """Agrega variables de calendario de la hora objetivo (t+1)."""
    df = df.copy()
    df["target_hour"] = df["hour_window"] + pd.Timedelta(hours=1)
    df["target_hour_of_day"] = df["target_hour"].dt.hour
    df["target_dayofweek"] = df["target_hour"].dt.dayofweek
    df["target_is_weekend"] = df["target_dayofweek"].isin([5, 6]).astype(int)
    df["target_hour_sin"] = np.sin(2 * np.pi * df["target_hour_of_day"] / 24)
    df["target_hour_cos"] = np.cos(2 * np.pi * df["target_hour_of_day"] / 24)
    return df


def construir_dataset_p1():
    """Pipeline completo de P1: parquet procesado -> tabla de modelado."""
    df_trips = pd.read_parquet(TRIPS_PROCESSED_PATH)
    df_stations = pd.read_parquet(STATIONS_PROCESSED_PATH)

    df_clean = limpiar_viajes(df_trips)
    df_filtrado = filtrar_estaciones_activas(df_clean, df_stations)
    df_p1 = add_calendar_features(build_hourly_demand_with_lags(df_filtrado))
    return df_p1, df_stations