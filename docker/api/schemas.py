from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


FEATURES_P1 = [
    "lag_1h", "lag_24h", "lag_168h",
    "target_hour_sin", "target_hour_cos", "target_is_weekend",
    "rolling_3h", "rolling_168h",
]

# --- P1: demanda horaria ---
class P1Input(BaseModel):
    """Datos para predecir la demanda de la hora siguiente en una estación.

    Las variables de calendario (seno, coseno y fin de semana) NO se envían:
    la API las calcula a partir de `target_hour`.
    """

    model_config = ConfigDict(
        extra="forbid",  # rechaza campos que no existen en el esquema
        json_schema_extra={
            "examples": [
                {
                    "target_hour": "2026-11-04T08:00:00",
                    "lag_1h": 5,
                    "lag_24h": 7.0,
                    "lag_168h": 6.0,
                    "rolling_3h": 4.67,
                    "rolling_168h": 5.2,
                }
            ]
        },
    )

    target_hour: datetime = Field(
        ..., examples=["2026-11-04T08:00:00"],
        description="Hora a predecir. Formato AAAA-MM-DDTHH:MM:SS, hora local, sin zona horaria",
    )
    lag_1h: int = Field(
        ..., ge=0, examples=[5],
        description="Demanda (viajes, entero) de la última hora observada",
    )
    lag_24h: float = Field(
        ..., ge=0, allow_inf_nan=False, examples=[7.0],
        description="Demanda 24 h antes de la hora objetivo",
    )
    lag_168h: float = Field(
        ..., ge=0, allow_inf_nan=False, examples=[6.0],
        description="Demanda 168 h (una semana) antes de la hora objetivo",
    )
    rolling_3h: float = Field(
        ..., ge=0, allow_inf_nan=False, examples=[4.67],
        description="Media de la demanda de las últimas 3 horas",
    )
    rolling_168h: float = Field(
        ..., ge=0, allow_inf_nan=False, examples=[5.2],
        description="Media de la demanda de las últimas 168 horas",
    )

    @field_validator("target_hour")
    @classmethod
    def sin_zona_horaria(cls, v: datetime) -> datetime:
        if v.tzinfo is not None:
            raise ValueError("target_hour debe ir sin zona horaria (hora local)")
        return v

    def to_dataframe(self) -> pd.DataFrame:
        h = self.target_hour
        fila = {
            "lag_1h": self.lag_1h,
            "lag_24h": self.lag_24h,
            "lag_168h": self.lag_168h,
            "target_hour_sin": float(np.sin(2 * np.pi * h.hour / 24)),
            "target_hour_cos": float(np.cos(2 * np.pi * h.hour / 24)),
            "target_is_weekend": int(h.weekday() in (5, 6)),
            "rolling_3h": self.rolling_3h,
            "rolling_168h": self.rolling_168h,
        }
        df = pd.DataFrame([fila], columns=FEATURES_P1)
        return df.astype({
            "lag_1h": "int64", "target_is_weekend": "int64",
            "lag_24h": "float64", "lag_168h": "float64",
            "target_hour_sin": "float64", "target_hour_cos": "float64",
            "rolling_3h": "float64", "rolling_168h": "float64",
        })

class P1Output(BaseModel):
    pregunta: str = "p1"
    modelo: str
    version: str
    run_id: str
    demanda_predicha: float = Field(..., ge=0, description="Viajes esperados en la hora objetivo (≥ 0)")

# --- P2: riesgo de desbalance ---
class P2Input(BaseModel):
    hora_del_dia: int = Field(..., ge=0, le=23)
    dia_semana: int = Field(..., ge=0, le=6)
    fin_de_semana: int = Field(..., ge=0, le=1)
    capacidad: float = Field(..., gt=0)
    salidas: float = Field(..., ge=0)
    llegadas: float = Field(..., ge=0)
    balance_acumulado_dia: float
    acumulado_relativo: float
    balance_ult_3h_relativo: float
    riesgo_vacia: int = Field(..., ge=0, le=1)
    riesgo_llena: int = Field(..., ge=0, le=1)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([self.model_dump()])[
            ["hora_del_dia", "dia_semana", "fin_de_semana", "capacidad",
             "salidas", "llegadas", "balance_acumulado_dia", "acumulado_relativo",
             "balance_ult_3h_relativo", "riesgo_vacia", "riesgo_llena"]
        ]


class P2Output(BaseModel):
    pregunta: str = "p2"
    modelo: str
    version: str
    run_id: str
    prob_riesgo: float
    pred_riesgo: int


# --- P3: clustering de estaciones ---
class P3Input(BaseModel):
    pct_manana: float = Field(..., ge=0, le=1)
    pct_mediodia: float = Field(..., ge=0, le=1)
    pct_tarde: float = Field(..., ge=0, le=1)
    pct_noche: float = Field(..., ge=0, le=1)
    pct_fin_de_semana: float = Field(..., ge=0, le=1)
    duracion_media: float = Field(..., ge=0)
    viajes_dia_log: float
    balance_neto: float = Field(..., ge=-1, le=1)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([self.model_dump()])[
            ["pct_manana", "pct_mediodia", "pct_tarde", "pct_noche",
             "pct_fin_de_semana", "duracion_media", "viajes_dia_log", "balance_neto"]
        ]


class P3Output(BaseModel):
    pregunta: str = "p3"
    modelo: str
    version: str
    run_id: str
    grupo_kmeans: int
