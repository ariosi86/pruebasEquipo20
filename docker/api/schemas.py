from pydantic import BaseModel, Field, ValidationError


# --- P1: demanda horaria ---
class P1Input(BaseModel):
    lag_1h: float = Field(..., description="Rezago a 1 hora")
    lag_24h: float = Field(..., description="Rezago a 24 horas")
    lag_168h: float = Field(..., description="Rezago a 168 horas")
    target_hour_sin: float = Field(..., description="Componente seno del ciclo horario")
    target_hour_cos: float = Field(..., description="Componente coseno del ciclo horario")
    target_is_weekend: int = Field(..., ge=0, le=1, description="1 si es fin de semana, 0 si no")
    rolling_3h: float = Field(..., description="Media móvil de 3 horas")
    rolling_168h: float = Field(..., description="Media móvil de 168 horas")

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame([self.model_dump()])[
            ["lag_1h", "lag_24h", "lag_168h",
             "target_hour_sin", "target_hour_cos", "target_is_weekend",
             "rolling_3h", "rolling_168h"]
        ]


class P1Output(BaseModel):
    pregunta: str = "p1"
    modelo: str
    version: str
    run_id: str
    demanda_predicha: float


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
