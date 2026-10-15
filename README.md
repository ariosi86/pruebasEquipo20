# TC5061-Proyecto-Equipo20


Diseñar, como equipo, una propuesta de solución de Machine Learning para un conjunto de preguntas sobre el sistema de bicicleta pública MiBici, que combine modelos de regresión, clasificación y clustering.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         proyecto_mibici and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── proyecto_mibici   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes proyecto_mibici a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

# Guía Rápida para el Equipo: Clonación, Entorno Virtual, Datos y Notebooks (MiBici)

Esta guía paso a paso contiene todo lo necesario para que cualquier integrante del equipo clone el repositorio, configure su entorno virtual utilizando **`requirements.txt`**, ejecute la ingesta de datos y comience a trabajar en su Jupyter Notebook con el código listo para las tres preguntas (**P1**, **P2** y **P3**).

---

## 1. Clonar el Repositorio y Configurar el Entorno Virtual

Sigue estos comandos en tu terminal según tu sistema operativo. Asegúrate de tener instalado **Python 3.10 o 3.11**.

### Paso 1: Clonar el proyecto
```bash
git clone [https://github.com/](https://github.com/)[ORGANIZACION_O_USUARIO]/TC5061-Proyecto-Equipo[NN].git

#(Seria la liga de este repo de prueba)

#abren la carpeta
cd TC5061-Proyecto-Equipo[NN]
```

### Paso 2: Crear el entorno virtual
```bash
# Crear el entorno con el nombre estándar .venv
python -m venv .venv
```

### Paso 3: Activar el entorno virtual

* **En macOS y Linux:**
  ```bash
  source .venv/bin/activate
  ```

* **En Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
  *(Si PowerShell arroja error de políticas de script, ejecuta antes: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

* **En Windows (Símbolo del sistema / CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

*(Sabrás que está activo porque aparecerá `(.venv)` al inicio de la línea de comandos).*

### Paso 4: Instalar dependencias desde `requirements.txt` y registrar el kernel
Actualiza `pip`, instala las librerías fijadas del proyecto, monta el paquete local y vincula el kernel de Jupyter:

```bash
# 1. Actualizar el gestor de paquetes
pip install --upgrade pip

# 2. Instalar dependencias oficiales fijadas en requirements.txt
pip install -r requirements.txt

# 3. Instalar el paquete local en modo editable para importar proyecto_mibici
pip install -e .

# 4. Vincular este entorno exacto a Jupyter
python -m ipykernel install --user --name=mibici-venv --display-name="Python (MiBici .venv)"
```

---

## 2. Descargar y Generar el Dataset (Ingesta Reproducible)

Con el entorno virtual activo `(.venv)`, corre el módulo de ingesta una sola vez desde la terminal:

```bash
python -m proyecto_mibici.dataset
```

**¿Qué hace este comando automáticamente?**
1. Descarga todos los archivos CSV mensuales de MiBici y la nomenclatura oficial sin pasos manuales.
2. Limpia formatos de texto, normaliza nombres de columnas y parsea fechas.
3. Genera los archivos en formato Apache Parquet de alta velocidad en:
   * `data/processed/trips.parquet`
   * `data/processed/stations.parquet`
4. Genera `data/manifest.json` con los hashes criptográficos SHA256 para certificar reproducibilidad idéntica.

---

## 3. Iniciar Jupyter y Crear tu Notebook

2. Crea tu archivo dentro de la carpeta `notebooks/` por ejemplo:
   * Para P1: `notebooks/1.0-iniciales-demanda_p1.ipynb`
   * Para P2: `notebooks/1.0-iniciales-desbalance_p2.ipynb`
   * Para P3: `notebooks/1.0-iniciales-clustering_p3.ipynb`
3. En la esquina superior derecha del notebook, asegúrate de que el Kernel seleccionado sea **`Python (MiBici .venv)`**.

---

## 4. Primera Celda del Notebook (Carga Rápida)

Coloca este bloque en la primera celda para garantizar que las rutas funcionen en cualquier computadora

Es importante para que se integre con la estructura de carpeta de cookiecutter
:

```python
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Resolver la raíz del repositorio sin rutas absolutas
ROOT_DIR = Path.cwd().resolve().parents[0] if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from proyecto_mibici.config import TRIPS_PROCESSED_PATH, STATIONS_PROCESSED_PATH

# Carga de datos columnar (toma entre 2 y 4 segundos)
df_trips = pd.read_parquet(TRIPS_PROCESSED_PATH)
df_stations = pd.read_parquet(STATIONS_PROCESSED_PATH)

print(f"Viajes cargados: {len(df_trips):,}")
print(f"Estaciones registradas: {len(df_stations):,}")
```

---

## 5. Recetas de Código: Construcción de la Variable $y$ y Features por Pregunta


Los archivos `trips.parquet` y `stations.parquet` **son inmutables**. Cada integrante ejecuta en su notebook el bloque correspondiente a su problema:

---

Esa es una sugerencia para calcular sus variables y pues la pregunta 1 y 2 no tienen directamente la variable y hay q calcularla.

Esto les serira de ayuda para crear las variables faltantes y puedan terminar sus modelos por que estaba revisando y no estan tan directos.

Esta documentado en el docx

### Pregunta 1 (P1): Demanda de Viajes por Estación (Regresión)

* **¿Qué predecimos ($y$)?** Número de salidas que tendrá una estación en la siguiente hora ($t+1$).
* **Granularidad:** `(hour_window, origen_id)`.

```python
# 1. Truncar inicio del viaje a intervalos de 1 hora
df_trips["hour_window"] = df_trips["inicio_del_viaje"].dt.floor("h")

# 2. Variable objetivo (y): Conteo de salidas
df_p1 = (
    df_trips.groupby(["hour_window", "origen_id"])
    .size()
    .reset_index(name="demand_count")  # <--- VARIABLE OBJETIVO Y
)

# 3. Features temporales (Rezagos para evitar Data Leakage)
df_p1 = df_p1.sort_values(["origen_id", "hour_window"]).reset_index(drop=True)
df_p1["lag_1h"] = df_p1.groupby("origen_id")["demand_count"].shift(1)
df_p1["lag_24h"] = df_p1.groupby("origen_id")["demand_count"].shift(24)
df_p1["lag_168h"] = df_p1.groupby("origen_id")["demand_count"].shift(168)  # Misma hora semana anterior

# Variables de calendario
df_p1["hour"] = df_p1["hour_window"].dt.hour
df_p1["dayofweek"] = df_p1["hour_window"].dt.dayofweek
df_p1["is_weekend"] = df_p1["dayofweek"].isin([5, 6]).astype(int)

# Transformación cíclica
df_p1["hour_sin"] = np.sin(2 * np.pi * df_p1["hour"] / 24.0)
df_p1["hour_cos"] = np.cos(2 * np.pi * df_p1["hour"] / 24.0)

# Limpiar nulos originados por los rezagos
df_p1 = df_p1.dropna().reset_index(drop=True)
```

---

### Pregunta 2 (P2): Riesgo de Desbalance Crítico (Clasificación Binaria)

* **¿Qué predecimos ($y$)?** $1$ si la estación entra en riesgo de quedarse vacía o saturarse en la siguiente hora; $0$ si opera en equilibrio normal.
* **Aproximación de riesgo (Rúbrica oficial):**
  $$\Delta = \text{Entradas} - \text{Salidas}$$
  $$y = 1 \quad \text{si} \quad \frac{\vert{}\Delta\vert{}}{\text{dpcapacity}} \ge 0.50, \quad \text{de lo contrario } 0$$

```python
# 1. Truncar marcas horarias
df_trips["hour_start"] = df_trips["inicio_del_viaje"].dt.floor("h")
df_trips["hour_end"] = df_trips["fin_del_viaje"].dt.floor("h")

# 2. Conteo de salidas por estación
salidas = (
    df_trips.groupby(["hour_start", "origen_id"])
    .size()
    .reset_index(name="departures")
    .rename(columns={"hour_start": "timestamp", "origen_id": "station_id"})
)

# 3. Conteo de entradas por estación
entradas = (
    df_trips.groupby(["hour_end", "destino_id"])
    .size()
    .reset_index(name="arrivals")
    .rename(columns={"hour_end": "timestamp", "destino_id": "station_id"})
)

# 4. Consolidar flujo horario
df_flow = pd.merge(salidas, entradas, on=["timestamp", "station_id"], how="outer").fillna(0)
df_flow["net_flow"] = df_flow["arrivals"] - df_flow["departures"]

# 5. Cruzar con capacidad física instalada
df_flow = df_flow.merge(df_stations[["id", "dpcapacity"]], left_on="station_id", right_on="id", how="inner")

# 6. Construcción de la etiqueta objetivo (y)
df_flow["stress_ratio"] = df_flow["net_flow"].abs() / df_flow["dpcapacity"]
df_flow["is_stressed"] = (df_flow["stress_ratio"] >= 0.50).astype(int)  # <--- VARIABLE OBJETIVO Y (0 o 1)

# 7. Features y rezagos
df_flow = df_flow.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
df_flow["net_flow_lag_1h"] = df_flow.groupby("station_id")["net_flow"].shift(1)
df_flow["net_flow_lag_24h"] = df_flow.groupby("station_id")["net_flow"].shift(24)

df_p2 = df_flow.dropna().reset_index(drop=True)
```

---

### Pregunta 3 (P3): Perfiles de Estaciones (Clustering No Supervisado)

* **¿Qué agrupamos?** Las estaciones físicas del sistema según patrones de uso anual y geolocalización.
* **Unidad de análisis:** 1 fila por estación (no viajes ni horas).

```python
# 1. Marcas temporales de cada viaje
hour = df_trips["inicio_del_viaje"].dt.hour
is_morning = hour.between(7, 10)
is_evening = hour.between(17, 20)
is_weekend = df_trips["inicio_del_viaje"].dt.dayofweek.isin([5, 6])
duration_min = (df_trips["fin_del_viaje"] - df_trips["inicio_del_viaje"]).dt.total_seconds() / 60.0

df_trips["is_morning"] = is_morning
df_trips["is_evening"] = is_evening
df_trips["is_weekend"] = is_weekend
df_trips["duration_min"] = duration_min

# 2. Resumen anual por estación
profiles = df_trips.groupby("origen_id").agg(
    total_trips=("viaje_id", "count"),
    morning_trips=("is_morning", "sum"),
    evening_trips=("is_evening", "sum"),
    weekend_trips=("is_weekend", "sum"),
    avg_duration_min=("duration_min", "mean")
).reset_index()

# 3. Métricas de perfil relativo
profiles["morning_evening_ratio"] = (profiles["morning_trips"] + 1) / (profiles["evening_trips"] + 1)
profiles["weekend_ratio"] = profiles["weekend_trips"] / (profiles["total_departures"] if "total_departures" in profiles else profiles["total_trips"] + 1)

# 4. Tabla final consolidada
df_p3 = df_stations.merge(profiles, left_on="id", right_on="origen_id", how="inner")
df_p3["daily_turnover"] = (df_p3["total_trips"] / 365.0) / df_p3["dpcapacity"]

# Columnas listas para StandardScaler y KMeans:
features_p3 = df_p3[["latitude", "longitude", "morning_evening_ratio", "weekend_ratio", "daily_turnover", "avg_duration_min"]]
```

---

## 6. Partición Temporal Estricta (Sin Data Leakage)

Para evaluar **P1** y **P2**, utiliza esta función en tu notebook antes de entrenar cualquier modelo:

```python
def split_temporal(df: pd.DataFrame, time_col: str):
    """Partición cronológica obligatoria por rúbrica para evitar fuga de información."""
    train_limit = pd.Timestamp("2025-12-31 23:59:59")
    val_limit = pd.Timestamp("2026-02-28 23:59:59")
    
    train = df[df[time_col] <= train_limit].copy()
    val = df[(df[time_col] > train_limit) & (df[time_col] <= val_limit)].copy()
    test = df[df[time_col] > val_limit].copy()
    
    return train, val, test

# Ejemplo de uso:
# train_df, val_df, test_df = split_temporal(df_p1, time_col="hour_window")
# print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
```