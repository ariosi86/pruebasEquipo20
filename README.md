# TC5061 - Proyecto MiBici (Equipo 20)

Solución integral de Machine Learning para el análisis, modelado predictivo y segmentación funcional del sistema de bicicleta pública compartida **MiBici** (Área Metropolitana de Guadalajara). El proyecto resuelve tres problemáticas operativas centrales combinando regresión, clasificación y clustering bajo estándares de MLOps: programación orientada a objetos (POO), Pipelines formales de scikit-learn, versionado criptográfico de datos, serialización defensiva, interfaces CLI con Typer y seguimiento de experimentos con MLflow.

---

## 1. Alcance de las Preguntas del Proyecto

| Pregunta | Tipo de Problema | Objetivo de Negocio / Técnico | Algoritmo y Pipeline |
|---|---|---|---|
| **P1** | Regresión Temporal | Estimar el volumen de demanda horaria de viajes por estación para la hora inmediata ($t+1$). | Baseline: `StandardScaler` + `LinearRegression`; Campeón: `LightGBM Tweedie`. |
| **P2** | Clasificación Binaria | Predecir si una estación entrará en estado de desbalance crítico (vacía o saturada) en $t+1$. | Baseline: `StandardScaler` + `LogisticRegression` balanceada; Campeón: `HistGradientBoostingClassifier` (GBM). |
| **P3** | Clustering No Supervisado | Identificar perfiles funcionales de uso en estaciones y contrastar el agrupamiento contra la división municipal (línea base). | Baseline: agrupación por municipio; Campeón: `StandardScaler` + `KMeans` ($K=4$). |

---

## 2. Organización del Repositorio

La arquitectura del proyecto sigue la estructura estándar de **Cookiecutter Data Science** con modularización independiente por pregunta:

```text
├── LICENSE                      <- Licencia de código abierto del proyecto
├── Makefile                     <- Comandos automatizados de apoyo (make data, make train)
├── README.md                    <- Documentación principal del repositorio
├── pyproject.toml               <- Metadatos del paquete local y configuración de herramientas
├── requirements.txt             <- Librerías y versiones fijadas del entorno de ejecución
├── setup.cfg                    <- Configuración de linters de código (flake8, black)
│
├── data/
│   ├── external/                <- Datos provenientes de fuentes externas secundarias
│   ├── interim/                 <- Datos intermedios en proceso de transformación
│   ├── processed/               <- Parquets procesados canónicos (trips.parquet y stations.parquet)
│   │   ├── train/               <- Partición de entrenamiento (Año 2025 completo)
│   │   └── test/                <- Partición de prueba (Abril a Septiembre de 2026)
│   ├── raw/                     <- Archivos CSV mensuales originales de MiBici (inmutables)
│   └── manifest.json            <- Registro criptográfico SHA-256 para validación de datos
│
├── docs/                        <- Documentación técnica extendida, model cards y guías metodológicas
├── models/                      <- Modelos y pipelines serializados en formato .pkl
├── notebooks/                   <- Jupyter Notebooks de análisis exploratorio (P1, P2 y P3)
├── references/                  <- Diccionarios de datos y manuales explicativos
├── reports/                     <- Tablas comparativas de métricas (.csv) y reportes generados
│   └── figures/                 <- Gráficas exportadas para reportes
│
├── tests/                       <- Suite de pruebas de integración y validación con pytest
│   └── test_pipeline.py         <- Pruebas unitarias de artefactos, modelos y pipelines
│
└── proyecto_mibici/             <- Paquete principal de Python
    ├── __init__.py              <- Metadatos del módulo principal (__version__, __author__)
    ├── config.py                <- Rutas resueltas con pathlib, constantes, URLs y semillas
    ├── dataset.py               <- Pipeline CLI para descarga, limpieza inicial y exportación a Parquet
    ├── features.py              <- Fachada unificada que expone la API pública de características
    ├── features_p1.py           <- Malla horaria, rezagos (1h, 24h, 168h), ventanas móviles y variables cíclicas
    ├── features_p2.py           <- Balance por día operativo (inicio 03:00 hrs) y estrés relativo
    ├── features_p3.py           <- Perfiles anuales de uso, distribuciones horarias y balance neto
    ├── pipeline.py              <- Orquestador maestro End-to-End (Datos + Train + Predict + Testing)
    ├── plots.py                 <- Visualizaciones generadas sobre backend headless 'Agg'
    └── modeling/
        ├── __init__.py          <- Exposición de las clases POO de modelado
        ├── model_p1.py          <- Clase POO DemandaLinealP1 (StandardScaler + LinearRegression)
        ├── model_p2.py          <- Clase POO RiesgoLogisticoP2 (StandardScaler + LogisticRegression)
        ├── model_p3.py          <- Clase POO ClusterEstacionesP3 (StandardScaler + KMeans)
        ├── train_p1.py          <- Script CLI de entrenamiento, evaluación y registro en MLflow (P1)
        ├── train_p2.py          <- Script CLI de entrenamiento, evaluación y registro en MLflow (P2)
        ├── train_p3.py          <- Script CLI de entrenamiento, contraste baseline y MLflow (P3)
        ├── predict_p1.py        <- Script CLI para inferencia sobre demanda horaria (P1)
        ├── predict_p2.py        <- Script CLI para inferencia sobre riesgo de desbalance (P2)
        ├── predict_p3.py        <- Script CLI para inferencia de clusters funcionales (P3)
        └── tracking.py          <- Módulo de integración defensiva con MLflow Tracking
```

---

## 3. Descripción de los Módulos de Código (`proyecto_mibici/`)

- **`config.py`**: Centraliza rutas absolutas mediante `pathlib.Path`, URLs de descarga de datos abiertos, semillas aleatorias (`RANDOM_SEED`), hiperparámetros base (`P2_UMBRAL`, `P3_K_ELEGIDO`), nombres de columnas predictoras y configuración de conexión a MLflow (`URI_MLFLOW_TRACKING`, `MLFLOW_EXPERIMENT_NAME`).
- **`dataset.py`**: Ejecuta el flujo ETL primario. Descarga el catálogo de estaciones y los registros históricos de viajes, maneja codificaciones conflictivas (`utf-8` / `latin-1`), normaliza nombres de columnas, tipa identificadores y fechas, exporta los archivos maestros `trips.parquet`, `stations.parquet` junto con las particiones específicas de entrenamiento (`trips_train_2025.parquet`) y prueba (`trips_test_2026_04_09.parquet`), y construye el archivo `data/manifest.json` con hashes SHA-256.
- **`features.py` (Fachada) y `features_p1.py`, `features_p2.py`, `features_p3.py`**: Implementan transformadores orientados a objetos compatibles con scikit-learn (`BaseEstimator`, `TransformerMixin`) junto con sus funciones orquestadoras:
  * **`features_p1.py`**: `LimpiadorFiltroEstacionesP1` y `GeneradorDemandaLagsP1` construyen la serie horaria completa rellenando horas sin viajes con cero, calculan rezagos (`lag_1h`, `lag_24h`, `lag_168h`), promedios móviles (`rolling_3h`, `rolling_168h`) y codificación trigonométrica del ciclo diario (`sin`/`cos`).
  * **`features_p2.py`**: `LimpiadorFlujoP2` y `ConstructorSerieRiesgoP2` eliminan viajes anómalos (<10 segundos entre estaciones distintas), estructuran el día operativo con corte a las 03:00 hrs, calculan el balance acumulado respecto a la capacidad instalada (`dpcapacity`) y etiquetan el estado de riesgo para la hora $t+1$.
  * **`features_p3.py`**: `LimpiadorPerfilP3` depura duraciones fuera de rango (>180 min) y sintetiza el comportamiento anual de cada estación en proporciones de uso por bloque horario, porcentaje de viajes en fin de semana, duración media, volumen diario en escala logarítmica y balance neto relativo.
- **`pipeline.py`**: Orquestador maestro que automatiza todo el ciclo End-to-End del proyecto en secuencia (Ingesta ETL, entrenamiento de P1, P2 y P3 con registro en MLflow, inferencia CLI de prueba y suite de validación con `pytest`).
- **`plots.py`**: Módulo de visualización configurado con backend `Agg` (sin requerir monitor gráfico) que genera las gráficas de evaluación para MLflow: series temporales y dispersión real vs. predicho (P1), matrices de confusión y curvas Precision-Recall (P2), y mapas geográficos de estaciones coloreados por clúster o municipio (P3).
- **`modeling/model_p1.py`, `model_p2.py`, `model_p3.py`**: Clases POO (`DemandaLinealP1`, `RiesgoLogisticoP2`, `ClusterEstacionesP3`) que encapsulan objetos `sklearn.pipeline.Pipeline` uniendo el escalamiento `StandardScaler` con el estimador final, ofreciendo métodos estandarizados `.fit()`, `.predict()`, `.save()` y `.load()`.
- **`modeling/tracking.py`**: Gestiona la trazabilidad en MLflow registrando el hash SHA-256 de los datos, el hash del commit de Git, hiperparámetros, métricas desglosadas por partición (`Train`, `Validación`, `Test`), artefactos de figuras, tablas CSV y el modelo con su firma (`infer_signature`), guardando simultáneamente copias locales de las figuras en `reports/figures/`.

---

## 4. Instalación y Configuración del Entorno

### Requisitos previos
- **Python 3.10 o 3.11** instalado en el sistema operativo.
- **Git** configurado en la terminal.

### Paso 1: Clonar el proyecto
```bash
git clone https://github.com/ariosi86/TC5061-Proyecto-Equipo20.git
cd TC5061-Proyecto-Equipo20
```

### Paso 2: Crear el entorno virtual
```bash
python -m venv .venv
```

### Paso 3: Activar el entorno virtual

- **macOS y Linux:**
  ```bash
  source .venv/bin/activate
  ```

- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
  *(En caso de bloqueo por políticas de ejecución en PowerShell, ejecutar antes: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

- **Windows (Símbolo del sistema / CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

### Paso 4: Instalar dependencias y montar el paquete local
```bash
# 1. Actualizar pip
pip install --upgrade pip

# 2. Instalar dependencias fijadas en requirements.txt
pip install -r requirements.txt

# 3. Instalar el paquete local proyecto_mibici en modo editable
pip install -e .

# 4. Registrar el kernel para uso en Jupyter Notebooks
python -m ipykernel install --user --name=mibici-venv --display-name="Python (MiBici .venv)"
```

---

## 5. Descarga y Procesamiento de Datos (Ingesta Reproducible)

Con el entorno virtual activo `(.venv)`, ejecuta el módulo de ingesta desde la raíz del proyecto:

```bash
python -m proyecto_mibici.dataset
```

**Salidas generadas automáticamente:**
1. Descarga de los archivos CSV originales en `data/raw/`.
2. Generación de los conjuntos procesados en formato Apache Parquet:
   * Maestro general: `data/processed/trips.parquet`
   * Catálogo de estaciones: `data/processed/stations.parquet`
   * Entrenamiento (2025): `data/processed/train/trips_train_2025.parquet`
   * Prueba (Abr-Sep 2026): `data/processed/test/trips_test_2026_04_09.parquet`
3. Creación de `data/manifest.json` con los hashes criptográficos SHA-256, conteo total de registros y rango de fechas.

---

## 6. Entrenamiento y Registro en MLflow (`train_p*.py`)

Cada script de entrenamiento construye su tabla de características desde los archivos Parquet procesados, aplica la partición temporal correspondiente para evitar fuga de información (*data leakage*), ajusta el Pipeline, guarda el artefacto `.pkl` en `models/` y registra la corrida completa en MLflow.

### Requisitos de MLflow
Antes de ejecutar los entrenamientos, inicia el servidor local de MLflow en una terminal secundaria con el entorno virtual activo:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
```

*La URI de seguimiento está definida en `proyecto_mibici/config.py` mediante `URI_MLFLOW_TRACKING` (`http://localhost:5000` o `sqlite:///mlflow.db`). Puedes consultar los experimentos en el navegador ingresando a `http://localhost:5000`.*

---

### P1: Entrenamiento de Demanda Horaria (Regresión)
Ejecuta un modelo individual:

```bash
python -m proyecto_mibici.modeling.train_p1 --model <nombre> --lr <learning_rate> --alpha <alpha> --max_iter <max_iter>
```

Modelos disponibles:

| `--model` | Descripción | Etapa |
|---|---|---|
| `baseline` | Regresión Lineal con medias móviles (línea base) | Baseline |
| `poisson` | Regresión de Poisson | Candidato |
| `lgbm_poisson` | LightGBM objective=poisson | Candidato |
| `lgbm_p_search` | Búsqueda anidada de hiperparámetros (objective=poisson) | Search |
| `lgbm_tweedie` | LightGBM objective=tweedie (campeón) | Candidato |
| `lgbm_t_search` | Búsqueda anidada de hiperparámetros (objective=tweedie) | Search |

- **Parámetros registrados en MLflow:**
  * `pregunta`: `"P1"`
  * `modelo`: Nombre del modelo (`DemandaLinealP1`, `PoissonRegression`, `LightGBM_poisson`, `LightGBM_tweedie`, etc.)
  * `features`: Lista de rezagos (`lag_1h`, `lag_24h`, `lag_168h`), medias móviles (`rolling_3h`, `rolling_168h`) y variables de calendario (`target_hour_sin`, `target_hour_cos`, `target_is_weekend`).
  * `train_inicio`, `train_fin`, `validation_inicio`, `validation_fin`, `test_inicio`, `test_fin`: Límites cronológicos exactos de cada partición.
  * `datos_version`: Hash SHA-256 corto del dataset procesado.
- **Métricas registradas:** `train_mae`, `train_rmse`, `test_mae`, `test_rmse`, `test_wape`, `test_sesgo`.
- **Artefactos generados:**
  * Local: `models/p1_linear_model.pkl` (baseline) / `models/p1_poisson_model.pkl` / modelos LightGBM.
  * MLflow: Pipeline scikit-learn con firma de entrada/salida, `manifest.json`, gráfica de serie temporal y gráfico de dispersión global.

---

### P2: Entrenamiento de Riesgo de Desbalance (Clasificación)
Ejecuta un modelo individual:

```bash
python -m proyecto_mibici.modeling.train_p2 --modelo <nombre> --umbral <umbral> --class-weight <peso> --max-iter <iter>
```

Modelos disponibles:

| `--modelo` | Descripción | Etapa |
|---|---|---|
| `baseline` | Persistencia ingenua (línea base) | Baseline |
| `logistic` | Regresión Logística (balanced / None) | Candidato |
| `logistic_search` | Búsqueda anidada de hiperparámetros (logística) | Search |
| `gbm` | `HistGradientBoostingClassifier` (campeón) | Candidato |
| `gbm_search` | Búsqueda anidada de hiperparámetros (GBM) | Search |

- **Parámetros registrados en MLflow:**
  * `pregunta`: `"P2"`
  * `modelo`: `LogisticRegression` o `HistGradientBoostingClassifier`
  * `umbral`: Umbral de decisión de riesgo (por defecto 0.5).
  * `class_weight`: Peso de clases para Regresión Logística (`balanced` o `None`).
  * `max_iter`: Iteraciones máximas.
  * `train_inicio`, `train_fin`, `validation_inicio`, `validation_fin`, `test_inicio`, `test_fin` y `datos_version`.
- **Métricas registradas:** `train_precision`, `train_recall`, `train_f1_score`, `train_pr_auc`, `test_precision`, `test_recall`, `test_f1_score`, `test_pr_auc`.
- **Artefactos generados:**
  * Local: `models/p2_logistic_model.pkl` / `models/p2_gbm_model.pkl`.
  * MLflow: Pipeline scikit-learn con firma de datos, `manifest.json`, matriz de confusión y curva Precision-Recall.

---

### P3: Entrenamiento de Clustering de Estaciones (No Supervisado)
```bash
python -m proyecto_mibici.modeling.train_p3
```

- **Parámetros registrados en MLflow:**
  * `pregunta`: `"P3"`
  * `modelo`: Registra dos corridas para contraste directo: `"AgrupacionMunicipio"` (Línea Base geográfica) y `"ClusterEstacionesP3"` (KMeans funcional).
  * `n_clusters`: `4` ($K$ elegido).
  * `n_init`: `10`.
  * `semilla`: `RANDOM_SEED`.
  * `variables`: Proporciones por bloque horario (`pct_madrugada`, `pct_manana`, `pct_mediodia`, `pct_tarde`, `pct_noche`), `pct_fin_de_semana`, `duracion_media`, `viajes_dia_log` y `balance_neto`.
  * `min_viajes_estacion`: `100`.
  * `duracion_max_min`: `180`.
- **Métricas registradas:** `train_silhouette`, `train_davies_bouldin`, `test_silhouette`, `test_davies_bouldin` en ambas corridas.
- **Artefactos generados:**
  * Local: `models/p3_kmeans_model.pkl` y tabla comparativa `reports/p3_comparacion.csv`.
  * MLflow: Pipeline scikit-learn con firma de datos, mapa de estaciones por clúster KMeans (`mapa_grupos_kmeans.png`) y mapa de la línea base municipal (`mapa_grupos_baseline.png`).

---

## 7. Inferencia por Línea de Comandos (`predict_p*.py`)

Los scripts de inferencia cargan los Pipelines serializados desde `models/` (aplicando internamente el `StandardScaler` ajustado en entrenamiento) y generan predicciones sobre nuevos datos mediante una interfaz CLI construida con **Typer**.

### Parámetros de los Scripts de Inferencia

| Opción CLI | Tipo | Obligatorio | Aplica a | Descripción |
|---|---|---|---|---|
| `--input`, `-i` | `PATH` | **Sí** | P1, P2, P3 | Ruta al archivo `.parquet` o `.csv` con las variables de entrada del modelo. |
| `--output`, `-o` | `PATH` | No | P1, P2, P3 | Ruta de salida (`.parquet` o `.csv`) para guardar el DataFrame con las predicciones anexadas. Si se omite, imprime una vista previa de 10 filas en consola. |
| `--umbral`, `-u` | `FLOAT` | No | Solo P2 | Umbral de decisión entre `0.0` y `1.0` para clasificar una estación en riesgo a partir de la probabilidad predicha (valor por defecto: `0.5`). |

---

### Ejemplos de Ejecución de Inferencia

#### 1. Inferencia de Demanda Horaria (P1)
Anexa la columna `demanda_predicha` sobre un archivo con la matriz de características horarias (rezagos y componentes temporales):

```bash
# Guardar resultados en archivo Parquet
python -m proyecto_mibici.modeling.predict_p1 \
  --input reports/predicciones_p1_test.parquet

# Mostrar vista previa en terminal sin guardar archivo
python -m proyecto_mibici.modeling.predict_p1 \
  --input reports/predicciones_p1_test.parquet \
  --output reports/mis_predicciones_p1.parquet



#### 2. Inferencia de Riesgo de Desbalance (P2)
Anexa las columnas `prob_riesgo` (probabilidad continua) y `pred_riesgo` (etiqueta booleana evaluada contra `--umbral`):

```bash
# Inferencia con umbral estándar (0.50) y vista previa en terminal
python -m proyecto_mibici.modeling.predict_p2 \
  --input reports/predicciones_p2_test.parquet \
  --umbral 0.5

# Inferencia con umbral sensible (0.35) exportando a CSV
python -m proyecto_mibici.modeling.predict_p2 \
  --input reports/predicciones_p2_test.parquet \
  --umbral 0.35 \
  --output reports/predicciones_p2_sensible.csv
```

#### 3. Inferencia de Clusters Funcionales (P3)
Escala las características y asigna la etiqueta de clúster a partir del catálogo procesado de estaciones:

```bash
# Asignar grupos funcionales y mostrar distribución en terminal
python -m proyecto_mibici.modeling.predict_p3 \
  --input reports/predicciones_p3_test.csv

# Exportar asignación final de grupos a una ruta personalizada
python -m proyecto_mibici.modeling.predict_p3 \
  --input reports/predicciones_p3_test.csv \
  --output reports/estaciones_clusterizadas.csv
```

---

## 8. Orquestador End-to-End e Integración de Pruebas (`pipeline.py`)

El proyecto incluye un script orquestador maestro que automatiza todo el ciclo de vida del sistema en una sola ejecución. Se encarga de coordinar la ingesta ETL, los tres entrenamientos con registro en MLflow, las pruebas de inferencia CLI y la validación con `pytest`.

Este script se creo para ejecutar la descarga, los entrenamientos, los tests y la generación de reportes y figuras, además de enviar a mlflow las configuraciones e hiperparametros de los 3 modelos.

Se genero de manera única de probar los modelos y que todo el proyecto funcione con metadatos fijos desde el script

### Parámetros del Orquestador Maestro

| Bandera CLI | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `--skip-data` | Booleano | `False` | Omite la descarga y el procesamiento inicial de datos si los archivos Parquet ya existen en `data/processed/`. |
| `--skip-tests` | Booleano | `False` | Omite la ejecución automática de las pruebas de inferencia y la suite de `pytest`. |

### Comandos de Ejecución Integral

* **Ejecutar todo el flujo desde cero (Ingesta + Entrenamiento + Inferencia de prueba + Pytest):**
  ```bash
  python -m proyecto_mibici.pipeline
  ```


* **Ejecutar únicamente ingesta y entrenamiento (forzando regenerar los archivos parquet):**
  ```bash
  python -m proyecto_mibici.pipeline --force-data
  ```

* **Ejecutar únicamente entrenamiento, inferencia y tests (omitiendo la descarga de datos):**
  ```bash
  python -m proyecto_mibici.pipeline --skip-data
  ```

* **Ejecutar únicamente ingesta y entrenamiento (omitiendo las pruebas finales):**
  ```bash
  python -m proyecto_mibici.pipeline --skip-tests
  ```


## 9. Entrega 2: Nuevos modelos, MLflow Project y selección del mejor modelo

La Entrega 2 extiende la Entrega 1 con modelos adicionales para P1 (Poisson, LightGBM), búsquedas de hiperparámetros organizadas como nested runs en MLflow, el proyecto MLflow para reproducción completa, el orquestador `run_train.py` y el registro de los mejores modelos en el MLflow Model Registry.

---

### 9.1 Nuevos modelos candidatos en P1 (Demanda Horaria)

Además del baseline de Regresión Lineal, P1 incluye candidatos de Regresión de Poisson y LightGBM (Poisson y Tweedie), con búsquedas anidadas de hiperparámetros.

Ejecuta un modelo individual:

```bash
python -m proyecto_mibici.modeling.train_p1 --model <nombre> --lr <learning_rate> --alpha <alpha> --max_iter <max_iter>
```

| Nombre modelo | `--model` | Descripción |
|---|---|---|
| Baseline (lineal v2) | `baseline` | Regresión Lineal con medias móviles |
| Poisson lineal (candidato) | `poisson` | Regresión de Poisson con `lr=0.05`, `alpha=0.0001`, `max_iter=300` |
| LightGBM-Poisson (candidato) | `lgbm_poisson` | LightGBM con objective=poisson |
| LightGBM-Poisson (nested search) | `lgbm_p_search` | Búsqueda anidada de hiperparámetros (objective=poisson) |
| LightGBM-Tweedie (candidato) | `lgbm_tweedie` | LightGBM con objective=tweedie |
| LightGBM-Tweedie (nested_search) | `lgbm_t_search` | Búsqueda anidada de hiperparámetros (objective=tweedie) |

Tags registrados en MLflow por modelo:

| Modelo | `model_family` | `stage_of_work` | Extra |
|---|---|---|---|
| baseline | linear | baseline | `baseline=true` |
| poisson | poisson_regression | candidato | |
| lgbm_poisson | candidato | lightgbm | |
| lgbm_p_search | search | lightgbm | |
| lgbm_tweedie | candidato | lightgbm | |
| lgbm_t_search | search | lightgbm | |

> **macOS:** la librería `lightgbm` requiere OpenMP para soporte multihilo. Instala con Homebrew:
> ```bash
> brew install libomp
> ```

---

### 9.2 MLflow Project y tracking avanzado

El proyecto está empaquetado como un **MLflow Project** para reproducir la ingesta y los entrenamientos desde cualquier entorno.

**Archivos del proyecto MLflow:**
- `MLproject`: define los entry points `ingest`, `train_p1`, `train_p2`, `train_p3` y `train`, con sus parámetros y variables de entorno.
- `python_env.yaml`: definición del entorno Python con las dependencias fijas.
- `run_train.py`: orquestador CLI que lanza todas las versiones de una pregunta mediante `mlflow run`.

#### 9.2.1 Levantar el servidor de MLflow

Desde la raíz del repositorio, con el entorno virtual activado:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
```

Accede a la interfaz en `http://localhost:5000`. En la terminal de trabajo define la URI:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

> **Importante:** antes de ejecutar cualquier comando `mlflow run`, exporta la URI de tracking en tu shell:
> ```bash
> export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
> ```
> O, si creaste el archivo `.env`, cárgalo antes:
> ```bash
> set -a && source .env && set +a
> ```
> Sin esta variable, `mlflow run` usa el file store local por defecto y los runs no quedan registrados en el servidor `http://127.0.0.1:5000`.
> Los hiperparámetros siempre se pasan como `-P clave=valor`. Los scripts usan `URI_MLFLOW_TRACKING` desde `config.py` para registrar parámetros, métricas y artefactos en MLflow.

#### 9.2.2 Reproducir la ingesta de datos

```bash
mlflow run . -e ingest --env-manager local --experiment-name mibici-equipo20
```

Equivalente a `python -m proyecto_mibici.dataset`.

#### 9.2.3 Entrenar modelos por pregunta

Hay dos formas equivalentes: con `run_train.py` o con `mlflow run` directamente. Ambos quedan registrados en MLflow con `source_type=PROJECT`.

**P1 — Demanda horaria (regresión):**

```bash
mlflow run . -e train_p1 --env-manager local --experiment-name mibici-equipo20 -P modelo=baseline
mlflow run . -e train_p1 --env-manager local --experiment-name mibici-equipo20 -P modelo=poisson -P lr=0.05 -P alpha=0.0001 -P max_iter=300
mlflow run . -e train_p1 --env-manager local --experiment-name mibici-equipo20 -P modelo=lgbm_tweedie
mlflow run . -e train_p1 --env-manager local --experiment-name mibici-equipo20 -P modelo=lgbm_t_search
mlflow run . -e train_p1 --env-manager local --experiment-name mibici-equipo20 -P modelo=lgbm_p_search
```

**P2 — Riesgo de desbalance (clasificación):**

```bash
mlflow run . -e train_p2 --env-manager local --experiment-name mibici-equipo20 -P modelo=baseline
mlflow run . -e train_p2 --env-manager local --experiment-name mibici-equipo20 -P modelo=logistic -P umbral=0.5 -P class_weight=balanced -P max_iter=1000
mlflow run . -e train_p2 --env-manager local --experiment-name mibici-equipo20 -P modelo=logistic -P umbral=0.5 -P class_weight=None -P max_iter=1000
mlflow run . -e train_p2 --env-manager local --experiment-name mibici-equipo20 -P modelo=logistic_search
mlflow run . -e train_p2 --env-manager local --experiment-name mibici-equipo20 -P modelo=gbm
mlflow run . -e train_p2 --env-manager local --experiment-name mibici-equipo20 -P modelo=gbm_search
```

**P3 — Clustering de estaciones (no supervisado):**

```bash
mlflow run . -e train_p3 --env-manager local --experiment-name mibici-equipo20 -P modelo=baseline
mlflow run . -e train_p3 --env-manager local --experiment-name mibici-equipo20 -P modelo=kmeans -P n_clusters=3 -P n_init=10
mlflow run . -e train_p3 --env-manager local --experiment-name mibici-equipo20 -P modelo=kmeans -P n_clusters=4 -P n_init=10
mlflow run . -e train_p3 --env-manager local --experiment-name mibici-equipo20 -P modelo=kmeans -P n_clusters=5 -P n_init=10
mlflow run . -e train_p3 --env-manager local --experiment-name mibici-equipo20 -P modelo=kmeans -P n_clusters=6 -P n_init=10
mlflow run . -e train_p3 --env-manager local --experiment-name mibici-equipo20 -P modelo=kmeans_search -P n_init=10
```

#### 9.2.4 Flujo completo recomendado

```bash
# 1. Levantar MLflow (terminal aparte)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000

# 2. En la terminal de trabajo
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# 3. Ingesta de datos
mlflow run . -e ingest --env-manager local --experiment-name mibici-equipo20

# 4. Entrenar modelos para producción (campeones)
python run_train.py train --pregunta p1 --versiones baseline,lgbm_tweedie
python run_train.py train --pregunta p2 --versiones baseline,gbm
python run_train.py train --pregunta p3 --versiones baseline,kmeans_k4
```

Cada run queda registrado en MLflow con sus parámetros, métricas, etiquetas (git commit, versión de datos, equipo) y artefactos (modelo con firma, figuras, `manifest.json`). El campo `Source` es consistente (`source_type=PROJECT`) en todas las corridas lanzadas vía `mlflow run`.

---

### 9.3 Orquestador `run_train.py`

`run_train.py` es un **atajo CLI (Typer)** para ejecutar todos los entrenamientos de una pregunta en un solo comando, en lugar de lanzar cada `mlflow run` uno a uno. **No contiene lógica de entrenamiento**: solo despacha cada versión candidata como un proyecto MLflow independiente mediante `mlflow run . -e train_pX -P ...`.

El entry point de MLflow (`train_p1`, `train_p2`, `train_p3`) invoca directamente al módulo de entrenamiento correspondiente, evitando recursión.

**Comando:**

```bash
python run_train.py train --pregunta <p1|p2|p3> [--versiones <lista>]
```

`--versiones` acepta `todos` (por defecto) o una lista separada por comas.

**Versiones candidatas por pregunta:**

| Pregunta | Versiones |
|---|---|
| P1 | baseline, poisson, lgbm_poisson, lgbm_poisson_search, lgbm_tweedie, lgbm_tweedie_search |
| P2 | baseline, logistic_balanced, logistic_sin_peso, logistic_umbral_sensible, logistic_search, gbm, gbm_search |
| P3 | baseline, kmeans_search, kmeans_k3, kmeans_k4, kmeans_k5, kmeans_k6 |

**Ejemplos:**

```bash
# Todas las versiones de P2
python run_train.py train --pregunta p2

# Solo baseline y el mejor candidato de P2
python run_train.py train --pregunta p2 --versiones baseline,gbm

# Búsqueda anidada de P2
python run_train.py train --pregunta p2 --versiones logistic_search
```

**Requisitos previos:**
1. Servidor de MLflow levantado (terminal aparte).
2. Datos procesados en `data/processed/` (generados con `python -m proyecto_mibici.dataset` o `mlflow run . -e ingest`).
3. Dependencias instaladas (`pip install -r requirements.txt`).
4. Todos los comandos `mlflow run` requieren `--env-manager local --experiment-name mibici-equipo20` porque MLflow 3.x intenta crear un entorno aislado con pyenv; estas flags hacen que use el `.venv` existente.

---

### 9.4 Registro en el Model Registry (`run_mejor_modelo.py`)

`run_mejor_modelo.py` registra en el MLflow Model Registry el modelo elegido de cada pregunta, conservando su linaje (experimento, corrida de origen, commit y versión de datos) y agregando una descripción a cada modelo.

**Esquema de ciclo de vida:** cada versión registrada se promueve al stage `Production` y se le asigna el alias `champion`, lo que permite cargar el modelo en producción por nombre y alias:

```python
import mlflow

# P1
modelo_p1 = mlflow.pyfunc.load_model("models:/mibici-equipo20-p1-demanda@champion")
# P2
modelo_p2 = mlflow.pyfunc.load_model("models:/mibici-equipo20-p2-riesgo@champion")
# P3
modelo_p3 = mlflow.pyfunc.load_model("models:/mibici-equipo20-p3-grupos@champion")
```

**Modelos registrados (1 campeón por pregunta):**

| Pregunta | Nombre en el Registry | Modelo elegido | Métrica de decisión (Test) | Justificación |
|---|---|---|---|---|
| P1 | `mibici-equipo20-p1-demanda` | LightGBM Regresión Tweedie | MAE 0.789 | Mejor MAE de validación (0.816) y prueba (0.789); −7.5 % vs baseline lineal (0.853). |
| P2 | `mibici-equipo20-p2-riesgo` | HistGradientBoostingClassifier (GBM), umbral 0.5 | F1 0.865 / PR-AUC 0.942 | Único candidato que supera al baseline de persistencia (F1 0.833); Regresión Logística se queda en F1 0.832 (no supera al baseline). |
| P3 | `mibici-equipo20-p3-grupos` | KMeans funcional K=4 | Silhouette 0.182 | Mejor Silhouette de prueba (0.182) y menor Davies-Bouldin (1.535) entre k=3..6; supera al baseline municipal (0.019 / 3.712). |

**Ejecución:**

```bash
python run_mejor_modelo.py
```

El script es idempotente: si ya existe una versión para la misma corrida de origen, la reusa y actualiza su descripción y tags en lugar de duplicarla.

**Comandos `mlflow run` para reproducir cada campeón desde un clon limpio:**

```bash
# 1. Levantar MLflow (terminal aparte)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# 2. Ingesta de datos
mlflow run . -e ingest --env-manager local --experiment-name mibici-equipo20

# 3. Entrenar campeones
mlflow run . -e train_p1 --env-manager local --experiment-name mibici-equipo20 -P modelo=lgbm_tweedie
mlflow run . -e train_p2 --env-manager local --experiment-name mibici-equipo20 -P modelo=gbm
mlflow run . -e train_p3 --env-manager local --experiment-name mibici-equipo20 -P modelo=kmeans -P n_clusters=4 -P n_init=10

# 4. Registrar campeones en el Model Registry
python run_mejor_modelo.py
```

**Verificación del alias champion en la UI de MLflow:**

Cada modelo registrado aparece en la sección **Models** de MLflow con:
- Alias `champion` → versión en producción (stage `Production`).
- Tags de linaje: `run_origen`, `git_commit`, `datos_version`, `pregunta`, `metrica_test`, `valor_metrica_test`.
- Descripción con el rendimiento frente al baseline y el costo de entrenamiento/inferencia.

---

## Pruebas Automatizadas y Gobernanza de Código

Para garantizar la calidad, reproducibilidad y seguridad del proyecto, el repositorio cuenta con un conjunto de pruebas automatizadas con **PyTest** y chequeos de seguridad.

### Estructura de Pruebas

#### P1


Las pruebas se ubican en la carpeta `tests/` organizadas por componentes:
```text
tests/
├── conftest.py            # datos sintéticos reutilizables (fixtures)
├── test_features_p1.py    # ingeniería de características de P1
├── test_model_p1.py       # modelos de P1 (baseline, lineal, Poisson)
├── test_train_p1.py       # métricas de evaluación y reproducibilidad del entrenamiento
└── test_predict_p1.py     # inferencia: carga de modelos y generación de predicciones
```
- `test_features_p1.py`: Limpieza de datos y generación de rezagos/malla horaria.
- `test_model_p1.py`: Reglas de negocio y restricciones de los modelos.
- `test_predict_p1.py`: Pipeline de inferencia y generación de reportes.
- `test_train_p1.py`: Pruebas de reproducibilidad con semillas fijas y métricas.

#### P3

Las pruebas de P3 se ubican en la carpeta `tests/` de la raíz:
```text
tests/
├── test_features_p3.py           # limpieza de viajes y perfil funcional de estaciones
├── test_model_p3.py              # modelos de P3 (baseline por municipio y KMeans)
├── test_reproducibilidad_p3.py   # reproducibilidad del entrenamiento de KMeans
└── test_deepchecks_p3.py         # validación de datos con DeepChecks (dataset real)
```
- `test_features_p3.py`: Limpieza de viajes (duplicados, duración y catálogo) y construcción del perfil de cada estación.
- `test_model_p3.py`: Asignación de grupos del baseline y de KMeans, errores esperados y guardado/carga del modelo.
- `test_reproducibilidad_p3.py`: Reproducibilidad de KMeans con semilla fija y control con otra semilla.
- `test_deepchecks_p3.py`: Integridad del perfil de estaciones y drift entre entrenamiento (2025) y prueba (may-sep 2026).

### Instrucciones de Ejecución

1. **Ejecutar suite completa de pruebas:**
```bash
pip install pytest          # (o: pip install -r requirements.txt)
pytest -v                   # todas las pruebas, con detalle
pytest proyecto_mibici/tests/test_features_p1.py -v #prueba por archivo
pytest proyecto_mibici/tests/test_model_p1.py -v
pytest proyecto_mibici/tests/test_predict_p1.py -v
pytest proyecto_mibici/tests/test_train_p1.py -v
pytest -k "reproducible"                  # solo pruebas cuyo nombre contenga la palabra
```

> Las pruebas **no usan el dataset real de MiBici**. Usan DataFrames pequeños y sintéticos definidos en `tests/conftest.py`, por lo que corren en pocos segundos y no dependen de archivos externos ni de MLflow.

3. **Ejecutar las pruebas de P3:**
```bash
pip install deepchecks==0.19.1 "scikit-learn>=1.5.1,<1.8" "setuptools<81"
pytest tests -v -k p3                        # todas las pruebas de P3
pytest tests/test_features_p3.py -v          # prueba por archivo
pytest tests/test_model_p3.py -v
pytest tests/test_reproducibilidad_p3.py -v
pytest tests/test_deepchecks_p3.py -v
```

> Las pruebas unitarias y de reproducibilidad de P3 usan datos sintéticos y no necesitan MLflow. Solo `test_deepchecks_p3.py` usa el dataset real (ver *Validación de datos (DeepChecks)*). DeepChecks 0.19.1 no importa con scikit-learn 1.8 o superior y requiere `setuptools<81`.

### Qué se prueba (P1: demanda horaria)

| Archivo | Componente | Casos cubiertos |
|---|---|---|
| `test_features_p1.py` | `LimpiadorFiltroEstacionesP1` | Elimina duplicados; descarta estaciones fuera de catálogo; descarta estaciones con exceso de horas vacías; falla si hay `viaje_id` repetido con datos distintos |
| `test_features_p1.py` | `GeneradorDemandaLagsP1` | Crea las columnas esperadas; sin nulos en rezagos ni target; rellena con 0 las horas sin viajes; semántica correcta de `lag_24h` y `lag_168h`; variables cíclicas en rango [-1, 1]; no modifica la entrada; resultado determinista; con pocos datos devuelve vacío sin error |
| `test_model_p1.py` | `BaselineIngenuoP1` | Estrategias `persistencia` y `promedio_24h_168h`; nulos tratados como 0; nunca predice negativos; estrategia inválida lanza `ValueError` |
| `test_model_p1.py` | `DemandaLinealP1` y `DemandaPoissonP1` | `fit`/`predict` con forma correcta y sin negativos; el pipeline expone el `scaler`; usa solo las `features` indicadas; falla si faltan columnas; `save`/`load` conserva las predicciones; la regresión lineal recupera una relación exacta |
| `test_train_p1.py` | `evaluar_metricas_regresion` | MAE, RMSE, WAPE y sesgo verificados contra valores calculados a mano; caso con reales en cero |
| `test_train_p1.py` | `calcular_metricas_por_estacion` y `registrar_fila_comparacion` | Una fila por estación; error absoluto total; unión con el catálogo de nombres; no modifica la entrada |
| `test_train_p1.py` | **Reproducibilidad del entrenamiento** | 	Con la misma semilla, LightGBM (objetivos Poisson y Tweedie), regresión lineal y Poisson producen predicciones idénticas; con submuestreo activo, otra semilla produce resultados distintos; las métricas de dos entrenamientos coinciden |
| `test_predict_p1.py` | `predecir_todos_los_modelos`, `generar_reporte_inferencia`, `cargar_modelo` | Columnas esperadas (4 baselines, v1, v2 y alias `demanda_predicha`); longitud correcta, sin NaN ni negativos; el alias coincide con v2; un modelo guardado y cargado predice igual que el original; fallback a `p1_linear_model.pkl`; error claro si falta el modelo o faltan columnas; no modifica la entrada; resultado determinista |

> Las pruebas de inferencia guardan los modelos en una carpeta temporal (`tmp_path`), por lo que no tocan el directorio `models/` real.

### Qué se prueba (P3: clustering de estaciones)

| Archivo | Componente | Casos cubiertos |
|---|---|---|
| `test_features_p3.py` | `limpiar_viajes_p3` | Calcula la duración en minutos; elimina viajes con `viaje_id` duplicado; elimina viajes de 0 min y de más de 180 min, y conserva el de 180 min exactos; elimina viajes con origen o destino fuera del catálogo |
| `test_features_p3.py` | `construir_perfil_estaciones` | Genera todas las variables de `P3_VARIABLES` con una fila por estación; las proporciones por rango horario suman 1; valores verificados a mano (`pct_manana`, `pct_fin_de_semana`, `duracion_media`, `balance_neto`); `pct_fin_de_semana` en [0, 1] y `balance_neto` en [-1, 1]; descarta estaciones con menos viajes que el mínimo; crea en 0 la columna de un rango horario sin viajes |
| `test_model_p3.py` | `BaselineMunicipioP3` | Un grupo por municipio en orden; usa el catálogo de estaciones si falta la columna; municipio no visto recibe -1; falla con `KeyError` sin municipio ni catálogo y con `ValueError` si predice sin entrenar |
| `test_model_p3.py` | `ClusterKMeansP3` | Una etiqueta válida por estación; recupera cuatro grupos sintéticos bien separados (ARI = 1); el `StandardScaler` está dentro del pipeline y aprende la media de entrenamiento; usa solo las variables indicadas, sin importar columnas extra ni su orden; falla con `ValueError` si predice sin entrenar; `save`/`load` conserva las etiquetas |
| `test_reproducibilidad_p3.py` | **Reproducibilidad del entrenamiento** | Dos entrenamientos de KMeans con `RANDOM_SEED` producen etiquetas, centros, silhouette y Davies-Bouldin idénticos; con otra semilla los centros cambian (control) |

> **Hallazgo:** la prueba `test_perfil_con_un_rango_horario_sin_viajes` detectó que `pd.crosstab` solo creaba columnas para los rangos horarios presentes en los datos. En un periodo sin viajes nocturnos, `pct_noche` no existía y el entrenamiento fallaba con `KeyError`. Se corrigió en `features_p3.py` con `reindex` sobre los cinco rangos horarios; con los datos completos el resultado no cambia.

### Reproducibilidad

El entrenamiento usa `RANDOM_SEED` (definido en `config.py`). Las pruebas de la sección *Reproducibilidad del entrenamiento* entrenan el mismo modelo dos veces con la misma semilla y verifican que las predicciones sean exactamente iguales (`np.testing.assert_array_equal`).

En P3, el azar de KMeans está en la posición inicial de los centros. `test_reproducibilidad_p3.py` usa perfiles sintéticos sin grupos marcados y un solo arranque (`n_init=1`), que es el caso en que la semilla decide el resultado. Con `RANDOM_SEED` dos entrenamientos dan etiquetas, centros y métricas idénticos; con otra semilla los centros cambian, lo que confirma que la igualdad se debe a la semilla.

### Resultados esperados
Resultado esperado: todas las pruebas en verde (PASSED) al final.

#### test_features_p1.py - 13 pruebas aprobadas

![Ejecución de PyTest](docs/images/test_features_p1.png)

#### test_train_p1.py - 18 pruebas aprobadas

![Ejecución de PyTest](docs/images/test_train_p1.png)


#### test_model_p1.py - 17 pruebas aprobadas

![Ejecución de PyTest](docs/images/test_model_p1.png)

### test_predict_p1.py - 9 pruebas aprobadas

![Ejecución de PyTest](docs/images/test_predict_p1.png)


#### P3 - 28 pruebas aprobadas

![Ejecución de PyTest P3](docs/images/evidencia_pytest_p3.png)

### Validación de datos (DeepChecks)

_Pendiente: documentar aquí los chequeos de integridad y de drift entre train/test, y cómo ejecutarlos._

#### P3

`tests/test_deepchecks_p3.py` valida el perfil **real** de estaciones que entra a KMeans, construido con las mismas funciones y fechas de `config.py` que `train_p3.py`: entrenamiento con 2025 y prueba con mayo a septiembre de 2026. P3 no tiene variable objetivo, así que los tests usan solo las ocho variables de `P3_VARIABLES`. Si los Parquet de `data/processed/` no existen, las pruebas se omiten (SKIPPED).

```bash
pytest tests/test_deepchecks_p3.py -v
```

| Prueba | Qué valida |
|---|---|
| `test_integridad_del_perfil_p3[entrenamiento]` | Sin nulos, sin tipos mezclados, ninguna variable constante y ninguna estación con perfil duplicado |
| `test_integridad_del_perfil_p3[prueba]` | Lo mismo para el perfil de prueba |
| `test_reglas_de_dominio_p3` | Una fila por estación; proporciones horarias que suman 1; `pct_fin_de_semana` en [0, 1]; `balance_neto` en [-1, 1]; duración media entre 0 y 180 min |
| `test_estaciones_de_2025_siguen_operando_en_2026` | Al menos 95 % de las estaciones de entrenamiento siguen activas en prueba |
| `test_sin_drift_en_las_estaciones_comunes` | Drift por variable (Kolmogorov-Smirnov < 0.2) y multivariado (< 0.25) entre los dos periodos, sobre las estaciones presentes en ambos |

**Resultados:**
- Perfiles limpios en ambos periodos: 0 nulos, 0 duplicados y ninguna variable constante.
- 371 estaciones en entrenamiento y 469 en prueba. De las 371 de 2025, 370 siguen operando en 2026; las 99 restantes de prueba son estaciones nuevas de la expansión de MiBici.
- Sistema completo (reporte informativo, sin condición): ninguna variable supera 0.2 por separado (máximo `pct_fin_de_semana`, 0.16), pero el drift multivariado es 0.33 (AUC del clasificador de dominio 0.66). Las variables que más distinguen los periodos son `viajes_dia_log` (36 %), `balance_neto` (28 %) y `pct_noche` (25 %), consistentes con estaciones recién abiertas y con menos viajes.
- Estaciones comunes (prueba con condición): máximo drift por variable 0.16 en `duracion_media` y drift multivariado 0.23 (AUC 0.62), debajo del límite de 0.25. Las 370 estaciones conservan su forma de uso, así que los grupos aprendidos en 2025 siguen siendo válidos para ellas.
- El drift multivariado de las estaciones comunes queda cerca del límite, impulsado por `duracion_media` y `balance_neto`. Conviene vigilarlo en el monitoreo de drift de la semana 10.

DeepChecks guarda cuatro reportes HTML en `reports/deepchecks/` (`p3_integridad_entrenamiento.html`, `p3_integridad_prueba.html`, `p3_drift_todas_las_estaciones.html` y `p3_drift_estaciones_comunes.html`), que se sobrescriben en cada corrida y no se suben al repositorio porque `reports/` está en el `.gitignore`.

![DeepChecks: integridad del perfil de estaciones de P3 (entrenamiento)](docs/images/evidencia_deepchecks_integridad_p3.png)

![DeepChecks: drift de P3 en las estaciones presentes en ambos periodos](docs/images/evidencia_deepchecks_drift_p3.png)

### Chequeos de seguridad

_Pendiente: documentar aquí el uso de `.gitignore`, variables de entorno y la salida de `bandit` / `pip-audit`._

Auditoría de seguridad de código
```bash
bandit -r src/
```

![Auditoría con Bandit](docs/images/evidencia_bandit.png)

Auditoría de dependencias
```bash
pip-audit
```
![Auditoría con Audit](docs/images/evidencia_audit.png)
### Evidencia de ejecución

_Pendiente: agregar capturas de `pytest -v` con todas las pruebas en verde (por ejemplo en `reports/figures/`)._