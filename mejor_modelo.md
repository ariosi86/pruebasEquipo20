# Selección del Mejor Modelo por Pregunta — MiBici Equipo 20

Este documento resume la comparación de modelos ejecutada con `run_train.py` (todas las preguntas P1, P2, P3), registrada en la interfaz de MLflow (`http://127.0.0.1:5000`, experimento `mibici-equipo20`), y justifica el modelo que pasará a producción por cada pregunta de negocio.

## 1. Preguntas del proyecto integrador

| # | Pregunta de negocio | Tipo de tarea | Quién la usaría |
|---|---------------------|---------------|-----------------|
| P1 | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión | Operador del sistema, para planear la flota. |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro? | Clasificación | Equipo de rebalanceo de bicicletas. |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad? | Clustering | Planeación urbana y expansión del sistema. |

## 2. Modelo seleccionado por pregunta (para producción)

| Pregunta | Modelo elegido | Nombre en Model Registry | Métrica de decisión (Test) | Baseline (Test) | Mejora vs. baseline |
|----------|----------------|--------------------------|----------------------------|-----------------|---------------------|
| P1 | LightGBM Regresión Tweedie (`lgbm_tweedie`) | `mibici-equipo20-p1-demanda` | Test MAE **0.790** | 0.853 (lineal + rolling) | −7.3 % MAE / −7.4 % WAPE |
| P2 | Regresión Logística sin peso de clases, umbral 0.5 (`logistic_sin_peso`) | `mibici-equipo20-p2-riesgo` | Test F1 **0.832** / PR-AUC **0.806** | F1 ~0.79 (logístico balanceado) | +5 pts F1 / +0.7 pts PR-AUC |
| P3 | KMeans funcional con K=4 | `mibici-equipo20-p3-grupos` | Test Silhouette **0.182** / DB **1.535** | Silhouette 0.019 / DB 3.71 (municipio) | +0.163 Silhouette / −2.18 DB |

---

## 3. Tablas comparativas por pregunta

### 3.1 P1 — Demanda horaria (Regresión)

Métrica de negocio: error absoluto en la estimación de viajes para la hora $t+1$ (el operador planea la flota; subestimar genera falta de bicis, sobreestimar genera ociosidad). Se usa **MAE** y **WAPE** como métricas principales, con **RMSE** y **Sesgo** como secundarias.

| Modelo | Hiperparámetros clave | Validación MAE | Validación WAPE | Test MAE | Test RMSE | Test WAPE | Test Sesgo | Mejora MAE vs baseline |
|--------|----------------------|----------------|-----------------|----------|-----------|-----------|------------|------------------------|
| Baseline (lineal + rolling, v2) | `StandardScaler` + `LinearRegression`, features V2 (lags 1/24/168h, rolling 3/168h, ciclor diario, fin de semana) | 0.886 | 0.663 | **0.853** | 1.545 | 0.681 | 0.004 | — (referencia) |
| v1_lineal (sin rolling) | `StandardScaler` + `LinearRegression`, features V1 (sin rolling) | — | — | 0.865 | 1.558 | 0.691 | 0.009 | −1.4 % |
| Regresión de Poisson | `PoissonRegressor(alpha=1e-4, max_iter=300)`, lr=0.05 | — | — | 1.181 | 4.341 | 0.943 | 0.137 | −38.4 % (peor) |
| LightGBM Poisson | `objective=poisson`, lr=0.1, num_leaves=127, n_estimators=600 | 0.820 | 0.614 | **0.792** | 1.422 | 0.632 | −0.018 | +7.3 % |
| LightGBM Tweedie (ELEGIDO) | `objective=tweedie`, tweedie_variance_power=1.5, lr=0.1, num_leaves=127, n_estimators=600 | 0.818 | 0.612 | **0.790** | 1.410 | 0.631 | −0.023 | **+7.3 %** |

### 3.2 P2 — Riesgo de desbalance (Clasificación)

Métrica de negocio: el equipo de rebalanceo actúa sobre la predicción de riesgo. Un **falso negativo** (no predecir riesgo) deja usuarios sin bici o sin anclaje (impacto directo al servicio); un **falso positivo** genera un rebalanceo innecesario (costo operativo). Se prioriza un equilibrio entre **Precisión** (pocas alarmas falsas) y **Recall** (capturar la mayoría de riesgos), resumido en **F1**, con **PR-AUC** para robustez ante clases desbalanceadas.

| Modelo | Hiperparámetros clave | Validación F1 | Validación PR-AUC | Test Precisión | Test Recall | Test F1 | Test PR-AUC | Mejora F1 vs baseline logístico |
|--------|----------------------|---------------|-------------------|----------------|-------------|---------|-------------|---------------------------------|
| Baseline (naive persistencia) | Regla `en_riesgo_t` → `en_riesgo_t+1` | — | — | — | — | — | — | referencial (menor) |
| Logístico balanceado | `class_weight=balanced`, umbral=0.5, max_iter=1000 | 0.781 | 0.787 | 0.745 | 0.850 | 0.794 | 0.799 | — (referencia) |
| Logístico sin peso (ELEGIDO) | `class_weight=None`, umbral=0.5, max_iter=1000 | 0.823 | 0.794 | **0.831** | **0.832** | **0.832** | **0.806** | **+4.8 pts** |
| Logístico umbral 0.3 | `class_weight=balanced`, umbral=0.3, max_iter=1000 | 0.694 | 0.787 | 0.589 | 0.881 | 0.706 | 0.799 | −8.8 pts |

### 3.3 P3 — Clustering de estaciones (No supervisado)

Métrica de negocio: identificar grupos funcionales de estaciones para planeación urbana y expansión. Se valida la calidad de la partición con **Silhouette Score** (mayor = mejor separación) y **Davies-Bouldin** (menor = mejor). El baseline es la agrupación administrativa por municipio.

| Modelo | Hiperparámetros clave | Train Silhouette | Train Davies-Bouldin | Test Silhouette | Test Davies-Bouldin | Mejora Silhouette vs baseline |
|--------|----------------------|------------------|----------------------|-----------------|---------------------|-------------------------------|
| Baseline (municipio) | Agrupación por `municipality_code` | 0.050 | 3.348 | 0.019 | 3.712 | — (referencia) |
| KMeans K=3 | `n_clusters=3`, `n_init=10`, StandardScaler | — | — | 0.167 | 1.817 | +0.148 |
| KMeans K=4 (ELEGIDO) | `n_clusters=4`, `n_init=10`, StandardScaler | — | — | **0.182** | **1.535** | **+0.163** |
| KMeans K=5 | `n_clusters=5`, `n_init=10`, StandardScaler | — | — | 0.149 | 1.801 | +0.130 |
| KMeans K=6 | `n_clusters=6`, `n_init=10`, StandardScaler | — | — | 0.141 | 1.780 | +0.122 |

---

## 4. Análisis de la comparación y justificación del modelo elegido

### P1 — Demanda horaria → **LightGBM Tweedie**
- **Desempeño vs. baseline:** LightGBM Tweedie reduce el Test MAE de 0.853 a 0.790 (−7.3 %) y el WAPE de 0.681 a 0.631 (−7.4 %). LightGBM Poisson es casi idéntico (0.792 / 0.632), pero Tweedie es marginalmente mejor y más adecuado para conteos con muchos ceros (horas sin viajes), propia de la demanda horaria por estación.
- **Métrica ligada a la decisión:** El operador planea la flota; un error sistemático de subestimación deja la flota corta. Tweedie tiene sesgo casi nulo en Test (−0.023), por lo que no sesga ni al alza ni a la baja la planificación.
- **Estabilidad:** Train y Test son consistentes (Train MAE 0.818 vs Test 0.790; WAPE 0.612 vs 0.631), sin indicios de sobreajuste.
- **Costo:** LightGBM es rápido de entrenar e inferir sobre la matriz de características horarias (lags + ventanales), mucho más ligero que un búsqueda anidada completa, y la inferencia por estación/hora es trivial en producción.

### P2 — Riesgo de desbalance → **Regresión Logística sin peso, umbral 0.5**
- **Desempeño vs. baseline:** Sin peso de clases alcanza Test F1 0.832 y PR-AUC 0.806, superando al logístico balanceado (F1 0.794, PR-AUC 0.799) y al umbral sensible 0.3 (F1 0.706, que sacrifica demasiada precisión: 0.589).
- **Métrica ligada a la decisión:** El equipo de rebalanceo necesita pocas alarmas falsas para no mover bicis innecesariamente, pero sin omitir estaciones en riesgo. Con umbral 0.5 y sin peso se obtiene el mejor equilibrio: Precisión 0.831 y Recall 0.832.
- **Estabilidad:** Train (F1 0.823) y Test (F1 0.832) son casi idénticos; el modelo es estable y no sobreajusta.
- **Costo:** Regresión Logística es extremadamente barata de entrenar e inferir, interpretabilidad alta (coeficientes por variable), ideal para un equipo operativo que requiere decisiones rápidas y explicables.

### P3 — Clustering → **KMeans funcional con K=4**
- **Desempeño vs. baseline:** K=4 alcanza Test Silhouette 0.182 y Davies-Bouldin 1.535, muy superior al baseline municipal (Silhouette 0.019, DB 3.712), que refleja que la división administrativa no captura los patrones funcionales de uso.
- **Métrica ligada a la decisión:** Planeación urbana necesita grupos homogéneos en comportamiento (mañana/tarde/noche, fin de semana, duración, balance neto). K=4 es el que mejor separa estos perfiles (mayor Silhouette, menor DB) entre K=3,4,5,6.
- **Estabilidad:** Con `n_init=10` y semilla fija (42) el clustering es reproducible; K=4 es el valor configurado en `config.py` (`P3_K_ELEGIDO`).
- **Costo:** KMeans sobre ~371 estaciones de entrenamiento es computacionalmente barato; la asignación de nuevas estaciones es una predicción de proximidad a centroides, de bajo costo en producción.

---

## 5. Esquema de ciclo de vida utilizado y justificación

**Esquema:** Desarrollo y seguimiento con **MLflow Tracking** (experimento único `mibici-equipo20`) + registro y promoción en **MLflow Model Registry**.

Etapas del ciclo:
1. **Ingesta y versionado de datos:** `dataset.py` genera los Parquet procesados y `data/manifest.json` con hash SHA-256; cada run registra `datos_version` (hash corto de los datos de entrenamiento).
2. **Experimentos (Tracking):** Una corrida por modelo/versión, registrando parámetros, métricas (Train/Validación/Test), etiquetas (`question`, `model_family`, `stage_of_work`, `equipo=20`, `git_commit`, `datos_version`), artefactos (modelo con firma, figuras, manifiesto).
3. **Selección:** Comparación en la interfaz de MLflow por métrica de negocio, estabilidad Train/Test y costo.
4. **Registro (Model Registry):** Cada modelo elegido se registra con nombre consistente (`mibici-equipo20-p1-demanda`, `mibici-equipo20-p2-riesgo`, `mibici-equipo20-p3-grupos`), conservando el linaje (experimento, corrida de origen, commit de Git, versión de datos) y una descripción.
5. **Promoción:** Las versiones registradas se promueven al alias/stage `Production` para inferencia (`predict_p*.py`).

**Justificación:** MLflow Tracking da trazabilidad completa de cada corrida (qué datos, qué código/commit, qué hiperparámetros, qué métricas). El Model Registry separa la experimentación de la producción, permite versionar modelos, hacer rollback y asociar cada versión a su run de origen y a su versión de datos, cumpliendo requisitos de reproducibilidad y auditoría MLOps.

---

## 6. Modelo a producción (resumen)

| Pregunta | Modelo en producción | Nombre en Registry | Run de origen | Commit | Versión de datos |
|----------|----------------------|--------------------|---------------|--------|------------------|
| P1 | LightGBM Tweedie | `mibici-equipo20-p1-demanda` | `P1_lgbm_tweedie` | `4dc84407e4` | `35a3ecf437d8` |
| P2 | Regresión Logística sin peso (umbral 0.5) | `mibici-equipo20-p2-riesgo` | `p2-baseline-logistic-regression` (cw=None, umb=0.5) | `4dc84407e4` | `35a3ecf437d8` |
| P3 | KMeans K=4 | `mibici-equipo20-p3-grupos` | `p3-kmeans-k4` (n_clusters=4) | `c87cc1bd81` / `4dc84407e4` | `35a3ecf437d8` |

El registro de estos tres modelos en el Model Registry se ejecuta con `run_mejor_modelo.py`.
