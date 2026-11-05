# Documentación de Contenedores — TC5061 Proyecto MiBici (Equipo 20)

Despliegue con **Docker Compose** de la API REST de predicción MiBici con **MLflow** como servidor de tracking y Model Registry, y **3 contenedores de API** (uno por pregunta: P1, P2, P3).

---

## 1. Arquitectura y puertos

| Servicio | Contenedor | Puerto host | Puerto interno | Descripción |
|----------|-----------|-------------|----------------|-------------|
| **mlflow** | `mlflow` | `5001` | `5000` | Servidor de tracking + Model Registry |
| **prep** | `prep` | — | — | Job: ingesta → entrenamientos P1/P2/P3 → registro de campeones |
| **api-p1** | `api-p1` | `8001` | `8000` | API P1: demanda horaria (regresión) |
| **api-p2** | `api-p2` | `8002` | `8000` | API P2: riesgo de desbalance (clasificación) |
| **api-p3** | `api-p3` | `8003` | `8000` | API P3: clustering de estaciones (no supervisado) |

Toda la comunicación entre contenedores usa la red interna `mibici-net`.  
Desde tu máquina (host), accedes a MLflow por `http://localhost:5001` y a cada API por su puerto correspondiente.

> El puerto `5000` de tu host puede estar ocupado por macOS (Control Center). Por eso MLflow se mapea a `5001:5000`.

---

## 1.1 Detalle de cada contenedor

### mlflow — Servidor de tracking y Model Registry

| Aspecto | Detalle |
|---------|---------|
| **Qué es** | Servidor MLflow que expone la API de tracking y el Model Registry. |
| **Qué hace** | Recibe runs de entrenamiento (parámetros, métricas, tags, artefactos), almacena los modelos registrados y los pone a disposición de las APIs a través del alias `@champion`. |
| **Cómo lo hace** | Ejecuta `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 --serve-artifacts`. Los artefactos (modelos `.pkl`) se guardan en el volumen compartido `mlruns`. El healthcheck consulta `http://localhost:5000`. |

### prep — Job de entrenamiento y registro de campeones

| Aspecto | Detalle |
|---------|---------|
| **Qué es** | Contenedor de un solo uso (job) que ejecuta todo el flujo de entrenamiento. |
| **Qué hace** | 1) Espera a que MLflow esté disponible. 2) Ejecuta `proyecto_mibici.dataset` para descargar y procesar datos de MiBici. 3) Entrena los 3 modelos (P1, P2, P3). 4) Registra los campeones en el Model Registry con alias `@champion` mediante `run_mejor_modelo.py`. |
| **Cómo lo hace** | Su `entrypoint.sh` ejecuta secuencialmente: `python -m proyecto_mibici.dataset` → `train_p1.py` → `train_p2.py` → `train_p3.py` → `run_mejor_modelo.py`. Termina con código 0 cuando finaliza. Las APIs dependen de que este contenedor termine con éxito (`service_completed_successfully`). |

### api-p1 — API de demanda horaria (P1)

| Aspecto | Detalle |
|---------|---------|
| **Qué es** | API REST con FastAPI para predicción de demanda horaria de bicicletas. |
| **Qué hace** | Expone `GET /health` (verifica que el modelo esté cargado) y `POST /predict` (recibe features de entrada y devuelve la demanda predicha). |
| **Cómo lo hace** | En cada petición, descarga el modelo desde MLflow Model Registry usando `mlflow.artifacts.download_artifacts()` + `joblib.load()`, ejecuta `modelo.predict(df)` con los datos de entrada y devuelve el resultado en JSON. Variables de entorno: `PREGUNTA=p1`, `MODELO_REGISTRY=mibici-equipo20-p1-demanda@champion`, `MLFLOW_TRACKING_URI=http://mlflow:5000`. |

### api-p2 — API de riesgo de desbalance (P2)

| Aspecto | Detalle |
|---------|---------|
| **Qué es** | API REST con FastAPI para predicción de riesgo de desbalance de estaciones. |
| **Qué hace** | Expone `GET /health` y `POST /predict`. Devuelve probabilidad de riesgo y clasificación binaria (0/1). |
| **Cómo lo hace** | Igual que api-p1: descarga el modelo desde MLflow en cada petición, ejecuta `modelo.predict(df)` y `modelo.predict_proba(df)[:, 1]`, y devuelve `prob_riesgo` y `pred_riesgo`. Variables de entorno: `PREGUNTA=p2`, `MODELO_REGISTRY=mibici-equipo20-p2-riesgo@champion`, `MLFLOW_TRACKING_URI=http://mlflow:5000`. |

### api-p3 — API de clustering de estaciones (P3)

| Aspecto | Detalle |
|---------|---------|
| **Qué es** | API REST con FastAPI para clustering funcional de estaciones. |
| **Qué hace** | Expone `GET /health` y `POST /predict`. Devuelve el número de grupo (cluster) al que pertenece una estación. |
| **Cómo lo hace** | Igual que las demás APIs: descarga el modelo KMeans desde MLflow en cada petición, ejecuta `modelo.predict(df)` y devuelve `grupo_kmeans`. Variables de entorno: `PREGUNTA=p3`, `MODELO_REGISTRY=mibici-equipo20-p3-grupos@champion`, `MLFLOW_TRACKING_URI=http://mlflow:5000`. |

---


## 2. Requisitos previos

- **Docker** instalado (versión 20.10+)
- **Docker Compose** (v2)
- Puerto **5001** libre en tu host (para MLflow)
- Puertos **8001, 8002, 8003** libres en tu host (para las APIs)

---

## 3. Comandos de Docker Compose

Ejecutar siempre desde la **raíz del repositorio**.

### 3.1 Construir todas las imágenes

```bash
docker compose -f docker/docker-compose.yml build
```

Esto construye 4 servicios: `mlflow`, `prep`, `api-p1`, `api-p2`, `api-p3`.

### 3.2 Levantar todo el flujo

```bash
docker compose -f docker/docker-compose.yml up -d
```

Secuencia automática:
1. `mlflow` se levanta y pasa el healthcheck.
2. `prep` espera a MLflow, ejecuta **ingesta → entrenamiento P1/P2/P3 → registro de campeones** en MLflow, y termina.
3. `api-p1`, `api-p2`, `api-p3` esperan a que `prep` termine (`service_completed_successfully`) y cargan los campeones del Model Registry.

### 3.3 Ver estado de los contenedores

```bash
docker compose -f docker/docker-compose.yml ps
```

Ejemplo esperado:

```
NAME         STATUS
mlflow       Up (healthy)
prep         Exited (0)
api-p1       Up
api-p2       Up
api-p3       Up
```

### 3.4 Ver logs

```bash
# Todos los servicios
docker compose -f docker/docker-compose.yml logs

# Un servicio específico
docker compose -f docker/docker-compose.yml logs mlflow
docker compose -f docker/docker-compose.yml logs prep
docker compose -f docker/docker-compose.yml logs api-p1
docker compose -f docker/docker-compose.yml logs api-p2
docker compose -f docker/docker-compose.yml logs api-p3
```

### 3.5 Detener y eliminar

```bash
# Detener contenedores (mantiene volúmenes e imágenes)
docker compose -f docker/docker-compose.yml down

# Detener y eliminar también los volúmenes nombrados (borra datos de MLflow)
docker compose -f docker/docker-compose.yml down -v
```

### 3.6 Reconstruir desde cero

Si hiciste cambios en el código de la API (por ejemplo, en `docker/api/model_loader.py`) y Docker los sirve desde caché, fuerza una reconstrucción sin caché:

```bash
docker compose -f docker/docker-compose.yml down -v
docker compose -f docker/docker-compose.yml build --no-cache api-p1 api-p2 api-p3
docker compose -f docker/docker-compose.yml up -d
```

---

## 4. Endpoints de las APIs

Todas las APIs exponen los mismos endpoints:

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/health` | Estado del servicio y si el modelo está cargado |
| `POST` | `/predict` | Recibe JSON con features, devuelve predicción |

El modelo se carga **en cada petición** desde el Model Registry de MLflow (descarga del artefacto + `joblib.load`), no va empaquetado en la imagen.

---

### 4.1 API P1 — Demanda horaria (puerto 8001)

Modelo: `mibici-equipo20-p1-demanda@champion` (LightGBM Tweedie)

**Health:**
```bash
curl http://localhost:8001/health
```

**Parámetros de entrada (`POST /predict`):**

| Parámetro | Tipo | Descripción | Ejemplo |
|-----------|------|-------------|---------|
| `lag_1h` | float | Rezago a 1 hora | `2` |
| `lag_24h` | float | Rezago a 24 horas | `3` |
| `lag_168h` | float | Rezago a 168 horas | `1` |
| `target_hour_sin` | float | Componente seno del ciclo horario | `0.5` |
| `target_hour_cos` | float | Componente coseno del ciclo horario | `0.8` |
| `target_is_weekend` | int (0/1) | 1 si es fin de semana, 0 si no | `0` |
| `rolling_3h` | float | Media móvil de 3 horas | `2.5` |
| `rolling_168h` | float | Media móvil de 168 horas | `2.0` |

**Ejemplo de predicción P1:**
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "lag_1h": 2,
    "lag_24h": 3,
    "lag_168h": 1,
    "target_hour_sin": 0.5,
    "target_hour_cos": 0.8,
    "target_is_weekend": 0,
    "rolling_3h": 2.5,
    "rolling_168h": 2.0
  }'
```

**Respuesta esperada:**
```json
{
  "pregunta": "p1",
  "modelo": "mibici-equipo20-p1-demanda",
  "version": "1",
  "run_id": "...",
  "demanda_predicha": 2.45
}
```

---

### 4.2 API P2 — Riesgo de desbalance (puerto 8002)

Modelo: `mibici-equipo20-p2-riesgo@champion` (HistGradientBoostingClassifier)

**Health:**
```bash
curl http://localhost:8002/health
```

**Parámetros de entrada (`POST /predict`):**

| Parámetro | Tipo | Rango | Descripción | Ejemplo |
|-----------|------|-------|-------------|---------|
| `hora_del_dia` | int | 0–23 | Hora del día | `14` |
| `dia_semana` | int | 0–6 | Día de la semana | `2` |
| `fin_de_semana` | int | 0/1 | 1 si es fin de semana | `0` |
| `capacidad` | float | > 0 | Capacidad de la estación | `30` |
| `salidas` | float | ≥ 0 | Salidas acumuladas | `45` |
| `llegadas` | float | ≥ 0 | Llegadas acumuladas | `40` |
| `balance_acumulado_dia` | float | — | Balance acumulado del día | `5` |
| `acumulado_relativo` | float | — | Acumulado relativo | `0.2` |
| `balance_ult_3h_relativo` | float | — | Balance últimas 3h relativo | `0.1` |
| `riesgo_vacia` | int | 0/1 | Riesgo de vacía | `0` |
| `riesgo_llena` | int | 0/1 | Riesgo de llena | `0` |

**Ejemplo de predicción P2:**
```bash
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hora_del_dia": 14,
    "dia_semana": 2,
    "fin_de_semana": 0,
    "capacidad": 30,
    "salidas": 45,
    "llegadas": 40,
    "balance_acumulado_dia": 5,
    "acumulado_relativo": 0.2,
    "balance_ult_3h_relativo": 0.1,
    "riesgo_vacia": 0,
    "riesgo_llena": 0
  }'
```

**Respuesta esperada:**
```json
{
  "pregunta": "p2",
  "modelo": "mibici-equipo20-p2-riesgo",
  "version": "1",
  "run_id": "...",
  "prob_riesgo": 0.0526,
  "pred_riesgo": 0
}
```

---

### 4.3 API P3 — Clustering de estaciones (puerto 8003)

Modelo: `mibici-equipo20-p3-grupos@champion` (KMeans K=4)

**Health:**
```bash
curl http://localhost:8003/health
```

**Parámetros de entrada (`POST /predict`):**

| Parámetro | Tipo | Rango | Descripción | Ejemplo |
|-----------|------|-------|-------------|---------|
| `pct_manana` | float | 0–1 | Porcentaje mañana | `0.2` |
| `pct_mediodia` | float | 0–1 | Porcentaje mediodía | `0.3` |
| `pct_tarde` | float | 0–1 | Porcentaje tarde | `0.25` |
| `pct_noche` | float | 0–1 | Porcentaje noche | `0.15` |
| `pct_fin_de_semana` | float | 0–1 | Porcentaje fin de semana | `0.1` |
| `duracion_media` | float | ≥ 0 | Duración media de viajes | `15` |
| `viajes_dia_log` | float | — | Viajes por día (log) | `5` |
| `balance_neto` | float | -1–1 | Balance neto de la estación | `0.0` |

**Ejemplo de predicción P3:**
```bash
curl -X POST http://localhost:8003/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pct_manana": 0.2,
    "pct_mediodia": 0.3,
    "pct_tarde": 0.25,
    "pct_noche": 0.15,
    "pct_fin_de_semana": 0.1,
    "duracion_media": 15,
    "viajes_dia_log": 5,
    "balance_neto": 0.0
  }'
```

**Respuesta esperada:**
```json
{
  "pregunta": "p3",
  "modelo": "mibici-equipo20-p3-grupos",
  "version": "1",
  "run_id": "...",
  "grupo_kmeans": 1
}
```

---

## 5. MLflow UI

Accede a la interfaz de MLflow desde tu navegador:

```
http://localhost:5001
```

Allí puedes ver:
- Experimentos y runs de entrenamiento
- Modelos registrados en el Model Registry con alias `@champion`
- Parámetros, métricas y artefactos de cada corrida

---

## 6. Solución de problemas comunes

| Problema | Causa probable | Solución |
|----------|---------------|----------|
| `mlflow` no está `healthy` | Puerto 5001 ocupado o error de SQLite | `docker compose logs mlflow` |
| `prep` no conecta a MLflow | Red interna de Docker no resuelve `mlflow:5000` | `docker compose logs prep` |
| API devuelve `"Modelo no disponible"` (503) | `prep` aún no terminó de registrar campeones | Esperar a que `prep` termine (`Exited 0`) |
| API devuelve 403 en MLflow | Host header no permitido | Verificar `--allowed-hosts` en `docker/mlflow/Containerfile` |
| `/predict` falla tras cambio de código | Capa `COPY` servida desde caché de Docker | `build --no-cache api-p1 api-p2 api-p3` |

---

## 7. Notas importantes

- Los campeones se registran en MLflow con el alias **`@champion`**. Cada vez que `prep` se ejecuta, vuelve a registrar y promover a Production/champion.
- El modelo se carga **en cada petición** desde el Model Registry de MLflow, no se empaqueta en la imagen de la API.
- Los volúmenes nombrados (`mlruns`) persisten entre reinicios. Usa `down -v` para limpiarlos completamente.
- Puerto de MLflow en host: `5001` (interno `5000`). Puertos de API: `8001`, `8002`, `8003`.
