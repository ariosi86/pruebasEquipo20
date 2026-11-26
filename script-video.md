# Script de Video — Entrega Proyecto Final (Equipo 20, TC5061)

- **Duración objetivo:** ~4 minutos 15 segundos (rango obligatorio: **2:00 a 5:00 min**, grabación continua sin cortes de edición).
- **Participantes:** 4 integrantes (1 persona opera el teclado/pantalla y los 4 narran por turnos).
- **Precondición antes de grabar:**
  1. Tener el stack levantado desde la raíz del repositorio:
     ```bash
     docker compose -f docker/docker-compose.yml --profile loadtest up -d --no-build
     ```
  2. Tener abiertas en el navegador 5 pestañas listas:
     - **Tab 1:** Presentación (`TC5061_Proyecto_Final_Equipo20.pdf`)
     - **Tab 2:** MLflow UI (`http://localhost:5001`)
     - **Tab 3:** Evidently UI (`http://localhost:8000`)
     - **Tab 4:** Locust UI (`http://localhost:8089`) y/o Swagger API P1 (`http://localhost:8001/docs`)
     - **Tab 5:** GitHub Actions & GHCR (`https://github.com/ariosi86/TC5061-Proyecto-Equipo20/actions/workflows/ci.yml`)
  3. Tener una terminal visible y limpia en la raíz del proyecto para ejecutar los comandos de la demo.

---

## Bloque 1 — Problema, Solución y Experimentación (`0:00 – 0:55`)
**Responsable:** Integrante 1 (Introducción y Arquitectura ML)  
**En pantalla:** Presentación (diapositivas de Problema, Solución y Estructura/MLflow) → cambio rápido a Tab 2 (MLflow UI).

**Guion:**
> "Hola, somos el Equipo 20 de TC5061. Nuestro proyecto resuelve tres problemas operativos del sistema de bicicleta pública **MiBici** en Guadalajara:
> 1. **P1 (Regresión):** predecir la demanda horaria de viajes por estación para la hora siguiente usando **LightGBM Tweedie**.
> 2. **P2 (Clasificación):** anticipar si una estación entrará en riesgo crítico de desbalance —vacía o llena— mediante **HistGradientBoostingClassifier**.
> 3. **P3 (Clustering):** segmentar estaciones en 4 perfiles funcionales con **KMeans**.
>
> Organizamos el repositorio bajo **Cookiecutter Data Science**, con transformadores orientados a objetos en scikit-learn, entry points reproducibles en `MLproject`, versionado de datos con SHA-256 en `manifest.json` y seguimiento completo en **MLflow**. Como vemos en la interfaz de MLflow (`http://localhost:5001`), los tres modelos campeones quedan registrados en el **Model Registry** con el alias `@champion`, su linaje, parámetros y métricas."

---

## Bloque 2 — Despliegue, APIs en Vivo y CI/CD (`0:55 – 2:10`)
**Responsable:** Integrante 2 (Operación y Demo Funcional en Terminal)  
**En pantalla:** Terminal (ejecución en vivo) → cambio rápido a Tab 5 (GitHub Actions: corrida en verde con `lint → test → build → smoke → publish` + Packages GHCR).

**Comandos a ejecutar en vivo en la terminal:**
```bash
# 1. Mostrar contenedores activos y saludables
docker compose -f docker/docker-compose.yml --profile loadtest ps

# 2. Verificar salud de las 3 APIs (muestra modelo @champion cargado)
curl -s http://localhost:8001/health | head -c 220; echo ""

# 3. Petición real de predicción P1 (Demanda horaria)
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"target_hour":"2026-06-15T18:00:00","lag_1h":5,"lag_24h":4.0,"lag_168h":6.0,"rolling_3h":4.5,"rolling_168h":3.8}'

# 4. Petición real de predicción P2 (Riesgo de desbalance)
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{"hora_del_dia":18,"dia_semana":1,"fin_de_semana":0,"capacidad":20,"salidas":8,"llegadas":2,"balance_acumulado_dia":-12,"acumulado_relativo":-0.6,"balance_ult_3h_relativo":-0.45,"riesgo_vacia":1,"riesgo_llena":0}'

# 5. Petición real de predicción P3 (Perfil de estación)
curl -X POST http://localhost:8003/predict \
  -H "Content-Type: application/json" \
  -d '{"pct_manana":0.23,"pct_mediodia":0.33,"pct_tarde":0.29,"pct_noche":0.12,"pct_fin_de_semana":0.20,"duracion_media":11.12,"viajes_dia_log":3.23,"balance_neto":-0.02}'
```

**Guion:**
> "Toda la plataforma se orquesta con **Docker Compose**: tenemos `mlflow`, un job `prep` que entrena y registra los campeones, **Evidently UI**, tres microservicios **FastAPI** (`api-p1`, `api-p2`, `api-p3`) y **Locust**.
> Aquí vemos con `docker compose ps` todos los contenedores saludables. Al consultar `/health` en P1 confirma que está sirviendo `mibici-equipo20-p1-demanda` versión 1 con alias `@champion`.
> Ahora enviamos peticiones reales `/predict`:
> - **P1** nos devuelve una demanda estimada (~7.34 viajes para esa hora).
> - **P2** nos devuelve la probabilidad de riesgo (~0.89) y la alerta binaria `pred_riesgo: 1`.
> - **P3** asigna la estación al clúster funcional correspondiente (`grupo_kmeans: 1`).
>
> Todo esto está respaldado por nuestro pipeline de **CI/CD en GitHub Actions** (`ci.yml`), como vemos aquí en la pestaña *Actions*: en cada push ejecuta `lint`, `pytest` en Python 3.10 y 3.11, `build` de las 4 imágenes, un `smoke` test levantando `mlflow`, `evidently` y las 3 APIs con Docker Compose, y finalmente `publish` en **GHCR** (`ghcr.io/ariosi86/mibici-*`)."

---

## Bloque 3 — Pruebas de Carga con Locust y FinOps (`2:10 – 3:15`)
**Responsable:** Integrante 3 (Rendimiento y Costos)  
**En pantalla:** Tab 4 (Locust UI en `http://localhost:8089` — lanzar una prueba rápida de 10 usuarios / spawn 5, o mostrar métricas) → diapositiva de resultados Locust y FinOps.

**Guion:**
> "Para validar la capacidad operativa integramos **Locust** contra las tres APIs. En nuestras pruebas de carga detectamos inicialmente que descargar el modelo desde MLflow en cada petición saturaba el servidor al 100% de CPU, limitando el sistema a **14 inferencias/s** con latencia p95 de **7.1 segundos** y **4.1% de errores** con 100 usuarios concurrentes.
> Implementamos una **caché con candado por TTL (`CACHE_TTL_SEG=60`)** en FastAPI, manteniendo la actualización automática desde el Model Registry. Con esto subimos a **70.9 inferencias/s**, bajamos la latencia p95 a **14 ms** con **0% de errores**, y alcanzamos un máximo de **1,657 inferencias/s** en prueba de estrés.
> En el análisis **FinOps**, para el tráfico estimado de MiBici (~14,700 inferencias diarias), el costo depende de la instancia encendida: recomendamos `t3.medium` (**$30.37 USD/mes**, 50% de ahorro frente a `t3.large`) y proponemos desacoplar MLflow en producción para bajar a `t3.small` Spot (**$5.31 USD/mes**)."

---

## Bloque 4 — Monitoreo con Evidently, A/B Testing, Gobernanza y Cierre (`3:15 – 4:15`)
**Responsable:** Integrante 4 (Monitoreo, Gobernanza y Conclusión)  
**En pantalla:** Tab 3 (Evidently UI en `http://localhost:8000`) + diapositiva de Drift, A/B Testing y Model Cards.

**Guion:**
> "Finalmente, para monitoreo y gobernanza integramos **Evidently UI** en Docker (`http://localhost:8000`) junto con `EvidentlyMonitor`, comparando datos de referencia 2025 contra producción en 2026:
> - En **P1** detectamos drift en 6 de 9 variables por crecimiento en el volumen de viajes; en **P3** en 6 de 8 variables por reducción en duración media; y en **P2** las distribuciones se mantuvieron estables según Evidently.
> - Cada detección alimenta un flujo de **A/B Testing** entre `@champion` y un `@challenger` reentrenado con reglas estadísticas estrictas (bootstrap e intervalos de confianza al 95%): en **P1** y **P3** la regla mantiene al campeón vigente, mientras que en **P2** promueve al challenger por mejora significativa en F1 y PR-AUC.
> - Todo el ciclo queda documentado en nuestras **Model Cards** (`docs/model_cards.md`) con usos previstos, limitaciones, sesgos y estrategia de reversión.
>
> Con esto demostramos un sistema MLOps completo, reproducible, monitoreado y listo para operar. Muchas gracias."

---

## Checklist rápido previo a grabar
- [ ] `git push` realizado y workflow **CI/CD MiBici** en verde en GitHub Actions (con las 4 imágenes visibles en **Packages** / GHCR).
- [ ] `docker compose -f docker/docker-compose.yml --profile loadtest ps` muestra todos los servicios `healthy` / `running`.
- [ ] Probar una vez los 3 `curl` de `/predict` antes de darle *Record*.
- [ ] Verificar cronómetro entre **3:30 y 4:30 min** para no salir del límite de 2–5 min.
