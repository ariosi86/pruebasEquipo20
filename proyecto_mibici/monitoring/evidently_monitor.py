#!/usr/bin/env python
"""
Clase reutilizable de monitoreo de drift con Evidently (Entregable 8).

Diseñada para que cualquier integrante del equipo la instancie con sus
propios conjuntos de referencia/actual y columnas, sin depender de P1.
No borra ni reemplaza nada de las entregas anteriores.

La clase incluye una capa de registro que usa la API real de Evidently Platform
para registrar:
- El proyecto de monitoreo (nombre, descripción, columnas, umbrales).
- Los datasets usados (referencia y actual, con estadísticas básicas).
- Los reportes generados (HTML, CSV, variables con drift, fecha).

Se conecta al servidor de Evidently UI en ``http://127.0.0.1:8000`` mediante
``RemoteWorkspace``. Si el servidor no está disponible, los registros se
guardan localmente como JSON bajo ``reports/drift/registro/`` para trazabilidad
y para el reporte PDF del entregable.

Ejemplo rápido (P1):
    from proyecto_mibici.monitoring import EvidentlyMonitor

    monitor = EvidentlyMonitor(
        numerical_columns=["lag_1h", "lag_24h", "lag_168h", "rolling_3h", "rolling_168h",
                           "target_hour_sin", "target_hour_cos"],
        categorical_columns=["target_is_weekend"],
        nombre_proyecto="p1-demanda-horaria",
    )
    resumen = monitor.comparar(df_referencia, df_actual, nombre="p1_ref2025_vs_oct2026")
    print(resumen)

Ejemplo genérico (P2 / P3 / cualquier otro):
    monitor = EvidentlyMonitor(
        numerical_columns=["edad", "saldo"],
        categorical_columns=["sexo", "region"],
        num_threshold=0.15,
        cat_threshold=0.10,
        nombre_proyecto="p2-riesgo-desequilibrio",
    )
    resumen = monitor.comparar(ref, cur, nombre="p2_ref_vs_actual", log_mlflow=False)
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import json
from pathlib import Path
import webbrowser

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from proyecto_mibici.config import (
    MLFLOW_EXPERIMENT_MONITOREO,
    RANDOM_SEED,
    REPORTS_DIR,
    URI_MLFLOW_TRACKING,
)

# URL del servidor de Evidently UI (API de la plataforma, no solo lectura del frontend).
EVIDENTLY_API_URL = "http://127.0.0.1:8000"


class EvidentlyMonitor:
    """Orquesta la comparación de drift de datos con Evidently y registra
    proyecto, datasets y reportes en la API de Evidently Platform.

    Se conecta al servidor de Evidently UI en ``http://127.0.0.1:8000`` mediante
    ``RemoteWorkspace``. Si el servidor no está disponible, los registros se
    guardan localmente como JSON bajo ``out_dir / "registro"``.

    Parametros
    ----------
    numerical_columns : iterable de str
        Columnas numéricas a comparar.
    categorical_columns : iterable de str, opcional
        Columnas categóricas/binarias a comparar.
    num_method : str
        Método de drift numérico de Evidently ("wasserstein", "ks", etc.).
    cat_method : str
        Método de drift categórico de Evidently ("jensenshannon", etc.).
    num_threshold : float
        Umbral de drift numérico (para Evidently y para la tabla de scores).
    cat_threshold : float
        Umbral de drift categórico (para Evidently).
    n_muestra : int
        Filas máximas por conjunto (Evidently es costoso con millones de filas).
    random_seed : int
        Semilla para el muestreo aleatorio.
    out_dir : str | Path | None
        Carpeta donde se guardan el HTML de Evidently y el CSV de scores.
        Por defecto: ``REPORTS_DIR / "drift"``.
    abrir_navegador : bool
        Si True, abre automáticamente el HTML generado en el navegador web
        después de crear el reporte. Por defecto: ``False``.
    registrar_local : bool
        Si True, guarda metadatos de proyecto, datasets y reportes como JSON
        bajo ``out_dir / "registro"`` (respaldo/trazabilidad). Por defecto: ``True``.
    usar_api_evidently : bool
        Si True, intenta registrar en la API de Evidently Platform en
        ``http://127.0.0.1:8000``. Si False o el servidor no está disponible,
        solo se usa registro local JSON. Por defecto: ``True``.
    """

    def __init__(
        self,
        numerical_columns: Iterable[str],
        categorical_columns: Iterable[str] | None = None,
        num_method: str = "wasserstein",
        cat_method: str = "jensenshannon",
        num_threshold: float = 0.1,
        cat_threshold: float = 0.1,
        n_muestra: int = 100_000,
        random_seed: int = RANDOM_SEED,
        out_dir: str | Path | None = None,
        abrir_navegador: bool = False,
        registrar_local: bool = True,
        usar_api_evidently: bool = True,
    ) -> None:
        self.numerical_columns = list(numerical_columns)
        self.categorical_columns = list(categorical_columns or [])
        self.num_method = num_method
        self.cat_method = cat_method
        self.num_threshold = num_threshold
        self.cat_threshold = cat_threshold
        self.n_muestra = n_muestra
        self.random_seed = random_seed
        self.out_dir = Path(out_dir) if out_dir is not None else REPORTS_DIR / "drift"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.abrir_navegador = abrir_navegador
        self.registrar_local = registrar_local
        self.usar_api_evidently = usar_api_evidently

        # Columnas que Evidently va a usar: todas las que pidió el usuario.
        self.all_columns = self.numerical_columns + self.categorical_columns

        # Carpetas de registro local
        self.registro_dir = self.out_dir / "registro"
        self.proyectos_dir = self.registro_dir / "proyectos"
        self.datasets_dir = self.registro_dir / "datasets"
        self.reportes_dir = self.registro_dir / "reportes"
        if self.registrar_local:
            self.proyectos_dir.mkdir(parents=True, exist_ok=True)
            self.datasets_dir.mkdir(parents=True, exist_ok=True)
            self.reportes_dir.mkdir(parents=True, exist_ok=True)

        # ID del proyecto (se asigna al registrar o al consultar la API de Evidently)
        self.proyecto_id: str | None = None
        self.proyecto_nombre: str | None = None

        # Workspace de Evidently (se inicializa en _conectar_evidently)
        self._ws = None
        self._evidently_disponible = False

        if self.usar_api_evidently:
            self._conectar_evidently()

        logger.info(
            f"EvidentlyMonitor listo | num={len(self.numerical_columns)} | cat={len(self.categorical_columns)} | umbral_num={self.num_threshold} | "
            f"umbral_cat={self.cat_threshold} | muestra={self.n_muestra} | out={self.out_dir} | registro={self.registrar_local} | api={self._evidently_disponible}"
        )

    # ------------------------------------------------------------------ #
    # Conexión a la API de Evidently Platform
    # ------------------------------------------------------------------ #
    def _conectar_evidently(self) -> None:
        """Intenta conectar con el servidor de Evidently UI en el puerto 8000."""
        try:
            from evidently.ui.workspace import RemoteWorkspace

            self._ws = RemoteWorkspace(base_url=EVIDENTLY_API_URL, verify=True)
            self._evidently_disponible = True
            logger.info(f"Conectado a Evidently Platform en {EVIDENTLY_API_URL}")
        except ImportError:
            logger.warning(
                "No se pudo importar RemoteWorkspace de evidently.ui.workspace. "
                "Se usará solo registro local JSON."
            )
            self._evidently_disponible = False
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"No se pudo conectar a Evidently Platform en {EVIDENTLY_API_URL}: {exc}. "
                "Se usará solo registro local JSON."
            )
            self._evidently_disponible = False

    # ------------------------------------------------------------------ #
    # Utilidades internas
    # ------------------------------------------------------------------ #
    @staticmethod
    def _muestrear(df: pd.DataFrame, n: int, random_seed: int) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        return df.sample(n=min(n, len(df)), random_state=random_seed).reset_index(drop=True)

    def _definicion(self) -> DataDefinition:
        return DataDefinition(
            numerical_columns=self.numerical_columns,
            categorical_columns=self.categorical_columns,
        )

    def _dataset_evidently(self, df: pd.DataFrame) -> Dataset:
        """Crea un objeto Dataset de Evidently para subir a la plataforma."""
        definicion = self._definicion()
        return Dataset.from_pandas(df[self.all_columns], data_definition=definicion)

    def _estadisticas_dataset(self, df: pd.DataFrame) -> dict:
        """Estadísticas básicas de un dataset para el registro."""
        stats = {
            "filas": len(df),
            "columnas": len(df.columns),
            "columnas_monitoreadas": list(self.all_columns),
            "fecha_registro": datetime.now(timezone.utc).isoformat(),
        }
        for c in self.numerical_columns:
            if c in df.columns:
                stats[f"media_{c}"] = float(df[c].mean())
                stats[f"std_{c}"] = float(df[c].std())
                stats[f"min_{c}"] = float(df[c].min())
                stats[f"max_{c}"] = float(df[c].max())
        for c in self.categorical_columns:
            if c in df.columns:
                stats[f"valores_unicos_{c}"] = int(df[c].nunique())
                stats[f"top_valor_{c}"] = (
                    str(df[c].mode().iloc[0]) if not df[c].mode().empty else None
                )
        return stats

    def _guardar_json(self, ruta: Path, datos: dict) -> None:
        """Guarda un diccionario como JSON con formato legible."""
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Registro de proyecto (API de Evidently + JSON local)
    # ------------------------------------------------------------------ #
    def registrar_proyecto(self, nombre: str, descripcion: str | None = None) -> str:
        """Registra (o recupera) el proyecto de monitoreo en Evidently Platform.

        Intenta crear o encontrar el proyecto en el servidor de Evidently UI
        mediante la API de RemoteWorkspace. Además guarda metadatos locales
        como JSON para trazabilidad.

        Parametros
        ----------
        nombre : str
            Nombre del proyecto (ej. "p1-demanda-horaria").
        descripcion : str | None
            Descripción opcional del proyecto.

        Returns
        -------
        str
            ID del proyecto (de Evidently Platform o generado localmente).
        """
        self.proyecto_nombre = nombre

        # 1. Intentar registrar en la API de Evidently Platform
        if self._evidently_disponible and self._ws is not None:
            try:
                # Buscar si ya existe un proyecto con este nombre
                existentes = self._ws.search_project(nombre)
                if existentes:
                    self.proyecto_id = str(existentes[0].id)
                    logger.info(
                        f"Proyecto existente en Evidently: {nombre} (id={self.proyecto_id})"
                    )
                else:
                    project_model = self._ws.create_project(name=nombre, description=descripcion)
                    self.proyecto_id = str(project_model.id)
                    logger.success(
                        f"Proyecto creado en Evidently: {nombre} (id={self.proyecto_id})"
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"No se pudo registrar el proyecto en Evidently Platform: {exc}")
                self.proyecto_id = None

        # 2. Fallback: ID generado localmente si no se pudo en la API
        if self.proyecto_id is None:
            self.proyecto_id = (
                f"local-{nombre}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            )

        # 3. Guardar metadatos locales (respaldo para el reporte PDF)
        if self.registrar_local:
            datos_proyecto = {
                "id": self.proyecto_id,
                "nombre": nombre,
                "descripcion": descripcion,
                "numerical_columns": self.numerical_columns,
                "categorical_columns": self.categorical_columns,
                "num_method": self.num_method,
                "cat_method": self.cat_method,
                "num_threshold": self.num_threshold,
                "cat_threshold": self.cat_threshold,
                "n_muestra": self.n_muestra,
                "random_seed": self.random_seed,
                "fecha_registro": datetime.now(timezone.utc).isoformat(),
                "evidently_api_url": EVIDENTLY_API_URL,
                "registrado_en_evidently": self._evidently_disponible,
            }
            self._guardar_json(self.proyectos_dir / f"{nombre}.json", datos_proyecto)
            logger.success(f"Proyecto registrado localmente: {nombre} (id={self.proyecto_id})")

        return self.proyecto_id

    # ------------------------------------------------------------------ #
    # Registro de datasets (API de Evidently + JSON local)
    # ------------------------------------------------------------------ #
    def registrar_dataset(
        self,
        df: pd.DataFrame,
        nombre: str,
        tipo: str,  # "referencia" o "actual"
    ) -> str:
        """Registra un dataset (referencia o actual) en Evidently Platform y localmente.

        Sube el dataset al servidor de Evidently mediante ``ws.add_dataset()``
        y guarda metadatos locales como JSON para trazabilidad.

        Parametros
        ----------
        df : pd.DataFrame
            Dataset a registrar.
        nombre : str
            Nombre del dataset (ej. "ref2025", "actual_oct2026").
        tipo : str
            Tipo de dataset: "referencia" o "actual".

        Returns
        -------
        str
            ID del dataset (de Evidently Platform o generado localmente).
        """
        if tipo not in ("referencia", "actual"):
            raise ValueError(f'tipo debe ser "referencia" o "actual", se recibió "{tipo}"')

        # 1. Intentar subir a la API de Evidently Platform
        dataset_id_evidently: str | None = None
        if self._evidently_disponible and self._ws is not None and self.proyecto_id:
            try:
                ds_evidently = self._dataset_evidently(df)
                dataset_id_evidently = self._ws.add_dataset(
                    project_id=self.proyecto_id,
                    dataset=ds_evidently,
                    name=nombre,
                    description=f"Dataset {tipo} para monitoreo de drift",
                )
                dataset_id_evidently = str(dataset_id_evidently)
                logger.success(
                    f"Dataset subido a Evidently: {nombre} ({tipo}) | id={dataset_id_evidently}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"No se pudo subir el dataset a Evidently Platform: {exc}")

        # 2. ID local (siempre se genera para trazabilidad)
        dataset_id_local = f"{tipo}-{nombre}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        dataset_id = dataset_id_evidently or dataset_id_local

        # 3. Guardar metadatos locales (respaldo para el reporte PDF)
        if self.registrar_local:
            datos_dataset = {
                "id": dataset_id,
                "id_evidently": dataset_id_evidently,
                "nombre": nombre,
                "tipo": tipo,
                "proyecto_id": self.proyecto_id,
                "proyecto_nombre": self.proyecto_nombre,
                "columnas_monitoreadas": list(self.all_columns),
                "estadisticas": self._estadisticas_dataset(df),
                "fecha_registro": datetime.now(timezone.utc).isoformat(),
            }
            self._guardar_json(self.datasets_dir / f"{dataset_id_local}.json", datos_dataset)
            logger.success(
                f"Dataset registrado localmente: {nombre} ({tipo}) | {len(df):,} filas | id={dataset_id_local}"
            )

        return dataset_id

    # ------------------------------------------------------------------ #
    # Registro de reportes (API de Evidently + JSON local)
    # ------------------------------------------------------------------ #
    def registrar_reporte(
        self,
        nombre: str,
        html_path: Path,
        scores_csv: Path,
        vars_con_drift: int,
        n_ref: int,
        n_actual: int,
        snapshot: Snapshot | None = None,
    ) -> str:
        """Registra un reporte de drift en Evidently Platform y localmente.

        Sube el reporte (Snapshot de Evidently) al servidor mediante
        ``ws.add_run()`` y guarda metadatos locales como JSON.

        Parametros
        ----------
        nombre : str
            Nombre del reporte (ej. "p1_ref2025_vs_oct2026").
        html_path : Path
            Ruta al archivo HTML de Evidently.
        scores_csv : Path
            Ruta al CSV de scores complementario.
        vars_con_drift : int
            Número de variables numéricas con drift detectado.
        n_ref : int
            Filas del conjunto de referencia usadas.
        n_actual : int
            Filas del conjunto actual usadas.
        snapshot : Snapshot | None
            Objeto Snapshot de Evidently (resultado de report.run()). Si se
            proporciona y la API está disponible, se sube a Evidently Platform.

        Returns
        -------
        str
            ID del reporte (de Evidently Platform o generado localmente).
        """
        # 1. Intentar subir el reporte a la API de Evidently Platform
        reporte_id_evidently: str | None = None
        if (
            self._evidently_disponible
            and self._ws is not None
            and self.proyecto_id
            and snapshot is not None
        ):
            try:
                snapshot_ref = self._ws.add_run(
                    project_id=self.proyecto_id,
                    run=snapshot,
                    include_data=True,
                    name=nombre,
                )
                reporte_id_evidently = str(snapshot_ref.id)
                logger.success(
                    f"Reporte subido a Evidently: {nombre} | id={reporte_id_evidently} | url={snapshot_ref.url}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"No se pudo subir el reporte a Evidently Platform: {exc}")

        # 2. ID local (siempre se genera para trazabilidad)
        reporte_id_local = (
            f"reporte-{nombre}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        )
        reporte_id = reporte_id_evidently or reporte_id_local

        # 3. Guardar metadatos locales (respaldo para el reporte PDF)
        if self.registrar_local:
            datos_reporte = {
                "id": reporte_id,
                "id_evidently": reporte_id_evidently,
                "nombre": nombre,
                "proyecto_id": self.proyecto_id,
                "proyecto_nombre": self.proyecto_nombre,
                "html_path": str(html_path),
                "scores_csv": str(scores_csv),
                "vars_con_drift": int(vars_con_drift),
                "n_ref": int(n_ref),
                "n_actual": int(n_actual),
                "num_threshold": self.num_threshold,
                "cat_threshold": self.cat_threshold,
                "fecha_registro": datetime.now(timezone.utc).isoformat(),
            }
            self._guardar_json(self.reportes_dir / f"{reporte_id_local}.json", datos_reporte)
            logger.success(
                f"Reporte registrado localmente: {nombre} | vars_con_drift={vars_con_drift} | id={reporte_id_local}"
            )

        return reporte_id

    # ------------------------------------------------------------------ #
    # Utilidades
    # ------------------------------------------------------------------ #
    def abrir_en_navegador(self, html_path: Path | str) -> None:
        """Abre un archivo HTML de Evidently en el navegador web por defecto."""
        ruta = Path(html_path).resolve()
        url = ruta.as_uri()
        logger.info(f"Abriendo reporte en el navegador: {url}")
        webbrowser.open(url)

    def tabla_scores(self, referencia: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
        """Tabla complementaria (fuera de Evidently) útil para el reporte PDF.

        Calcula, por variable: medias, desviaciones estándares y una métrica de
        distancia normalizada (Wasserstein para numéricas). Marca ``drift=True``
        cuando excede ``num_threshold``. Para columnas categóricas deja las
        métricas numéricas en NaN (Evidently ya calcula Jensen-Shannon en el
        HTML).
        """
        filas = []
        for c in self.all_columns:
            es_numerica = c in self.numerical_columns
            fila = {
                "variable": c,
                "tipo": "numerica" if es_numerica else "categorica",
            }
            if es_numerica:
                sd = float(referencia[c].std())
                w_norm = (
                    float(wasserstein_distance(referencia[c], actual[c])) / sd
                    if sd > 0
                    else np.nan
                )
                fila.update(
                    {
                        "media_ref": float(referencia[c].mean()),
                        "media_actual": float(actual[c].mean()),
                        "std_ref": sd,
                        "std_actual": float(actual[c].std()),
                        "wasserstein_norm": w_norm,
                        "drift": bool(w_norm > self.num_threshold),
                    }
                )
            else:
                fila.update(
                    {
                        "media_ref": np.nan,
                        "media_actual": np.nan,
                        "std_ref": np.nan,
                        "std_actual": np.nan,
                        "wasserstein_norm": np.nan,
                        "drift": np.nan,
                    }
                )
            filas.append(fila)
        return pd.DataFrame(filas)

    # ------------------------------------------------------------------ #
    # API principal
    # ------------------------------------------------------------------ #
    def comparar(
        self,
        referencia: pd.DataFrame,
        actual: pd.DataFrame,
        nombre: str,
        *,
        mlflow_run_name: str | None = None,
        log_mlflow: bool = True,
        abrir_navegador: bool | None = None,
        registrar_proyecto_nombre: str | None = None,
        registrar_proyecto_descripcion: str | None = None,
    ) -> dict:
        """Ejecuta la comparación completa de drift con registro en Evidently.

        Pasos:
        1. Registra el proyecto en Evidently Platform (si se indica nombre).
        2. Registra los datasets de referencia y actual en Evidently Platform.
        3. Muestrea referencia y actual (si exceden ``n_muestra``).
        4. Genera el HTML de Evidently (data drift + data summary).
        5. (Opcional) Abre el HTML en el navegador web.
        6. Calcula la tabla de scores complementaria y la guarda como CSV.
        7. Registra el reporte (run) en Evidently Platform.
        8. Guarda metadatos locales JSON para trazabilidad.
        9. (Opcional) Loguea parámetros/métricas/artefactos a MLflow.

        Parametros
        ----------
        referencia : pd.DataFrame
            Conjunto de referencia (periodo de entrenamiento).
        actual : pd.DataFrame
            Conjunto actual (periodo de producción / posterior).
        nombre : str
            Identificador del comparativo (se usa en nombres de archivo).
        mlflow_run_name : str | None
            Nombre del run de MLflow. Si es None y ``log_mlflow=True``,
            se usa ``drift_<nombre>``.
        log_mlflow : bool
            Si True, registra el experimento en MLflow.
        abrir_navegador : bool | None
            Si True, abre el HTML en el navegador. Si None, usa el valor
            de ``self.abrir_navegador`` (por defecto False).
        registrar_proyecto_nombre : str | None
            Nombre del proyecto a registrar en Evidently. Si None, no se registra proyecto.
        registrar_proyecto_descripcion : str | None
            Descripción del proyecto.

        Returns
        -------
        dict
            Resumen con claves: ``nombre``, ``n_ref``, ``n_actual``,
            ``vars_con_drift`` (solo numéricas), ``html`` (ruta del HTML),
            ``scores_csv`` (ruta del CSV), ``proyecto_id``,
            ``dataset_referencia_id``, ``dataset_actual_id``, ``reporte_id``.
        """
        if referencia.empty or actual.empty:
            raise ValueError("Referencia y actual no pueden estar vacías.")

        faltantes_ref = [c for c in self.all_columns if c not in referencia.columns]
        faltantes_cur = [c for c in self.all_columns if c not in actual.columns]
        if faltantes_ref or faltantes_cur:
            raise ValueError(
                f"Columnas faltantes: referencia={faltantes_ref} | actual={faltantes_cur}"
            )

        # 0. Registrar proyecto (si se pide)
        if registrar_proyecto_nombre:
            self.registrar_proyecto(
                registrar_proyecto_nombre, descripcion=registrar_proyecto_descripcion
            )

        # 1. Registrar datasets (referencia y actual) en Evidently Platform
        dataset_ref_id = self.registrar_dataset(referencia, f"ref-{nombre}", tipo="referencia")
        dataset_cur_id = self.registrar_dataset(actual, f"cur-{nombre}", tipo="actual")

        # 2. Muestrear para el reporte de Evidently
        ref_m = self._muestrear(referencia, self.n_muestra, self.random_seed)
        cur_m = self._muestrear(actual, self.n_muestra, self.random_seed)

        # 3. Generar reporte HTML de Evidently y obtener el Snapshot
        ds_ref = self._dataset_evidently(ref_m)
        ds_cur = self._dataset_evidently(cur_m)
        report = Report(
            [
                DataDriftPreset(
                    num_method=self.num_method,
                    num_threshold=self.num_threshold,
                    cat_method=self.cat_method,
                    cat_threshold=self.cat_threshold,
                ),
                DataSummaryPreset(),
            ]
        )
        snapshot = report.run(current_data=ds_cur, reference_data=ds_ref)
        salida_html = self.out_dir / f"drift_{nombre}.html"
        snapshot.save_html(str(salida_html))
        logger.success(f"Reporte HTML de Evidently -> {salida_html}")

        # 4. Abrir en navegador (si se solicita)
        abrir = self.abrir_navegador if abrir_navegador is None else abrir_navegador
        if abrir:
            self.abrir_en_navegador(salida_html)

        # 5. Tabla de scores complementaria
        scores = self.tabla_scores(ref_m, cur_m)
        scores_csv = self.out_dir / f"drift_{nombre}_scores.csv"
        scores.to_csv(scores_csv, index=False)
        logger.info("\n" + scores.round(4).to_string(index=False))

        vars_con_drift = int(
            scores.loc[scores["tipo"] == "numerica", "drift"].fillna(False).astype(bool).sum()
        )
        logger.info(
            f"Variables con drift (numéricas, umbral={self.num_threshold}): {vars_con_drift}"
        )

        # 6. Registrar reporte en Evidently Platform (subir el Snapshot)
        reporte_id = self.registrar_reporte(
            nombre=nombre,
            html_path=salida_html,
            scores_csv=scores_csv,
            vars_con_drift=vars_con_drift,
            n_ref=len(ref_m),
            n_actual=len(cur_m),
            snapshot=snapshot,
        )

        resumen = {
            "nombre": nombre,
            "n_ref": len(ref_m),
            "n_actual": len(cur_m),
            "vars_con_drift": vars_con_drift,
            "html": str(salida_html),
            "scores_csv": str(scores_csv),
            "proyecto_id": self.proyecto_id,
            "proyecto_nombre": self.proyecto_nombre,
            "dataset_referencia_id": dataset_ref_id,
            "dataset_actual_id": dataset_cur_id,
            "reporte_id": reporte_id,
        }

        # 7. MLflow (opcional)
        if log_mlflow:
            mlflow.set_tracking_uri(URI_MLFLOW_TRACKING)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_MONITOREO)
            run_name = mlflow_run_name or f"drift_{nombre}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(
                    {
                        "n_muestra": self.n_muestra,
                        "num_method": self.num_method,
                        "cat_method": self.cat_method,
                        "num_threshold": self.num_threshold,
                        "cat_threshold": self.cat_threshold,
                        "n_ref": len(ref_m),
                        "n_actual": len(cur_m),
                        "nombre": nombre,
                        "proyecto_id": self.proyecto_id or "",
                        "evidently_api": self._evidently_disponible,
                    }
                )
                mlflow.log_metric("vars_con_drift", float(vars_con_drift))
                mlflow.log_artifacts(str(self.out_dir), artifact_path=f"drift_{nombre}")
                logger.success(f"Run de MLflow '{run_name}' registrado con artefactos de drift.")

        return resumen


# ---------------------------------------------------------------------- #
# Importación tardía de Snapshot para anotaciones de tipo
# ---------------------------------------------------------------------- #
from evidently.core.report import Snapshot

# ---------------------------------------------------------------------- #
# CLI opcional: permite ejecutar este módulo directamente como script
# ---------------------------------------------------------------------- #
import typer

app = typer.Typer(help="Monitoreo de drift con Evidently (clase reutilizable).")


@app.command()
def ejemplo_p1():
    """Ejemplo de uso con P1 (solo si existen los datos de modelado)."""
    from proyecto_mibici.config import (
        P1_ACTUAL_FIN,
        P1_FEATURES_V2,
        P1_TEST_FIN,
        P1_VAL_INICIO,
    )
    from proyecto_mibici.features import construir_dataset_p1

    df_p1, _ = construir_dataset_p1()
    t = df_p1["target_hour"]
    actual = df_p1[(t >= pd.Timestamp(P1_TEST_FIN)) & (t < pd.Timestamp(P1_ACTUAL_FIN))]
    ref = df_p1[t < pd.Timestamp(P1_VAL_INICIO)]

    monitor = EvidentlyMonitor(
        numerical_columns=[c for c in P1_FEATURES_V2 if c != "target_is_weekend"],
        categorical_columns=["target_is_weekend"],
        nombre_proyecto="p1-demanda-horaria",
        descripcion_proyecto="Monitoreo de drift P1: referencia 2025 vs octubre 2026",
    )
    resumen = monitor.comparar(ref, actual)
    logger.info(f"Resumen P1: {resumen}")


if __name__ == "__main__":
    app()
