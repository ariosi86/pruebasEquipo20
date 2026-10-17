from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Rutas de artefactos de datos y metadatos
MANIFEST_PATH = DATA_DIR / "manifest.json"
STATIONS_PROCESSED_PATH = PROCESSED_DATA_DIR / "stations.parquet"
TRIPS_PROCESSED_PATH = PROCESSED_DATA_DIR / "trips.parquet"

# ==========================================
# 2. PARÁMETROS GLOBALES Y TRACKING
# ==========================================
RANDOM_SEED = 42
MLFLOW_EXPERIMENT_NAME = "mibici-equipo20"


# ==========================================
# 3. FUENTES Y ENLACES DE DESCARGA DIRECTA
# ==========================================
#  los enlaces directos obtenidos de https://mibici.net/es/datos-abiertos/
MIBICI_HOST = "https://www.mibici.net"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-419,es;q=0.9,en;q=0.8",
    "Referer": "https://mibici.net/es/datos-abiertos/",
}

DATA_URLS = {
    # Nomenclatura más reciente
    "stations": f"{MIBICI_HOST}/site/assets/files/1118/nomenclatura_2026_09.csv",
    
    # Año 2025 completo (12 meses cerrados)
   
    "trips": {
    "2025-01": f"{MIBICI_HOST}/site/assets/files/97653/datos_abiertos_2025_01.csv",
    "2025-02": f"{MIBICI_HOST}/site/assets/files/99879/datos_abiertos_2025_02.csv",
    "2025-03": f"{MIBICI_HOST}/site/assets/files/101986/datos_abiertos_2025_03.csv",
    "2025-04": f"{MIBICI_HOST}/site/assets/files/103813/datos_abiertos_2025_04.csv",
    "2025-05": f"{MIBICI_HOST}/site/assets/files/105513/datos_abiertos_2025_05.csv",
    "2025-06": f"{MIBICI_HOST}/site/assets/files/107088/datos_abiertos_2025_06.csv",
    "2025-07": f"{MIBICI_HOST}/site/assets/files/108898/datos_abiertos_2025_07.csv",
    "2025-08": f"{MIBICI_HOST}/site/assets/files/111021/datos_abiertos_2025_08.csv",
    "2025-09": f"{MIBICI_HOST}/site/assets/files/113216/datos_abiertos_2025_09.csv",
    "2025-10": f"{MIBICI_HOST}/site/assets/files/115104/datos_abiertos_2025_10.csv",
    "2025-11": f"{MIBICI_HOST}/site/assets/files/116773/datos_abiertos_2025_11.csv",
    "2025-12": f"{MIBICI_HOST}/site/assets/files/118153/datos_abiertos_2025_12.csv",

    #Año 2026 (9 meses cerrados hasta septiembre)
    "2026-01": f"{MIBICI_HOST}/site/assets/files/120163/datos_abiertos_2026_01.csv",
    "2026-02": f"{MIBICI_HOST}/site/assets/files/121761/datos_abiertos_2026_02.csv",
    "2026-03": f"{MIBICI_HOST}/site/assets/files/123236/datos_abiertos_2026_03.csv",
    "2026-04": f"{MIBICI_HOST}/site/assets/files/124927/datos_abiertos_2026_04.csv",
    "2026-05": f"{MIBICI_HOST}/site/assets/files/125943/datos_abiertos_2026_05.csv",
    "2026-06": f"{MIBICI_HOST}/site/assets/files/127734/datos_abiertos_2026_06.csv",
    "2026-07": f"{MIBICI_HOST}/site/assets/files/129006/datos_abiertos_2026_07.csv",
    "2026-08": f"{MIBICI_HOST}/site/assets/files/130362/datos_abiertos_2026_08.csv",
    "2026-09": f"{MIBICI_HOST}/site/assets/files/131825/datos_abiertos_2026_09.csv",
    }
}

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
