# app/config/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Rutas de archivos
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"

RUTA_RECETAS = DATA_DIR / "recetas.csv"
RUTA_NUTRICION = DATA_DIR / "tabla_composicion_alimentos.csv"
RUTA_PORCIONES_12_14 = DATA_DIR / "porciones_12_14.csv"
RUTA_PORCIONES_15_17 = DATA_DIR / "porciones_15_17.csv"
RUTA_CHROMA = DATA_DIR / "chroma_db"

# Configuración OpenAI - NUNCA hardcodear la API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "400"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

# Configuración del vectorstore
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_COLLECTION_RECETAS = "recetas_v2"
CHROMA_COLLECTION_NUTRICION = "nutricion_v2"

# Configuración de logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Validación de configuración crítica
def validar_configuracion():
    """Valida que todas las configuraciones críticas estén presentes."""
    errores = []

    if not OPENAI_API_KEY:
        errores.append("❌ OPENAI_API_KEY no está configurada en las variables de entorno")

    archivos_requeridos = [
        RUTA_RECETAS,
        RUTA_NUTRICION,
        RUTA_PORCIONES_12_14,
        RUTA_PORCIONES_15_17
    ]

    for archivo in archivos_requeridos:
        if not archivo.exists():
            errores.append(f"❌ Archivo no encontrado: {archivo}")

    if errores:
        print("\n".join(errores))
        return False

    print("✅ Configuración validada correctamente")
    return True

# =====================================================
