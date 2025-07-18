# app/config.py

import os

# === RUTAS DE DATOS ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta al archivo con recetas preprocesadas
RUTA_RECETAS = os.path.join(BASE_DIR, "..", "data", "recetas.csv")
RUTA_NUTRICION = os.path.join(BASE_DIR, "..", "data", "nutricion_completa.csv")

# Ruta al índice vectorial de Chroma
RUTA_CHROMA = os.path.join(BASE_DIR, "..", "data", "chroma_index")



# Parámetros de generación por defecto
GEN_MAX_LENGTH = 180
GEN_TEMPERATURE = 0.8
GEN_TOP_K = 50
GEN_TOP_P = 0.95

# === CONFIGURACIÓN CHROMA ===
CHROMA_COLLECTION = "recetas"
CHROMA_PERSIST = True  # para mantener la base en disco
