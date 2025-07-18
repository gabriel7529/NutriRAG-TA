from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .router import router

app = FastAPI(
    title="NutriRAG API",
    description="API para generar planes nutricionales personalizados para adolescentes peruanos con anemia o sobrepeso.",
    version="1.0.0"
)

# Habilita CORS por si se conecta desde Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas desde router.py
app.include_router(router)
