from fastapi import APIRouter
from .models import PerfilUsuario
from .utils.retrievalqa import generar_plan

router = APIRouter()

@router.post("/generate_plan")
def generar_plan_nutricional(perfil: PerfilUsuario):
    return generar_plan(perfil.dict())
