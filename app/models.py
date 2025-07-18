

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class SexoEnum(str, Enum):
    M = "Masculino"
    F = "Femenino"

class PerfilUsuario(BaseModel):
    nombre: str = Field(..., min_length=1)
    edad: int = Field(..., ge=12, le=17)
    sexo: SexoEnum
    peso: float = Field(..., ge=30.0, le=120.0)
    altura: int = Field(..., ge=130, le=200)
    condiciones: List[str] = Field(default_factory=list)
    actividad_fisica: str
    preferencias: List[str] = Field(default_factory=list)
    alergias: Optional[str] = None
