
def calcular_get(edad: int, sexo: str) -> int:
    if sexo.upper() == "M":
        return 2200 if edad < 15 else 2500
    else:
        return 2000 if edad < 15 else 2100

def distribuir_calorias(get: int) -> dict:
    return {
        "desayuno": int(get * 0.20),
        "almuerzo": int(get * 0.40),
        "cena": int(get * 0.30),
        "refrigerio": int(get * 0.10),
    }

def filtrar_por_condicion(recetas: list, condicion: str) -> list:
    if condicion == "anemia":
        return [r for r in recetas if r["hierro_mg"] >= 3.5]
    if condicion == "sobrepeso":
        return [r for r in recetas if r["energia_kcal"] <= 800]
    return recetas
