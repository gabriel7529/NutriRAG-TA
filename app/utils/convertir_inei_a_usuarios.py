import pandas as pd

# === Ruta de tu archivo original ===
input_file = "data/BD_ADOLESCENTES_2017.xlsx"
output_file = "data/usuarios.csv"

# === Funciones de diagnóstico clínico ===
def diagnosticar_anemia(edad, sexo, hb):
    if pd.isnull(hb):
        return "desconocido"
    if edad >= 15:
        return "anemia" if (sexo == "M" and hb < 13.0) or (sexo == "F" and hb < 12.0) else "normal"
    elif edad >= 12:
        return "anemia" if hb < 12.0 else "normal"
    return "no_aplica"

def diagnosticar_sobrepeso(imc):
    if imc > 25:
        return "sobrepeso"
    elif imc < 18.5:
        return "delgadez"
    else:
        return "normal"

# === Cargar y procesar datos ===
df = pd.read_excel(input_file)

# Renombrar columnas para facilitar
df = df.rename(columns={
    "ID": "id_usuario",
    "P401_SEXO": "sexo",
    "P406_EDAD_AÑOS": "edad",
    "P408_PESO_NETO": "peso_kg",
    "P410_TALLA": "talla_cm",
    "P420_VALOR_HB": "hemoglobina"
})

# Mapear sexo numérico → texto
df["sexo"] = df["sexo"].map({1: "M", 2: "F"})

# Calcular IMC
df["talla_m"] = df["talla_cm"] / 100
df["imc"] = df["peso_kg"] / (df["talla_m"] ** 2)

# Aplicar diagnósticos
df["anemia"] = df.apply(lambda row: diagnosticar_anemia(row["edad"], row["sexo"], row["hemoglobina"]), axis=1)
df["nutricional"] = df["imc"].apply(diagnosticar_sobrepeso)

# Unificar diagnóstico
def combinar_diagnostico(row):
    if row["anemia"] == "anemia" and row["nutricional"] == "sobrepeso":
        return "ambos"
    elif row["anemia"] == "anemia":
        return "anemia"
    elif row["nutricional"] == "sobrepeso":
        return "sobrepeso"
    else:
        return "normal"

df["diagnostico"] = df.apply(combinar_diagnostico, axis=1)

# Seleccionar columnas finales
final = df[[
    "id_usuario", "edad", "sexo", "peso_kg", "talla_cm", "hemoglobina", "diagnostico"
]].copy()

# Guardar archivo limpio
final.to_csv(output_file, index=False)
print(f"✅ Archivo generado: {output_file}")
