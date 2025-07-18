import requests
from bs4 import BeautifulSoup
import pandas as pd

def extraer_tabla_porciones(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    tabla = soup.find('table')  # Encuentra la primera tabla

    df = pd.read_html(str(tabla))[0]  # Convierte HTML a DataFrame
    return df

# URLs de las páginas
url_12_14 = "https://alimentacionsaludable.ins.gob.pe/adolescentes/porciones-recomendadas/adolescentes-de-12-14-anos-0"
url_15_17 = "https://alimentacionsaludable.ins.gob.pe/adolescentes/porciones-recomendadas/adolescentes-de-15-17-anos-0"

# Extraer dataframes
df_12_14 = extraer_tabla_porciones(url_12_14)
df_15_17 = extraer_tabla_porciones(url_15_17)

# Guardar como CSV
df_12_14.to_csv("data/porciones_12_14.csv", index=False)
df_15_17.to_csv("data/porciones_15_17.csv", index=False)

# También podrías convertir a JSON
json_12_14 = df_12_14.set_index(df_12_14.columns[0]).to_dict()[df_12_14.columns[1]]
json_15_17 = df_15_17.set_index(df_15_17.columns[0]).to_dict()[df_15_17.columns[1]]
