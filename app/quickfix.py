# quick_fix.py - Script para solucionar rápidamente el problema

import os
import pandas as pd
import sys
from pathlib import Path

def verificar_y_reparar():
    """Verifica y repara el vectorstore paso a paso."""

    print("🔧 REPARACIÓN RÁPIDA DEL VECTORSTORE")
    print("=" * 50)

    # 1. Verificar estructura de archivos
    print("\n1️⃣ Verificando archivos...")

    # Ajusta estas rutas según tu proyecto
    rutas = {
        "recetas": "data/recetas.csv",  # Cambia por tu ruta real
        "nutricion": "data/tabla_nutricion.csv",  # Cambia por tu ruta real
        "chroma": "data/chroma_db"  # Cambia por tu ruta real
    }

    # Si tus rutas son diferentes, úsalas aquí:
    # rutas = {
    #     "recetas": "ruta/a/tu/archivo/recetas.csv",
    #     "nutricion": "ruta/a/tu/archivo/nutricion.csv",
    #     "chroma": "ruta/a/tu/directorio/chroma"
    # }

    for nombre, ruta in rutas.items():
        if os.path.exists(ruta):
            if nombre in ["recetas", "nutricion"]:
                try:
                    df = pd.read_csv(ruta)
                    print(f"  ✅ {nombre}: {len(df)} filas")
                except Exception as e:
                    print(f"  ❌ {nombre}: Error leyendo - {e}")
            else:
                print(f"  ✅ {nombre}: Directorio existe")
        else:
            print(f"  ❌ {nombre}: No encontrado - {ruta}")

    # 2. Verificar contenido de recetas
    print("\n2️⃣ Verificando contenido de recetas...")

    try:
        df_recetas = pd.read_csv(rutas["recetas"])
        print(f"  📊 Total recetas: {len(df_recetas)}")
        print(f"  📋 Columnas: {list(df_recetas.columns)}")

        # Verificar columnas críticas
        columnas_necesarias = ["nombre", "ingredientes", "preparacion"]
        faltantes = [col for col in columnas_necesarias if col not in df_recetas.columns]

        if faltantes:
            print(f"  ⚠️ Columnas faltantes: {faltantes}")
        else:
            print(f"  ✅ Columnas principales presentes")

        # Mostrar muestra
        print(f"\n  📝 Muestra de datos:")
        for i, row in df_recetas.head(2).iterrows():
            nombre = row.get('nombre', 'Sin nombre')
            print(f"    - {nombre}")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

    # 3. Recrear vectorstore
    print("\n3️⃣ Recreando vectorstore...")

    try:
        # Eliminar directorio existente
        import shutil
        if os.path.exists(rutas["chroma"]):
            shutil.rmtree(rutas["chroma"])
            print(f"  🗑️ Directorio anterior eliminado")

        # Crear nuevo directorio
        os.makedirs(rutas["chroma"], exist_ok=True)
        print(f"  📁 Nuevo directorio creado")

        # Ahora crear vectorstore manualmente
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma

        print(f"  🧠 Inicializando embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        print(f"  📚 Procesando recetas...")

        # Procesar recetas paso a paso
        textos = []
        metadatos = []

        for idx, row in df_recetas.iterrows():
            # Crear texto simple
            nombre = str(row.get('nombre', f'Receta {idx}'))
            ingredientes = str(row.get('ingredientes', 'Sin ingredientes'))[:200]

            texto = f"Receta: {nombre}. Ingredientes: {ingredientes}"
            textos.append(texto)

            # Metadatos simples (solo tipos primitivos)
            metadata = {
                'id': f'receta_{idx}',
                'nombre': nombre,
                'energia_kcal': float(row.get('energia_kcal', 0)) if pd.notna(row.get('energia_kcal')) else 0.0,
                'proteinas_g': float(row.get('proteinas_g', 0)) if pd.notna(row.get('proteinas_g')) else 0.0,
                'hierro_mg': float(row.get('hierro_mg', 0)) if pd.notna(row.get('hierro_mg')) else 0.0,
                'condicion_principal': 'anemia' if float(row.get('hierro_mg', 0)) >= 3 else 'general'
            }
            metadatos.append(metadata)

        print(f"  🔄 Creando vectorstore con {len(textos)} documentos...")

        # Crear vectorstore
        vectordb = Chroma(
            collection_name="recetas_test",
            persist_directory=rutas["chroma"],
            embedding_function=embeddings
        )

        # Agregar en lotes pequeños
        batch_size = 10
        for i in range(0, len(textos), batch_size):
            batch_textos = textos[i:i+batch_size]
            batch_metadatos = metadatos[i:i+batch_size]

            vectordb.add_texts(texts=batch_textos, metadatas=batch_metadatos)
            print(f"    ✅ Lote {i//batch_size + 1} agregado")

        vectordb.persist()
        print(f"  💾 Vectorstore persistido")

        # 4. Probar búsqueda
        print(f"\n4️⃣ Probando búsqueda...")

        resultados = vectordb.similarity_search("pescado", k=3)
        print(f"  🔍 Búsqueda 'pescado': {len(resultados)} resultados")

        if len(resultados) > 0:
            for i, doc in enumerate(resultados[:2]):
                nombre = doc.metadata.get('nombre', 'Sin nombre')
                print(f"    {i+1}. {nombre}")

        resultados_anemia = vectordb.similarity_search("anemia hierro", k=3)
        print(f"  🔍 Búsqueda 'anemia hierro': {len(resultados_anemia)} resultados")

        if len(resultados) > 0 and len(resultados_anemia) > 0:
            print(f"\n🎉 VECTORSTORE CREADO EXITOSAMENTE")
            print(f"   Total documentos: {len(textos)}")
            print(f"   Búsquedas funcionando: ✅")
            return True
        else:
            print(f"\n❌ PROBLEMA: Vectorstore creado pero búsquedas fallan")
            return False

    except Exception as e:
        print(f"  ❌ Error creando vectorstore: {e}")
        import traceback
        print(f"  📍 Detalle: {traceback.format_exc()}")
        return False

def crear_archivo_recetas_prueba():
    """Crea un archivo de recetas de prueba si no existe."""
    print(f"\n🆘 CREANDO ARCHIVO DE RECETAS DE PRUEBA")

    # Datos de prueba basados en tu CSV original
    recetas_prueba = [
        {
            "nombre": "Arvejitas Con Anchoveta Rebozada",
            "ingredientes": "anchoveta, arvejas, arroz, huevos, cebolla, aceite",
            "preparacion": "Lavar anchovetas, rebozar con huevo, freír. Cocinar arvejas.",
            "energia_kcal": 450.0,
            "proteinas_g": 25.0,
            "hierro_mg": 4.5,
            "raciones": 4
        },
        {
            "nombre": "Pescado Saltado Con Frijoles",
            "ingredientes": "pescado, frijoles, cebolla, tomate, ají amarillo",
            "preparacion": "Saltear pescado con verduras, agregar frijoles cocidos.",
            "energia_kcal": 380.0,
            "proteinas_g": 22.0,
            "hierro_mg": 3.8,
            "raciones": 4
        },
        {
            "nombre": "Escabeche de Pescado",
            "ingredientes": "pescado, cebolla, vinagre, ají amarillo, camote",
            "preparacion": "Freír pescado, preparar escabeche con vinagre y verduras.",
            "energia_kcal": 520.0,
            "proteinas_g": 28.0,
            "hierro_mg": 5.2,
            "raciones": 4
        }
    ]

    df_prueba = pd.DataFrame(recetas_prueba)

    # Crear directorio si no existe
    os.makedirs("data", exist_ok=True)

    # Guardar archivo
    archivo_prueba = "data/recetas_prueba.csv"
    df_prueba.to_csv(archivo_prueba, index=False)

    print(f"  ✅ Archivo creado: {archivo_prueba}")
    print(f"  📊 Recetas de prueba: {len(df_prueba)}")

    return archivo_prueba

if __name__ == "__main__":
    print("🚀 INICIANDO REPARACIÓN...")

    # Opción 1: Intentar reparar con archivos existentes
    if verificar_y_reparar():
        print(f"\n✅ REPARACIÓN EXITOSA")
    else:
        print(f"\n⚠️ REPARACIÓN FALLÓ")

        # Opción 2: Crear archivos de prueba
        respuesta = input(f"\n¿Crear archivos de prueba? (s/n): ")
        if respuesta.lower() in ['s', 'si', 'y', 'yes']:
            archivo_prueba = crear_archivo_recetas_prueba()
            print(f"\nUsando archivo de prueba para verificar...")

            # Actualizar rutas para usar archivo de prueba
            # y repetir verificación...
