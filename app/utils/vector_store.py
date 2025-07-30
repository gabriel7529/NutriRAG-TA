
import os
import pandas as pd
import shutil
import logging
import numpy as np
import random
import time
import gc
from typing import List, Dict, Tuple, Optional
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from ..config import RUTA_NUTRICION, RUTA_CHROMA, RUTA_RECETAS

logger = logging.getLogger(__name__)

class NutritionalRAGSystemFixed:
    """Sistema RAG corregido que evita respuestas template y genera variedad."""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore_alimentos = None
        self.vectorstore_recetas = None
        self.tabla_nutricional = None
        self.recetas_df = None

        # Cache para evitar repeticiones en una sesión
        self.cache_busquedas = {}
        self.contador_consultas = {}

        # Configuración de condiciones CORREGIDA
        self.condiciones_nutrientes = {
            "anemia": {
                "nutrientes_prioritarios": ["HIERRO_MG"],
                "hierro_minimo": 2.0,
                "proteinas_minimo": 10.0,
                "energia_max": 1000,
                "keywords": ["pescado", "carne", "hígado", "sangre", "quinua", "lentejas", "espinaca", "anchoveta"],
                "boost_factor": 2.0
            },
            "sobrepeso": {
                "nutrientes_prioritarios": ["FIBRA_DIETARIA_G", "PROTEINAS_G"],
                "energia_max": 150,
                "fibra_minimo": 2.0,
                "proteinas_minimo": 8.0,
                "keywords": ["verduras", "frutas", "ensalada", "brócoli", "apio", "lechuga", "tomate"],
                "boost_factor": 1.5
            },
            "diabetes": {
                "nutrientes_prioritarios": ["FIBRA_DIETARIA_G"],
                "energia_max": 200,
                "fibra_minimo": 3.0,
                "keywords": ["integral", "verduras", "legumbres", "quinua"],
                "boost_factor": 1.8
            },
            "general": {
                "nutrientes_prioritarios": ["PROTEINAS_G", "CALCIO_MG"],
                "proteinas_minimo": 5.0,
                "calcio_minimo": 30.0,
                "keywords": ["balanceado", "nutritivo", "completo"],
                "boost_factor": 1.0
            }
        }

    def inicializar_sistema(self) -> bool:
        """Inicializa todo el sistema RAG."""
        try:
            logger.info("🚀 Inicializando sistema RAG nutricional corregido...")

            if not self._cargar_datos():
                return False

            if not self._crear_vectorstore_alimentos_mejorado():
                return False

            if not self._crear_vectorstore_recetas_mejorado():
                return False

            logger.info("✅ Sistema RAG corregido inicializado")
            return True

        except Exception as e:
            logger.error(f"❌ Error inicializando sistema: {e}")
            return False

    def _cargar_datos(self) -> bool:
        """Carga y valida los datos."""
        try:
            if not os.path.exists(RUTA_NUTRICION):
                logger.error(f"Archivo nutricional no encontrado: {RUTA_NUTRICION}")
                return False

            self.tabla_nutricional = pd.read_csv(RUTA_NUTRICION)
            logger.info(f"📊 Tabla nutricional cargada: {len(self.tabla_nutricional)} alimentos")

            if not os.path.exists(RUTA_RECETAS):
                logger.error(f"Archivo de recetas no encontrado: {RUTA_RECETAS}")
                return False

            self.recetas_df = pd.read_csv(RUTA_RECETAS)
            logger.info(f"📊 Recetas cargadas: {len(self.recetas_df)} disponibles")

            self._limpiar_datos_mejorado()

            if len(self.tabla_nutricional) < 10:
                logger.error("❌ Muy pocos alimentos en la tabla nutricional")
                return False

            if len(self.recetas_df) < 5:
                logger.error("❌ Muy pocas recetas disponibles")
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Error cargando datos: {e}")
            return False

    def _limpiar_datos_mejorado(self):
        """Limpia y normaliza los datos con validaciones mejoradas."""
        self.tabla_nutricional = self.tabla_nutricional.fillna(0)

        columnas_requeridas = ['NOMBRE_ALIMENTO', 'ENERGIA_KCAL', 'PROTEINAS_G', 'HIERRO_MG']
        columnas_faltantes = [col for col in columnas_requeridas if col not in self.tabla_nutricional.columns]

        if columnas_faltantes:
            logger.error(f"❌ Columnas faltantes en tabla nutricional: {columnas_faltantes}")
            return False

        columnas_numericas = ['ENERGIA_KCAL', 'PROTEINAS_G', 'HIERRO_MG', 'CALCIO_MG',
                             'VITAMINA_C_MG', 'FIBRA_DIETARIA_G']

        for col in columnas_numericas:
            if col in self.tabla_nutricional.columns:
                self.tabla_nutricional[col] = pd.to_numeric(
                    self.tabla_nutricional[col], errors='coerce'
                ).fillna(0)

        self.tabla_nutricional = self.tabla_nutricional[
            (self.tabla_nutricional['NOMBRE_ALIMENTO'].notna()) &
            (self.tabla_nutricional['NOMBRE_ALIMENTO'].str.len() > 2)
        ]

        self.recetas_df = self.recetas_df.fillna({
            'nombre': 'Receta sin nombre',
            'ingredientes': 'Sin ingredientes',
            'preparacion': 'Sin preparación',
            'energia_kcal': 0,
            'proteinas_g': 0,
            'hierro_mg': 0
        })

        self.recetas_df = self.recetas_df.drop_duplicates(subset=['nombre'])

        logger.info(f"🧹 Datos limpiados: {len(self.tabla_nutricional)} alimentos, {len(self.recetas_df)} recetas")

    def _crear_vectorstore_alimentos_mejorado(self) -> bool:
        """Crea vectorstore mejorado para alimentos con manejo de errores corregido."""
        try:
            logger.info("🥗 Creando vectorstore de alimentos mejorado...")

            textos_alimentos = []
            metadatos_alimentos = []

            for idx, row in self.tabla_nutricional.iterrows():
                nombre = str(row.get('NOMBRE_ALIMENTO', 'Sin nombre')).strip()

                if not nombre or nombre == 'Sin nombre' or len(nombre) < 3:
                    continue

                energia = float(row.get('ENERGIA_KCAL', 0))
                proteinas = float(row.get('PROTEINAS_G', 0))
                hierro = float(row.get('HIERRO_MG', 0))
                calcio = float(row.get('CALCIO_MG', 0))
                fibra = float(row.get('FIBRA_DIETARIA_G', 0))
                vitamina_c = float(row.get('VITAMINA_C_MG', 0))

                texto_base = f"Alimento nutritivo: {nombre}. "

                if energia > 0:
                    nivel_energia = "alto" if energia > 300 else "moderado" if energia > 150 else "bajo"
                    texto_base += f"Energía {nivel_energia}: {energia} kcal por 100g. "

                if proteinas > 0:
                    nivel_proteina = "excelente" if proteinas > 15 else "bueno" if proteinas > 8 else "moderado"
                    texto_base += f"Fuente {nivel_proteina} de proteínas: {proteinas}g. "

                if hierro > 0:
                    nivel_hierro = "muy rico" if hierro > 5 else "rico" if hierro > 2 else "contiene"
                    texto_base += f"{nivel_hierro} en hierro: {hierro}mg. "

                if calcio > 0:
                    nivel_calcio = "excelente" if calcio > 100 else "bueno" if calcio > 50 else "contiene"
                    texto_base += f"{nivel_calcio} fuente de calcio: {calcio}mg. "

                if fibra > 0:
                    nivel_fibra = "alto" if fibra > 5 else "moderado" if fibra > 2 else "bajo"
                    texto_base += f"Contenido {nivel_fibra} de fibra: {fibra}g. "

                categorias = []
                for condicion, config in self.condiciones_nutrientes.items():
                    es_recomendado = self._evaluar_alimento_para_condicion(row, condicion)
                    if es_recomendado["recomendado"]:
                        categorias.append(condicion)
                        texto_base += f"Recomendado para {condicion}: {es_recomendado['razon']}. "

                if not categorias:
                    categorias.append("general")
                    texto_base += "Alimento de uso general. "

                palabras_nombre = nombre.lower().split()
                if len(palabras_nombre) > 1:
                    texto_base += f"También conocido como: {' '.join(palabras_nombre)}. "

                grupo_alimento = self._clasificar_grupo_alimento(nombre.lower())
                texto_base += f"Grupo alimentario: {grupo_alimento}. "

                textos_alimentos.append(texto_base)

                metadatos_alimentos.append({
                    'id_alimento': f'alimento_{idx}',
                    'nombre': nombre,
                    'energia_kcal': energia,
                    'proteinas_g': proteinas,
                    'hierro_mg': hierro,
                    'calcio_mg': calcio,
                    'fibra_g': fibra,
                    'vitamina_c_mg': vitamina_c,
                    'categorias_recomendado': ','.join(categorias),
                    'grupo_alimento': grupo_alimento,
                    'nivel_hierro': "alto" if hierro > 3 else "medio" if hierro > 1 else "bajo",
                    'nivel_energia': "alto" if energia > 300 else "medio" if energia > 150 else "bajo",
                    'apto_anemia': hierro >= 2.0,
                    'apto_sobrepeso': energia <= 150 and fibra >= 2.0,
                    'indice_nutricional': round((proteinas * 2 + hierro * 3 + fibra * 1.5) / 3, 2)
                })

            if not textos_alimentos:
                logger.error("❌ No se pudieron crear textos para alimentos")
                return False

            # CORRECCIÓN: Manejo mejorado de ChromaDB con archivos bloqueados
            chroma_alimentos = os.path.join(RUTA_CHROMA, "alimentos_v2")

            # Intentar limpiar con reintentos
            max_reintentos = 3
            for intento in range(max_reintentos):
                try:
                    if os.path.exists(chroma_alimentos):
                        gc.collect()  # Limpiar memoria
                        shutil.rmtree(chroma_alimentos)
                        logger.info(f"🧹 Directorio limpiado en intento {intento + 1}")
                    break
                except PermissionError as e:
                    if intento < max_reintentos - 1:
                        logger.warning(f"⚠️ Archivo en uso, reintentando...")
                        time.sleep(2)
                    else:
                        # Usar nombre alternativo
                        suffix = random.randint(1000, 9999)
                        chroma_alimentos = os.path.join(RUTA_CHROMA, f"alimentos_v2_{suffix}")
                        logger.info(f"🔄 Usando directorio alternativo: {chroma_alimentos}")

            # Crear vectorstore con fallback
            try:
                collection_name = f"alimentos_nutricionales_{random.randint(100, 999)}"
                self.vectorstore_alimentos = Chroma(
                    collection_name=collection_name,
                    persist_directory=chroma_alimentos,
                    embedding_function=self.embeddings
                )
            except Exception as chroma_error:
                logger.warning(f"⚠️ ChromaDB falló: {chroma_error}")

                # Fallback a FAISS
                try:
                    from langchain.vectorstores import FAISS
                    self.vectorstore_alimentos = FAISS.from_texts(
                        texts=textos_alimentos,
                        metadatas=metadatos_alimentos,
                        embedding=self.embeddings
                    )
                    logger.info("✅ Usando FAISS como fallback")
                except Exception as faiss_error:
                    logger.error(f"❌ FAISS también falló: {faiss_error}")
                    return False

            # Agregar documentos en lotes
            if hasattr(self.vectorstore_alimentos, 'add_texts'):
                batch_size = 25
                for i in range(0, len(textos_alimentos), batch_size):
                    try:
                        batch_textos = textos_alimentos[i:i+batch_size]
                        batch_metadatos = metadatos_alimentos[i:i+batch_size]

                        self.vectorstore_alimentos.add_texts(
                            texts=batch_textos,
                            metadatas=batch_metadatos
                        )
                    except Exception as batch_error:
                        logger.warning(f"⚠️ Error en lote {i}: {batch_error}")
                        continue

                # Persistir si es posible
                if hasattr(self.vectorstore_alimentos, 'persist'):
                    try:
                        self.vectorstore_alimentos.persist()
                    except:
                        logger.warning("⚠️ No se pudo persistir vectorstore")

            logger.info(f"✅ Vectorstore alimentos mejorado: {len(textos_alimentos)} documentos únicos")
            return True

        except Exception as e:
            logger.error(f"❌ Error creando vectorstore alimentos: {e}")
            return False

    def _evaluar_alimento_para_condicion(self, row: pd.Series, condicion: str) -> Dict:
        """Evalúa si un alimento es recomendado para una condición específica."""
        config = self.condiciones_nutrientes.get(condicion, {})

        energia = float(row.get('ENERGIA_KCAL', 0))
        proteinas = float(row.get('PROTEINAS_G', 0))
        hierro = float(row.get('HIERRO_MG', 0))
        fibra = float(row.get('FIBRA_DIETARIA_G', 0))

        if condicion == "anemia":
            if hierro >= config.get("hierro_minimo", 2.0):
                return {"recomendado": True, "razon": f"alto contenido de hierro ({hierro}mg)"}
            elif proteinas >= config.get("proteinas_minimo", 10.0):
                return {"recomendado": True, "razon": f"buena fuente de proteínas ({proteinas}g)"}

        elif condicion == "sobrepeso":
            if energia <= config.get("energia_max", 150) and fibra >= config.get("fibra_minimo", 2.0):
                return {"recomendado": True, "razon": f"bajo en calorías ({energia} kcal) y rico en fibra ({fibra}g)"}
            elif energia <= config.get("energia_max", 150):
                return {"recomendado": True, "razon": f"bajo en calorías ({energia} kcal)"}

        elif condicion == "general":
            if proteinas >= 5.0 or hierro >= 1.0:
                return {"recomendado": True, "razon": "nutritivo balanceado"}

        return {"recomendado": False, "razon": "no cumple criterios específicos"}

    def _clasificar_grupo_alimento(self, nombre: str) -> str:
        """Clasifica un alimento en su grupo correspondiente."""
        nombre = nombre.lower()

        if any(word in nombre for word in ['pescado', 'anchoveta', 'atún', 'pota', 'marino']):
            return "pescados_mariscos"
        elif any(word in nombre for word in ['pollo', 'carne', 'res', 'cerdo', 'hígado']):
            return "carnes_aves"
        elif any(word in nombre for word in ['leche', 'queso', 'yogurt', 'lácteo']):
            return "lacteos"
        elif any(word in nombre for word in ['frijol', 'lenteja', 'arveja', 'garbanzo', 'pallares']):
            return "legumbres"
        elif any(word in nombre for word in ['arroz', 'pan', 'trigo', 'quinua', 'avena']):
            return "cereales"
        elif any(word in nombre for word in ['papa', 'camote', 'yuca', 'olluco']):
            return "tuberculos"
        elif any(word in nombre for word in ['tomate', 'lechuga', 'zanahoria', 'cebolla', 'apio']):
            return "verduras"
        elif any(word in nombre for word in ['manzana', 'naranja', 'plátano', 'limón']):
            return "frutas"
        else:
            return "otros"

    def buscar_alimentos_recomendados_mejorado(self, condicion: str, k: int = 8, diversidad: bool = True) -> List[Dict]:
        """Búsqueda mejorada de alimentos con diversidad garantizada."""
        try:
            if not self.vectorstore_alimentos:
                logger.error("Vectorstore de alimentos no inicializado")
                return []

            consulta_key = f"{condicion}_{k}_{diversidad}"

            if consulta_key not in self.contador_consultas:
                self.contador_consultas[consulta_key] = 0
            self.contador_consultas[consulta_key] += 1

            config = self.condiciones_nutrientes.get(condicion.lower(), self.condiciones_nutrientes["general"])

            queries = self._generar_queries_diversas(condicion, config)

            todos_resultados = {}

            for i, query in enumerate(queries):
                try:
                    k_por_query = max(3, k // len(queries) + 2)

                    resultados = self.vectorstore_alimentos.similarity_search(
                        query,
                        k=k_por_query * 2
                    )

                    for doc in resultados:
                        metadata = doc.metadata
                        nombre = metadata.get('nombre', '')

                        if nombre and nombre not in todos_resultados:
                            score = self._calcular_score_alimento(metadata, condicion, query)

                            if diversidad:
                                score = self._aplicar_factor_diversidad(nombre, todos_resultados, score)

                            metadata['search_score'] = score
                            metadata['query_origen'] = f"query_{i+1}"
                            todos_resultados[nombre] = metadata

                except Exception as e:
                    logger.warning(f"Error en query {i+1}: {e}")
                    continue

            if not todos_resultados:
                logger.warning(f"No se encontraron alimentos para {condicion}")
                return []

            alimentos_filtrados = []
            for metadata in todos_resultados.values():
                if self._cumple_criterios_condicion(metadata, condicion):
                    alimentos_filtrados.append(metadata)

            alimentos_ordenados = sorted(alimentos_filtrados, key=lambda x: x.get('search_score', 0), reverse=True)

            if diversidad:
                alimentos_diversos = self._seleccionar_con_diversidad_grupos(alimentos_ordenados, k)
            else:
                alimentos_diversos = alimentos_ordenados[:k]

            logger.info(f"✅ Encontrados {len(alimentos_diversos)} alimentos para {condicion} (consulta #{self.contador_consultas[consulta_key]})")

            return alimentos_diversos

        except Exception as e:
            logger.error(f"❌ Error buscando alimentos: {e}")
            return []

    def _generar_queries_diversas(self, condicion: str, config: Dict) -> List[str]:
        """Genera queries diversas para obtener variedad de alimentos."""
        queries = []

        queries.append(f"alimentos nutritivos recomendados para {condicion}")

        nutrientes_prioritarios = config.get("nutrientes_prioritarios", [])
        for nutriente in nutrientes_prioritarios:
            if nutriente == "HIERRO_MG":
                queries.append("alimentos ricos en hierro pescado carne hígado")
            elif nutriente == "FIBRA_DIETARIA_G":
                queries.append("alimentos con fibra verduras frutas cereales integrales")
            elif nutriente == "PROTEINAS_G":
                queries.append("alimentos proteicos pescado pollo huevos lácteos")

        keywords = config.get("keywords", [])
        if keywords:
            grupos_keywords = [keywords[i:i+3] for i in range(0, len(keywords), 3)]
            for grupo in grupos_keywords:
                queries.append(f"alimentos nutritivos {' '.join(grupo)}")

        if condicion.lower() == "anemia":
            queries.extend([
                "pescados mariscos ricos hierro proteínas",
                "carnes rojas hígado sangre hierro",
                "legumbres quinua cereales hierro vegetal"
            ])
        elif condicion.lower() == "sobrepeso":
            queries.extend([
                "verduras frescas bajas calorías fibra",
                "frutas dietéticas pocas calorías",
                "alimentos ligeros bajo contenido energético"
            ])

        queries_unicas = []
        for q in queries:
            if q not in queries_unicas:
                queries_unicas.append(q)

        return queries_unicas[:6]

    def _calcular_score_alimento(self, metadata: Dict, condicion: str, query: str) -> float:
        """Calcula score personalizado para un alimento."""
        score = 0.0

        categorias = metadata.get('categorias_recomendado', '').lower()
        if condicion.lower() in categorias:
            score += 10.0
        elif 'general' in categorias:
            score += 5.0

        config = self.condiciones_nutrientes.get(condicion.lower(), {})

        if condicion.lower() == "anemia":
            hierro = metadata.get('hierro_mg', 0)
            proteinas = metadata.get('proteinas_g', 0)

            if hierro >= 5.0:
                score += 15.0
            elif hierro >= 3.0:
                score += 10.0
            elif hierro >= 1.0:
                score += 5.0

            if proteinas >= 15.0:
                score += 8.0
            elif proteinas >= 10.0:
                score += 5.0

        elif condicion.lower() == "sobrepeso":
            energia = metadata.get('energia_kcal', 0)
            fibra = metadata.get('fibra_g', 0)

            if energia <= 100:
                score += 12.0
            elif energia <= 150:
                score += 8.0
            elif energia <= 200:
                score += 4.0
            else:
                score -= 5.0

            if fibra >= 5.0:
                score += 8.0
            elif fibra >= 2.0:
                score += 5.0

        indice_nutricional = metadata.get('indice_nutricional', 0)
        score += indice_nutricional * 0.5

        grupo = metadata.get('grupo_alimento', '')
        if grupo in ['pescados_mariscos', 'carnes_aves']:
            score += 2.0

        return round(score, 2)

    def _aplicar_factor_diversidad(self, nombre: str, resultados_existentes: Dict, score: float) -> float:
        """Aplica factor de diversidad para evitar alimentos muy similares."""
        if not resultados_existentes:
            return score

        palabras_nombre = set(nombre.lower().split())

        for nombre_existente in resultados_existentes.keys():
            palabras_existente = set(nombre_existente.lower().split())

            interseccion = palabras_nombre.intersection(palabras_existente)
            union = palabras_nombre.union(palabras_existente)

            if union:
                similitud = len(interseccion) / len(union)
                if similitud > 0.5:
                    score *= 0.7

        return score

    def _cumple_criterios_condicion(self, metadata: Dict, condicion: str) -> bool:
        """Verifica si un alimento cumple los criterios específicos de la condición."""
        config = self.condiciones_nutrientes.get(condicion.lower(), {})

        nombre = metadata.get('nombre', '')
        if not nombre or len(nombre) < 3:
            return False

        if condicion.lower() == "anemia":
            hierro = metadata.get('hierro_mg', 0)
            proteinas = metadata.get('proteinas_g', 0)
            return hierro >= 1.0 or proteinas >= 8.0

        elif condicion.lower() == "sobrepeso":
            energia = metadata.get('energia_kcal', 0)
            return energia <= 250

        elif condicion.lower() == "diabetes":
            energia = metadata.get('energia_kcal', 0)
            fibra = metadata.get('fibra_g', 0)
            return energia <= 200 or fibra >= 2.0

        return True

    def _seleccionar_con_diversidad_grupos(self, alimentos: List[Dict], k: int) -> List[Dict]:
        """Selecciona alimentos asegurando diversidad de grupos alimentarios."""
        if len(alimentos) <= k:
            return alimentos

        seleccionados = []
        grupos_usados = set()

        for alimento in alimentos:
            if len(seleccionados) >= k:
                break

            grupo = alimento.get('grupo_alimento', 'otros')
            if grupo not in grupos_usados:
                seleccionados.append(alimento)
                grupos_usados.add(grupo)

        for alimento in alimentos:
            if len(seleccionados) >= k:
                break

            if alimento not in seleccionados:
                seleccionados.append(alimento)

        return seleccionados[:k]

    def _crear_vectorstore_recetas_mejorado(self) -> bool:
        """Crea vectorstore mejorado para recetas."""
        try:
            logger.info("🍽️ Creando vectorstore de recetas mejorado...")

            textos_recetas = []
            metadatos_recetas = []

            for idx, row in self.recetas_df.iterrows():
                nombre = str(row.get('nombre', 'Sin nombre')).strip()

                if not nombre or nombre == 'Sin nombre':
                    continue

                ingredientes = str(row.get('ingredientes', ''))
                preparacion = str(row.get('preparacion', ''))
                energia = float(row.get('energia_kcal', 0))
                proteinas = float(row.get('proteinas_g', 0))
                hierro = float(row.get('hierro_mg', 0))

                alimentos_identificados = self._analizar_ingredientes_mejorado(ingredientes)

                texto = f"Receta saludable: {nombre}. "

                if energia > 0:
                    nivel_energia = "alta" if energia > 600 else "moderada" if energia > 400 else "ligera"
                    texto += f"Receta {nivel_energia} en energía: {energia} kcal por porción. "

                if hierro > 0:
                    nivel_hierro = "excelente" if hierro > 5 else "buena" if hierro > 2 else "moderada"
                    texto += f"{nivel_hierro} fuente de hierro: {hierro}mg. "

                if proteinas > 0:
                    nivel_proteinas = "rica" if proteinas > 20 else "buena" if proteinas > 10 else "moderada"
                    texto += f"Receta {nivel_proteinas} en proteínas: {proteinas}g. "

                if alimentos_identificados:
                    nombres_alimentos = [a['nombre'] for a in alimentos_identificados]
                    texto += f"Contiene ingredientes nutritivos: {', '.join(nombres_alimentos)}. "

                    hierro_ingredientes = sum(a.get('hierro_mg', 0) for a in alimentos_identificados)
                    proteinas_ingredientes = sum(a.get('proteinas_g', 0) for a in alimentos_identificados)

                    if hierro_ingredientes > 0:
                        texto += f"Los ingredientes aportan {hierro_ingredientes:.1f}mg adicionales de hierro. "
                    if proteinas_ingredientes > 0:
                        texto += f"Los ingredientes suman {proteinas_ingredientes:.1f}g de proteínas extra. "

                condiciones_apropiadas = self._clasificar_receta_mejorada(energia, hierro, proteinas, alimentos_identificados)
                texto += f"Apropiada para: {', '.join(condiciones_apropiadas)}. "

                tipo_preparacion = self._analizar_tipo_preparacion(preparacion)
                texto += f"Método de preparación: {tipo_preparacion}. "

                ingredientes_clave = self._extraer_ingredientes_clave(ingredientes)
                if ingredientes_clave:
                    texto += f"Ingredientes principales: {', '.join(ingredientes_clave)}. "

                textos_recetas.append(texto)

                metadatos_recetas.append({
                    'id_receta': f'receta_{idx}',
                    'nombre': nombre,
                    'energia_kcal': energia,
                    'proteinas_g': proteinas,
                    'hierro_mg': hierro,
                    'condiciones_apropiadas': ','.join(condiciones_apropiadas),
                    'alimentos_identificados': ','.join([a['nombre'] for a in alimentos_identificados]),
                    'num_alimentos_nutritivos': len(alimentos_identificados),
                    'tipo_preparacion': tipo_preparacion,
                    'ingredientes_clave': ','.join(ingredientes_clave),
                    'hierro_total': hierro + sum(a.get('hierro_mg', 0) for a in alimentos_identificados),
                    'proteinas_total': proteinas + sum(a.get('proteinas_g', 0) for a in alimentos_identificados),
                    'dificultad': self._evaluar_dificultad(preparacion),
                    'tiempo_estimado': self._estimar_tiempo_preparacion(preparacion),
                    'ingredientes_originales': ingredientes[:300],
                    'preparacion_original': preparacion[:300]
                })

            # Crear vectorstore con manejo de errores similar al de alimentos
            chroma_recetas = os.path.join(RUTA_CHROMA, "recetas_v2")

            max_reintentos = 3
            for intento in range(max_reintentos):
                try:
                    if os.path.exists(chroma_recetas):
                        gc.collect()
                        shutil.rmtree(chroma_recetas)
                    break
                except PermissionError:
                    if intento < max_reintentos - 1:
                        time.sleep(2)
                    else:
                        suffix = random.randint(1000, 9999)
                        chroma_recetas = os.path.join(RUTA_CHROMA, f"recetas_v2_{suffix}")

            try:
                collection_name = f"recetas_mejoradas_{random.randint(100, 999)}"
                self.vectorstore_recetas = Chroma(
                    collection_name=collection_name,
                    persist_directory=chroma_recetas,
                    embedding_function=self.embeddings
                )
            except Exception as chroma_error:
                logger.warning(f"⚠️ ChromaDB recetas falló: {chroma_error}")
                try:
                    from langchain.vectorstores import FAISS
                    self.vectorstore_recetas = FAISS.from_texts(
                        texts=textos_recetas,
                        metadatas=metadatos_recetas,
                        embedding=self.embeddings
                    )
                    logger.info("✅ Usando FAISS para recetas")
                except Exception as faiss_error:
                    logger.error(f"❌ FAISS recetas falló: {faiss_error}")
                    return False

            if hasattr(self.vectorstore_recetas, 'add_texts'):
                batch_size = 20
                for i in range(0, len(textos_recetas), batch_size):
                    try:
                        batch_textos = textos_recetas[i:i+batch_size]
                        batch_metadatos = metadatos_recetas[i:i+batch_size]

                        self.vectorstore_recetas.add_texts(
                            texts=batch_textos,
                            metadatas=batch_metadatos
                        )
                    except Exception as batch_error:
                        logger.warning(f"⚠️ Error en lote recetas {i}: {batch_error}")

                if hasattr(self.vectorstore_recetas, 'persist'):
                    try:
                        self.vectorstore_recetas.persist()
                    except:
                        logger.warning("⚠️ No se pudo persistir vectorstore recetas")

            logger.info(f"✅ Vectorstore recetas mejorado: {len(textos_recetas)} recetas únicas")
            return True

        except Exception as e:
            logger.error(f"❌ Error creando vectorstore recetas: {e}")
            return False

    def _analizar_ingredientes_mejorado(self, ingredientes: str) -> List[Dict]:
        """Análisis mejorado de ingredientes con mejor coincidencia."""
        alimentos_encontrados = []

        if not isinstance(ingredientes, str) or len(ingredientes) < 5:
            return alimentos_encontrados

        ingredientes_lower = ingredientes.lower()

        sinonimos = {
            'pescado': ['pescado', 'pez', 'anchoveta', 'atún', 'bonito'],
            'pollo': ['pollo', 'ave', 'gallina'],
            'carne': ['carne', 'res', 'vacuno'],
            'arroz': ['arroz', 'grano'],
            'papa': ['papa', 'patata'],
            'cebolla': ['cebolla', 'bulbo'],
            'tomate': ['tomate', 'jitomate'],
            'leche': ['leche', 'lácteo'],
            'huevo': ['huevo', 'clara', 'yema']
        }

        for _, alimento in self.tabla_nutricional.iterrows():
            nombre_alimento = str(alimento.get('NOMBRE_ALIMENTO', '')).lower()

            encontrado = False
            palabra_encontrada = ""

            if nombre_alimento in ingredientes_lower:
                encontrado = True
                palabra_encontrada = nombre_alimento
            else:
                palabras_alimento = nombre_alimento.split()
                for palabra in palabras_alimento:
                    if len(palabra) > 3 and palabra in ingredientes_lower:
                        encontrado = True
                        palabra_encontrada = palabra
                        break

                if not encontrado:
                    for base, variaciones in sinonimos.items():
                        if any(var in nombre_alimento for var in variaciones):
                            if any(var in ingredientes_lower for var in variaciones):
                                encontrado = True
                                palabra_encontrada = base
                                break

            if encontrado:
                alimentos_encontrados.append({
                    'nombre': str(alimento.get('NOMBRE_ALIMENTO', '')),
                    'energia_kcal': float(alimento.get('ENERGIA_KCAL', 0)),
                    'proteinas_g': float(alimento.get('PROTEINAS_G', 0)),
                    'hierro_mg': float(alimento.get('HIERRO_MG', 0)),
                    'calcio_mg': float(alimento.get('CALCIO_MG', 0)),
                    'palabra_encontrada': palabra_encontrada
                })

        alimentos_unicos = {}
        for alimento in alimentos_encontrados:
            nombre = alimento['nombre']
            if nombre not in alimentos_unicos:
                alimentos_unicos[nombre] = alimento

        return list(alimentos_unicos.values())

    def _clasificar_receta_mejorada(self, energia: float, hierro: float, proteinas: float, alimentos: List[Dict]) -> List[str]:
        """Clasificación mejorada de recetas por condiciones."""
        condiciones = []

        hierro_total = hierro + sum(a.get('hierro_mg', 0) for a in alimentos)
        proteinas_total = proteinas + sum(a.get('proteinas_g', 0) for a in alimentos)

        if hierro_total >= 5.0:
            condiciones.append("anemia")
        elif hierro_total >= 3.0 or proteinas_total >= 25.0:
            condiciones.append("anemia_leve")

        if energia <= 300:
            condiciones.append("sobrepeso")
        elif energia <= 500:
            condiciones.append("sobrepeso_moderado")

        if energia <= 400 and proteinas_total >= 15.0:
            condiciones.append("diabetes")

        if proteinas_total >= 10.0:
            condiciones.append("general")

        if not condiciones:
            condiciones.append("ocasional")

        return condiciones

    def _analizar_tipo_preparacion(self, preparacion: str) -> str:
        """Analiza el tipo de preparación de la receta."""
        if not preparacion:
            return "simple"

        preparacion_lower = preparacion.lower()

        if any(word in preparacion_lower for word in ['freír', 'frito', 'aceite caliente']):
            return "frito"
        elif any(word in preparacion_lower for word in ['hervir', 'sancochar', 'cocinar']):
            return "cocido"
        elif any(word in preparacion_lower for word in ['horno', 'hornear']):
            return "horneado"
        elif any(word in preparacion_lower for word in ['vapor', 'al vapor']):
            return "al_vapor"
        elif any(word in preparacion_lower for word in ['guisar', 'guiso', 'estofar']):
            return "guisado"
        elif any(word in preparacion_lower for word in ['saltear', 'saltado']):
            return "salteado"
        else:
            return "mixto"

    def _extraer_ingredientes_clave(self, ingredientes: str) -> List[str]:
        """Extrae los ingredientes más importantes de la lista."""
        if not ingredientes:
            return []

        items = ingredientes.split(',')
        ingredientes_clave = []

        for item in items[:5]:
            item_limpio = item.strip()
            if len(item_limpio) > 3:
                palabras = item_limpio.split()
                ingrediente = ' '.join([p for p in palabras if not any(c.isdigit() for c in p)])[:30]
                if ingrediente:
                    ingredientes_clave.append(ingrediente)

        return ingredientes_clave

    def _evaluar_dificultad(self, preparacion: str) -> str:
        """Evalúa la dificultad de preparación."""
        if not preparacion:
            return "fácil"

        num_pasos = len([s for s in preparacion.split('.') if s.strip()])

        if num_pasos <= 3:
            return "fácil"
        elif num_pasos <= 6:
            return "intermedio"
        else:
            return "avanzado"

    def _estimar_tiempo_preparacion(self, preparacion: str) -> str:
        """Estima el tiempo de preparación."""
        if not preparacion:
            return "15-30 min"

        preparacion_lower = preparacion.lower()

        if any(word in preparacion_lower for word in ['rápido', 'minutos', 'freír']):
            return "15-30 min"
        elif any(word in preparacion_lower for word in ['sancochar', 'cocinar', 'hervir']):
            return "30-45 min"
        elif any(word in preparacion_lower for word in ['horno', 'guisar', 'estofar']):
            return "45-60 min"
        else:
            return "30-45 min"

    def buscar_recetas_por_condicion_mejorado(self, condicion: str, alimentos_recomendados: List[str] = None, k: int = 10) -> List[Dict]:
        """Búsqueda mejorada de recetas con mayor diversidad."""
        try:
            if not self.vectorstore_recetas:
                logger.error("Vectorstore de recetas no inicializado")
                return []

            consulta_key = f"recetas_{condicion}_{k}"
            if consulta_key not in self.contador_consultas:
                self.contador_consultas[consulta_key] = 0
            self.contador_consultas[consulta_key] += 1

            queries = self._generar_queries_recetas_diversas(condicion, alimentos_recomendados)

            recetas_encontradas = {}

            for i, query in enumerate(queries):
                try:
                    k_por_query = max(3, k // len(queries) + 1)
                    resultados = self.vectorstore_recetas.similarity_search(query, k=k_por_query * 2)

                    for doc in resultados:
                        nombre = doc.metadata.get('nombre', '')
                        if nombre and nombre not in recetas_encontradas:
                            score = self._calcular_score_receta_mejorado(doc.metadata, condicion, alimentos_recomendados)

                            if self.contador_consultas[consulta_key] > 1:
                                score += np.random.uniform(-2, 2)

                            doc.metadata['relevance_score'] = score
                            doc.metadata['query_origen'] = f"query_{i+1}"
                            recetas_encontradas[nombre] = doc.metadata

                except Exception as e:
                    logger.warning(f"Error en query recetas {i+1}: {e}")
                    continue

            if not recetas_encontradas:
                logger.warning(f"No se encontraron recetas para {condicion}")
                return []

            recetas_filtradas = []
            for metadata in recetas_encontradas.values():
                if self._cumple_criterios_receta(metadata, condicion):
                    recetas_filtradas.append(metadata)

            recetas_ordenadas = sorted(
                recetas_filtradas,
                key=lambda x: x.get('relevance_score', 0) + np.random.uniform(-1, 1),
                reverse=True
            )

            recetas_diversas = self._seleccionar_recetas_diversas(recetas_ordenadas, k)

            logger.info(f"✅ Encontradas {len(recetas_diversas)} recetas para {condicion} (consulta #{self.contador_consultas[consulta_key]})")

            return recetas_diversas

        except Exception as e:
            logger.error(f"❌ Error buscando recetas: {e}")
            return []

    def _generar_queries_recetas_diversas(self, condicion: str, alimentos_recomendados: List[str] = None) -> List[str]:
        """Genera queries diversas para recetas."""
        queries = []

        queries.append(f"recetas saludables nutritivas para {condicion}")

        tipos_preparacion = ["frito", "cocido", "guisado", "salteado"]
        for tipo in tipos_preparacion:
            queries.append(f"recetas {tipo} {condicion}")

        if alimentos_recomendados:
            for alimento in alimentos_recomendados[:3]:
                queries.append(f"recetas con {alimento} nutritivas")

        if condicion.lower() == "anemia":
            queries.extend([
                "recetas ricas hierro pescado anchoveta",
                "recetas proteicas carne pollo hígado",
                "recetas nutritivas sangrecita vísceras"
            ])
        elif condicion.lower() == "sobrepeso":
            queries.extend([
                "recetas ligeras bajas calorías verduras",
                "recetas dietéticas saludables fibra",
                "recetas light pescado verduras"
            ])

        return queries[:8]

    def _calcular_score_receta_mejorado(self, metadata: Dict, condicion: str, alimentos_recomendados: List[str] = None) -> float:
        """Cálculo mejorado de score para recetas."""
        score = 0.0

        condiciones_apropiadas = metadata.get('condiciones_apropiadas', '').lower()
        if condicion.lower() in condiciones_apropiadas:
            score += 15.0
        elif 'general' in condiciones_apropiadas:
            score += 8.0

        if condicion.lower() == "anemia":
            hierro_total = metadata.get('hierro_total', 0)
            if hierro_total >= 5.0:
                score += 12.0
            elif hierro_total >= 3.0:
                score += 8.0
            elif hierro_total >= 1.0:
                score += 4.0

        elif condicion.lower() == "sobrepeso":
            energia = metadata.get('energia_kcal', 0)
            if energia <= 300:
                score += 12.0
            elif energia <= 500:
                score += 8.0
            elif energia <= 700:
                score += 4.0

        if alimentos_recomendados:
            alimentos_identificados = metadata.get('alimentos_identificados', '').lower()
            for alimento in alimentos_recomendados:
                if alimento.lower() in alimentos_identificados:
                    score += 3.0

        num_alimentos = metadata.get('num_alimentos_nutritivos', 0)
        score += min(num_alimentos * 1.5, 6.0)

        tipo_prep = metadata.get('tipo_preparacion', '')
        if tipo_prep in ['cocido', 'al_vapor', 'guisado']:
            score += 2.0
        elif tipo_prep in ['salteado', 'horneado']:
            score += 1.0

        return round(score, 2)

    def _cumple_criterios_receta(self, metadata: Dict, condicion: str) -> bool:
        """Verifica si una receta cumple criterios básicos."""
        nombre = metadata.get('nombre', '')
        if not nombre or len(nombre) < 3:
            return False

        energia = metadata.get('energia_kcal', 0)

        if condicion.lower() == "sobrepeso":
            return energia <= 800
        elif condicion.lower() == "anemia":
            hierro_total = metadata.get('hierro_total', 0)
            return hierro_total >= 0.5

        return True

    def _seleccionar_recetas_diversas(self, recetas: List[Dict], k: int) -> List[Dict]:
        """Selecciona recetas con diversidad de tipos de preparación."""
        if len(recetas) <= k:
            return recetas

        seleccionadas = []
        tipos_usados = set()

        for receta in recetas:
            if len(seleccionadas) >= k:
                break

            tipo_prep = receta.get('tipo_preparacion', 'mixto')
            if tipo_prep not in tipos_usados:
                seleccionadas.append(receta)
                tipos_usados.add(tipo_prep)

        for receta in recetas:
            if len(seleccionadas) >= k:
                break

            if receta not in seleccionadas:
                seleccionadas.append(receta)

        return seleccionadas[:k]

    def generar_recomendacion_completa_mejorada(self, condicion: str) -> Dict:
        """Genera recomendación completa con sistema mejorado."""
        try:
            logger.info(f"🔍 Generando recomendación mejorada para: {condicion}")

            alimentos_recomendados = self.buscar_alimentos_recomendados_mejorado(condicion, k=8, diversidad=True)
            nombres_alimentos = [a['nombre'] for a in alimentos_recomendados]

            recetas_recomendadas = self.buscar_recetas_por_condicion_mejorado(
                condicion,
                nombres_alimentos,
                k=10
            )

            analisis_nutricional = self._generar_analisis_nutricional_mejorado(condicion, alimentos_recomendados, recetas_recomendadas)

            verificacion_diversidad = self._verificar_diversidad_resultados(alimentos_recomendados, recetas_recomendadas)

            respuesta = {
                "condicion": condicion,
                "alimentos_recomendados": {
                    "lista": alimentos_recomendados[:6],
                    "total_encontrados": len(alimentos_recomendados),
                    "diversidad": verificacion_diversidad["diversidad_alimentos"]
                },
                "recetas_recomendadas": {
                    "lista": recetas_recomendadas[:8],
                    "total_encontradas": len(recetas_recomendadas),
                    "diversidad": verificacion_diversidad["diversidad_recetas"]
                },
                "analisis_nutricional": analisis_nutricional,
                "match_alimentos_recetas": self._analizar_coincidencias_mejorado(nombres_alimentos, recetas_recomendadas),
                "recomendaciones_adicionales": self._generar_recomendaciones_adicionales(condicion, alimentos_recomendados)
            }

            logger.info(f"✅ Recomendación mejorada generada: {len(alimentos_recomendados)} alimentos, {len(recetas_recomendadas)} recetas")

            return respuesta

        except Exception as e:
            logger.error(f"❌ Error generando recomendación mejorada: {e}")
            return {"error": str(e)}

    def _generar_analisis_nutricional_mejorado(self, condicion: str, alimentos: List[Dict], recetas: List[Dict]) -> Dict:
        """Análisis nutricional más detallado."""
        if not alimentos and not recetas:
            return {"mensaje": "No hay datos para analizar"}

        analisis = {}

        if alimentos:
            total_energia_alimentos = sum(a.get('energia_kcal', 0) for a in alimentos)
            total_hierro_alimentos = sum(a.get('hierro_mg', 0) for a in alimentos)
            total_proteinas_alimentos = sum(a.get('proteinas_g', 0) for a in alimentos)

            analisis["alimentos"] = {
                "promedio_energia_kcal": round(total_energia_alimentos / len(alimentos), 1),
                "promedio_hierro_mg": round(total_hierro_alimentos / len(alimentos), 1),
                "promedio_proteinas_g": round(total_proteinas_alimentos / len(alimentos), 1),
                "alimento_mas_nutritivo": max(alimentos, key=lambda x: x.get('indice_nutricional', 0))['nombre']
            }

        if recetas:
            total_energia_recetas = sum(r.get('energia_kcal', 0) for r in recetas)
            total_hierro_recetas = sum(r.get('hierro_mg', 0) for r in recetas)

            analisis["recetas"] = {
                "promedio_energia_kcal": round(total_energia_recetas / len(recetas), 1),
                "promedio_hierro_mg": round(total_hierro_recetas / len(recetas), 1),
                "receta_mas_nutritiva": max(recetas, key=lambda x: x.get('relevance_score', 0))['nombre']
            }

        analisis["evaluacion_condicion"] = self._evaluar_adecuacion_condicion(condicion, alimentos, recetas)

        return analisis

    def _verificar_diversidad_resultados(self, alimentos: List[Dict], recetas: List[Dict]) -> Dict:
        """Verifica la diversidad de los resultados."""
        diversidad = {
            "diversidad_alimentos": {},
            "diversidad_recetas": {}
        }

        if alimentos:
            grupos_alimentos = {}
            for alimento in alimentos:
                grupo = alimento.get('grupo_alimento', 'otros')
                grupos_alimentos[grupo] = grupos_alimentos.get(grupo, 0) + 1

            diversidad["diversidad_alimentos"] = {
                "grupos_representados": len(grupos_alimentos),
                "distribucion_grupos": grupos_alimentos,
                "score_diversidad": min(len(grupos_alimentos) * 2, 10)
            }

        if recetas:
            tipos_preparacion = {}
            for receta in recetas:
                tipo = receta.get('tipo_preparacion', 'mixto')
                tipos_preparacion[tipo] = tipos_preparacion.get(tipo, 0) + 1

            diversidad["diversidad_recetas"] = {
                "tipos_preparacion": len(tipos_preparacion),
                "distribucion_tipos": tipos_preparacion,
                "score_diversidad": min(len(tipos_preparacion) * 2, 10)
            }

        return diversidad

    def _analizar_coincidencias_mejorado(self, alimentos_recomendados: List[str], recetas: List[Dict]) -> Dict:
        """Análisis mejorado de coincidencias."""
        if not recetas:
            return {"mensaje": "No hay recetas para analizar"}

        total_recetas = len(recetas)
        recetas_con_alimentos = 0
        coincidencias_detalladas = []

        for receta in recetas:
            alimentos_en_receta = receta.get('alimentos_identificados', '').lower()
            coincidencias_receta = []

            for alimento in alimentos_recomendados:
                if alimento.lower() in alimentos_en_receta:
                    coincidencias_receta.append(alimento)

            if coincidencias_receta:
                recetas_con_alimentos += 1
                coincidencias_detalladas.append({
                    "receta": receta.get('nombre', ''),
                    "alimentos_coincidentes": coincidencias_receta,
                    "score": receta.get('relevance_score', 0),
                    "tipo_preparacion": receta.get('tipo_preparacion', '')
                })

        porcentaje_match = (recetas_con_alimentos / total_recetas * 100) if total_recetas > 0 else 0

        return {
            "porcentaje_coincidencia": round(porcentaje_match, 1),
            "recetas_con_alimentos_recomendados": recetas_con_alimentos,
            "total_recetas": total_recetas,
            "coincidencias_detalladas": sorted(coincidencias_detalladas, key=lambda x: x['score'], reverse=True)[:5],
            "calidad_match": "excelente" if porcentaje_match > 70 else "buena" if porcentaje_match > 50 else "moderada"
        }

    def _evaluar_adecuacion_condicion(self, condicion: str, alimentos: List[Dict], recetas: List[Dict]) -> Dict:
        """Evalúa qué tan adecuados son los resultados para la condición."""
        evaluacion = {"cumple_criterios": True, "observaciones": [], "score": 0}

        config = self.condiciones_nutrientes.get(condicion.lower(), {})

        if condicion.lower() == "anemia" and alimentos:
            hierro_promedio = sum(a.get('hierro_mg', 0) for a in alimentos) / len(alimentos)
            if hierro_promedio >= 3.0:
                evaluacion["score"] += 8
                evaluacion["observaciones"].append("Excelente contenido de hierro en alimentos")
            elif hierro_promedio >= 1.0:
                evaluacion["score"] += 5
                evaluacion["observaciones"].append("Buen contenido de hierro")
            else:
                evaluacion["observaciones"].append("Considerar alimentos con más hierro")

        elif condicion.lower() == "sobrepeso" and alimentos:
            energia_promedio = sum(a.get('energia_kcal', 0) for a in alimentos) / len(alimentos)
            if energia_promedio <= 150:
                evaluacion["score"] += 8
                evaluacion["observaciones"].append("Alimentos apropiados para control de peso")
            elif energia_promedio <= 250:
                evaluacion["score"] += 5
                evaluacion["observaciones"].append("Alimentos moderadamente apropiados")

        evaluacion["nivel"] = "excelente" if evaluacion["score"] >= 7 else "bueno" if evaluacion["score"] >= 4 else "aceptable"

        return evaluacion

    def _generar_recomendaciones_adicionales(self, condicion: str, alimentos: List[Dict]) -> Dict:
        """CORREGIDO: Genera recomendaciones adicionales personalizadas."""
        recomendaciones = {
            "consejos_nutricionales": [],
            "sugerencias_preparacion": [],
            "alimentos_complementarios": []
        }

        if condicion.lower() == "anemia":
            recomendaciones["consejos_nutricionales"].extend([
                "Combinar alimentos ricos en hierro con vitamina C para mejor absorción",
                "Evitar té y café durante las comidas principales",
                "Consumir estos alimentos regularmente por al menos 3 meses"
            ])
            recomendaciones["sugerencias_preparacion"].extend([
                "Cocinar en ollas de hierro puede aumentar el contenido del mineral",
                "Preferir preparaciones al vapor o guisadas sobre fritas"
            ])

        elif condicion.lower() == "sobrepeso":
            recomendaciones["consejos_nutricionales"].extend([
                "Consumir porciones moderadas y masticar lentamente",
                "Beber agua antes de las comidas",
                "Incluir vegetales en cada comida principal"
            ])
            recomendaciones["sugerencias_preparacion"].extend([
                "Preferir métodos de cocción sin aceite: al vapor, hervido, a la plancha",
                "Usar especias y hierbas para dar sabor sin calorías extra"
            ])

        # CORRECCIÓN: Evitar operación set - list
        grupos_presentes = set(a.get('grupo_alimento', '') for a in alimentos if a.get('grupo_alimento'))
        grupos_deseados = {'pescados_mariscos', 'verduras', 'frutas', 'cereales'}
        grupos_faltantes = grupos_deseados - grupos_presentes  # Ambos son sets ahora

        for grupo in grupos_faltantes:
            if grupo == 'pescados_mariscos':
                recomendaciones["alimentos_complementarios"].append("Incluir más pescados como atún, bonito o sardinas")
            elif grupo == 'verduras':
                recomendaciones["alimentos_complementarios"].append("Agregar más verduras de hoja verde como espinaca o acelga")
            elif grupo == 'frutas':
                recomendaciones["alimentos_complementarios"].append("Incluir frutas cítricas ricas en vitamina C")

        return recomendaciones


# =============================================================================
# FUNCIONES GLOBALES ACTUALIZADAS CON MANEJO DE ERRORES

_sistema_rag_mejorado = None

def inicializar_sistema_rag_mejorado() -> bool:
    """Inicializa el sistema RAG mejorado global con cleanup."""
    global _sistema_rag_mejorado
    try:
        # Limpiar instancia previa
        if _sistema_rag_mejorado is not None:
            if hasattr(_sistema_rag_mejorado, 'vectorstore_alimentos'):
                _sistema_rag_mejorado.vectorstore_alimentos = None
            if hasattr(_sistema_rag_mejorado, 'vectorstore_recetas'):
                _sistema_rag_mejorado.vectorstore_recetas = None

            _sistema_rag_mejorado.cache_busquedas.clear()
            _sistema_rag_mejorado.contador_consultas.clear()

            gc.collect()
            logger.info("🧹 Instancia previa limpiada")

        _sistema_rag_mejorado = NutritionalRAGSystemFixed()
        return _sistema_rag_mejorado.inicializar_sistema()
    except Exception as e:
        logger.error(f"❌ Error inicializando sistema RAG mejorado: {e}")
        return False

def buscar_plan_nutricional_sin_template(condicion: str) -> Dict:
    """Función principal que genera un plan nutricional SIN respuestas template."""
    global _sistema_rag_mejorado
    if _sistema_rag_mejorado is None:
        if not inicializar_sistema_rag_mejorado():
            return {"error": "No se pudo inicializar el sistema RAG mejorado"}

    return _sistema_rag_mejorado.generar_recomendacion_completa_mejorada(condicion)

def limpiar_cache_busquedas():
    """Limpia el cache de búsquedas para obtener resultados frescos."""
    global _sistema_rag_mejorado
    if _sistema_rag_mejorado is not None:
        _sistema_rag_mejorado.cache_busquedas.clear()
        _sistema_rag_mejorado.contador_consultas.clear()
        logger.info("🧹 Cache de búsquedas limpiado")

def verificar_diversidad_sistema() -> Dict:
    """Verifica la diversidad del sistema RAG."""
    global _sistema_rag_mejorado

    if _sistema_rag_mejorado is None:
        return {"error": "Sistema no inicializado"}

    condiciones_prueba = ["anemia", "sobrepeso", "general"]
    resultados_diversidad = {}

    for condicion in condiciones_prueba:
        try:
            resultados_consultas = []
            for i in range(3):
                resultado = _sistema_rag_mejorado.generar_recomendacion_completa_mejorada(condicion)
                if "error" not in resultado:
                    alimentos = [a["nombre"] for a in resultado["alimentos_recomendados"]["lista"]]
                    recetas = [r["nombre"] for r in resultado["recetas_recomendadas"]["lista"]]
                    resultados_consultas.append({"alimentos": alimentos, "recetas": recetas})

            if len(resultados_consultas) >= 2:
                alimentos_consulta1 = set(resultados_consultas[0]["alimentos"])
                alimentos_consulta2 = set(resultados_consultas[1]["alimentos"])

                recetas_consulta1 = set(resultados_consultas[0]["recetas"])
                recetas_consulta2 = set(resultados_consultas[1]["recetas"])

                diversidad_alimentos = len(alimentos_consulta1.union(alimentos_consulta2)) / max(len(alimentos_consulta1), len(alimentos_consulta2), 1)
                diversidad_recetas = len(recetas_consulta1.union(recetas_consulta2)) / max(len(recetas_consulta1), len(recetas_consulta2), 1)

                resultados_diversidad[condicion] = {
                    "diversidad_alimentos": round(diversidad_alimentos, 2),
                    "diversidad_recetas": round(diversidad_recetas, 2),
                    "evaluacion": "excelente" if diversidad_alimentos > 1.5 else "buena" if diversidad_alimentos > 1.2 else "necesita_mejora"
                }

        except Exception as e:
            resultados_diversidad[condicion] = {"error": str(e)}

    return {
        "sistema_funcionando": True,
        "diversidad_por_condicion": resultados_diversidad,
        "recomendacion": "Sistema funcionando con diversidad mejorada" if all(
            r.get("diversidad_alimentos", 0) > 1.0 for r in resultados_diversidad.values() if "error" not in r
        ) else "Considerar limpiar cache si ve repeticiones"
    }

# Mantener funciones de compatibilidad
def get_vectorstore():
    """Función de compatibilidad para obtener vectorstore de recetas."""
    global _sistema_rag_mejorado
    if _sistema_rag_mejorado is None:
        if not inicializar_sistema_rag_mejorado():
            raise Exception("No se pudo inicializar el sistema RAG mejorado")
    return _sistema_rag_mejorado.vectorstore_recetas

def get_vectorstore_nutricion():
    """Función de compatibilidad para obtener vectorstore de nutrición."""
    global _sistema_rag_mejorado
    if _sistema_rag_mejorado is None:
        if not inicializar_sistema_rag_mejorado():
            raise Exception("No se pudo inicializar el sistema RAG mejorado")
    return _sistema_rag_mejorado.vectorstore_alimentos
