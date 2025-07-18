import os
import pandas as pd
import shutil
import logging
from typing import List, Dict, Tuple, Optional
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from ..config import RUTA_NUTRICION, RUTA_CHROMA, RUTA_RECETAS

logger = logging.getLogger(__name__)

class NutritionalRAGSystem:
    """Sistema RAG completo que combina tabla nutricional con recetas."""

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

        # Mapeo de condiciones a necesidades nutricionales
        self.condiciones_nutrientes = {
            "anemia": {
                "nutrientes_clave": ["HIERRO_MG", "VITAMINA_C_MG", "PROTEINAS_G"],
                "valores_minimos": {"HIERRO_MG": 3.0, "PROTEINAS_G": 15.0},
                "alimentos_priorizados": ["pescado", "carne", "hígado", "sangre", "quinua", "lentejas", "espinaca"]
            },
            "sobrepeso": {
                "nutrientes_clave": ["ENERGIA_KCAL", "FIBRA_DIETARIA_G", "PROTEINAS_G"],
                "valores_maximos": {"ENERGIA_KCAL": 200},
                "valores_minimos": {"FIBRA_DIETARIA_G": 2.0, "PROTEINAS_G": 10.0},
                "alimentos_priorizados": ["verduras", "frutas", "pescado magro", "cereales integrales"]
            },
            "general": {
                "nutrientes_clave": ["PROTEINAS_G", "CALCIO_MG", "HIERRO_MG"],
                "valores_minimos": {"PROTEINAS_G": 8.0, "CALCIO_MG": 50.0},
                "alimentos_priorizados": ["pescado", "lácteos", "cereales", "legumbres", "verduras"]
            }
        }

    def inicializar_sistema(self) -> bool:
        """Inicializa todo el sistema RAG."""
        try:
            logger.info("🚀 Inicializando sistema RAG nutricional...")

            # 1. Cargar datos
            if not self._cargar_datos():
                return False

            # 2. Crear vectorstores
            if not self._crear_vectorstore_alimentos():
                return False

            if not self._crear_vectorstore_recetas():
                return False

            logger.info("✅ Sistema RAG inicializado correctamente")
            return True

        except Exception as e:
            logger.error(f"❌ Error inicializando sistema: {e}")
            return False

    def _cargar_datos(self) -> bool:
        """Carga los datos de nutrición y recetas."""
        try:
            # Cargar tabla nutricional
            if not os.path.exists(RUTA_NUTRICION):
                logger.error(f"Archivo nutricional no encontrado: {RUTA_NUTRICION}")
                return False

            self.tabla_nutricional = pd.read_csv(RUTA_NUTRICION)
            logger.info(f"📊 Tabla nutricional: {len(self.tabla_nutricional)} alimentos")

            # Cargar recetas
            if not os.path.exists(RUTA_RECETAS):
                logger.error(f"Archivo de recetas no encontrado: {RUTA_RECETAS}")
                return False

            self.recetas_df = pd.read_csv(RUTA_RECETAS)
            logger.info(f"📊 Recetas: {len(self.recetas_df)} disponibles")

            # Limpiar datos
            self._limpiar_datos()

            return True

        except Exception as e:
            logger.error(f"❌ Error cargando datos: {e}")
            return False

    def _limpiar_datos(self):
        """Limpia y prepara los datos para el vectorstore."""

        # Limpiar tabla nutricional
        self.tabla_nutricional = self.tabla_nutricional.fillna(0)

        # Convertir columnas numéricas
        columnas_numericas = ['ENERGIA_KCAL', 'PROTEINAS_G', 'HIERRO_MG', 'CALCIO_MG',
                             'VITAMINA_C_MG', 'FIBRA_DIETARIA_G']

        for col in columnas_numericas:
            if col in self.tabla_nutricional.columns:
                self.tabla_nutricional[col] = pd.to_numeric(
                    self.tabla_nutricional[col], errors='coerce'
                ).fillna(0)

        # Limpiar recetas
        self.recetas_df = self.recetas_df.fillna({
            'nombre': 'Sin nombre',
            'ingredientes': 'Sin ingredientes',
            'preparacion': 'Sin preparación',
            'energia_kcal': 0,
            'proteinas_g': 0,
            'hierro_mg': 0
        })

        logger.info("🧹 Datos limpiados correctamente")

    def _crear_vectorstore_alimentos(self) -> bool:
        """Crea vectorstore para alimentos con clasificación nutricional."""
        try:
            logger.info("🥗 Creando vectorstore de alimentos...")

            # Preparar textos y metadatos para alimentos
            textos_alimentos = []
            metadatos_alimentos = []

            for idx, row in self.tabla_nutricional.iterrows():
                # Crear texto descriptivo para el alimento
                nombre = str(row.get('NOMBRE_ALIMENTO', 'Sin nombre'))
                energia = row.get('ENERGIA_KCAL', 0)
                proteinas = row.get('PROTEINAS_G', 0)
                hierro = row.get('HIERRO_MG', 0)
                calcio = row.get('CALCIO_MG', 0)
                fibra = row.get('FIBRA_DIETARIA_G', 0)
                vitamina_c = row.get('VITAMINA_C_MG', 0)

                # Texto enriquecido para búsqueda semántica
                texto = f"Alimento: {nombre}. "

                # Información nutricional
                if energia > 0:
                    texto += f"Energía: {energia} kcal por 100g. "
                if proteinas > 0:
                    texto += f"Proteínas: {proteinas}g. "
                if hierro > 0:
                    texto += f"Hierro: {hierro}mg. "
                if calcio > 0:
                    texto += f"Calcio: {calcio}mg. "
                if fibra > 0:
                    texto += f"Fibra: {fibra}g. "
                if vitamina_c > 0:
                    texto += f"Vitamina C: {vitamina_c}mg. "

                # Clasificación por condiciones
                condiciones_recomendado = []

                # Para anemia (hierro alto)
                if hierro >= 3.0:
                    condiciones_recomendado.append("anemia")
                    texto += "Excelente para anemia por su alto contenido de hierro. "
                elif hierro >= 1.0:
                    condiciones_recomendado.append("anemia_leve")
                    texto += "Bueno para anemia por su contenido de hierro. "

                # Para sobrepeso (bajo en calorías, alto en fibra)
                if energia <= 200 and fibra >= 2.0:
                    condiciones_recomendado.append("sobrepeso")
                    texto += "Ideal para control de peso por ser bajo en calorías y rico en fibra. "
                elif energia <= 300:
                    condiciones_recomendado.append("sobrepeso_moderado")
                    texto += "Apropiado para control de peso moderado. "

                # General (balanceado)
                if proteinas >= 8.0 and calcio >= 50.0:
                    condiciones_recomendado.append("general")
                    texto += "Alimento nutritivo balanceado. "

                if not condiciones_recomendado:
                    condiciones_recomendado.append("general")

                textos_alimentos.append(texto)

                # Metadatos (solo tipos primitivos)
                metadatos_alimentos.append({
                    'id_alimento': f'alimento_{idx}',
                    'nombre': nombre,
                    'energia_kcal': float(energia),
                    'proteinas_g': float(proteinas),
                    'hierro_mg': float(hierro),
                    'calcio_mg': float(calcio),
                    'fibra_g': float(fibra),
                    'vitamina_c_mg': float(vitamina_c),
                    'condiciones_recomendado': ', '.join(condiciones_recomendado),
                    'codigo': str(row.get('CODIGO', f'A{idx}'))
                })

            # Crear vectorstore
            chroma_alimentos = os.path.join(RUTA_CHROMA, "alimentos")
            if os.path.exists(chroma_alimentos):
                shutil.rmtree(chroma_alimentos)

            self.vectorstore_alimentos = Chroma(
                collection_name="alimentos_nutricionales",
                persist_directory=RUTA_CHROMA,
                embedding_function=self.embeddings
            )

            # Agregar en lotes
            batch_size = 50
            for i in range(0, len(textos_alimentos), batch_size):
                batch_textos = textos_alimentos[i:i+batch_size]
                batch_metadatos = metadatos_alimentos[i:i+batch_size]

                self.vectorstore_alimentos.add_texts(
                    texts=batch_textos,
                    metadatas=batch_metadatos
                )

            self.vectorstore_alimentos.persist()
            logger.info(f"✅ Vectorstore alimentos creado: {len(textos_alimentos)} documentos")

            return True

        except Exception as e:
            logger.error(f"❌ Error creando vectorstore alimentos: {e}")
            return False

    def _crear_vectorstore_recetas(self) -> bool:
        """Crea vectorstore para recetas con análisis de ingredientes."""
        try:
            logger.info("🍽️ Creando vectorstore de recetas...")

            textos_recetas = []
            metadatos_recetas = []

            for idx, row in self.recetas_df.iterrows():
                nombre = str(row.get('nombre', 'Sin nombre'))
                ingredientes = str(row.get('ingredientes', 'Sin ingredientes'))
                preparacion = str(row.get('preparacion', 'Sin preparación'))
                energia = row.get('energia_kcal', 0)
                proteinas = row.get('proteinas_g', 0)
                hierro = row.get('hierro_mg', 0)

                # Análisis de ingredientes para encontrar alimentos de la tabla nutricional
                alimentos_encontrados = self._analizar_ingredientes(ingredientes)

                # Crear texto enriquecido
                texto = f"Receta: {nombre}. "

                # Información nutricional
                if energia > 0:
                    texto += f"Aporta {energia} calorías. "
                if proteinas > 0:
                    texto += f"Contiene {proteinas}g de proteínas. "
                if hierro > 0:
                    texto += f"Rico en hierro con {hierro}mg. "

                # Alimentos identificados
                if alimentos_encontrados:
                    nombres_alimentos = [a['nombre'] for a in alimentos_encontrados]
                    texto += f"Contiene alimentos nutritivos: {', '.join(nombres_alimentos)}. "

                    # Sumar valores nutricionales de los alimentos identificados
                    hierro_total = sum(a.get('hierro_mg', 0) for a in alimentos_encontrados)
                    energia_alimentos = sum(a.get('energia_kcal', 0) for a in alimentos_encontrados)

                    if hierro_total > 0:
                        texto += f"Los ingredientes aportan {hierro_total:.1f}mg de hierro adicional. "

                # Clasificación por condición
                condicion_principal = self._clasificar_receta(energia, hierro, alimentos_encontrados)
                texto += f"Recomendada para: {condicion_principal}. "

                # Ingredientes y preparación (resumidos)
                ingredientes_resumido = ingredientes[:300] + "..." if len(ingredientes) > 300 else ingredientes
                preparacion_resumida = preparacion[:300] + "..." if len(preparacion) > 300 else preparacion

                texto += f"Ingredientes: {ingredientes_resumido}. "
                texto += f"Preparación: {preparacion_resumida}."

                textos_recetas.append(texto)

                # Metadatos
                metadatos_recetas.append({
                    'id_receta': f'receta_{idx}',
                    'nombre': nombre,
                    'energia_kcal': float(energia),
                    'proteinas_g': float(proteinas),
                    'hierro_mg': float(hierro),
                    'condicion_principal': condicion_principal,
                    'alimentos_identificados': ', '.join([a['nombre'] for a in alimentos_encontrados]),
                    'num_alimentos_nutritivos': len(alimentos_encontrados),
                    'ingredientes_originales': ingredientes[:500],  # Truncar para evitar problemas
                    'preparacion_original': preparacion[:500]
                })

            # Crear vectorstore
            chroma_recetas = os.path.join(RUTA_CHROMA, "recetas")
            if os.path.exists(chroma_recetas):
                shutil.rmtree(chroma_recetas)

            self.vectorstore_recetas = Chroma(
                collection_name="recetas_enriquecidas",
                persist_directory=RUTA_CHROMA,
                embedding_function=self.embeddings
            )

            # Agregar en lotes
            batch_size = 30
            for i in range(0, len(textos_recetas), batch_size):
                batch_textos = textos_recetas[i:i+batch_size]
                batch_metadatos = metadatos_recetas[i:i+batch_size]

                self.vectorstore_recetas.add_texts(
                    texts=batch_textos,
                    metadatas=batch_metadatos
                )

            self.vectorstore_recetas.persist()
            logger.info(f"✅ Vectorstore recetas creado: {len(textos_recetas)} documentos")

            return True

        except Exception as e:
            logger.error(f"❌ Error creando vectorstore recetas: {e}")
            return False

    def _analizar_ingredientes(self, ingredientes: str) -> List[Dict]:
        """Analiza los ingredientes de una receta y encuentra alimentos de la tabla nutricional."""
        alimentos_encontrados = []

        if not isinstance(ingredientes, str):
            return alimentos_encontrados

        ingredientes_lower = ingredientes.lower()

        # Buscar alimentos de la tabla nutricional en los ingredientes
        for _, alimento in self.tabla_nutricional.iterrows():
            nombre_alimento = str(alimento.get('NOMBRE_ALIMENTO', '')).lower()

            # Buscar coincidencias (palabras clave)
            palabras_clave = self._extraer_palabras_clave(nombre_alimento)

            for palabra in palabras_clave:
                if palabra in ingredientes_lower and len(palabra) > 3:  # Evitar coincidencias muy cortas
                    alimentos_encontrados.append({
                        'nombre': str(alimento.get('NOMBRE_ALIMENTO', '')),
                        'energia_kcal': alimento.get('ENERGIA_KCAL', 0),
                        'proteinas_g': alimento.get('PROTEINAS_G', 0),
                        'hierro_mg': alimento.get('HIERRO_MG', 0),
                        'calcio_mg': alimento.get('CALCIO_MG', 0),
                        'palabra_encontrada': palabra
                    })
                    break  # Solo una coincidencia por alimento

        return alimentos_encontrados

    def _extraer_palabras_clave(self, nombre_alimento: str) -> List[str]:
        """Extrae palabras clave de un nombre de alimento."""
        # Eliminar caracteres especiales y dividir
        import re
        palabras = re.findall(r'\b\w+\b', nombre_alimento.lower())

        # Filtrar palabras muy comunes o muy cortas
        palabras_excluidas = {'de', 'con', 'en', 'la', 'el', 'y', 'o', 'a', 'para', 'por'}
        palabras_clave = [p for p in palabras if len(p) > 3 and p not in palabras_excluidas]

        return palabras_clave

    def _clasificar_receta(self, energia: float, hierro: float, alimentos: List[Dict]) -> str:
        """Clasifica una receta según la condición más apropiada."""

        # Calcular hierro total (receta + alimentos)
        hierro_total = hierro + sum(a.get('hierro_mg', 0) for a in alimentos)

        # Para anemia (priorizar hierro)
        if hierro_total >= 4.0:
            return "anemia"
        elif hierro_total >= 2.0:
            return "anemia_leve"

        # Para sobrepeso (priorizar bajas calorías)
        if energia <= 400:
            return "sobrepeso"
        elif energia <= 600:
            return "sobrepeso_moderado"

        # General
        return "general"

    def buscar_alimentos_recomendados(self, condicion: str, k: int = 8) -> List[Dict]:
        """Busca alimentos recomendados para una condición específica."""
        try:
            if not self.vectorstore_alimentos:
                logger.error("Vectorstore de alimentos no inicializado")
                return []

            # Crear query específica para la condición
            config_condicion = self.condiciones_nutrientes.get(condicion.lower(), self.condiciones_nutrientes["general"])

            # Query semántica
            alimentos_priorizados = config_condicion.get("alimentos_priorizados", [])
            query_alimentos = f"alimentos nutritivos para {condicion} {' '.join(alimentos_priorizados[:3])}"

            # Buscar en vectorstore
            resultados = self.vectorstore_alimentos.similarity_search(query_alimentos, k=k*2)

            # Filtrar y rankear por condición específica
            alimentos_filtrados = []
            for doc in resultados:
                metadata = doc.metadata
                condiciones_rec = metadata.get('condiciones_recomendado', '').lower()

                if condicion.lower() in condiciones_rec or 'general' in condiciones_rec:
                    alimentos_filtrados.append(metadata)

            return alimentos_filtrados[:k]

        except Exception as e:
            logger.error(f"❌ Error buscando alimentos: {e}")
            return []

    def buscar_recetas_por_condicion(self, condicion: str, alimentos_recomendados: List[str] = None, k: int = 10) -> List[Dict]:
        """Busca recetas que contengan alimentos recomendados para una condición."""
        try:
            if not self.vectorstore_recetas:
                logger.error("Vectorstore de recetas no inicializado")
                return []

            # Crear queries múltiples
            queries = []

            # Query por condición
            queries.append(f"recetas recomendadas para {condicion}")

            # Queries por alimentos específicos
            if alimentos_recomendados:
                for alimento in alimentos_recomendados[:3]:  # Top 3 alimentos
                    queries.append(f"recetas con {alimento} para {condicion}")

            # Query nutricional específica
            if condicion.lower() == "anemia":
                queries.append("recetas ricas en hierro pescado anchoveta")
            elif condicion.lower() == "sobrepeso":
                queries.append("recetas ligeras bajas calorías verduras")

            # Realizar búsquedas y combinar resultados
            recetas_encontradas = {}

            for query in queries:
                resultados = self.vectorstore_recetas.similarity_search(query, k=k//len(queries) + 2)

                for doc in resultados:
                    nombre = doc.metadata.get('nombre', '')
                    if nombre and nombre not in recetas_encontradas:
                        # Calcular score de relevancia
                        score = self._calcular_score_receta(doc.metadata, condicion, alimentos_recomendados)
                        doc.metadata['relevance_score'] = score
                        recetas_encontradas[nombre] = doc.metadata

            # Ordenar por score y devolver top k
            recetas_ordenadas = sorted(
                recetas_encontradas.values(),
                key=lambda x: x.get('relevance_score', 0),
                reverse=True
            )

            return recetas_ordenadas[:k]

        except Exception as e:
            logger.error(f"❌ Error buscando recetas: {e}")
            return []

    def _calcular_score_receta(self, metadata: Dict, condicion: str, alimentos_recomendados: List[str] = None) -> float:
        """Calcula un score de relevancia para una receta."""
        score = 0.0

        # Score por condición principal
        condicion_receta = metadata.get('condicion_principal', '').lower()
        if condicion.lower() in condicion_receta:
            score += 10.0
        elif 'general' in condicion_receta:
            score += 5.0

        # Score por alimentos identificados
        alimentos_identificados = metadata.get('alimentos_identificados', '').lower()
        if alimentos_recomendados:
            for alimento in alimentos_recomendados:
                if alimento.lower() in alimentos_identificados:
                    score += 3.0

        # Score por valores nutricionales
        if condicion.lower() == "anemia":
            hierro = metadata.get('hierro_mg', 0)
            if hierro >= 4.0:
                score += 8.0
            elif hierro >= 2.0:
                score += 4.0

        elif condicion.lower() == "sobrepeso":
            energia = metadata.get('energia_kcal', 0)
            if energia <= 400:
                score += 8.0
            elif energia <= 600:
                score += 4.0

        # Score por número de alimentos nutritivos identificados
        num_alimentos = metadata.get('num_alimentos_nutritivos', 0)
        score += num_alimentos * 1.0

        return score

    def generar_recomendacion_completa(self, condicion: str) -> Dict:
        """Genera una recomendación completa combinando alimentos y recetas."""
        try:
            logger.info(f"🔍 Generando recomendación para: {condicion}")

            # 1. Buscar alimentos recomendados
            alimentos_recomendados = self.buscar_alimentos_recomendados(condicion, k=8)
            nombres_alimentos = [a['nombre'] for a in alimentos_recomendados]

            # 2. Buscar recetas que contengan esos alimentos
            recetas_recomendadas = self.buscar_recetas_por_condicion(
                condicion,
                nombres_alimentos,
                k=10
            )

            # 3. Preparar respuesta completa
            respuesta = {
                "condicion": condicion,
                "alimentos_recomendados": {
                    "lista": alimentos_recomendados[:5],  # Top 5
                    "total_encontrados": len(alimentos_recomendados)
                },
                "recetas_recomendadas": {
                    "lista": recetas_recomendadas[:6],  # Top 6
                    "total_encontradas": len(recetas_recomendadas)
                },
                "analisis_nutricional": self._generar_analisis_nutricional(condicion, alimentos_recomendados),
                "match_alimentos_recetas": self._analizar_coincidencias(nombres_alimentos, recetas_recomendadas)
            }

            logger.info(f"✅ Recomendación generada: {len(alimentos_recomendados)} alimentos, {len(recetas_recomendadas)} recetas")

            return respuesta

        except Exception as e:
            logger.error(f"❌ Error generando recomendación: {e}")
            return {"error": str(e)}

    def _generar_analisis_nutricional(self, condicion: str, alimentos: List[Dict]) -> Dict:
        """Genera análisis nutricional de los alimentos recomendados."""
        if not alimentos:
            return {"mensaje": "No hay alimentos para analizar"}

        # Calcular promedios
        total_energia = sum(a.get('energia_kcal', 0) for a in alimentos)
        total_hierro = sum(a.get('hierro_mg', 0) for a in alimentos)
        total_proteinas = sum(a.get('proteinas_g', 0) for a in alimentos)
        total_calcio = sum(a.get('calcio_mg', 0) for a in alimentos)

        num_alimentos = len(alimentos)

        return {
            "promedio_energia_kcal": round(total_energia / num_alimentos, 1),
            "promedio_hierro_mg": round(total_hierro / num_alimentos, 1),
            "promedio_proteinas_g": round(total_proteinas / num_alimentos, 1),
            "promedio_calcio_mg": round(total_calcio / num_alimentos, 1),
            "alimento_mas_nutritivo": max(alimentos, key=lambda x: x.get('hierro_mg', 0) + x.get('proteinas_g', 0))['nombre'],
            "cumple_requerimientos": self._evaluar_requerimientos(condicion, total_hierro/num_alimentos, total_energia/num_alimentos)
        }

    def _evaluar_requerimientos(self, condicion: str, hierro_promedio: float, energia_promedio: float) -> Dict:
        """Evalúa si los alimentos cumplen los requerimientos nutricionales."""
        config = self.condiciones_nutrientes.get(condicion.lower(), {})

        evaluacion = {"cumple": True, "observaciones": []}

        if condicion.lower() == "anemia":
            if hierro_promedio < 3.0:
                evaluacion["cumple"] = False
                evaluacion["observaciones"].append("Necesita alimentos con más hierro")

        elif condicion.lower() == "sobrepeso":
            if energia_promedio > 200:
                evaluacion["cumple"] = False
                evaluacion["observaciones"].append("Algunos alimentos son altos en calorías")

        return evaluacion

    def _analizar_coincidencias(self, alimentos_recomendados: List[str], recetas: List[Dict]) -> Dict:
        """Analiza qué tan bien coinciden los alimentos recomendados con las recetas encontradas."""
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
                    "score": receta.get('relevance_score', 0)
                })

        porcentaje_match = (recetas_con_alimentos / total_recetas * 100) if total_recetas > 0 else 0

        return {
            "porcentaje_coincidencia": round(porcentaje_match, 1),
            "recetas_con_alimentos_recomendados": recetas_con_alimentos,
            "total_recetas": total_recetas,
            "coincidencias_detalladas": coincidencias_detalladas[:3]  # Top 3
        }


# =============================================================================
# FUNCIONES DE CONVENIENCIA PARA MANTENER COMPATIBILIDAD

# Instancia global del sistema
_sistema_rag = None

def inicializar_sistema_rag() -> bool:
    """Inicializa el sistema RAG global."""
    global _sistema_rag
    try:
        _sistema_rag = NutritionalRAGSystem()
        return _sistema_rag.inicializar_sistema()
    except Exception as e:
        logger.error(f"❌ Error inicializando sistema RAG: {e}")
        return False

def get_vectorstore():
    """Función de compatibilidad para obtener vectorstore de recetas."""
    global _sistema_rag
    if _sistema_rag is None:
        if not inicializar_sistema_rag():
            raise Exception("No se pudo inicializar el sistema RAG")
    return _sistema_rag.vectorstore_recetas

def get_vectorstore_nutricion():
    """Función de compatibilidad para obtener vectorstore de nutrición."""
    global _sistema_rag
    if _sistema_rag is None:
        if not inicializar_sistema_rag():
            raise Exception("No se pudo inicializar el sistema RAG")
    return _sistema_rag.vectorstore_alimentos

def buscar_plan_nutricional_completo(condicion: str) -> Dict:
    """
    Función principal que genera un plan nutricional completo.
    Esta es la función que debes usar en tu retrievalqa_improved.py
    """
    global _sistema_rag
    if _sistema_rag is None:
        if not inicializar_sistema_rag():
            return {"error": "No se pudo inicializar el sistema RAG"}

    return _sistema_rag.generar_recomendacion_completa(condicion)

def verificar_sistema_rag() -> Dict:
    """Verifica el estado del sistema RAG."""
    global _sistema_rag

    estado = {
        "sistema_inicializado": _sistema_rag is not None,
        "vectorstore_alimentos": False,
        "vectorstore_recetas": False,
        "datos_cargados": False,
        "estadisticas": {}
    }

    if _sistema_rag is not None:
        estado["vectorstore_alimentos"] = _sistema_rag.vectorstore_alimentos is not None
        estado["vectorstore_recetas"] = _sistema_rag.vectorstore_recetas is not None
        estado["datos_cargados"] = (_sistema_rag.tabla_nutricional is not None and
                                   _sistema_rag.recetas_df is not None)

        if estado["datos_cargados"]:
            estado["estadisticas"] = {
                "total_alimentos": len(_sistema_rag.tabla_nutricional),
                "total_recetas": len(_sistema_rag.recetas_df),
                "alimentos_con_hierro_alto": len(_sistema_rag.tabla_nutricional[
                    _sistema_rag.tabla_nutricional['HIERRO_MG'] >= 3.0
                ]),
                "recetas_con_valores_nutricionales": len(_sistema_rag.recetas_df[
                    (_sistema_rag.recetas_df['energia_kcal'] > 0) |
                    (_sistema_rag.recetas_df['hierro_mg'] > 0)
                ])
            }

    return estado

# =============================================================================
# ACTUALIZACIÓN PARA TU retrievalqa_improved.py

def actualizar_retrievalqa_improved():
    """
    Genera el código actualizado para tu retrievalqa_improved.py
    """

    codigo_actualizado = '''
# ACTUALIZACIÓN PARA TU app/utils/retrievalqa_improved.py

# Agregar esta importación al inicio:
from .vector_store_complete import buscar_plan_nutricional_completo, verificar_sistema_rag

class NutritionalPlanner:
    # ... (mantener todo tu código existente)

    def _buscar_recetas_semanticas_mejorada(self, condiciones: List[str], k: int = 15) -> List[dict]:
        """Búsqueda semántica mejorada usando el sistema RAG completo."""

        condicion_principal = condiciones[0].lower() if condiciones else "general"

        try:
            # Usar el nuevo sistema RAG
            recomendacion_completa = buscar_plan_nutricional_completo(condicion_principal)

            if "error" in recomendacion_completa:
                logger.warning(f"Error en sistema RAG: {recomendacion_completa['error']}")
                # Fallback al método original
                return self._buscar_recetas_semanticas(condiciones, k)

            # Extraer recetas del sistema RAG
            recetas_recomendadas = recomendacion_completa.get("recetas_recomendadas", {}).get("lista", [])

            # Convertir formato para compatibilidad
            recetas_convertidas = []
            for receta in recetas_recomendadas:
                recetas_convertidas.append({
                    "nombre": receta.get("nombre", ""),
                    "energia_kcal": receta.get("energia_kcal", 0),
                    "proteinas_g": receta.get("proteinas_g", 0),
                    "hierro_mg": receta.get("hierro_mg", 0),
                    "condicion_principal": receta.get("condicion_principal", "general"),
                    "alimentos_identificados": receta.get("alimentos_identificados", ""),
                    "ingredientes": receta.get("ingredientes_originales", ""),
                    "preparacion": receta.get("preparacion_original", ""),
                    "relevance_score": receta.get("relevance_score", 0)
                })

            logger.info(f"✅ Sistema RAG encontró {len(recetas_convertidas)} recetas para {condicion_principal}")
            return recetas_convertidas

        except Exception as e:
            logger.error(f"❌ Error en sistema RAG: {e}")
            # Fallback al método original
            return self._buscar_recetas_semanticas(condiciones, k)

    def _obtener_alimentos_clave_mejorada(self, condiciones: List[str]) -> List[str]:
        """Obtiene alimentos clave usando el sistema RAG completo."""

        condicion_principal = condiciones[0].lower() if condiciones else "general"

        try:
            # Usar el nuevo sistema RAG
            recomendacion_completa = buscar_plan_nutricional_completo(condicion_principal)

            if "error" not in recomendacion_completa:
                alimentos_recomendados = recomendacion_completa.get("alimentos_recomendados", {}).get("lista", [])
                nombres_alimentos = [a["nombre"] for a in alimentos_recomendados]

                logger.info(f"✅ Sistema RAG encontró {len(nombres_alimentos)} alimentos para {condicion_principal}")
                return nombres_alimentos[:8]

        except Exception as e:
            logger.error(f"❌ Error obteniendo alimentos del sistema RAG: {e}")

        # Fallback al método original
        return self._obtener_alimentos_clave(condiciones)

    def generar_plan(self, perfil: dict) -> dict:
        """Genera un plan nutricional personalizado usando RAG mejorado."""

        # ... (mantener validaciones existentes)

        try:
            # Usar el sistema RAG completo para obtener recomendaciones
            condicion_principal = condiciones[0].lower()

            # Obtener recomendación completa del sistema RAG
            recomendacion_rag = buscar_plan_nutricional_completo(condicion_principal)

            if "error" not in recomendacion_rag:
                # Usar datos del sistema RAG
                recetas_encontradas = recomendacion_rag["recetas_recomendadas"]["lista"]
                alimentos_recomendados_rag = [a["nombre"] for a in recomendacion_rag["alimentos_recomendados"]["lista"]]

                # Convertir formato para compatibilidad con tu código existente
                recetas_convertidas = []
                for receta in recetas_encontradas:
                    recetas_convertidas.append({
                        "nombre": receta.get("nombre", ""),
                        "energia_kcal": receta.get("energia_kcal", 0),
                        "proteinas_g": receta.get("proteinas_g", 0),
                        "hierro_mg": receta.get("hierro_mg", 0)
                    })

                logger.info(f"🎯 Usando sistema RAG: {len(recetas_convertidas)} recetas, {len(alimentos_recomendados_rag)} alimentos")

            else:
                # Fallback a métodos originales
                logger.warning("Usando métodos fallback")
                recetas_convertidas = self._buscar_recetas_semanticas(condiciones)
                alimentos_recomendados_rag = self._obtener_alimentos_clave(condiciones)

            # Continuar con tu lógica existente usando recetas_convertidas y alimentos_recomendados_rag
            # ...

            # En la respuesta final, agregar información del análisis RAG
            plan_final = {
                # ... tu código existente

                # Agregar información del sistema RAG
                "analisis_rag": recomendacion_rag.get("analisis_nutricional", {}),
                "coincidencias_alimentos_recetas": recomendacion_rag.get("match_alimentos_recetas", {}),
                "alimentos_tabla_nutricional": alimentos_recomendados_rag[:5],  # Top 5 de la tabla
            }

            return plan_final

        except Exception as e:
            logger.error(f"❌ Error generando plan: {e}")
            return {"error": f"Error interno: {str(e)}"}
'''

    return codigo_actualizado

# =============================================================================
# SCRIPT DE PRUEBA E INICIALIZACIÓN

def probar_sistema_completo():
    """Prueba completa del sistema RAG."""
    print("🧪 PROBANDO SISTEMA RAG COMPLETO")
    print("=" * 50)

    # 1. Verificar estado inicial
    print("\n1️⃣ Verificando estado del sistema...")
    estado = verificar_sistema_rag()

    for key, value in estado.items():
        if key != "estadisticas":
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")

    if estado.get("estadisticas"):
        print(f"\n📊 Estadísticas:")
        for key, value in estado["estadisticas"].items():
            print(f"  - {key}: {value}")

    # 2. Inicializar si es necesario
    if not estado["sistema_inicializado"]:
        print(f"\n2️⃣ Inicializando sistema...")
        if inicializar_sistema_rag():
            print("  ✅ Sistema inicializado correctamente")
        else:
            print("  ❌ Error inicializando sistema")
            return False

    # 3. Probar recomendaciones
    print(f"\n3️⃣ Probando recomendaciones por condición...")

    condiciones_prueba = ["anemia", "sobrepeso", "general"]

    for condicion in condiciones_prueba:
        print(f"\n🔍 Condición: {condicion}")

        try:
            recomendacion = buscar_plan_nutricional_completo(condicion)

            if "error" in recomendacion:
                print(f"  ❌ Error: {recomendacion['error']}")
                continue

            alimentos = recomendacion["alimentos_recomendados"]["lista"]
            recetas = recomendacion["recetas_recomendadas"]["lista"]

            print(f"  📊 Alimentos encontrados: {len(alimentos)}")
            print(f"  📊 Recetas encontradas: {len(recetas)}")

            # Mostrar muestra
            if alimentos:
                print(f"  🥗 Alimentos top:")
                for i, alimento in enumerate(alimentos[:3], 1):
                    nombre = alimento["nombre"]
                    hierro = alimento.get("hierro_mg", 0)
                    print(f"    {i}. {nombre} (Hierro: {hierro} mg)")

            if recetas:
                print(f"  🍽️ Recetas top:")
                for i, receta in enumerate(recetas[:3], 1):
                    nombre = receta["nombre"]
                    score = receta.get("relevance_score", 0)
                    print(f"    {i}. {nombre} (Score: {score:.1f})")

            # Análisis de coincidencias
            coincidencias = recomendacion.get("match_alimentos_recetas", {})
            porcentaje = coincidencias.get("porcentaje_coincidencia", 0)
            print(f"  🎯 Coincidencia alimentos-recetas: {porcentaje:.1f}%")

        except Exception as e:
            print(f"  ❌ Error probando {condicion}: {e}")

    print(f"\n🎉 PRUEBA COMPLETADA")
    return True

def generar_codigo_integracion():
    """Genera archivos de código para integrar el sistema."""

    # Código para config.py
    config_code = '''
# Agregar a tu app/config.py:

# Rutas para el sistema RAG completo
RUTA_NUTRICION = "data/nutricion_completa.csv"  # Tu tabla nutricional
RUTA_RECETAS = "data/recetas.csv"     # Tu archivo de recetas
RUTA_CHROMA = "data/vectorstore_rag"            # Directorio para vectorstores

# Configuración del sistema RAG
RAG_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",
    "batch_size_alimentos": 50,
    "batch_size_recetas": 30,
    "max_alimentos_recomendados": 8,
    "max_recetas_recomendadas": 10
}
'''

    # Código para usar en tu API
    api_code = '''
# Ejemplo de uso en tu endpoint de API:

from app.utils.vector_store_complete import buscar_plan_nutricional_completo, inicializar_sistema_rag

@app.post("/api/generar-plan")
async def generar_plan_endpoint(perfil: PerfilUsuario):
    try:
        # Inicializar sistema si es necesario
        if not inicializar_sistema_rag():
            return {"error": "No se pudo inicializar el sistema RAG"}

        # Obtener condición principal
        condicion = perfil.condiciones[0] if perfil.condiciones else "general"

        # Generar recomendación completa usando RAG
        recomendacion_rag = buscar_plan_nutricional_completo(condicion)

        if "error" in recomendacion_rag:
            return {"error": recomendacion_rag["error"]}

        # Procesar con tu lógica existente
        planner = NutritionalPlanner()
        plan_final = planner.generar_plan(perfil.dict())

        # Enriquecer con datos del sistema RAG
        plan_final["alimentos_tabla_nutricional"] = [
            a["nombre"] for a in recomendacion_rag["alimentos_recomendados"]["lista"]
        ]

        plan_final["analisis_nutricional_detallado"] = recomendacion_rag["analisis_nutricional"]
        plan_final["coincidencias_ingredientes"] = recomendacion_rag["match_alimentos_recetas"]

        return plan_final

    except Exception as e:
        return {"error": f"Error interno: {str(e)}"}
'''

    # Guardar archivos
    with open('rag_config.py', 'w', encoding='utf-8') as f:
        f.write(config_code)

    with open('rag_api_example.py', 'w', encoding='utf-8') as f:
        f.write(api_code)

    print("📄 Archivos de integración generados:")
    print("  - rag_config.py")
    print("  - rag_api_example.py")

if __name__ == "__main__":
    print("🚀 SISTEMA RAG NUTRICIONAL COMPLETO")
    print("=" * 50)

    # Probar sistema
    if probar_sistema_completo():
        # Generar código de integración
        generar_codigo_integracion()

        print(f"\n✅ SISTEMA LISTO PARA USAR")
        print(f"📋 Próximos pasos:")
        print(f"1. Ajustar rutas en tu config.py")
        print(f"2. Integrar con tu retrievalqa_improved.py")
        print(f"3. Probar con tu API")
    else:
        print(f"\n❌ SISTEMA NO ESTÁ LISTO")
        print(f"Revisar logs para más detalles")
