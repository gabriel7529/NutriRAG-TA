# app/utils/retrievalqa_improved.py - CORREGIDO PARA NUEVA IMPLEMENTACIÓN

from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

# IMPORTACIONES CORREGIDAS - usar la nueva implementación
from .vector_store import (
    buscar_plan_nutricional_sin_template,  # Función principal corregida
    inicializar_sistema_rag_mejorado,      # Nueva función de inicialización
    limpiar_cache_busquedas,               # Para obtener variedad
    verificar_diversidad_sistema           # Para verificar funcionamiento
)

from .llm_generator import (
    generar_explicacion,
    generar_consejos_nutricionales,
    generar_recetas_peruanas_con_llm  # Para fallback LLM
)
from .clinic_rules import calcular_get, distribuir_calorias

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NutritionalPlannerFixed:
    """Planificador nutricional corregido que usa el sistema RAG mejorado."""

    def __init__(self):
        self.sistema_inicializado = False
        self.ultimo_resultado_rag = None
        self._inicializar_rag()

    def _inicializar_rag(self):
        """Inicializa el sistema RAG mejorado."""
        try:
            logger.info("🚀 Inicializando sistema RAG mejorado...")

            # Usar la nueva función de inicialización
            if inicializar_sistema_rag_mejorado():
                self.sistema_inicializado = True
                logger.info("✅ Sistema RAG mejorado inicializado correctamente")

                # Verificar diversidad del sistema
                verificacion = verificar_diversidad_sistema()
                if verificacion.get("sistema_funcionando", False):
                    logger.info("✅ Sistema de diversidad funcionando correctamente")
                else:
                    logger.warning("⚠️ Sistema de diversidad necesita atención")

            else:
                logger.warning("⚠️ No se pudo inicializar sistema RAG, usando métodos fallback")
                self.sistema_inicializado = False

        except Exception as e:
            logger.error(f"❌ Error inicializando RAG: {e}")
            self.sistema_inicializado = False

    def _normalizar_sexo(self, sexo: str) -> str:
        """Normaliza el sexo a formato de una letra (M/F)."""
        if not isinstance(sexo, str):
            raise ValueError("El sexo debe ser una cadena de texto")

        sexo_normalizado = sexo.strip().upper()

        mapeo_masculino = ['M', 'MASCULINO', 'MALE', 'HOMBRE', 'H']
        mapeo_femenino = ['F', 'FEMENINO', 'FEMALE', 'MUJER', 'FEM']

        if sexo_normalizado in mapeo_masculino:
            return 'M'
        elif sexo_normalizado in mapeo_femenino:
            return 'F'
        else:
            raise ValueError(f"Sexo no reconocido: '{sexo}'. Valores válidos: {mapeo_masculino + mapeo_femenino}")

    def validar_perfil(self, perfil: dict) -> Tuple[bool, str]:
        """Valida el perfil del usuario con normalización de datos."""
        required_fields = ["edad", "sexo"]

        for field in required_fields:
            if field not in perfil:
                return False, f"Campo requerido faltante: {field}"

        edad = perfil["edad"]
        if not isinstance(edad, (int, float)) or edad < 12 or edad > 17:
            return False, "La edad debe ser un número entre 12 y 17 años"

        try:
            sexo_original = perfil["sexo"]
            sexo_normalizado = self._normalizar_sexo(sexo_original)
            perfil["sexo_normalizado"] = sexo_normalizado
            logger.info(f"✅ Sexo normalizado: '{sexo_original}' → '{sexo_normalizado}'")
        except ValueError as e:
            return False, str(e)

        # Validar y normalizar condiciones
        condiciones = perfil.get("condiciones", [])
        if not isinstance(condiciones, list):
            if isinstance(condiciones, str):
                perfil["condiciones"] = [condiciones] if condiciones.strip() else ["general"]
            else:
                perfil["condiciones"] = ["general"]

        condiciones_validas = ["anemia", "sobrepeso", "diabetes", "general", "ninguna"]
        condiciones_normalizadas = []

        for condicion in perfil["condiciones"]:
            condicion_lower = condicion.lower().strip()
            if condicion_lower in condiciones_validas:
                condiciones_normalizadas.append(condicion_lower)
            else:
                logger.warning(f"⚠️ Condición no reconocida: '{condicion}', usando 'general'")
                condiciones_normalizadas.append("general")

        perfil["condiciones_normalizadas"] = condiciones_normalizadas

        return True, "Perfil válido"

    def _obtener_recomendaciones_rag_mejorado(self, condicion: str, forzar_diversidad: bool = False) -> Dict:
        """
        Obtiene recomendaciones usando el sistema RAG mejorado sin templates.
        """
        try:
            logger.info(f"🔍 Obteniendo recomendaciones RAG mejorado para: {condicion}")

            # Si queremos forzar diversidad, limpiar cache
            if forzar_diversidad:
                logger.info("🧹 Limpiando cache para obtener diversidad")
                limpiar_cache_busquedas()

            # Verificar que el sistema esté inicializado
            if not self.sistema_inicializado:
                logger.warning("⚠️ Sistema RAG no inicializado, reintentando...")
                self._inicializar_rag()

            if not self.sistema_inicializado:
                logger.error("❌ Sistema RAG no disponible")
                return None

            # Usar la nueva función principal sin templates
            recomendacion_completa = buscar_plan_nutricional_sin_template(condicion)

            if "error" in recomendacion_completa:
                logger.warning(f"⚠️ Error en sistema RAG: {recomendacion_completa['error']}")
                return None

            # Verificar calidad de resultados
            alimentos_rag = recomendacion_completa.get("alimentos_recomendados", {}).get("lista", [])
            recetas_rag = recomendacion_completa.get("recetas_recomendadas", {}).get("lista", [])

            logger.info(f"📊 RAG mejorado: {len(alimentos_rag)} alimentos, {len(recetas_rag)} recetas")

            # Verificar diversidad
            diversidad_alimentos = recomendacion_completa.get("alimentos_recomendados", {}).get("diversidad", {})
            diversidad_recetas = recomendacion_completa.get("recetas_recomendadas", {}).get("diversidad", {})

            score_diversidad_alimentos = diversidad_alimentos.get("score_diversidad", 0)
            score_diversidad_recetas = diversidad_recetas.get("score_diversidad", 0)

            logger.info(f"🌈 Diversidad - Alimentos: {score_diversidad_alimentos}, Recetas: {score_diversidad_recetas}")

            # Almacenar último resultado para análisis
            self.ultimo_resultado_rag = recomendacion_completa

            return recomendacion_completa

        except Exception as e:
            logger.error(f"❌ Error obteniendo recomendaciones RAG: {e}")
            return None

    def _complementar_con_llm_si_necesario(self, recomendacion_rag: Dict, condicion: str) -> Dict:
        """
        Complementa las recomendaciones RAG con recetas LLM si es necesario.
        """
        try:
            recetas_rag = recomendacion_rag.get("recetas_recomendadas", {}).get("lista", [])
            alimentos_rag = recomendacion_rag.get("alimentos_recomendados", {}).get("lista", [])

            # Determinar si necesitamos más recetas
            necesita_complemento = len(recetas_rag) < 6

            if necesita_complemento:
                logger.info(f"🤖 Complementando con LLM: solo {len(recetas_rag)} recetas encontradas")

                # Extraer nombres de alimentos para LLM
                nombres_alimentos = [a.get("nombre", "") for a in alimentos_rag[:5]]

                try:
                    # Generar recetas adicionales con LLM
                    recetas_llm = generar_recetas_peruanas_con_llm(
                        nombres_alimentos,
                        condicion,
                        num_recetas=4
                    )

                    if recetas_llm:
                        logger.info(f"✅ LLM generó {len(recetas_llm)} recetas adicionales")

                        # Combinar recetas RAG con recetas LLM
                        recetas_combinadas = recetas_rag.copy()

                        for receta_llm in recetas_llm:
                            # Marcar como generada por LLM
                            receta_llm["generada_por_llm"] = True
                            receta_llm["relevance_score"] = receta_llm.get("relevance_score", 5.0)
                            recetas_combinadas.append(receta_llm)

                        # Actualizar la recomendación
                        recomendacion_rag["recetas_recomendadas"]["lista"] = recetas_combinadas
                        recomendacion_rag["recetas_recomendadas"]["total_encontradas"] = len(recetas_combinadas)
                        recomendacion_rag["recetas_generadas_por_llm"] = True
                        recomendacion_rag["num_recetas_llm"] = len(recetas_llm)

                        logger.info(f"🔄 Total después de complemento: {len(recetas_combinadas)} recetas")

                except Exception as e:
                    logger.warning(f"⚠️ Error complementando con LLM: {e}")

            return recomendacion_rag

        except Exception as e:
            logger.error(f"❌ Error en complemento LLM: {e}")
            return recomendacion_rag

    def _convertir_formato_rag_mejorado(self, recomendacion_rag: Dict) -> Tuple[List[Dict], List[str]]:
        """
        Convierte el formato del RAG mejorado al formato esperado por el resto del código.
        """
        try:
            # Extraer recetas
            recetas_rag = recomendacion_rag.get("recetas_recomendadas", {}).get("lista", [])
            alimentos_rag = recomendacion_rag.get("alimentos_recomendados", {}).get("lista", [])

            # Convertir recetas al formato esperado
            recetas_convertidas = []

            for receta in recetas_rag:
                receta_convertida = {
                    "nombre": receta.get("nombre", ""),
                    "energia_kcal": float(receta.get("energia_kcal", 0)),
                    "proteinas_g": float(receta.get("proteinas_g", 0)),
                    "hierro_mg": float(receta.get("hierro_mg", 0)),
                    "hierro_total": float(receta.get("hierro_total", receta.get("hierro_mg", 0))),
                    "proteinas_total": float(receta.get("proteinas_total", receta.get("proteinas_g", 0))),
                    "condiciones_apropiadas": receta.get("condiciones_apropiadas", ""),
                    "alimentos_identificados": receta.get("alimentos_identificados", ""),
                    "ingredientes": receta.get("ingredientes_originales", receta.get("ingredientes", "")),
                    "preparacion": receta.get("preparacion_original", receta.get("preparacion", "")),
                    "relevance_score": float(receta.get("relevance_score", 0)),
                    "num_alimentos_nutritivos": int(receta.get("num_alimentos_nutritivos", 0)),
                    "tipo_preparacion": receta.get("tipo_preparacion", "mixto"),
                    "dificultad": receta.get("dificultad", "intermedio"),
                    "tiempo_estimado": receta.get("tiempo_estimado", "30-45 min"),
                    "generada_por_llm": bool(receta.get("generada_por_llm", False)),

                    # Campos adicionales para compatibilidad
                    "tiempo_preparacion": receta.get("tiempo_estimado", "30-45 min"),
                    "porciones": int(receta.get("porciones", 4)),
                    "beneficios_nutricionales": receta.get("beneficios_nutricionales", "")
                }

                recetas_convertidas.append(receta_convertida)

            # Extraer nombres de alimentos
            nombres_alimentos = [a.get("nombre", "") for a in alimentos_rag]

            logger.info(f"✅ Convertido: {len(recetas_convertidas)} recetas, {len(nombres_alimentos)} alimentos")

            return recetas_convertidas, nombres_alimentos

        except Exception as e:
            logger.error(f"❌ Error convirtiendo formato RAG: {e}")
            return [], []

    def _calcular_score_total_mejorado(self, receta: Dict, condicion: str) -> float:
        """
        Calcula score total mejorado que considera tanto RAG como características específicas.
        """
        score = 0.0

        try:
            # 1. Score del sistema RAG (peso alto)
            score_rag = float(receta.get("relevance_score", 0))
            score += score_rag * 3.0  # Dar mucho peso al score del RAG

            # 2. Score por valores nutricionales totales (considerando ingredientes)
            if condicion.lower() == "anemia":
                hierro_total = float(receta.get("hierro_total", receta.get("hierro_mg", 0)))
                proteinas_total = float(receta.get("proteinas_total", receta.get("proteinas_g", 0)))

                if hierro_total >= 8.0:
                    score += 15.0
                elif hierro_total >= 5.0:
                    score += 12.0
                elif hierro_total >= 3.0:
                    score += 8.0
                elif hierro_total >= 1.0:
                    score += 4.0

                if proteinas_total >= 25.0:
                    score += 10.0
                elif proteinas_total >= 15.0:
                    score += 6.0
                elif proteinas_total >= 10.0:
                    score += 3.0

            elif condicion.lower() == "sobrepeso":
                energia = float(receta.get("energia_kcal", 0))

                if energia <= 300:
                    score += 15.0
                elif energia <= 500:
                    score += 10.0
                elif energia <= 700:
                    score += 5.0
                else:
                    score -= 3.0  # Penalizar muy altas calorías

            # 3. Score por diversidad de alimentos nutritivos
            num_alimentos = int(receta.get("num_alimentos_nutritivos", 0))
            score += min(num_alimentos * 2.0, 10.0)  # Máximo 10 puntos

            # 4. Score por condiciones apropiadas
            condiciones_apropiadas = receta.get("condiciones_apropiadas", "").lower()
            if condicion.lower() in condiciones_apropiadas:
                score += 8.0
            elif "general" in condiciones_apropiadas:
                score += 3.0

            # 5. Bonus por recetas generadas por LLM (están específicamente diseñadas)
            if receta.get("generada_por_llm", False):
                score += 5.0

            # 6. Score por tipo de preparación (preferir métodos saludables)
            tipo_prep = receta.get("tipo_preparacion", "").lower()
            if tipo_prep in ["cocido", "al_vapor", "guisado"]:
                score += 3.0
            elif tipo_prep in ["salteado", "horneado"]:
                score += 2.0
            elif tipo_prep == "frito":
                score -= 2.0  # Penalizar frituras

        except Exception as e:
            logger.warning(f"Error calculando score para {receta.get('nombre', 'unknown')}: {e}")

        return round(score, 2)

    def _filtrar_recetas_inteligente_rag(self, recetas: List[Dict], condicion: str, calorias_objetivo: int) -> List[Dict]:
        """
        Filtrado inteligente específico para el sistema RAG mejorado.
        """
        if not recetas:
            return []

        logger.info(f"🔍 Filtrando {len(recetas)} recetas para {condicion}")

        # Calcular scores para todas las recetas
        recetas_con_score = []

        for receta in recetas:
            score = self._calcular_score_total_mejorado(receta, condicion)
            recetas_con_score.append((receta, score))

        # Ordenar por score descendente
        recetas_con_score.sort(key=lambda x: x[1], reverse=True)

        # Aplicar filtros específicos por condición
        recetas_filtradas = []

        for receta, score in recetas_con_score:
            incluir = False

            # Criterios específicos por condición
            if condicion.lower() == "anemia":
                hierro_total = float(receta.get("hierro_total", receta.get("hierro_mg", 0)))
                # Ser más permisivo con recetas LLM o con buen score
                if (receta.get("generada_por_llm", False) or
                    hierro_total >= 1.0 or
                    score >= 8.0):
                    incluir = True

            elif condicion.lower() == "sobrepeso":
                energia = float(receta.get("energia_kcal", 0))
                # Permitir recetas LLM o con energía controlada
                if (receta.get("generada_por_llm", False) or
                    energia <= 800 or
                    score >= 8.0):
                    incluir = True

            else:  # general y otros
                # Para condiciones generales, ser más permisivo
                if score >= 5.0:
                    incluir = True

            if incluir:
                recetas_filtradas.append(receta)

        # Asegurar diversidad de tipos de preparación
        recetas_diversas = self._asegurar_diversidad_preparacion(recetas_filtradas)

        # Limitar a 8 recetas máximo
        resultado_final = recetas_diversas[:8]

        logger.info(f"✅ Filtradas: {len(resultado_final)} recetas finales")

        # Log de estadísticas
        recetas_llm = len([r for r in resultado_final if r.get("generada_por_llm", False)])
        if recetas_llm > 0:
            logger.info(f"🤖 Incluye {recetas_llm} recetas generadas por LLM")

        return resultado_final

    def _asegurar_diversidad_preparacion(self, recetas: List[Dict]) -> List[Dict]:
        """
        Asegura diversidad en los tipos de preparación.
        """
        if len(recetas) <= 6:
            return recetas

        diversas = []
        tipos_usados = set()

        # Primera pasada: una de cada tipo
        for receta in recetas:
            tipo = receta.get("tipo_preparacion", "mixto")
            if tipo not in tipos_usados:
                diversas.append(receta)
                tipos_usados.add(tipo)

        # Segunda pasada: completar con las mejores
        for receta in recetas:
            if len(diversas) >= 8:
                break
            if receta not in diversas:
                diversas.append(receta)

        return diversas

    def _distribuir_menu_por_tiempo_mejorado(self, recetas: List[Dict], distribucion: Dict) -> Dict:
        """
        Distribución mejorada que considera el tipo de preparación y horarios apropiados.
        """
        menu_distribuido = {
            "desayuno": [],
            "almuerzo": [],
            "cena": [],
            "refrigerio": []
        }

        if not recetas:
            return menu_distribuido

        # Clasificar recetas por energía y tipo
        recetas_ligeras = []      # <= 350 kcal
        recetas_medianas = []     # 350-600 kcal
        recetas_completas = []    # > 600 kcal

        for receta in recetas:
            energia = receta.get("energia_kcal", 0)
            tipo_prep = receta.get("tipo_preparacion", "")

            if energia <= 350:
                recetas_ligeras.append(receta)
            elif energia <= 600:
                recetas_medianas.append(receta)
            else:
                recetas_completas.append(receta)

        # Asignar por horarios

        # Desayuno: preferir ligeras o medianas
        candidatos_desayuno = recetas_ligeras + recetas_medianas
        if candidatos_desayuno:
            menu_distribuido["desayuno"] = [candidatos_desayuno[0]["nombre"]]

        # Almuerzo: preferir completas o medianas
        candidatos_almuerzo = recetas_completas + recetas_medianas
        if candidatos_almuerzo:
            # Evitar repetir la del desayuno
            for candidato in candidatos_almuerzo:
                if candidato["nombre"] not in menu_distribuido["desayuno"]:
                    menu_distribuido["almuerzo"] = [candidato["nombre"]]
                    break

        # Cena: medianas o ligeras
        candidatos_cena = recetas_medianas + recetas_ligeras
        if candidatos_cena:
            for candidato in candidatos_cena:
                if (candidato["nombre"] not in menu_distribuido["desayuno"] and
                    candidato["nombre"] not in menu_distribuido["almuerzo"]):
                    menu_distribuido["cena"] = [candidato["nombre"]]
                    break

        # Refrigerio: ligeras
        if recetas_ligeras:
            for candidato in recetas_ligeras:
                if candidato["nombre"] not in [
                    menu_distribuido["desayuno"],
                    menu_distribuido["almuerzo"],
                    menu_distribuido["cena"]
                ]:
                    menu_distribuido["refrigerio"] = [candidato["nombre"]]
                    break

        return menu_distribuido

    def generar_plan(self, perfil: dict) -> dict:
        """
        Genera un plan nutricional usando el sistema RAG mejorado sin templates.
        """

        # 1. Validar entrada
        es_valido, mensaje = self.validar_perfil(perfil)
        if not es_valido:
            return {"error": mensaje}

        # 2. Extraer datos del perfil
        edad = perfil["edad"]
        sexo = perfil["sexo_normalizado"]
        sexo_original = perfil["sexo"]
        peso = perfil.get("peso", "No especificado")
        altura = perfil.get("altura", "No especificado")
        condiciones = perfil.get("condiciones_normalizadas", ["general"])

        if not condiciones or "ninguna" in condiciones:
            condiciones = ["general"]

        condicion_principal = condiciones[0].lower()

        logger.info(f"🎯 Generando plan para: {edad} años, {sexo} ({sexo_original}), condición: {condicion_principal}")

        try:
            # 3. Cálculos nutricionales básicos
            get = calcular_get(edad, sexo)
            distribucion = distribuir_calorias(get)

            # 4. Obtener recomendaciones del sistema RAG mejorado
            logger.info("🔍 Consultando sistema RAG mejorado...")

            recomendacion_rag = self._obtener_recomendaciones_rag_mejorado(condicion_principal)

            if recomendacion_rag:
                logger.info("✅ Sistema RAG mejorado respondió correctamente")

                # 5. Complementar con LLM si es necesario
                recomendacion_completa = self._complementar_con_llm_si_necesario(
                    recomendacion_rag, condicion_principal
                )

                # 6. Convertir formato
                recetas_encontradas, alimentos_recomendados = self._convertir_formato_rag_mejorado(
                    recomendacion_completa
                )

                # Información adicional del análisis RAG
                analisis_nutricional = recomendacion_completa.get("analisis_nutricional", {})
                coincidencias_alimentos = recomendacion_completa.get("match_alimentos_recetas", {})
                recomendaciones_adicionales = recomendacion_completa.get("recomendaciones_adicionales", {})
                tiene_recetas_llm = recomendacion_completa.get("recetas_generadas_por_llm", False)

                logger.info(f"📊 Datos procesados: {len(recetas_encontradas)} recetas, {len(alimentos_recomendados)} alimentos")
                if tiene_recetas_llm:
                    logger.info("🤖 Incluye recetas generadas por LLM")

            else:
                # Fallback completo
                logger.warning("⚠️ Sistema RAG no disponible, usando fallback completo")
                recetas_encontradas = self._buscar_recetas_fallback_completo(condicion_principal)
                alimentos_recomendados = self._obtener_alimentos_fallback(condiciones)
                analisis_nutricional = {}
                coincidencias_alimentos = {}
                recomendaciones_adicionales = {}
                tiene_recetas_llm = True  # En fallback, todas son LLM

            # Verificar que tenemos recetas
            if not recetas_encontradas:
                return {
                    "error": "No se pudieron generar recetas para la condición especificada. El sistema necesita datos adicionales."
                }

            # 7. Filtrado inteligente específico para RAG
            recetas_filtradas = self._filtrar_recetas_inteligente_rag(
                recetas_encontradas,
                condicion_principal,
                get
            )

            # 8. Distribuir menú por horarios mejorado
            menu_distribuido = self._distribuir_menu_por_tiempo_mejorado(recetas_filtradas, distribucion)

            # 9. Preparar datos para LLM (explicación)
            nombres_recetas = [r["nombre"] for r in recetas_filtradas[:4]]
            condiciones_str = ", ".join(condiciones)

            # Crear perfil enriquecido
            perfil_str = (
                f"Adolescente de {edad} años, sexo {sexo_original}, peso {peso}, altura {altura}. "
                f"Diagnóstico: {condiciones_str}. "
                f"Requerimiento energético: {get} kcal/día. "
                f"Alimentos clave recomendados: {', '.join(alimentos_recomendados[:5])}. "
            )

            # Agregar información nutricional si disponible
            if analisis_nutricional and "alimentos" in analisis_nutricional:
                info_alimentos = analisis_nutricional["alimentos"]
                hierro_promedio = info_alimentos.get("promedio_hierro_mg", 0)
                energia_promedio = info_alimentos.get("promedio_energia_kcal", 0)
                perfil_str += f"Los alimentos recomendados aportan en promedio {hierro_promedio} mg de hierro y {energia_promedio} kcal por 100g. "

            # 10. Generar explicación con LLM
            try:
                recetas_info = ", ".join(nombres_recetas)
                if tiene_recetas_llm:
                    recetas_info += " (incluye recetas creadas específicamente para este perfil nutricional)"

                explicacion = generar_explicacion(perfil_str, recetas_info, max_tokens=400)
                consejos = generar_consejos_nutricionales(perfil_str, max_tokens=250)

            except Exception as e:
                logger.warning(f"⚠️ Error generando explicación con LLM: {e}")
                explicacion = self._generar_explicacion_fallback(condicion_principal, nombres_recetas)
                consejos = self._generar_consejos_fallback(condicion_principal)

            # 11. Preparar respuesta final enriquecida
            plan_final = {
                "perfil": {
                    "edad": edad,
                    "sexo": sexo_original,
                    "peso": peso,
                    "altura": altura,
                    "condiciones": condiciones
                },
                "requerimientos": {
                    "calorias_diarias": get,
                    "distribucion_horaria": distribucion
                },
                "menu_sugerido": nombres_recetas,
                "menu_por_horario": menu_distribuido,
                "alimentos_recomendados": alimentos_recomendados[:8],
                "explicacion_profesional": explicacion,
                "consejos_adicionales": consejos,

                # Información enriquecida del sistema RAG mejorado
                "alimentos_tabla_nutricional": [
                    {
                        "nombre": a.get("nombre", ""),
                        "hierro_mg": float(a.get("hierro_mg", 0)),
                        "energia_kcal": float(a.get("energia_kcal", 0)),
                        "proteinas_g": float(a.get("proteinas_g", 0)),
                        "calcio_mg": float(a.get("calcio_mg", 0)),
                        "fibra_g": float(a.get("fibra_g", 0)),
                        "grupo_alimento": a.get("grupo_alimento", "otros"),
                        "indice_nutricional": float(a.get("indice_nutricional", 0))
                    }
                    for a in (recomendacion_rag.get("alimentos_recomendados", {}).get("lista", []) if recomendacion_rag else [])
                ][:6],

                # Análisis nutricional detallado del sistema RAG
                "analisis_nutricional_detallado": analisis_nutricional,
                "coincidencias_ingredientes": coincidencias_alimentos,
                "recomendaciones_adicionales": recomendaciones_adicionales,

                # Recetas con información completa del RAG
                "recetas_detalladas": [
                    {
                        "nombre": r["nombre"],
                        "energia_kcal": float(r.get("energia_kcal", 0)),
                        "hierro_mg": float(r.get("hierro_mg", 0)),
                        "hierro_total": float(r.get("hierro_total", r.get("hierro_mg", 0))),
                        "proteinas_g": float(r.get("proteinas_g", 0)),
                        "proteinas_total": float(r.get("proteinas_total", r.get("proteinas_g", 0))),
                        "condiciones_apropiadas": r.get("condiciones_apropiadas", ""),
                        "alimentos_identificados": r.get("alimentos_identificados", ""),
                        "score_relevancia": float(r.get("relevance_score", 0)),
                        "tipo_preparacion": r.get("tipo_preparacion", "mixto"),
                        "dificultad": r.get("dificultad", "intermedio"),
                        "tiempo_estimado": r.get("tiempo_estimado", "30-45 min"),
                        "num_alimentos_nutritivos": int(r.get("num_alimentos_nutritivos", 0)),

                        # Información de la receta
                        "ingredientes": self._truncar_texto(r.get("ingredientes", ""), 400),
                        "preparacion": self._truncar_texto(r.get("preparacion", ""), 400),
                        "porciones": int(r.get("porciones", 4)),
                        "beneficios_nutricionales": r.get("beneficios_nutricionales", ""),

                        # Metadata del sistema
                        "generada_por_ia": bool(r.get("generada_por_llm", False)),
                        "query_origen": r.get("query_origen", ""),
                        "sistema_origen": "RAG_mejorado" if not r.get("generada_por_llm", False) else "LLM_complemento"
                    }
                    for r in recetas_filtradas[:6]
                ],

                # Métricas del sistema mejorado
                "metricas_sistema": {
                    "version_sistema": "RAG_mejorado_v2",
                    "total_recetas_analizadas": len(recetas_encontradas),
                    "recetas_seleccionadas": len(recetas_filtradas),
                    "total_alimentos_recomendados": len(alimentos_recomendados),
                    "sistema_rag_usado": recomendacion_rag is not None,
                    "recetas_generadas_por_ia": tiene_recetas_llm,
                    "recetas_ia_count": len([r for r in recetas_filtradas if r.get("generada_por_llm", False)]),
                    "porcentaje_coincidencia_ingredientes": float(coincidencias_alimentos.get("porcentaje_coincidencia", 0)) if coincidencias_alimentos else 0,
                    "diversidad_alimentos": (recomendacion_rag.get("alimentos_recomendados", {}).get("diversidad", {}).get("score_diversidad", 0) if recomendacion_rag else 0),
                    "diversidad_recetas": (recomendacion_rag.get("recetas_recomendadas", {}).get("diversidad", {}).get("score_diversidad", 0) if recomendacion_rag else 0),
                    "calidad_match": coincidencias_alimentos.get("calidad_match", "moderada") if coincidencias_alimentos else "no_disponible"
                }
            }

            # Log de métricas finales
            metricas = plan_final["metricas_sistema"]
            logger.info(f"✅ Plan generado exitosamente:")
            logger.info(f"   - {metricas['recetas_seleccionadas']} recetas finales")
            logger.info(f"   - {metricas['total_alimentos_recomendados']} alimentos recomendados")
            logger.info(f"   - Diversidad alimentos: {metricas['diversidad_alimentos']}")
            logger.info(f"   - Diversidad recetas: {metricas['diversidad_recetas']}")

            if metricas['recetas_generadas_por_ia']:
                logger.info(f"   - {metricas['recetas_ia_count']} recetas generadas por IA")

            return plan_final

        except Exception as e:
            logger.error(f"❌ Error generando plan: {e}")
            return {
                "error": f"Error interno al generar el plan: {str(e)}",
                "perfil_recibido": perfil,
                "sistema_rag_disponible": self.sistema_inicializado
            }

    def _truncar_texto(self, texto: str, max_length: int) -> str:
        """Trunca texto manteniendo palabras completas."""
        if not texto or len(texto) <= max_length:
            return texto

        truncado = texto[:max_length]
        ultimo_espacio = truncado.rfind(' ')

        if ultimo_espacio > max_length * 0.8:  # Si el último espacio está en el último 20%
            return truncado[:ultimo_espacio] + "..."
        else:
            return truncado + "..."

    def _buscar_recetas_fallback_completo(self, condicion: str) -> List[Dict]:
        """Fallback completo usando solo LLM cuando el sistema RAG falla."""
        logger.warning(f"🔄 Activando fallback completo para {condicion}")

        # Alimentos básicos según condición para el LLM
        alimentos_basicos = {
            "anemia": ["pescado", "quinua", "lentejas", "espinaca", "hígado", "anchoveta"],
            "sobrepeso": ["verduras", "frutas", "pescado magro", "cereales integrales", "ensaladas"],
            "diabetes": ["verduras", "pescado", "quinua", "legumbres", "cereales integrales"],
            "general": ["pescado", "quinua", "verduras", "frutas", "legumbres", "cereales"]
        }

        alimentos_para_condicion = alimentos_basicos.get(condicion, alimentos_basicos["general"])

        try:
            # Intentar generar recetas con LLM
            recetas_llm = generar_recetas_peruanas_con_llm(
                alimentos_para_condicion,
                condicion,
                num_recetas=6
            )

            if recetas_llm:
                logger.info(f"✅ Fallback LLM generó {len(recetas_llm)} recetas")
                # Marcar todas como generadas por LLM
                for receta in recetas_llm:
                    receta["generada_por_llm"] = True
                    receta["relevance_score"] = 7.0  # Score alto para recetas específicas
                    receta["sistema_origen"] = "LLM_fallback"

                return recetas_llm
            else:
                logger.warning("⚠️ LLM fallback no generó recetas")

        except Exception as e:
            logger.error(f"❌ Error en fallback LLM: {e}")

        # Fallback final: recetas hardcoded
        return self._recetas_hardcoded_fallback(condicion)

    def _recetas_hardcoded_fallback(self, condicion: str) -> List[Dict]:
        """Recetas hardcoded como último recurso."""
        logger.warning("🔄 Usando recetas hardcoded de emergencia")

        recetas_emergencia = {
            "anemia": [
                {
                    "nombre": "Pescado a la plancha con quinua",
                    "energia_kcal": 350,
                    "hierro_mg": 4.5,
                    "proteinas_g": 25,
                    "ingredientes": "Pescado fresco, quinua, verduras",
                    "preparacion": "Cocinar pescado a la plancha, hervir quinua, acompañar con verduras",
                    "generada_por_llm": False,
                    "relevance_score": 6.0,
                    "tipo_preparacion": "plancha"
                },
                {
                    "nombre": "Lentejas guisadas con espinaca",
                    "energia_kcal": 280,
                    "hierro_mg": 3.8,
                    "proteinas_g": 18,
                    "ingredientes": "Lentejas, espinaca, cebolla, ajo",
                    "preparacion": "Guisar lentejas con verduras y espinaca fresca",
                    "generada_por_llm": False,
                    "relevance_score": 5.5,
                    "tipo_preparacion": "guisado"
                }
            ],
            "sobrepeso": [
                {
                    "nombre": "Ensalada de verduras con atún",
                    "energia_kcal": 220,
                    "hierro_mg": 2.0,
                    "proteinas_g": 20,
                    "ingredientes": "Verduras mixtas, atún, limón",
                    "preparacion": "Mezclar verduras frescas con atún en agua",
                    "generada_por_llm": False,
                    "relevance_score": 5.0,
                    "tipo_preparacion": "crudo"
                },
                {
                    "nombre": "Sopa de verduras con quinua",
                    "energia_kcal": 180,
                    "hierro_mg": 1.5,
                    "proteinas_g": 8,
                    "ingredientes": "Verduras variadas, quinua, caldo",
                    "preparacion": "Hervir verduras con quinua en caldo ligero",
                    "generada_por_llm": False,
                    "relevance_score": 4.5,
                    "tipo_preparacion": "cocido"
                }
            ]
        }

        recetas_default = [
            {
                "nombre": "Pescado con verduras al vapor",
                "energia_kcal": 300,
                "hierro_mg": 3.0,
                "proteinas_g": 22,
                "ingredientes": "Pescado, verduras mixtas",
                "preparacion": "Cocinar al vapor pescado con verduras",
                "generada_por_llm": False,
                "relevance_score": 5.0,
                "tipo_preparacion": "vapor"
            }
        ]

        return recetas_emergencia.get(condicion, recetas_default)

    def _obtener_alimentos_fallback(self, condiciones: List[str]) -> List[str]:
        """Alimentos fallback específicos por condición."""
        condicion_principal = condiciones[0].lower() if condiciones else "general"

        alimentos_fallback = {
            "anemia": ["Pescado", "Quinua", "Lentejas", "Espinaca", "Hígado de pollo", "Anchoveta"],
            "sobrepeso": ["Verduras de hoja verde", "Frutas frescas", "Pescado magro", "Cereales integrales", "Brócoli"],
            "diabetes": ["Verduras", "Pescado", "Quinua", "Legumbres", "Cereales integrales"],
            "general": ["Pescado", "Quinua", "Verduras variadas", "Frutas de estación", "Legumbres"]
        }

        return alimentos_fallback.get(condicion_principal, alimentos_fallback["general"])

    def _generar_explicacion_fallback(self, condicion: str, recetas: List[str]) -> str:
        """Explicación fallback mejorada."""
        recetas_str = ', '.join(recetas[:2]) if recetas else "las recetas seleccionadas"

        explicaciones = {
            "anemia": f"Este plan nutricional prioriza alimentos ricos en hierro como {recetas_str}. "
                     "El hierro presente en pescados, quinua y legumbres ayuda a mejorar los niveles de hemoglobina. "
                     "Se recomienda combinar con alimentos ricos en vitamina C para optimizar la absorción del hierro. "
                     "Este plan está diseñado específicamente para adolescentes con anemia ferropénica.",

            "sobrepeso": f"Las recetas seleccionadas ({recetas_str}) son opciones nutritivas con control calórico. "
                        "Priorizan proteínas magras, verduras y preparaciones saludables que contribuyen al control del peso "
                        "mientras aportan los nutrientes esenciales para el crecimiento adolescente. "
                        "Se enfoca en la saciedad y el valor nutricional.",

            "general": f"Este plan incluye recetas variadas ({recetas_str}) que aportan los nutrientes necesarios "
                      "para el crecimiento y desarrollo durante la adolescencia. Las recetas están adaptadas a "
                      "ingredientes peruanos y preferencias locales, asegurando un aporte nutricional balanceado."
        }

        return explicaciones.get(condicion, explicaciones["general"])

    def _generar_consejos_fallback(self, condicion: str) -> str:
        """Consejos fallback específicos por condición."""
        consejos_base = [
            "Mantén horarios regulares de comida (desayuno, almuerzo, cena y refrigerio saludable).",
            "Consume al menos 8 vasos de agua al día para mantener una hidratación adecuada.",
            "Incluye variedad de colores en tus comidas a través de frutas y verduras frescas."
        ]

        consejos_especificos = {
            "anemia": [
                "Combina alimentos ricos en hierro con frutas cítricas para mejorar la absorción.",
                "Evita tomar té o café durante las comidas principales ya que pueden interferir con la absorción del hierro.",
                "Incluye regularmente alimentos de origen animal ricos en hierro hemo."
            ],
            "sobrepeso": [
                "Practica actividad física regular, al menos 60 minutos diarios.",
                "Controla las porciones y come lentamente para reconocer las señales de saciedad.",
                "Prefiere métodos de cocción saludables como al vapor, a la plancha o hervido."
            ],
            "general": [
                "Mantén un equilibrio entre todos los grupos de alimentos.",
                "Limita el consumo de alimentos ultraprocesados y azúcares añadidos.",
                "Asegúrate de dormir entre 8-10 horas diarias para un desarrollo óptimo."
            ]
        }

        consejos_condicion = consejos_especificos.get(condicion, consejos_especificos["general"])
        todos_consejos = consejos_base + consejos_condicion

        return " ".join(todos_consejos)

    def verificar_estado_sistema(self) -> Dict:
        """Verifica el estado actual del sistema RAG."""
        try:
            estado = {
                "sistema_inicializado": self.sistema_inicializado,
                "ultimo_resultado_disponible": self.ultimo_resultado_rag is not None,
                "timestamp": self._get_timestamp()
            }

            if self.sistema_inicializado:
                # Verificar diversidad del sistema
                verificacion_diversidad = verificar_diversidad_sistema()
                estado["diversidad_sistema"] = verificacion_diversidad

            return estado

        except Exception as e:
            return {
                "error": str(e),
                "sistema_inicializado": False,
                "timestamp": self._get_timestamp()
            }

    def limpiar_cache_sistema(self):
        """Limpia el cache del sistema para obtener nuevas recomendaciones."""
        try:
            if self.sistema_inicializado:
                limpiar_cache_busquedas()
                logger.info("🧹 Cache del sistema limpiado")
            else:
                logger.warning("⚠️ Sistema no inicializado, no se puede limpiar cache")
        except Exception as e:
            logger.error(f"❌ Error limpiando cache: {e}")

    def _get_timestamp(self) -> str:
        """Obtiene timestamp actual."""
        from datetime import datetime
        return datetime.now().isoformat()


# Función de conveniencia mejorada para mantener compatibilidad
def generar_plan(perfil: dict) -> dict:
    """
    Función wrapper mejorada que usa el sistema RAG sin templates.
    Mantiene compatibilidad con el código existente.
    """
    planner = NutritionalPlannerFixed()
    return planner.generar_plan(perfil)

def verificar_sistema_rag_mejorado() -> dict:
    """Verifica el estado del sistema RAG mejorado."""
    try:
        planner = NutritionalPlannerFixed()
        return planner.verificar_estado_sistema()
    except Exception as e:
        return {"error": str(e), "sistema_funcionando": False}

def limpiar_cache_recomendaciones():
    """Limpia el cache para obtener recomendaciones frescas."""
    try:
        planner = NutritionalPlannerFixed()
        planner.limpiar_cache_sistema()
    except Exception as e:
        logger.error(f"Error limpiando cache: {e}")

# Función de prueba para verificar funcionamiento
def probar_sistema_completo():
    """Función de prueba completa del sistema mejorado."""
    print("🧪 PROBANDO SISTEMA RAG MEJORADO")
    print("=" * 50)

    # Perfiles de prueba
    perfiles_prueba = [
        {
            "edad": 15,
            "sexo": "femenino",
            "peso": "55 kg",
            "altura": "1.60 m",
            "condiciones": ["anemia"]
        },
        {
            "edad": 16,
            "sexo": "masculino",
            "peso": "70 kg",
            "altura": "1.70 m",
            "condiciones": ["sobrepeso"]
        }
    ]

    try:
        planner = NutritionalPlannerFixed()

        # Verificar estado inicial
        estado = planner.verificar_estado_sistema()
        print(f"📊 Estado del sistema: {estado}")

        # Probar con cada perfil
        for i, perfil in enumerate(perfiles_prueba, 1):
            print(f"\n--- PRUEBA {i}: {perfil['condiciones'][0].upper()} ---")

            resultado = planner.generar_plan(perfil)

            if "error" in resultado:
                print(f"❌ Error: {resultado['error']}")
            else:
                metricas = resultado.get("metricas_sistema", {})
                print(f"✅ Plan generado:")
                print(f"   - Recetas: {metricas.get('recetas_seleccionadas', 0)}")
                print(f"   - Alimentos: {metricas.get('total_alimentos_recomendados', 0)}")
                print(f"   - Sistema RAG usado: {metricas.get('sistema_rag_usado', False)}")
                print(f"   - Recetas IA: {metricas.get('recetas_ia_count', 0)}")
                print(f"   - Diversidad alimentos: {metricas.get('diversidad_alimentos', 0)}")

        print(f"\n🎉 Prueba completada")
        return True

    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False

if __name__ == "__main__":
    probar_sistema_completo()
