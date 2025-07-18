# app/utils/retrievalqa_improved.py - CON GENERADOR LLM DE RECETAS

from typing import Dict, List, Optional, Tuple
import logging
from .vector_store import (
    buscar_plan_nutricional_completo,
    verificar_sistema_rag,
    inicializar_sistema_rag
)
from .llm_generator import (
    generar_explicacion,
    generar_consejos_nutricionales,
    generar_recetas_peruanas_con_llm  # Nueva función
)
from .clinic_rules import calcular_get, distribuir_calorias

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NutritionalPlanner:
    def __init__(self):
        # Inicializar sistema RAG al crear el planner
        self._inicializar_rag()

    def _inicializar_rag(self):
        """Inicializa el sistema RAG si no está listo."""
        try:
            estado = verificar_sistema_rag()
            if not estado["sistema_inicializado"]:
                logger.info("🔄 Inicializando sistema RAG...")
                if inicializar_sistema_rag():
                    logger.info("✅ Sistema RAG inicializado correctamente")
                else:
                    logger.warning("⚠️ No se pudo inicializar sistema RAG, usando métodos fallback")
            else:
                logger.info("✅ Sistema RAG ya inicializado")
        except Exception as e:
            logger.warning(f"⚠️ Error inicializando RAG: {e}")

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

    def _obtener_recomendaciones_rag_con_llm_fallback(self, condicion: str) -> Dict:
        """
        Obtiene recomendaciones usando RAG y si no hay recetas, las genera con LLM.
        """
        try:
            logger.info(f"🔍 Obteniendo recomendaciones RAG para: {condicion}")

            # Intentar usar sistema RAG completo
            recomendacion_completa = buscar_plan_nutricional_completo(condicion)

            if "error" in recomendacion_completa:
                logger.warning(f"⚠️ Error en sistema RAG: {recomendacion_completa['error']}")
                return None

            # Verificar si tenemos recetas
            recetas_rag = recomendacion_completa.get("recetas_recomendadas", {}).get("lista", [])
            alimentos_rag = recomendacion_completa.get("alimentos_recomendados", {}).get("lista", [])

            logger.info(f"📊 RAG inicial: {len(alimentos_rag)} alimentos, {len(recetas_rag)} recetas")

            # Si no hay recetas pero sí alimentos, generar recetas con LLM
            if len(recetas_rag) == 0 and len(alimentos_rag) > 0:
                logger.info("🤖 No hay recetas en vectorstore, generando con LLM...")

                nombres_alimentos = [a.get("nombre", "") for a in alimentos_rag]
                recetas_generadas_llm = generar_recetas_peruanas_con_llm(
                    alimentos_recomendados=nombres_alimentos,
                    condicion=condicion,
                    num_recetas=6
                )

                if recetas_generadas_llm:
                    # Actualizar la recomendación con las recetas generadas
                    recomendacion_completa["recetas_recomendadas"]["lista"] = recetas_generadas_llm
                    recomendacion_completa["recetas_recomendadas"]["total_encontradas"] = len(recetas_generadas_llm)
                    recomendacion_completa["recetas_generadas_por_llm"] = True

                    logger.info(f"✅ Generadas {len(recetas_generadas_llm)} recetas con LLM")
                else:
                    logger.warning("⚠️ No se pudieron generar recetas con LLM")

            return recomendacion_completa

        except Exception as e:
            logger.error(f"❌ Error obteniendo recomendaciones: {e}")
            return None

    def _convertir_recetas_rag_a_formato_compatible(self, recetas_rag: List[Dict]) -> List[Dict]:
        """Convierte las recetas del formato RAG al formato esperado por el código existente."""
        recetas_convertidas = []

        for receta in recetas_rag:
            receta_convertida = {
                "nombre": receta.get("nombre", ""),
                "energia_kcal": float(receta.get("energia_kcal", 0)),
                "proteinas_g": float(receta.get("proteinas_g", 0)),
                "hierro_mg": float(receta.get("hierro_mg", 0)),
                "condicion_principal": receta.get("condicion_principal", "general"),
                "alimentos_identificados": receta.get("alimentos_identificados", ""),
                "ingredientes": receta.get("ingredientes_originales", receta.get("ingredientes", "")),
                "preparacion": receta.get("preparacion_original", receta.get("preparacion", "")),
                "relevance_score": receta.get("relevance_score", 0),
                "num_alimentos_nutritivos": receta.get("num_alimentos_nutritivos", 0),
                "generada_por_llm": receta.get("generada_por_llm", False),
                "tiempo_preparacion": receta.get("tiempo_preparacion", "30 minutos"),
                "porciones": receta.get("porciones", 4),
                "beneficios_nutricionales": receta.get("beneficios_nutricionales", "")
            }
            recetas_convertidas.append(receta_convertida)

        return recetas_convertidas

    def _filtrar_recetas_inteligente_mejorado(self, recetas: List[dict], condicion: str, calorias_objetivo: int) -> List[dict]:
        """Filtrado inteligente mejorado que considera el score del RAG y recetas generadas por LLM."""

        def calcular_score_total(receta):
            score = 0

            # Score del sistema RAG (si está disponible)
            score_rag = receta.get("relevance_score", 0)
            score += score_rag * 2  # Dar más peso al score del RAG

            # Bonus para recetas generadas por LLM (están hechas específicamente)
            if receta.get("generada_por_llm", False):
                score += 15.0  # Bonus alto para recetas LLM

            # Score por condición específica (lógica original)
            if condicion == "anemia":
                hierro = receta.get("hierro_mg", 0)
                if hierro >= 5:
                    score += 10
                elif hierro >= 3:
                    score += 7
                elif hierro >= 1:
                    score += 3
            elif condicion == "sobrepeso":
                calorias = receta.get("energia_kcal", 0)
                if calorias <= 400:
                    score += 10
                elif calorias <= 600:
                    score += 7
                elif calorias <= 800:
                    score += 3

            # Score por alimentos nutritivos identificados
            num_alimentos = receta.get("num_alimentos_nutritivos", 0)
            score += num_alimentos * 1.5

            # Score por variedad nutricional
            proteinas = receta.get("proteinas_g", 0)
            if proteinas >= 20:
                score += 5
            elif proteinas >= 10:
                score += 3

            return score

        # Calcular scores y ordenar
        recetas_con_score = [(r, calcular_score_total(r)) for r in recetas]
        recetas_con_score.sort(key=lambda x: x[1], reverse=True)

        # Filtro principal - ser más flexible con recetas LLM
        recetas_filtradas = []

        for receta, score in recetas_con_score:
            # Si es generada por LLM, es automáticamente válida
            if receta.get("generada_por_llm", False):
                recetas_filtradas.append(receta)
            # Si no, aplicar filtros tradicionales
            elif condicion == "anemia":
                if receta.get("hierro_mg", 0) >= 1.0 or score >= 5.0:
                    recetas_filtradas.append(receta)
            elif condicion == "sobrepeso":
                if receta.get("energia_kcal", 0) <= 800 or score >= 5.0:
                    recetas_filtradas.append(receta)
            else:
                recetas_filtradas.append(receta)

        # Asegurar mínimo de recetas
        if len(recetas_filtradas) < 4:
            logger.warning(f"Pocas recetas filtradas para {condicion}, tomando top por score")
            recetas_filtradas = [r for r, s in recetas_con_score[:8]]

        return recetas_filtradas[:8]  # Top 8 recetas

    def _distribuir_menu_por_tiempo(self, recetas: List[dict], distribucion: dict) -> dict:
        """Distribuye las recetas por tiempo de comida según sus calorías."""
        menu_distribuido = {
            "desayuno": [],
            "almuerzo": [],
            "cena": [],
            "refrigerio": []
        }

        # Clasificar recetas por calorías
        recetas_ligeras = [r for r in recetas if r.get("energia_kcal", 0) <= 300]
        recetas_medianas = [r for r in recetas if 300 < r.get("energia_kcal", 0) <= 600]
        recetas_completas = [r for r in recetas if r.get("energia_kcal", 0) > 600]

        # Asignar por tiempo de comida
        if recetas_ligeras:
            menu_distribuido["refrigerio"] = [recetas_ligeras[0]["nombre"]]
            if len(recetas_ligeras) > 1:
                menu_distribuido["desayuno"] = [recetas_ligeras[1]["nombre"]]

        if recetas_completas:
            menu_distribuido["almuerzo"] = [recetas_completas[0]["nombre"]]
            if len(recetas_completas) > 1:
                menu_distribuido["cena"] = [recetas_completas[1]["nombre"]]

        if recetas_medianas:
            if not menu_distribuido["desayuno"]:
                menu_distribuido["desayuno"] = [recetas_medianas[0]["nombre"]]
            elif not menu_distribuido["cena"]:
                menu_distribuido["cena"] = [recetas_medianas[0]["nombre"]]

        return menu_distribuido

    def generar_plan(self, perfil: dict) -> dict:
        """Genera un plan nutricional personalizado usando RAG mejorado con LLM fallback."""

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

        logger.info(f"Generando plan para: {edad} años, {sexo} ({sexo_original}), condición: {condicion_principal}")

        try:
            # 3. Cálculos nutricionales básicos
            get = calcular_get(edad, sexo)
            distribucion = distribuir_calorias(get)

            # 4. Obtener recomendaciones usando sistema RAG + LLM fallback
            recomendacion_rag = self._obtener_recomendaciones_rag_con_llm_fallback(condicion_principal)

            if recomendacion_rag:
                # Usar recomendaciones del sistema híbrido RAG + LLM
                logger.info("🎯 Usando recomendaciones del sistema RAG + LLM")

                recetas_rag = recomendacion_rag["recetas_recomendadas"]["lista"]
                alimentos_rag = recomendacion_rag["alimentos_recomendados"]["lista"]

                # Convertir formato para compatibilidad
                recetas_encontradas = self._convertir_recetas_rag_a_formato_compatible(recetas_rag)
                alimentos_recomendados = [a["nombre"] for a in alimentos_rag]

                # Información adicional del análisis RAG
                analisis_nutricional = recomendacion_rag.get("analisis_nutricional", {})
                coincidencias_alimentos = recomendacion_rag.get("match_alimentos_recetas", {})
                recetas_generadas_llm = recomendacion_rag.get("recetas_generadas_por_llm", False)

                logger.info(f"✅ Sistema híbrido proporcionó {len(recetas_encontradas)} recetas y {len(alimentos_recomendados)} alimentos")
                if recetas_generadas_llm:
                    logger.info("🤖 Incluye recetas generadas por LLM")

            else:
                # Fallback completo a métodos tradicionales
                logger.warning("⚠️ Usando métodos fallback completos")
                recetas_encontradas = self._buscar_recetas_fallback(condiciones)
                alimentos_recomendados = self._obtener_alimentos_fallback(condiciones)
                analisis_nutricional = {}
                coincidencias_alimentos = {}
                recetas_generadas_llm = False

            # Verificar que tenemos recetas
            if not recetas_encontradas:
                return {
                    "error": "No se pudieron generar recetas para la condición especificada. Contacte soporte técnico."
                }

            # 5. Filtrado inteligente con el nuevo sistema
            recetas_filtradas = self._filtrar_recetas_inteligente_mejorado(
                recetas_encontradas,
                condicion_principal,
                get
            )

            # 6. Distribuir menú por horarios
            menu_distribuido = self._distribuir_menu_por_tiempo(recetas_filtradas, distribucion)

            # 7. Preparar datos para el LLM
            nombres_recetas = [r["nombre"] for r in recetas_filtradas[:4]]
            condiciones_str = ", ".join(condiciones)

            # Crear perfil enriquecido con información nutricional
            perfil_str = (
                f"Adolescente de {edad} años, sexo {sexo_original}, peso {peso}, altura {altura}. "
                f"Diagnóstico: {condiciones_str}. "
                f"Requerimiento energético: {get} kcal/día. "
                f"Alimentos clave recomendados: {', '.join(alimentos_recomendados[:5])}. "
            )

            # Agregar información del análisis nutricional si está disponible
            if analisis_nutricional:
                hierro_promedio = analisis_nutricional.get("promedio_hierro_mg", 0)
                energia_promedio = analisis_nutricional.get("promedio_energia_kcal", 0)
                perfil_str += f"Los alimentos recomendados aportan en promedio {hierro_promedio} mg de hierro y {energia_promedio} kcal por 100g."

            # 8. Generar explicación con LLM
            try:
                # Incluir información sobre recetas generadas por IA
                recetas_info = ", ".join(nombres_recetas)
                if recetas_generadas_llm:
                    recetas_info += " (algunas recetas fueron creadas específicamente para este perfil nutricional)"

                explicacion = generar_explicacion(
                    perfil_str,
                    recetas_info,
                    max_tokens=400
                )
                consejos = generar_consejos_nutricionales(perfil_str, max_tokens=250)
            except Exception as e:
                logger.warning(f"Error generando explicación con LLM: {e}")
                explicacion = self._generar_explicacion_fallback(condicion_principal, nombres_recetas)
                consejos = self._generar_consejos_fallback(condicion_principal)

            # 9. Preparar respuesta final enriquecida
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

                # Información enriquecida del sistema RAG
                "alimentos_tabla_nutricional": [
                    {
                        "nombre": a["nombre"],
                        "hierro_mg": a.get("hierro_mg", 0),
                        "energia_kcal": a.get("energia_kcal", 0),
                        "proteinas_g": a.get("proteinas_g", 0)
                    }
                    for a in (recomendacion_rag.get("alimentos_recomendados", {}).get("lista", []) if recomendacion_rag else [])
                ][:5],

                "analisis_nutricional_detallado": analisis_nutricional,
                "coincidencias_ingredientes": coincidencias_alimentos,

                "recetas_detalladas": [
                    {
                        "nombre": r["nombre"],
                        "energia_kcal": r.get("energia_kcal", 0),
                        "hierro_mg": r.get("hierro_mg", 0),
                        "proteinas_g": r.get("proteinas_g", 0),
                        "alimentos_identificados": r.get("alimentos_identificados", ""),
                        "score_relevancia": r.get("relevance_score", 0),
                        "ingredientes": r.get("ingredientes", "")[:300] + "..." if len(r.get("ingredientes", "")) > 300 else r.get("ingredientes", ""),
                        "preparacion": r.get("preparacion", "")[:300] + "..." if len(r.get("preparacion", "")) > 300 else r.get("preparacion", ""),
                        "tiempo_preparacion": r.get("tiempo_preparacion", "30 minutos"),
                        "porciones": r.get("porciones", 4),
                        "beneficios_nutricionales": r.get("beneficios_nutricionales", ""),
                        "generada_por_ia": r.get("generada_por_llm", False)
                    }
                    for r in recetas_filtradas[:4]
                ],

                # Métricas del sistema
                "metricas_sistema": {
                    "total_recetas_analizadas": len(recetas_encontradas),
                    "recetas_seleccionadas": len(recetas_filtradas),
                    "total_alimentos_recomendados": len(alimentos_recomendados),
                    "sistema_rag_usado": recomendacion_rag is not None,
                    "recetas_generadas_por_ia": recetas_generadas_llm,
                    "porcentaje_coincidencia_ingredientes": coincidencias_alimentos.get("porcentaje_coincidencia", 0) if coincidencias_alimentos else 0,
                    "recetas_ia_count": len([r for r in recetas_filtradas if r.get("generada_por_llm", False)])
                }
            }

            logger.info(f"✅ Plan generado exitosamente: {len(nombres_recetas)} recetas, {len(alimentos_recomendados)} alimentos")
            if recetas_generadas_llm:
                logger.info(f"🤖 Incluye {plan_final['metricas_sistema']['recetas_ia_count']} recetas generadas por IA")

            return plan_final

        except Exception as e:
            logger.error(f"❌ Error generando plan: {e}")
            return {
                "error": f"Error interno al generar el plan: {str(e)}",
                "perfil_recibido": perfil
            }

    def _buscar_recetas_fallback(self, condiciones: List[str]) -> List[Dict]:
        """Método fallback para buscar recetas si el sistema RAG falla completamente."""
        # En caso extremo, usar el generador LLM directamente
        condicion_principal = condiciones[0].lower() if condiciones else "general"

        # Alimentos básicos según condición
        alimentos_basicos = {
            "anemia": ["pescado", "quinua", "lentejas", "espinaca", "hígado"],
            "sobrepeso": ["verduras", "frutas", "pescado magro", "cereales integrales"],
            "general": ["pescado", "quinua", "verduras", "frutas", "legumbres"]
        }

        alimentos_para_condicion = alimentos_basicos.get(condicion_principal, alimentos_basicos["general"])

        try:
            from .llm_generator import generar_recetas_peruanas_con_llm
            recetas_llm = generar_recetas_peruanas_con_llm(alimentos_para_condicion, condicion_principal, 4)
            if recetas_llm:
                logger.info("✅ Fallback completo: usando solo recetas LLM")
                return recetas_llm
        except Exception as e:
            logger.warning(f"⚠️ Error en fallback LLM: {e}")

        # Fallback final: recetas hardcodeadas
        return self._recetas_hardcoded_fallback(condicion_principal)

    def _recetas_hardcoded_fallback(self, condicion: str) -> List[Dict]:
        """Último fallback: recetas básicas hardcodeadas."""
        recetas_basicas = [
            {
                "nombre": "Pescado con Quinua Tradicional",
                "energia_kcal": 420,
                "proteinas_g": 25,
                "hierro_mg": 3.5,
                "ingredientes": "pescado fresco, quinua, verduras, especias peruanas",
                "preparacion": "Cocinar quinua, preparar pescado con especias tradicionales, servir con verduras",
                "generada_por_llm": False
            },
            {
                "nombre": "Lentejas Guisadas Peruanas",
                "energia_kcal": 280,
                "proteinas_g": 18,
                "hierro_mg": 4.2,
                "ingredientes": "lentejas, verduras, ají amarillo, especias",
                "preparacion": "Guisar lentejas con sofrito peruano tradicional",
                "generada_por_llm": False
            }
        ]

        return recetas_basicas

    def _obtener_alimentos_fallback(self, condiciones: List[str]) -> List[str]:
        """Método fallback para obtener alimentos si el sistema RAG falla."""
        condicion_principal = condiciones[0].lower() if condiciones else "general"

        if condicion_principal == "anemia":
            return ["pescado", "quinua", "lentejas", "espinaca", "hígado"]
        elif condicion_principal == "sobrepeso":
            return ["verduras", "frutas", "pescado magro", "cereales integrales"]
        else:
            return ["pescado", "quinua", "verduras", "frutas", "legumbres"]

    def _generar_explicacion_fallback(self, condicion: str, recetas: List[str]) -> str:
        """Explicación de fallback sin LLM."""
        if condicion == "anemia":
            return (
                f"Este plan nutricional prioriza alimentos ricos en hierro como {', '.join(recetas[:2])}. "
                "El hierro presente en pescados y legumbres ayuda a mejorar los niveles de hemoglobina. "
                "Se recomienda combinar con alimentos ricos en vitamina C para mejorar la absorción."
            )
        elif condicion == "sobrepeso":
            return (
                f"Las recetas seleccionadas ({', '.join(recetas[:2])}) son opciones nutritivas y balanceadas "
                "con control calórico. Priorizan proteínas magras, verduras y preparaciones saludables "
                "que ayudan a mantener un peso adecuado mientras aportan nutrientes esenciales."
            )
        else:
            return (
                f"Este plan incluye recetas variadas ({', '.join(recetas[:2])}) que aportan "
                "los nutrientes necesarios para el crecimiento y desarrollo durante la adolescencia, "
                "considerando las preferencias de la cocina peruana."
            )

    def _generar_consejos_fallback(self, condicion: str) -> str:
        """Consejos de fallback sin LLM."""
        consejos_base = [
            "Mantén horarios regulares de comida (desayuno, almuerzo, cena y un refrigerio).",
            "Consume al menos 8 vasos de agua al día.",
            "Incluye variedad de colores en tus comidas (frutas y verduras).",
            "Evita el exceso de azúcar y alimentos ultraprocesados."
        ]

        if condicion == "anemia":
            consejos_base.append("Combina alimentos ricos en hierro con cítricos para mejorar la absorción.")
        elif condicion == "sobrepeso":
            consejos_base.append("Practica actividad física regular y controla las porciones.")

        return " ".join(consejos_base)


# Función de conveniencia para mantener compatibilidad
def generar_plan(perfil: dict) -> dict:
    """Función wrapper para mantener compatibilidad con el código existente."""
    planner = NutritionalPlanner()
    return planner.generar_plan(perfil)
