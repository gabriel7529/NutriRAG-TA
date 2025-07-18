import os
from openai import OpenAI
from typing import Optional, List, Dict
import logging
import json
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración OpenAI desde variables de entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "400"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

logger = logging.getLogger(__name__)

# Inicializar cliente de forma segura
def get_openai_client() -> Optional[OpenAI]:
    """Inicializa el cliente OpenAI de forma segura."""
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY no configurada")
        return None

    try:
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.error(f"Error inicializando cliente OpenAI: {e}")
        return None

def generar_recetas_peruanas_con_llm(alimentos_recomendados: List[str], condicion: str, num_recetas: int = 4) -> List[Dict]:
    """
    Genera recetas peruanas usando LLM cuando no se encuentran en el vectorstore.

    Args:
        alimentos_recomendados: Lista de alimentos que deben incluirse
        condicion: Condición nutricional (anemia, sobrepeso, etc.)
        num_recetas: Número de recetas a generar

    Returns:
        Lista de diccionarios con recetas generadas
    """
    client = get_openai_client()
    if not client:
        logger.warning("⚠️ OpenAI no disponible, usando recetas predefinidas")
        return generar_recetas_fallback(alimentos_recomendados, condicion, num_recetas)

    # Preparar prompt específico para la condición
    condicion_info = {
        "anemia": {
            "objetivo": "aumentar absorción de hierro y hemoglobina",
            "nutrientes_clave": "hierro, vitamina C, proteínas",
            "preparaciones": "cocido, salteado, estofado (para mejor absorción del hierro)"
        },
        "sobrepeso": {
            "objetivo": "control calórico manteniendo nutrientes esenciales",
            "nutrientes_clave": "proteínas magras, fibra, vitaminas",
            "preparaciones": "al vapor, a la plancha, hervido, horneado (bajo en grasas)"
        },
        "general": {
            "objetivo": "nutrición balanceada para crecimiento",
            "nutrientes_clave": "proteínas, calcio, hierro, vitaminas",
            "preparaciones": "variadas técnicas culinarias peruanas"
        }
    }

    info_condicion = condicion_info.get(condicion.lower(), condicion_info["general"])
    alimentos_texto = ", ".join(alimentos_recomendados[:6])  # Máximo 6 alimentos

    prompt = f"""
Eres un chef nutricionista especializado en cocina peruana tradicional. Necesitas crear {num_recetas} recetas AUTÉNTICAS peruanas para un adolescente con {condicion}.

ALIMENTOS QUE DEBES INCLUIR: {alimentos_texto}

OBJETIVO NUTRICIONAL: {info_condicion['objetivo']}
NUTRIENTES CLAVE: {info_condicion['nutrientes_clave']}
PREPARACIONES RECOMENDADAS: {info_condicion['preparaciones']}

REQUISITOS:
1. Recetas 100% peruanas tradicionales
2. Usar al menos 2-3 de los alimentos recomendados por receta
3. Ingredientes fáciles de conseguir en Perú
4. Porciones apropiadas para adolescentes
5. Técnicas de cocina que preserven/potencien los nutrientes

FORMATO DE RESPUESTA (JSON válido):
{{
  "recetas": [
    {{
      "nombre": "Nombre de la receta peruana",
      "ingredientes": "Lista detallada de ingredientes con cantidades",
      "preparacion": "Pasos de preparación claros y específicos",
      "tiempo_preparacion": "X minutos",
      "porciones": 4,
      "alimentos_nutritivos_incluidos": ["alimento1", "alimento2"],
      "beneficios_nutricionales": "Por qué es buena para {condicion}",
      "valores_estimados": {{
        "energia_kcal": 450,
        "proteinas_g": 25,
        "hierro_mg": 4.5
      }}
    }}
  ]
}}

Genera recetas como: Pescado a la chorrillana, Quinotto de verduras, Estofado de lentejas con acelgas, Saltado de quinua, etc.
"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un chef nutricionista peruano experto. Creas recetas tradicionales "
                        "del Perú que son nutritivas y apropiadas para condiciones específicas. "
                        "Siempre respondes en formato JSON válido."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1500,  # Más tokens para recetas completas
            temperature=0.8,  # Más creatividad para las recetas
            top_p=0.9
        )

        contenido = response.choices[0].message.content.strip()

        # Limpiar el JSON si viene con ```json
        if contenido.startswith("```json"):
            contenido = contenido.replace("```json", "").replace("```", "").strip()

        # Parsear JSON
        try:
            recetas_json = json.loads(contenido)
            recetas_generadas = recetas_json.get("recetas", [])

            # Convertir al formato esperado por el sistema
            recetas_convertidas = []
            for receta in recetas_generadas:
                receta_convertida = {
                    "nombre": receta.get("nombre", "Receta Peruana"),
                    "ingredientes": receta.get("ingredientes", ""),
                    "preparacion": receta.get("preparacion", ""),
                    "energia_kcal": receta.get("valores_estimados", {}).get("energia_kcal", 400),
                    "proteinas_g": receta.get("valores_estimados", {}).get("proteinas_g", 20),
                    "hierro_mg": receta.get("valores_estimados", {}).get("hierro_mg", 3.0),
                    "condicion_principal": condicion,
                    "alimentos_identificados": ", ".join(receta.get("alimentos_nutritivos_incluidos", [])),
                    "relevance_score": 9.0,  # Score alto porque están hechas específicamente
                    "tiempo_preparacion": receta.get("tiempo_preparacion", "30 minutos"),
                    "porciones": receta.get("porciones", 4),
                    "beneficios_nutricionales": receta.get("beneficios_nutricionales", ""),
                    "generada_por_llm": True
                }
                recetas_convertidas.append(receta_convertida)

            logger.info(f"✅ Generadas {len(recetas_convertidas)} recetas peruanas con LLM")
            return recetas_convertidas

        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parseando JSON de recetas: {e}")
            logger.debug(f"Contenido recibido: {contenido[:200]}...")
            return generar_recetas_fallback(alimentos_recomendados, condicion, num_recetas)

    except Exception as e:
        logger.error(f"❌ Error generando recetas con LLM: {e}")
        return generar_recetas_fallback(alimentos_recomendados, condicion, num_recetas)

def generar_recetas_fallback(alimentos_recomendados: List[str], condicion: str, num_recetas: int = 4) -> List[Dict]:
    """Genera recetas de fallback cuando el LLM no está disponible."""

    # Plantillas de recetas peruanas por condición
    plantillas_recetas = {
        "anemia": [
            {
                "nombre": "Pescado a la Chorrillana con Quinua",
                "ingredientes": "500g pescado fresco, 1 taza quinua, 2 tomates, 1 cebolla, ají amarillo, ajos, culantro",
                "preparacion": "1. Cocinar quinua. 2. Freír pescado. 3. Preparar chorrillana con tomate, cebolla y ají. 4. Servir pescado sobre quinua con chorrillana.",
                "energia_kcal": 480,
                "proteinas_g": 28,
                "hierro_mg": 5.2
            },
            {
                "nombre": "Estofado de Lentejas con Acelgas",
                "ingredientes": "1 taza lentejas, 3 tazas acelgas, 1 cebolla, tomates, ají panca, comino, ajos",
                "preparacion": "1. Cocinar lentejas. 2. Preparar sofrito con cebolla, ajo y ají. 3. Agregar acelgas y lentejas. 4. Cocinar 15 minutos.",
                "energia_kcal": 320,
                "proteinas_g": 18,
                "hierro_mg": 4.8
            },
            {
                "nombre": "Saltado de Quinua con Hígado",
                "ingredientes": "300g hígado de pollo, 1 taza quinua cocida, cebolla, tomate, ají amarillo, sillao, vinagre",
                "preparacion": "1. Cortar hígado en tiras. 2. Saltear con cebolla y ají. 3. Agregar tomate y quinua. 4. Sazonar con sillao.",
                "energia_kcal": 420,
                "proteinas_g": 25,
                "hierro_mg": 8.5
            },
            {
                "nombre": "Sopa de Quinua con Sangrecita",
                "ingredientes": "1 taza quinua, 200g sangrecita, verduras mixtas, cebolla, ajos, hierbas aromáticas",
                "preparacion": "1. Dorar sangrecita. 2. Preparar caldo con verduras. 3. Agregar quinua y cocinar. 4. Servir caliente.",
                "energia_kcal": 380,
                "proteinas_g": 22,
                "hierro_mg": 12.0
            }
        ],
        "sobrepeso": [
            {
                "nombre": "Pescado al Vapor con Verduras",
                "ingredientes": "400g pescado blanco, brócoli, zanahoria, zapallito, hierbas aromáticas, limón",
                "preparacion": "1. Cocinar pescado al vapor. 2. Cocer verduras al vapor. 3. Servir con limón y hierbas.",
                "energia_kcal": 280,
                "proteinas_g": 25,
                "hierro_mg": 2.1
            },
            {
                "nombre": "Ensalada de Quinua con Verduras",
                "ingredientes": "1 taza quinua cocida, tomate, pepino, lechuga, cebolla morada, limón, aceite de oliva",
                "preparacion": "1. Mezclar quinua fría con verduras picadas. 2. Aliñar con limón y aceite. 3. Servir fresco.",
                "energia_kcal": 240,
                "proteinas_g": 12,
                "hierro_mg": 2.8
            },
            {
                "nombre": "Caldo de Verduras con Pollo",
                "ingredientes": "200g pechuga de pollo, acelgas, espinacas, zanahoria, apio, cebolla, hierbas",
                "preparacion": "1. Cocinar pollo en caldo de verduras. 2. Agregar verduras verdes. 3. Servir caliente.",
                "energia_kcal": 220,
                "proteinas_g": 20,
                "hierro_mg": 1.8
            },
            {
                "nombre": "Tortilla de Verduras Peruana",
                "ingredientes": "3 huevos, espinacas, tomate, cebolla, ají amarillo, queso fresco, hierbas",
                "preparacion": "1. Batir huevos con verduras picadas. 2. Cocinar en sartén antiadherente. 3. Servir con ensalada.",
                "energia_kcal": 260,
                "proteinas_g": 18,
                "hierro_mg": 2.5
            }
        ]
    }

    # Seleccionar plantillas según condición
    recetas_base = plantillas_recetas.get(condicion.lower(), plantillas_recetas["anemia"])

    # Adaptar recetas para incluir alimentos recomendados
    recetas_adaptadas = []
    for i, receta_base in enumerate(recetas_base[:num_recetas]):
        receta_adaptada = receta_base.copy()

        # Intentar incluir alimentos recomendados en la receta
        alimentos_incluidos = []
        for alimento in alimentos_recomendados[:3]:  # Máximo 3 por receta
            if any(keyword in alimento.lower() for keyword in ['pescado', 'pollo', 'carne']):
                alimentos_incluidos.append(alimento)
            elif any(keyword in alimento.lower() for keyword in ['quinua', 'arroz', 'avena']):
                alimentos_incluidos.append(alimento)
            elif any(keyword in alimento.lower() for keyword in ['verdura', 'vegetal', 'lechuga']):
                alimentos_incluidos.append(alimento)

        # Formatear según el formato esperado
        receta_adaptada.update({
            "condicion_principal": condicion,
            "alimentos_identificados": ", ".join(alimentos_incluidos),
            "relevance_score": 7.0,  # Score medio para fallback
            "tiempo_preparacion": "25-30 minutos",
            "porciones": 4,
            "beneficios_nutricionales": f"Receta tradicional peruana adaptada para {condicion}",
            "generada_por_llm": False
        })

        recetas_adaptadas.append(receta_adaptada)

    logger.info(f"✅ Generadas {len(recetas_adaptadas)} recetas fallback para {condicion}")
    return recetas_adaptadas

def generar_explicacion_segura(perfil: str, recetas: str, max_tokens: int = None) -> str:
    """Genera explicación nutricional con manejo de errores robusto."""
    client = get_openai_client()
    if not client:
        return generar_explicacion_fallback(perfil, recetas)

    max_tokens = max_tokens or OPENAI_MAX_TOKENS

    prompt = (
        f"Eres un nutricionista especializado en adolescentes peruanos. "
        f"Perfil del paciente: {perfil}. "
        f"Recetas del plan: {recetas}. "
        f"Explica profesionalmente por qué este plan es adecuado, "
        f"considerando la cultura alimentaria peruana y las necesidades específicas. "
        f"Máximo 300 palabras, tono profesional pero amigable."
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un nutricionista especializado en adolescentes peruanos. "
                        "Proporciona explicaciones científicas pero comprensibles, "
                        "considerando la cultura alimentaria local."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=OPENAI_TEMPERATURE,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )

        respuesta = response.choices[0].message.content.strip()
        logger.info("✅ Explicación generada con OpenAI")
        return respuesta

    except Exception as e:
        logger.warning(f"⚠️ Error con OpenAI, usando fallback: {e}")
        return generar_explicacion_fallback(perfil, recetas)

def generar_consejos_seguros(perfil: str, max_tokens: int = None) -> str:
    """Genera consejos nutricionales con fallback robusto."""
    client = get_openai_client()
    if not client:
        return generar_consejos_fallback(perfil)

    max_tokens = max_tokens or 200

    prompt = (
        f"Basándote en: {perfil}, proporciona 4 consejos nutricionales "
        f"específicos y prácticos para este adolescente peruano. "
        f"Incluye horarios, hidratación y hábitos culturalmente apropiados."
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un nutricionista que da consejos prácticos para "
                        "adolescentes peruanos, considerando su cultura y estilo de vida."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9
        )

        respuesta = response.choices[0].message.content.strip()
        logger.info("✅ Consejos generados con OpenAI")
        return respuesta

    except Exception as e:
        logger.warning(f"⚠️ Error con OpenAI, usando fallback: {e}")
        return generar_consejos_fallback(perfil)

def generar_explicacion_fallback(perfil: str, recetas: str) -> str:
    """Explicación de respaldo sin IA."""
    return (
        f"Plan nutricional personalizado basado en recetas tradicionales peruanas: {recetas}. "
        f"Este plan ha sido diseñado considerando el perfil nutricional específico ({perfil}) "
        f"y las recomendaciones nutricionales para adolescentes. Las recetas seleccionadas "
        f"aportan los nutrientes esenciales para el crecimiento y desarrollo, priorizando "
        f"ingredientes locales y preparaciones familiares de la cocina peruana."
    )

def generar_consejos_fallback(perfil: str) -> str:
    """Consejos de respaldo sin IA."""
    return (
        "Consejos importantes: 1) Mantén horarios regulares de comida, "
        "2) Consume al menos 8 vasos de agua diarios, "
        "3) Incluye frutas y verduras de colores variados, "
        "4) Evita bebidas azucaradas y comida chatarra excesiva."
    )

def verificar_conexion_openai() -> bool:
    """Verifica si la conexión con OpenAI funciona."""
    client = get_openai_client()
    if not client:
        return False

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        logger.info("✅ Conexión OpenAI verificada")
        return True
    except Exception as e:
        logger.error(f"❌ Error verificando OpenAI: {e}")
        return False

# Funciones de compatibilidad para el sistema existente
def generar_explicacion(perfil: str, recetas: str, max_length: int = 250) -> str:
    """Función de compatibilidad que llama a la versión segura."""
    return generar_explicacion_segura(perfil, recetas, max_length)

def generar_consejos_nutricionales(perfil: str) -> str:
    """Función de compatibilidad que llama a la versión segura."""
    return generar_consejos_seguros(perfil)
