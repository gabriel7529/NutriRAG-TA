import streamlit as st
import requests
import json
from datetime import datetime
import time

# Configuración de la página
st.set_page_config(
    page_title="NutriPlan AI - Planificador Nutricional",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para un diseño moderno
st.markdown("""
<style>
    /* Importar fuentes de Google */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    /* Variables CSS */
    :root {
        --primary-color: #2E7D32;
        --secondary-color: #4CAF50;
        --accent-color: #81C784;
        --background-color: #F8F9FA;
        --text-color: #2C3E50;
        --card-background: #FFFFFF;
        --shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        --border-radius: 12px;
    }

    /* Estilos generales */
    .main {
        font-family: 'Poppins', sans-serif;
    }

    /* Header personalizado */
    .header-container {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        padding: 2rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        box-shadow: var(--shadow);
        color: white;
        text-align: center;
    }

    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }

    .header-subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
    }

    /* Formulario de perfil */
    .profile-card {
        background: var(--card-background);
        padding: 2rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        margin-bottom: 2rem;
    }

    .profile-title {
        color: var(--primary-color);
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--accent-color);
        padding-bottom: 0.5rem;
    }

    /* Contenedor de resultados */
    .result-container {
        background: var(--card-background);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        padding: 2rem;
        margin-bottom: 2rem;
    }

    .section-title {
        color: var(--primary-color);
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid var(--accent-color);
        padding-left: 1rem;
    }

    .recipe-card {
        background: linear-gradient(135deg, #E8F5E8, #F1F8E9);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid var(--secondary-color);
    }

    .nutrition-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    .nutrition-item {
        background: var(--card-background);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        text-align: center;
    }

    /* Botones personalizados */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.4);
    }

    /* Métricas */
    .metric-card {
        background: var(--card-background);
        padding: 1.5rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        text-align: center;
        margin-bottom: 1rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.9rem;
        color: var(--text-color);
        font-weight: 500;
    }

    /* Alertas personalizadas */
    .custom-alert {
        padding: 1rem;
        border-radius: var(--border-radius);
        margin: 1rem 0;
        border-left: 4px solid;
    }

    .alert-success {
        background: #d4edda;
        border-color: #28a745;
        color: #155724;
    }

    .alert-warning {
        background: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }

    .alert-info {
        background: #cce7ff;
        border-color: #007bff;
        color: #004085;
    }

    /* Animaciones */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }

        .profile-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Configuración de la API
API_BASE_URL = "http://127.0.0.1:8000"

# Inicializar el estado de la sesión
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}
if 'profile_created' not in st.session_state:
    st.session_state.profile_created = False
if 'nutrition_plan' not in st.session_state:
    st.session_state.nutrition_plan = None

# Header principal
st.markdown("""
<div class="header-container fade-in">
    <div class="header-title">🥗 NutriPlan AI</div>
    <div class="header-subtitle">Planificador Nutricional Personalizado para Adolescentes Peruanos</div>
</div>
""", unsafe_allow_html=True)

# Sidebar para configuración del perfil
with st.sidebar:
    st.markdown("""
    <div class="profile-title">👤 Perfil del Usuario</div>
    """, unsafe_allow_html=True)

    # Formulario de perfil
    with st.form("profile_form"):
        st.markdown("### Información Personal")

        nombre = st.text_input("Nombre", placeholder="Ingresa tu nombre")
        edad = st.number_input("Edad", min_value=12, max_value=17, value=15)
        sexo = st.selectbox("Sexo", ["Femenino", "Masculino"])

        st.markdown("### Información Física")
        peso = st.number_input("Peso (kg)", min_value=30.0, max_value=120.0, value=60.0, step=0.1)
        altura = st.number_input("Altura (cm)", min_value=130, max_value=200, value=160)

        st.markdown("### Condiciones de Salud")
        condiciones = st.multiselect(
            "Condiciones presentes:",
            ["anemia", "sobrepeso", "obesidad"],
            default=[]
        )

        st.markdown("### Estilo de Vida")
        actividad_fisica = st.selectbox(
            "Nivel de actividad física:",
            ["sedentaria", "ligera", "moderada", "intensa"],
            index=0
        )

        st.markdown("### Preferencias Alimentarias")
        preferencias_disponibles = [
            "pescado", "quinua", "verduras", "frutas", "legumbres",
            "arroz", "pollo", "huevos", "lácteos", "cereales"
        ]
        preferencias = st.multiselect(
            "Alimentos preferidos:",
            preferencias_disponibles,
            default=["pescado", "quinua", "verduras"]
        )

        alergias = st.text_input("Alergias", placeholder="Ej: ninguna, mariscos, etc.", value="ninguna")

        submitted = st.form_submit_button("💾 Crear Plan Nutricional", use_container_width=True)

        if submitted and nombre:
            st.session_state.user_profile = {
                "nombre": nombre,
                "edad": int(edad),
                "sexo": sexo,
                "peso": float(peso),
                "altura": int(altura),
                "condiciones": condiciones if condiciones else [],
                "actividad_fisica": actividad_fisica,
                "preferencias": preferencias,
                "alergias": alergias
            }
            st.session_state.profile_created = True

            # Llamar a la API para generar el plan
            with st.spinner("🤖 Generando tu plan nutricional personalizado..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/generate_plan",
                        json=st.session_state.user_profile,
                        timeout=120  # 2 minutos para procesos largos
                    )

                    if response.status_code == 200:
                        st.session_state.nutrition_plan = response.json()
                        st.success("✅ ¡Plan nutricional generado correctamente!")
                    else:
                        st.error(f"❌ Error del servidor: {response.status_code}")
                        st.error(f"Detalle: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("❌ No se pudo conectar con el servidor. Asegúrate de que la API esté ejecutándose.")
                except requests.exceptions.Timeout:
                    st.error("❌ La solicitud tardó demasiado tiempo. Intenta de nuevo.")
                except Exception as e:
                    st.error(f"❌ Error inesperado: {str(e)}")

# Área principal
if st.session_state.profile_created and st.session_state.nutrition_plan:

    # Mostrar métricas del usuario
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.user_profile['edad']}</div>
            <div class="metric-label">Años</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        imc = st.session_state.user_profile['peso'] / ((st.session_state.user_profile['altura']/100) ** 2)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{imc:.1f}</div>
            <div class="metric-label">IMC</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        calorias = st.session_state.nutrition_plan.get('requerimientos', {}).get('calorias_diarias', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{calorias}</div>
            <div class="metric-label">Calorías/día</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        hierro_promedio = st.session_state.nutrition_plan.get('analisis_nutricional_detallado', {}).get('promedio_hierro_mg', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{hierro_promedio:.1f}</div>
            <div class="metric-label">mg Hierro</div>
        </div>
        """, unsafe_allow_html=True)

    # Mostrar el plan nutricional
    st.markdown(f"""
    <div class="result-container fade-in">
        <h2 style="color: var(--primary-color); text-align: center;">
            🎯 Plan Nutricional para {st.session_state.user_profile['nombre']}
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # Distribución de calorías por horario
    st.markdown('<div class="section-title">📊 Distribución de Calorías Diarias</div>', unsafe_allow_html=True)

    distribucion = st.session_state.nutrition_plan.get('requerimientos', {}).get('distribucion_horaria', {})

    col1, col2, col3, col4 = st.columns(4)
    horarios = ['desayuno', 'almuerzo', 'cena', 'refrigerio']
    iconos = ['🌅', '☀️', '🌙', '🍎']

    for col, horario, icono in zip([col1, col2, col3, col4], horarios, iconos):
        with col:
            calorias = distribucion.get(horario, 0)
            st.markdown(f"""
            <div class="nutrition-item">
                <div style="font-size: 2rem;">{icono}</div>
                <div style="font-weight: 600; text-transform: capitalize;">{horario}</div>
                <div style="color: var(--primary-color); font-weight: 700;">{calorias} kcal</div>
            </div>
            """, unsafe_allow_html=True)

    # Menu sugerido
    st.markdown('<div class="section-title">🍽️ Menú Sugerido</div>', unsafe_allow_html=True)

    menu_por_horario = st.session_state.nutrition_plan.get('menu_por_horario', {})

    for horario in ['desayuno', 'almuerzo', 'cena', 'refrigerio']:
        platos = menu_por_horario.get(horario, [])
        if platos:
            st.markdown(f"**{horario.title()}:** {', '.join(platos)}")
        else:
            st.markdown(f"**{horario.title()}:** No especificado")

    # Recetas detalladas
    st.markdown('<div class="section-title">👨‍🍳 Recetas Detalladas</div>', unsafe_allow_html=True)

    recetas = st.session_state.nutrition_plan.get('recetas_detalladas', [])

    for receta in recetas:
        st.markdown(f"""
        <div class="recipe-card">
            <h3 style="color: var(--primary-color); margin-bottom: 1rem;">
                🥘 {receta.get('nombre', 'Sin nombre')}
            </h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
                <div><strong>⚡ Energía:</strong> {receta.get('energia_kcal', 0)} kcal</div>
                <div><strong>🥩 Proteínas:</strong> {receta.get('proteinas_g', 0)} g</div>
                <div><strong>🩸 Hierro:</strong> {receta.get('hierro_mg', 0)} mg</div>
                <div><strong>⏱️ Tiempo:</strong> {receta.get('tiempo_preparacion', 'No especificado')}</div>
                <div><strong>👥 Porciones:</strong> {receta.get('porciones', 0)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Mostrar ingredientes y preparación en expander
        with st.expander(f"Ver receta completa de {receta.get('nombre', 'Sin nombre')}"):
            st.markdown("**Ingredientes y Preparación:**")
            ingredientes = receta.get('ingredientes', 'No disponible')
            if len(ingredientes) > 200:
                ingredientes = ingredientes[:200] + "..."
            st.write(ingredientes)

            preparacion = receta.get('preparacion', 'No disponible')
            if len(preparacion) > 200:
                preparacion = preparacion[:200] + "..."
            st.markdown("**Pasos:**")
            st.write(preparacion)

    # Alimentos recomendados
    st.markdown('<div class="section-title">🥬 Alimentos Recomendados</div>', unsafe_allow_html=True)

    alimentos_recomendados = st.session_state.nutrition_plan.get('alimentos_recomendados', [])

    if alimentos_recomendados:
        cols = st.columns(3)
        for i, alimento in enumerate(alimentos_recomendados):
            with cols[i % 3]:
                st.markdown(f"• {alimento}")

    # Tabla nutricional
    st.markdown('<div class="section-title">📋 Información Nutricional Detallada</div>', unsafe_allow_html=True)

    tabla_nutricional = st.session_state.nutrition_plan.get('alimentos_tabla_nutricional', [])

    if tabla_nutricional:
        st.dataframe(
            tabla_nutricional,
            use_container_width=True,
            hide_index=True
        )

    # Explicación profesional
    st.markdown('<div class="section-title">👩‍⚕️ Recomendaciones Profesionales</div>', unsafe_allow_html=True)

    explicacion = st.session_state.nutrition_plan.get('explicacion_profesional', '')
    if explicacion:
        st.markdown(f"""
        <div class="custom-alert alert-info">
            {explicacion}
        </div>
        """, unsafe_allow_html=True)

    # Consejos adicionales
    consejos = st.session_state.nutrition_plan.get('consejos_adicionales', '')
    if consejos:
        st.markdown('<div class="section-title">💡 Consejos Adicionales</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="custom-alert alert-success">
            {consejos}
        </div>
        """, unsafe_allow_html=True)

    # Análisis de cumplimiento
    analisis = st.session_state.nutrition_plan.get('analisis_nutricional_detallado', {})
    cumplimiento = analisis.get('cumple_requerimientos', {})

    if cumplimiento:
        st.markdown('<div class="section-title">⚠️ Análisis de Requerimientos</div>', unsafe_allow_html=True)

        cumple = cumplimiento.get('cumple', False)
        observaciones = cumplimiento.get('observaciones', [])

        if cumple:
            st.markdown("""
            <div class="custom-alert alert-success">
                ✅ Tu plan nutricional cumple con los requerimientos necesarios.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="custom-alert alert-warning">
                ⚠️ Hay algunos aspectos a mejorar en tu alimentación:
            </div>
            """, unsafe_allow_html=True)

            for observacion in observaciones:
                st.markdown(f"• {observacion}")

    # Botón para generar nuevo plan
    if st.button("🔄 Generar Nuevo Plan", use_container_width=True):
        st.session_state.nutrition_plan = None
        st.session_state.profile_created = False
        st.rerun()

elif st.session_state.profile_created:
    # Si hay perfil pero no plan (error en API)
    st.markdown("""
    <div class="custom-alert alert-warning">
        ⚠️ Hubo un problema al generar tu plan nutricional.
        Por favor, intenta crear el plan nuevamente desde la barra lateral.
    </div>
    """, unsafe_allow_html=True)

else:
    # Mostrar mensaje de bienvenida si no hay perfil
    st.markdown("""
    <div class="profile-card fade-in">
        <div class="profile-title">🚀 ¡Bienvenido a NutriPlan AI!</div>
        <p style="font-size: 1.1rem; color: var(--text-color); line-height: 1.6;">
            Para comenzar, por favor completa tu perfil en la barra lateral izquierda.
            Esto nos permitirá generar recomendaciones nutricionales personalizadas
            específicamente para ti.
        </p>
        <p style="font-size: 1rem; color: var(--secondary-color); font-weight: 500;">
            📋 Información necesaria:
        </p>
        <ul style="color: var(--text-color); line-height: 1.8;">
            <li>Datos personales (nombre, edad, sexo)</li>
            <li>Medidas físicas (peso, altura)</li>
            <li>Condiciones de salud</li>
            <li>Nivel de actividad física</li>
            <li>Preferencias y restricciones alimentarias</li>
        </ul>

        <div style="margin-top: 2rem; padding: 1rem; background: #E3F2FD; border-radius: 8px; border-left: 4px solid #2196F3;">
            <h4 style="color: #1976D2; margin-bottom: 0.5rem;">🔧 Estado del Servidor</h4>
            <p style="margin: 0; color: #0D47A1;">
                Asegúrate de que tu API esté ejecutándose en:
                <code style="background: #BBDEFB; padding: 2px 6px; border-radius: 4px;">http://127.0.0.1:8000</code>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
---
<div style="text-align: center; color: var(--text-color); font-size: 0.9rem; margin-top: 2rem;">
    <p>💚 Desarrollado con amor para la salud de los adolescentes peruanos</p>
    <p>🏥 Basado en guías del Ministerio de Salud del Perú y la OMS</p>
    <p style="font-size: 0.8rem; opacity: 0.7;">
        © 2024 NutriPlan AI - Universidad Nacional de San Agustín
    </p>
</div>
""", unsafe_allow_html=True)
