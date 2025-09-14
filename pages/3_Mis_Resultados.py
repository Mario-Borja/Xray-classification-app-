# -----------------------------------------------------------------------------
# Archivo: pages/page_6.py
# Contenido: Vista de Resultados y Reportes para el Paciente (con Imágenes)
# -----------------------------------------------------------------------------
import streamlit as st
import pandas as pd # Importar pandas si se usa para fechas o dataframes
from PIL import Image # Importar Image de Pillow para manejar las imágenes
import io # Importar io si se necesitara para bytes (aunque aquí esperamos objetos PIL)

# --- Verificación de Autenticación ---
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("Por favor, inicia sesión para acceder a esta página.")
    st.stop()

# --- Configuración de Página ---
st.set_page_config(page_title="Mis Resultados Médicos", layout="wide")
st.markdown("# Página 6: Mis Resultados de Radiografía de Tórax 📄")
st.sidebar.header("Mis Resultados")

# --- Lógica para Mostrar Resultados ---

# Obtener el ID del paciente logueado (asumimos que es el username)
current_patient_id = st.session_state.get('username')

st.write(f"Bienvenido/a, **{current_patient_id}**. Aquí puedes ver tus resultados.")

# Simulación de Carga de Datos
report_data = st.session_state.get('report_data_to_save', None)
patient_has_results = False

# Variables para las imágenes, se llenarán desde report_data si existe
original_image_display = None
gradcam_image_display = None

# Verificar que el reporte sea para el paciente actual
if report_data and report_data.get('patient_id') == current_patient_id:
    st.success("Se encontró un reporte asociado a tu cuenta:")
    patient_has_results = True

    # Extraer la información del reporte
    date_generated = report_data.get('date_generated', 'No disponible')
    cnn_prediction = report_data.get('cnn_prediction', 'No disponible')
    cnn_probability = report_data.get('cnn_probability', 0.0)
    llm_report_text = report_data.get('llm_report_text', 'Reporte no disponible.')
    doctor_id = report_data.get('doctor_id', 'No especificado')

    # --- MODIFICACIÓN: Recuperar imágenes desde el diccionario report_data ---
    original_image_display = report_data.get('original_image_pil', None)
    gradcam_image_display = report_data.get('gradcam_image_pil', None)
    # --- FIN MODIFICACIÓN ---

    # Mostrar la información
    st.markdown("---")
    st.subheader(f"Reporte Generado el: {date_generated}")
    st.markdown(f"**Generado por Dr(a):** {doctor_id}")
    st.markdown("---")

    # Sección de Resultados de IA
    st.markdown("### Resultados del Análisis por Inteligencia Artificial (IA)")
    col1_ia, col2_ia = st.columns(2)
    with col1_ia:
        st.metric(label="Principal Patología Sugerida por IA", value=cnn_prediction)
    with col2_ia:
        st.metric(label="Confianza de la IA (Probabilidad)", value=f"{cnn_probability:.2%}")

    # Mostrar Imágenes
    st.markdown("---")
    st.markdown("### Imágenes Analizadas")
    
    col_img1, col_img2 = st.columns(2)

    with col_img1:
        if original_image_display:
            st.image(original_image_display, caption="Tu Radiografía Original", use_column_width=True)
        else:
            st.info("La imagen original no está disponible para visualización en este reporte.")

    with col_img2:
        if gradcam_image_display:
            st.image(gradcam_image_display, caption=f"Mapa de Calor (GradCAM) para '{cnn_prediction}'", use_column_width=True)
        else:
            st.info("El mapa de calor GradCAM no está disponible para visualización en este reporte.")

    # Sección del Reporte del Médico (asistido por LLM)
    st.markdown("---") # Separador antes del reporte
    st.markdown("### Reporte Radiológico (Generado con Asistencia de IA)")
    st.markdown(llm_report_text) # Usar st.markdown para formatear el texto del reporte


    st.markdown("---")
    st.info("""
    **Nota Importante:** Este reporte ha sido generado con asistencia de inteligencia artificial y
    revisado (o generado) por un profesional médico. Las imágenes y los resultados de la IA son herramientas
    de apoyo y no reemplazan el diagnóstico ni el juicio clínico de tu médico. Por favor, discute estos
    resultados e imágenes con tu médico tratante.
    """)

else:
    # Si no hay datos en session_state o no coinciden con el paciente actual
    st.info(f"No se encontraron resultados o reportes recientes asociados a tu cuenta ({current_patient_id}).")
    st.write("Si esperabas ver un reporte, por favor, asegúrate de que tu médico lo haya generado y asociado correctamente.")

# Limpiar los datos de sesión después de mostrarlos (opcional, para evitar que se muestren siempre)
# Esta lógica de limpieza es más relevante si el usuario navega fuera y vuelve,
# o si quieres que el reporte solo se vea una vez hasta que se genere uno nuevo.
# if patient_has_results and st.sidebar.button("Marcar reporte como visto/cerrar"):
#     if 'report_data_to_save' in st.session_state: 
#         del st.session_state.report_data_to_save 
#         # Ya no necesitamos borrar las claves individuales de imagen aquí, 
#         # ya que las leemos desde report_data_to_save
#     st.rerun()

