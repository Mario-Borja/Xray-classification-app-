# -----------------------------------------------------------------------------
# Archivo: pages/page_5.py
# Contenido: Clasificador de Patologías Pulmonares con Rayos X, GradCAM y Reporte IA
# -----------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import cv2 # Para OpenCV en GradCAM
import base64 # Para codificar imágenes para el LLM
import io # Para manejar bytes de imagen

# --- MODIFICACIÓN: Importar OpenAI ---
from openai import OpenAI
# --- FIN MODIFICACIÓN ---

# --- Verificación de Autenticación ---
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("Por favor, inicia sesión para acceder a esta página.")
    st.stop()

# --- Configuración y Constantes ---
st.set_page_config(page_title="Diagnóstico Rayos X", layout="wide") # Llamar solo una vez
st.markdown("# Página 5: Clasificador de Patologías Pulmonares con Asistencia de IA 🩺")
st.sidebar.header("Predicción con Rayos X")

# Definir rutas
BASE_APP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_APP_PATH, "models/")
BASE_MODEL_WEIGHTS_FILENAME = "densenet.hdf5"
FINE_TUNED_WEIGHTS_FILENAME = "pretrained_model.h5"
base_model_weights_path = os.path.join(MODEL_PATH, BASE_MODEL_WEIGHTS_FILENAME)
fine_tuned_weights_path = os.path.join(MODEL_PATH, FINE_TUNED_WEIGHTS_FILENAME)

DATA_PATH = os.path.join(BASE_APP_PATH, "data/")
TRAIN_CSV_PATH = os.path.join(DATA_PATH, "train-small.csv")

IMG_HEIGHT = 224
IMG_WIDTH = 224
ACTUAL_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]
PATIENT_ID_COL = 'PatientId' 

# --- MODIFICACIÓN: Configuración del Cliente OpenAI ---
# ¡¡¡ADVERTENCIA DE SEGURIDAD!!!
# Poner tu clave API directamente en el código es riesgoso si compartes el código o lo subes a un repositorio.
# Para pruebas locales, puedes reemplazar "sk-TU_CLAVE_API_SECRETA_AQUI" con tu clave real.
# Para producción, usa variables de entorno o los Secrets de Streamlit Cloud.

OPENAI_API_KEY_PLACEHOLDER = "sk-TU_CLAVE_API_SECRETA_AQUI"
OPENAI_API_KEY_HARDCODED = "sk-Pa28ta3mfTOfAnhXCDDoAlzxnH5_LE-pCwSlPDccL5T3BlbkFJj2Qwr83HP_DJ4QOspfOdSuKpp5HpNO5VtH3LwVma4A"  # <--- TU CLAVE REAL ESTÁ AQUÍ (según tu mensaje)

client = None

# 1. Intentar con la clave hardcodeada SI ES DIFERENTE DEL PLACEHOLDER
if OPENAI_API_KEY_HARDCODED != OPENAI_API_KEY_PLACEHOLDER and OPENAI_API_KEY_HARDCODED.startswith("sk-"):
    try:
        client = OpenAI(api_key=OPENAI_API_KEY_HARDCODED)
        st.sidebar.info("Cliente OpenAI inicializado con clave API hardcodeada (SOLO PARA PRUEBAS).")
    except Exception as e:
        st.sidebar.error(f"Error OpenAI Client con clave hardcodeada: {e}")
        client = None # Asegurar que client es None si falla

# 2. Si la clave hardcodeada no se usó o falló, intentar con st.secrets (para Streamlit Cloud)
if client is None:
    try:
        # Intentar leer desde st.secrets si está disponible
        # Esto no dará error si st.secrets no existe, pero .get() devolverá None si la clave no está
        api_key_from_secrets = st.secrets.get("OPENAI_API_KEY")
        if api_key_from_secrets:
            client = OpenAI(api_key=api_key_from_secrets)
            # st.sidebar.info("Cliente OpenAI inicializado con st.secrets.") # Opcional
    except AttributeError: # st.secrets no existe (ej. localmente sin configurar secrets.toml)
        pass # Continuar al siguiente método
    except Exception as e: # Otro error al inicializar con secrets
        st.sidebar.warning(f"Error al intentar usar st.secrets para OpenAI: {e}")
        client = None

# 3. Si aún no hay cliente, intentar con variable de entorno (último recurso antes de advertir)
if client is None:
    try:
        client = OpenAI() # Intenta leer desde la variable de entorno OPENAI_API_KEY automáticamente
        # st.sidebar.info("Cliente OpenAI inicializado con variable de entorno.") # Opcional
    except Exception as e: # Captura errores si OpenAI() falla (ej. variable no seteada y no hay clave por defecto)
        # No mostramos error aquí todavía, lo haremos después si client sigue siendo None
        pass

# 4. Advertencia final si client sigue siendo None
if client is None:
    st.sidebar.warning("API Key de OpenAI no configurada correctamente (ni hardcodeada, ni en st.secrets, ni como variable de entorno). La generación de reportes no funcionará.")
# --- FIN MODIFICACIÓN ---


# --- Funciones Auxiliares y de Modelo ---
@st.cache_data
def compute_class_freqs_for_loss(train_csv_path_func, label_columns_func):
    try:
        df = pd.read_csv(train_csv_path_func)
        for col in label_columns_func:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            else:
                print(f"Error: La columna '{col}' no se encontró en {train_csv_path_func}")
                return None, None
        freqs = {}
        total_examples = len(df)
        if total_examples == 0: return None, None
        pos_counts = df[label_columns_func].sum()
        neg_counts = total_examples - pos_counts
        pos_freqs = pos_counts / total_examples
        neg_freqs = neg_counts / total_examples
        pos_weights_arr = np.array([neg_freqs[label] for label in label_columns_func])
        neg_weights_arr = np.array([pos_freqs[label] for label in label_columns_func])
        return pos_weights_arr, neg_weights_arr
    except Exception as e:
        print(f"Error al calcular frecuencias para la pérdida: {e}")
        return None, None

pos_loss_weights, neg_loss_weights = compute_class_freqs_for_loss(TRAIN_CSV_PATH, ACTUAL_LABELS)

def get_weighted_loss(pos_weights_arr_func, neg_weights_arr_func, epsilon=1e-7):
    if pos_weights_arr_func is None or neg_weights_arr_func is None:
        print("ADVERTENCIA: Usando binary_crossentropy estándar porque los pesos para la pérdida ponderada no pudieron ser calculados.")
        return 'binary_crossentropy'
    def weighted_loss(y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        log_y_pred = K.log(y_pred + epsilon); log_1_y_pred = K.log(1.0 - y_pred + epsilon)
        loss_pos = pos_weights_arr_func * y_true * log_y_pred
        loss_neg = neg_weights_arr_func * (1.0 - y_true) * log_1_y_pred
        loss = - (loss_pos + loss_neg)
        loss_per_example = K.mean(loss, axis=1); batch_loss = K.mean(loss_per_example)
        return batch_loss
    return weighted_loss

custom_loss_func = get_weighted_loss(pos_loss_weights, neg_loss_weights)

@st.cache_resource
def load_xray_model():
    try:
        base_model_arch = DenseNet121(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        if os.path.exists(base_model_weights_path):
            base_model_arch.load_weights(base_model_weights_path)
        else:
            st.sidebar.warning(f"Pesos base '{BASE_MODEL_WEIGHTS_FILENAME}' no encontrados.")
        x = base_model_arch.output
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
        num_classes = len(ACTUAL_LABELS)
        predictions_layer = Dense(num_classes, activation='sigmoid', name='predictions')(x)
        final_model = Model(inputs=base_model_arch.input, outputs=predictions_layer, name='DenseNet121_XRay_App')
        if os.path.exists(fine_tuned_weights_path):
            final_model.load_weights(fine_tuned_weights_path)
        else:
            st.sidebar.error(f"¡Error Crítico! '{FINE_TUNED_WEIGHTS_FILENAME}' no encontrado.")
            return None
        final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                            loss=custom_loss_func,
                            metrics=[tf.keras.metrics.AUC(multi_label=True, name='auc')])
        st.sidebar.success("Modelo de Rayos X cargado.")
        return final_model
    except Exception as e:
        st.sidebar.error(f"Error al cargar modelo Rayos X: {e}")
        return None

model = load_xray_model()

def preprocess_image(image_pil, target_size):
    if image_pil.mode != 'RGB': image_pil = image_pil.convert('RGB')
    img = image_pil.resize(target_size); img_array = keras.utils.img_to_array(img)
    img_array = img_array / 255.0; img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_last_conv_layer_name(model_to_inspect):
    for layer in reversed(model_to_inspect.layers):
        if isinstance(layer, keras.layers.Conv2D): return layer.name
    return None

def make_gradcam_heatmap(img_array, grad_model_input, last_conv_layer_name_local, pred_index=None):
    grad_model = Model(inputs=[grad_model_input.inputs], outputs=[grad_model_input.get_layer(last_conv_layer_name_local).output, grad_model_input.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None: pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    if grads is None: return None
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)); last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]; heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + K.epsilon()); return heatmap.numpy()

def display_gradcam_pil(original_pil_image, heatmap_np, alpha=0.5):
    img_for_cv = np.array(original_pil_image.convert('RGB'))
    heatmap_resized = cv2.resize(heatmap_np, (img_for_cv.shape[1], img_for_cv.shape[0])); heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET); jet_heatmap_rgb = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img_np = jet_heatmap_rgb * alpha + img_for_cv * (1 - alpha); superimposed_img_np = np.clip(superimposed_img_np, 0, 255).astype(np.uint8)
    return Image.fromarray(superimposed_img_np)

def image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def call_openai_vision_api_streamlit(system_prompt, user_prompt_text, original_image_b64, gradcam_image_b64):
    if not client: # client se define globalmente arriba
        st.error("Cliente de OpenAI no inicializado. Verifica la configuración de la API Key.")
        return "Error: Cliente de OpenAI no configurado."
    try:
        with st.spinner("Generando reporte con IA... Esto puede tardar un momento."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{original_image_b64}", "detail": "low"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{gradcam_image_b64}", "detail": "low"}},
                    ]},
                ],
                max_tokens=1200
            )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error durante la llamada a la API de OpenAI: {e}")
        return f"Error al generar reporte: {e}"

if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'llm_report' not in st.session_state:
    st.session_state.llm_report = None
if 'original_image_for_report' not in st.session_state:
    st.session_state.original_image_for_report = None
if 'gradcam_image_for_report' not in st.session_state:
    st.session_state.gradcam_image_for_report = None

st.write("Sube una imagen de radiografía de tórax para obtener una predicción de patologías y generar un reporte asistido por IA.")
uploaded_file = st.file_uploader("Elige una imagen...", type=["png", "jpg", "jpeg"], key="file_uploader_page5")

if model is None:
    st.error("El modelo de IA no pudo ser cargado. Por favor, revisa los logs y la configuración.")
else:
    if uploaded_file is not None:
        original_image_pil = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagen Original Subida")
            st.image(original_image_pil, use_column_width=True)
        with col2:
            if st.button("Predecir Patología", key="predict_button_page5", type="primary"):
                st.session_state.prediction_results = None
                st.session_state.llm_report = None
                st.session_state.original_image_for_report = None
                st.session_state.gradcam_image_for_report = None
                with st.spinner("Analizando imagen..."):
                    processed_img_array = preprocess_image(original_image_pil.copy(), (IMG_HEIGHT, IMG_WIDTH))
                    predictions_raw = model.predict(processed_img_array)[0]
                    predicted_pathology_index = np.argmax(predictions_raw)
                    predicted_pathology_label = ACTUAL_LABELS[predicted_pathology_index]
                    predicted_pathology_prob = predictions_raw[predicted_pathology_index]
                    st.session_state.prediction_results = {
                        "label": predicted_pathology_label,
                        "probability": predicted_pathology_prob,
                        "all_probabilities": predictions_raw
                    }
                    st.session_state.original_image_for_report = original_image_pil.copy()
                    last_conv_name = get_last_conv_layer_name(model)
                    if last_conv_name:
                        heatmap_np = make_gradcam_heatmap(processed_img_array, model, last_conv_name, pred_index=predicted_pathology_index)
                        if heatmap_np is not None:
                            gradcam_image_pil = display_gradcam_pil(original_image_pil.copy(), heatmap_np)
                            st.session_state.gradcam_image_for_report = gradcam_image_pil
                        else:
                            st.session_state.gradcam_image_for_report = None
                            st.warning("No se pudo generar el mapa de calor GradCAM.")
                    else:
                        st.session_state.gradcam_image_for_report = None
                        st.warning("No se pudo encontrar la capa convolucional para GradCAM.")
            
            if st.session_state.prediction_results:
                results = st.session_state.prediction_results
                st.subheader("Resultado de la Predicción IA:")
                st.success(f"Patología Principal Predicha: **{results['label']}**")
                st.info(f"Probabilidad: **{results['probability']:.2%}**")
                if st.session_state.gradcam_image_for_report:
                    st.subheader("Mapa de Calor GradCAM")
                    st.image(st.session_state.gradcam_image_for_report, caption=f"GradCAM para '{results['label']}'", use_column_width=True)
                with st.expander("Ver todas las probabilidades por patología"):
                    probs_df = pd.DataFrame({'Patología': ACTUAL_LABELS, 'Probabilidad': results['all_probabilities']})
                    st.dataframe(probs_df.sort_values(by='Probabilidad', ascending=False).style.format({"Probabilidad": "{:.2%}"}))
                st.markdown("---")
                if st.button("Generar Reporte Radiológico Asistido por IA", key="generate_report_button"):
                    if client and st.session_state.original_image_for_report and st.session_state.gradcam_image_for_report:
                        original_b64 = image_to_base64(st.session_state.original_image_for_report)
                        gradcam_b64 = image_to_base64(st.session_state.gradcam_image_for_report)
                        system_prompt_llm = "Eres un radiólogo experto con especialización en patologías pulmonares y análisis de radiografías de tórax. Tu tarea es generar un reporte radiológico conciso y profesional basado en la información proporcionada. Sé objetivo y basa tus conclusiones en la evidencia visual y los datos de la IA."
                        user_prompt_text_llm = f"""
Por favor, genera un reporte radiológico para el siguiente caso.
**Información del Caso Proporcionada por IA:**
- **Predicción Principal de Patología:** {results['label']}
- **Probabilidad Asociada por IA:** {results['probability']:.2%}
- **Contexto Visual:** Te proporciono dos imágenes:
    1. La radiografía de tórax original.
    2. Un mapa de calor GradCAM que resalta las regiones que la IA consideró importantes para su predicción de '{results['label']}'.
**Instrucciones para el Reporte:**
1.  **Encabezado:** Incluye: Paciente ID (usar "No especificado por ahora"), Fecha del Examen (usar fecha actual), Tipo de Examen ("Radiografía de Tórax").
2.  **Hallazgos de IA:** Resume brevemente la predicción de la IA y su probabilidad. Describe dónde se localiza la activación principal del GradCAM.
3.  **Interpretación Radiológica Detallada:**
    * Comenta brevemente la calidad técnica de la imagen original.
    * Correlaciona la predicción de la IA ({results['label']}) con los hallazgos visuales en la radiografía original, prestando especial atención a las áreas resaltadas por el GradCAM.
        * Si la predicción es visualmente consistente, describe los signos radiológicos que la apoyan.
        * Si la predicción NO es claramente visible o parece incorrecta, justifica tu discrepancia.
        * Si el GradCAM es difuso o no es útil, indícalo.
    * Describe brevemente otros hallazgos pulmonares, pleurales, mediastínicos o cardíacos relevantes.
4.  **Conclusión/Impresión Diagnóstica:** Ofrece una conclusión concisa. Indica si los hallazgos de la IA son apoyados por tu interpretación visual. Sugiere posibles diagnósticos diferenciales y recomendaciones.
**Por favor, genera el reporte.**
"""
                        report_text = call_openai_vision_api_streamlit(
                            system_prompt_llm, user_prompt_text_llm, original_b64, gradcam_b64
                        )
                        st.session_state.llm_report = report_text
                    else:
                        st.error("No se pueden generar el reporte. Asegúrate de que el cliente OpenAI esté configurado y las imágenes estén disponibles.")
            
            if st.session_state.llm_report:
                st.markdown("---")
                st.subheader("Reporte Radiológico Generado por IA (Borrador)")
                st.markdown(st.session_state.llm_report)
                st.markdown("---")
                st.subheader("Asociar Resultados a Paciente")
                mock_patient_ids = ['usuario1', 'usuario2'] # Reemplaza con tu lógica real para obtener pacientes
                if not mock_patient_ids:
                    st.warning("No hay pacientes disponibles para asociar.")
                else:
                    selected_patient_id_for_report = st.selectbox(
                        "Seleccionar Paciente para asociar este reporte:",
                        options=mock_patient_ids, index=0, key="patient_select_for_report"
                    )
                    if st.button("Guardar y Asociar Reporte al Paciente", key="save_report_button"):
                        st.session_state.report_data_to_save = {
                            "patient_id": selected_patient_id_for_report,
                            "doctor_id": st.session_state.get('username'),
                            "date_generated": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "cnn_prediction": st.session_state.prediction_results['label'],
                            "cnn_probability": st.session_state.prediction_results['probability'],
                            "llm_report_text": st.session_state.llm_report,
                            "original_image_pil": st.session_state.get('original_image_for_report'), # <-- Añadir esto
                            "gradcam_image_pil": st.session_state.get('gradcam_image_for_report')     # <-- Añadir esto

                        }
                        st.success(f"Reporte y resultados asociados (simulado) al paciente: {selected_patient_id_for_report}")
                        st.info("En una implementación completa, estos datos se guardarían en una base de datos.")
                        st.json(st.session_state.report_data_to_save)
                        st.markdown(f"Para ver los resultados del paciente (funcionalidad de `page_6.py`), necesitarías implementar esa página para que recupere y muestre la información guardada para el paciente `{selected_patient_id_for_report}`.")
                        st.session_state.prediction_results = None; st.session_state.llm_report = None
                        st.session_state.original_image_for_report = None; st.session_state.gradcam_image_for_report = None
                        st.rerun()
    else:
        st.info("Sube una imagen para comenzar.")

