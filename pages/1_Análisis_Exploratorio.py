# -----------------------------------------------------------------------------
# Archivo: pages/page_4.py
# Contenido: Exploración del Dataset de Rayos X y Análisis de Desbalance.
# -----------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import random

# --- Verificación de Autenticación ---
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("Por favor, inicia sesión para acceder a esta página.")
    st.stop()

# --- Configuración y Constantes ---
st.set_page_config(page_title="Exploración Rayos X", layout="wide")
st.markdown("# Exploración del Dataset y Desbalance de Clases 📊")
st.sidebar.header("Exploración Dataset")

# Definir rutas relativas al directorio principal de la app (donde está app.py)
BASE_DATA_PATH = "data/"
IMAGE_PATH = os.path.join(BASE_DATA_PATH, "images-small")
TRAIN_CSV_PATH = os.path.join(BASE_DATA_PATH, "train-small.csv")
VALID_CSV_PATH = os.path.join(BASE_DATA_PATH, "valid-small.csv")
TEST_CSV_PATH = os.path.join(BASE_DATA_PATH, "test.csv")

# Lista de columnas de etiquetas
LABEL_COLS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]
IMAGE_COL = 'Image' # Columna con nombres de archivo
PATIENT_ID_COL = 'PatientId'
# --- Funciones Cacheadas ---

@st.cache_data # Cachear la carga de datos
def load_data(csv_path):
    """Carga el archivo CSV y realiza limpieza básica."""
    try:
        df = pd.read_csv(csv_path)
        # Asegurar que las columnas de etiquetas existan y sean numéricas
        actual_labels = [col for col in LABEL_COLS if col in df.columns]
        for col in actual_labels:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        # Crear columna FullPath
        if IMAGE_COL in df.columns:
             # Verificar si IMAGE_PATH existe antes de crear rutas completas
             if not os.path.isdir(IMAGE_PATH):
                  st.error(f"El directorio de imágenes '{IMAGE_PATH}' no fue encontrado. Verifica la estructura de carpetas.")
                  # Devolver None o un df vacío podría ser una opción, pero lanzar error es más claro
                  raise FileNotFoundError(f"Directorio no encontrado: {IMAGE_PATH}")
             df['FullPath'] = df[IMAGE_COL].apply(lambda x: os.path.join(IMAGE_PATH, x))
        else:
             st.error(f"La columna '{IMAGE_COL}' no se encuentra en {os.path.basename(csv_path)}")
             df['FullPath'] = None # O manejar de otra forma

        return df, actual_labels
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo CSV: {csv_path}")
        return None, []
    except Exception as e:
        st.error(f"Error al cargar o procesar {csv_path}: {e}")
        return None, []

@st.cache_data
def compute_class_freqs(labels_df, label_columns):
    """Calcula la frecuencia de casos positivos y negativos."""
    freqs = {}
    total_examples = len(labels_df)
    if total_examples == 0: return freqs
    for label in label_columns:
        pos_count = labels_df[label].sum()
        neg_count = total_examples - pos_count
        freqs[label] = (pos_count / total_examples, neg_count / total_examples)
    return freqs

@st.cache_data
def calculate_loss_weights(class_freqs):
    """Calcula los pesos w_pos y w_neg."""
    pos_weights = {}
    neg_weights = {}
    for label, (freq_pos, freq_neg) in class_freqs.items():
        pos_weights[label] = freq_neg
        neg_weights[label] = freq_pos
    return pos_weights, neg_weights

# --- Carga de Datos Principal (Entrenamiento) ---

st.header("1. Carga y Visión General de los Datos")
train_df, actual_label_cols = load_data(TRAIN_CSV_PATH)

if train_df is not None:
    st.success(f"Datos de entrenamiento cargados ({train_df.shape[0]} filas).")
    if st.checkbox("Mostrar datos crudos (primeras 100 filas)", False):
        st.dataframe(train_df.head(100))

    st.markdown(f"Columnas de patologías identificadas: `{', '.join(actual_label_cols)}`")

else:
    st.error("No se pudieron cargar los datos de entrenamiento. No se puede continuar con el análisis.")
    st.stop() # Detener si no hay datos

# --- Carga de Datos Principal (Entrenamiento, Validación, Prueba) ---
# Modificado para cargar los 3 datasets necesarios para el análisis de superposición
st.header("1. Carga y Visión General de los Datos")
train_df, actual_label_cols = load_data(TRAIN_CSV_PATH)
valid_df, _ = load_data(VALID_CSV_PATH) # No necesitamos las etiquetas de validación aquí
test_df, _ = load_data(TEST_CSV_PATH)   # No necesitamos las etiquetas de prueba aquí

data_loaded_successfully = True
if train_df is None:
    st.error("Fallo al cargar datos de entrenamiento.")
    data_loaded_successfully = False
if valid_df is None:
    st.error("Fallo al cargar datos de validación.")
    data_loaded_successfully = False
if test_df is None:
    st.error("Fallo al cargar datos de prueba.")
    data_loaded_successfully = False

if data_loaded_successfully:
    st.success(f"Datos cargados: Entrenamiento ({train_df.shape[0]} filas), Validación ({valid_df.shape[0]} filas), Prueba ({test_df.shape[0]} filas).")
    if st.checkbox("Mostrar datos crudos de entrenamiento (primeras 100 filas)", False):
        st.dataframe(train_df.head(100))
    st.markdown(f"Columnas de patologías identificadas: `{', '.join(actual_label_cols)}`")
else:
    st.error("No se pudieron cargar todos los conjuntos de datos necesarios. No se puede continuar con el análisis completo.")
    st.stop()

# --- Análisis de Frecuencia de Etiquetas ---
st.header("2. Análisis del Desbalance de Clases")

class_frequencies = compute_class_freqs(train_df, actual_label_cols)

if not class_frequencies:
     st.warning("No se pudieron calcular las frecuencias de clase.")
else:
    # Preparar datos para gráficos
    freq_data = []
    pos_freq_map = {}
    for label, (freq_pos, freq_neg) in class_frequencies.items():
        freq_data.append({'Patología': label, 'Tipo': 'Positivo', 'Frecuencia': freq_pos})
        freq_data.append({'Patología': label, 'Tipo': 'Negativo', 'Frecuencia': freq_neg})
        pos_freq_map[label] = freq_pos

    freq_df = pd.DataFrame(freq_data)
    # Ordenar por frecuencia positiva para visualización
    order = sorted(actual_label_cols, key=lambda l: pos_freq_map.get(l, 0), reverse=True)

    # Gráfico 1: Frecuencias Positivas vs Negativas
    st.subheader("Frecuencia de Casos Positivos vs. Negativos")
    fig1, ax1 = plt.subplots(figsize=(14, 6)) # Crear figura y ejes con Matplotlib
    sns.barplot(data=freq_df, x='Patología', y='Frecuencia', hue='Tipo', order=order,
                palette={'Positivo': 'coral', 'Negativo': 'skyblue'}, ax=ax1)
    ax1.set_title('Frecuencia de Casos Positivos vs. Negativos por Patología', fontsize=16)
    ax1.set_xlabel('Patología', fontsize=12)
    ax1.set_ylabel('Frecuencia', fontsize=12)
    ax1.tick_params(axis='x', rotation=75)
    ax1.legend(title='Tipo de Caso')
    plt.tight_layout()
    st.pyplot(fig1) # Mostrar el gráfico de Matplotlib en Streamlit
    st.markdown("""
    **Observación:** Se evidencia claramente el gran desbalance: la barra 'Negativo' (azul) es mucho más alta que la 'Positivo' (coral) para casi todas las patologías. Las patologías más comunes (ej. Infiltration) aún son mucho menos frecuentes que la ausencia de la misma. Patologías como 'Hernia' son extremadamente raras en este conjunto.
    """)

    # --- Cálculo y Visualización de Pesos ---
    st.header("3. Pérdida Ponderada para Contrarrestar el Desbalance")
    st.markdown("""
    Para evitar que el modelo ignore las clases minoritarias (las patologías, que son las que nos interesan detectar), podemos usar una **función de pérdida ponderada**. Asignamos un peso mayor a los errores en la clase minoritaria (positiva) y un peso menor a los errores en la clase mayoritaria (negativa).

    Calculamos los pesos como:
    * `Peso Positivo (w_pos) = Frecuencia Negativa`
    * `Peso Negativo (w_neg) = Frecuencia Positiva`
    """)

    pos_weights, neg_weights = calculate_loss_weights(class_frequencies)

    # Mostrar pesos en una tabla expandible
    with st.expander("Ver Pesos Calculados (w_pos, w_neg)"):
        weights_df = pd.DataFrame({
            'Patología': actual_label_cols,
            'w_pos (freq_neg)': [pos_weights.get(l, 0) for l in actual_label_cols],
            'w_neg (freq_pos)': [neg_weights.get(l, 0) for l in actual_label_cols]
        }).set_index('Patología')
        st.dataframe(weights_df.style.format("{:.4f}"))

    # Gráfico 2: Contribución Ponderada Esperada
    st.subheader("Contribución Esperada a la Pérdida (Después de Ponderar)")
    weighted_contribution_data = []
    for label in actual_label_cols:
        freq_pos, freq_neg = class_frequencies.get(label, (0, 1))
        w_p = pos_weights.get(label, 0)
        w_n = neg_weights.get(label, 0)
        total_pos_contrib = freq_pos * w_p
        total_neg_contrib = freq_neg * w_n
        weighted_contribution_data.append({'Patología': label, 'Tipo': 'Positivo Ponderado', 'Contribución': total_pos_contrib})
        weighted_contribution_data.append({'Patología': label, 'Tipo': 'Negativo Ponderado', 'Contribución': total_neg_contrib})

    weighted_contrib_df = pd.DataFrame(weighted_contribution_data)

    fig2, ax2 = plt.subplots(figsize=(14, 6)) # Crear figura y ejes
    sns.barplot(data=weighted_contrib_df, x='Patología', y='Contribución', hue='Tipo', order=order,
                palette={'Positivo Ponderado': 'lightcoral', 'Negativo Ponderado': 'lightblue'}, ax=ax2)
    ax2.set_title('Contribución Total Esperada a la Pérdida (Ponderada)', fontsize=16)
    ax2.set_xlabel('Patología', fontsize=12)
    ax2.set_ylabel('Contribución Ponderada (freq * peso)', fontsize=12)
    ax2.tick_params(axis='x', rotation=75)
    ax2.legend(title='Tipo de Caso (Ponderado)')
    ax2.set_ylim(bottom=0) # Asegurar que el eje Y empiece en 0
    plt.tight_layout()
    st.pyplot(fig2) # Mostrar gráfico en Streamlit
    st.markdown("""
    **Observación:** Después de aplicar la ponderación, la contribución total esperada de los casos positivos y negativos a la función de pérdida se vuelve **igual** para cada patología. Esto fuerza al modelo a prestar la misma atención a acertar en casos positivos y negativos, independientemente de cuán rara sea la patología.
    """)


# --- Exploración de Imágenes ---
st.header("4. Exploración de Imágenes de Ejemplo")
st.markdown("Selecciona un índice para ver la radiografía y sus etiquetas asociadas.")

if train_df is not None and 'FullPath' in train_df.columns and not train_df['FullPath'].isnull().all():
    # Slider para seleccionar índice de imagen
    max_index = len(train_df) - 1
    selected_index = st.slider("Selecciona Índice de Imagen:", 0, max_index, random.randint(0, max_index))

    col1, col2 = st.columns([2, 1]) # Columna para imagen, columna para etiquetas

    with col1:
        image_path = train_df.loc[selected_index, 'FullPath']
        image_filename = train_df.loc[selected_index, IMAGE_COL]

        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                st.image(image, caption=f"Imagen: {image_filename} (Índice: {selected_index})", use_column_width=True)
            except Exception as e:
                st.error(f"No se pudo cargar la imagen {image_filename}: {e}")
        else:
            st.error(f"Ruta de imagen no encontrada o inválida para el índice {selected_index}: {image_path}")

    with col2:
        st.subheader("Etiquetas Asociadas:")
        labels = train_df.loc[selected_index, actual_label_cols]
        positive_labels = labels[labels == 1].index.tolist()

        if positive_labels:
            for label in positive_labels:
                st.success(f"✓ {label}") # Marca positiva
            # Mostrar también las negativas podría ser útil
            negative_labels = labels[labels == 0].index.tolist()
            with st.expander("Ver etiquetas negativas"):
                 for label in negative_labels:
                      st.write(f"- {label}")

        else:
            st.info("No se encontraron patologías positivas para esta imagen (según las etiquetas).")

        # Mostrar otras infos si existen
        other_cols = ['Patient Age', 'Patient Gender', 'View Position']
        for col in other_cols:
            if col in train_df.columns:
                st.write(f"**{col}:** {train_df.loc[selected_index, col]}")


else:
    st.warning("No se puede mostrar la exploración de imágenes porque la columna 'FullPath' falta o está vacía, o las imágenes no se encontraron.")


# --- Resumen Preprocesamiento ---
st.header("5. Resumen del Preprocesamiento (para el Modelo)")
st.markdown("""
Basado en la exploración, los pasos clave para preparar estas imágenes para un modelo CNN serían:

1.  **Carga en Escala de Grises:** Leer las imágenes como monocromáticas.
2.  **Redimensionamiento:** Cambiar el tamaño a uno fijo (ej. 224x224 píxeles).
3.  **Normalización:** Escalar los valores de píxeles del rango [0, 255] al rango [0, 1].
4.  **Aumento de Datos (Solo Entrenamiento):** Aplicar transformaciones aleatorias (rotaciones, flips, zoom, etc.) a las imágenes de entrenamiento para mejorar la robustez del modelo.

Estos pasos se implementarían típicamente usando generadores de datos como `ImageDataGenerator` de Keras o pipelines `tf.data`.
""")

# --- NUEVA SECCIÓN: Histograma Interactivo ---
st.header("6. Análisis Interactivo de Intensidad de Píxeles")
st.markdown("El histograma muestra la distribución de los valores de intensidad de los píxeles (0=negro, 255=blanco) para la imagen seleccionada arriba con el slider.")

if train_df is not None and 'FullPath' in train_df.columns:
    # Reutilizar selected_index del slider anterior
    hist_image_path = train_df.loc[selected_index, 'FullPath']
    hist_image_filename = train_df.loc[selected_index, IMAGE_COL]

    if hist_image_path and os.path.exists(hist_image_path):
        try:
            hist_image = Image.open(hist_image_path).convert('L') # Cargar en escala de grises
            hist_image_array = np.array(hist_image)

            # Crear el histograma con Matplotlib
            fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
            ax_hist.hist(hist_image_array.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.8)
            ax_hist.set_title(f'Histograma de Intensidad de Píxeles (Imagen: {hist_image_filename})')
            ax_hist.set_xlabel('Intensidad de Píxel (0-255)')
            ax_hist.set_ylabel('Frecuencia (Número de Píxeles)')
            ax_hist.grid(axis='y', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig_hist) # Mostrar en Streamlit

        except Exception as e:
            st.error(f"No se pudo generar el histograma para la imagen {hist_image_filename}: {e}")
    else:
        st.warning(f"No se puede generar el histograma: Imagen no encontrada para el índice {selected_index}.")
else:
    st.warning("Carga los datos primero para ver el histograma.")


# --- NUEVA SECCIÓN: Visor por Paciente ---
st.header("7. Visor de Imágenes por Paciente")

if train_df is not None and PATIENT_ID_COL in train_df.columns:
    # Obtener lista única de IDs de paciente y ordenarla
    patient_ids = sorted(train_df[PATIENT_ID_COL].unique())

    # Permitir seleccionar un paciente
    selected_patient_id = st.selectbox("Selecciona un ID de Paciente:", options=patient_ids, index=0)

    if selected_patient_id:
        st.subheader(f"Radiografías para el Paciente ID: {selected_patient_id}")

        # Filtrar el dataframe por el paciente seleccionado
        patient_df = train_df[train_df[PATIENT_ID_COL] == selected_patient_id].reset_index() # Reset index para iterar fácil

        if not patient_df.empty:
            # Mostrar imágenes en columnas (ej. 3 columnas)
            num_cols = 3
            cols = st.columns(num_cols)
            col_idx = 0

            for index, row in patient_df.iterrows():
                current_col = cols[col_idx % num_cols] # Ciclar entre columnas
                with current_col:
                    img_filename = row[IMAGE_COL]
                    img_path = row['FullPath']
                    st.markdown(f"**Imagen:** {img_filename}")

                    if img_path and os.path.exists(img_path):
                        try:
                            image = Image.open(img_path)
                            st.image(image, use_column_width=True)
                        except Exception as e:
                            st.error(f"Error al cargar {img_filename}: {e}", icon="⚠️")
                    else:
                        st.warning(f"Imagen no encontrada: {img_filename}", icon="⚠️")

                    # Mostrar etiquetas para esta imagen
                    st.markdown("**Etiquetas:**")
                    labels_patient = row[actual_label_cols]
                    positive_labels_patient = labels_patient[labels_patient == 1].index.tolist()
                    negative_labels_patient = labels_patient[labels_patient == 0].index.tolist()

                    if positive_labels_patient:
                        for label in positive_labels_patient:
                            st.success(f"✓ {label}")
                    else:
                        st.info("Ninguna patología positiva.")

                    # Expander opcional para negativas
                    with st.expander("Ver negativas"):
                         if negative_labels_patient:
                              st.markdown('\n'.join([f"- {label}" for label in negative_labels_patient]))
                         else:
                              st.write("Todas las etiquetas son positivas.")
                    st.markdown("---") # Separador entre imágenes en la misma columna
                col_idx += 1 # Pasar a la siguiente columna (o volver a la primera)

        else:
            st.info(f"No se encontraron imágenes para el paciente ID: {selected_patient_id}")
    else:
        st.info("Selecciona un ID de paciente de la lista.")

else:
    st.warning("No se puede mostrar el visor por paciente: Carga los datos primero y asegúrate de que exista la columna 'Patient ID'.")

# --- NUEVA SECCIÓN: Análisis de Superposición de Pacientes ---
st.header("8. Análisis de Superposición de Pacientes (Fuga de Datos)")
st.markdown("""
Es crucial verificar si los mismos pacientes aparecen en diferentes conjuntos de datos (entrenamiento, validación, prueba). Si un paciente está en el conjunto de entrenamiento y también en el de validación/prueba, el modelo podría aprender características específicas de ese paciente, llevando a una **evaluación demasiado optimista** de su rendimiento (fuga de datos).
""")

# Verificar si todos los dataframes necesarios están cargados
if data_loaded_successfully:
    # Extraer IDs únicos (usar cache para esta operación podría ser útil si los datasets son grandes)
    @st.cache_data
    def get_patient_ids(df, id_col):
        if id_col in df.columns:
            return set(df[id_col].astype(str).str.strip().unique())
        else:
            st.error(f"Columna '{id_col}' no encontrada en uno de los dataframes.")
            return set() # Devuelve conjunto vacío si la columna no existe

    train_patient_ids = get_patient_ids(train_df, PATIENT_ID_COL)
    valid_patient_ids = get_patient_ids(valid_df, PATIENT_ID_COL)
    test_patient_ids = get_patient_ids(test_df, PATIENT_ID_COL)

    if train_patient_ids and valid_patient_ids and test_patient_ids: # Asegurarse que no estén vacíos
        # Calcular superposiciones
        overlap_train_valid = train_patient_ids.intersection(valid_patient_ids)
        overlap_train_test = train_patient_ids.intersection(test_patient_ids)
        overlap_valid_test = valid_patient_ids.intersection(test_patient_ids)
        n_overlap_tv = len(overlap_train_valid)
        n_overlap_tt = len(overlap_train_test)
        n_overlap_vt = len(overlap_valid_test)

        st.subheader("Resultados de la Superposición:")
        col_ov1, col_ov2, col_ov3 = st.columns(3)
        with col_ov1:
            st.metric("Entrenamiento ∩ Validación", f"{n_overlap_tv} Pacientes")
        with col_ov2:
            st.metric("Entrenamiento ∩ Prueba", f"{n_overlap_tt} Pacientes")
        with col_ov3:
            st.metric("Validación ∩ Prueba", f"{n_overlap_vt} Pacientes")

        if n_overlap_tv > 0 or n_overlap_tt > 0 or n_overlap_vt > 0:
            st.warning("¡Se detectó superposición de pacientes! Esto puede causar fuga de datos.", icon="⚠️")

            # Mostrar IDs superpuestos en expanders
            if n_overlap_tv > 0:
                with st.expander(f"Ver los {n_overlap_tv} IDs superpuestos (Entrenamiento ∩ Validación)"):
                    st.write(sorted(list(overlap_train_valid)))
            if n_overlap_tt > 0:
                with st.expander(f"Ver los {n_overlap_tt} IDs superpuestos (Entrenamiento ∩ Prueba)"):
                    st.write(sorted(list(overlap_train_test)))
            if n_overlap_vt > 0:
                 with st.expander(f"Ver los {n_overlap_vt} IDs superpuestos (Validación ∩ Prueba)"):
                    st.write(sorted(list(overlap_valid_test)))

            # Explicar y mostrar el efecto de la limpieza
            st.subheader("Impacto de la Limpieza (Simulado)")
            st.markdown("""
            Para evitar la fuga de datos, los pacientes superpuestos deben eliminarse de los conjuntos de validación y prueba antes de la evaluación final. A continuación se muestra cuántos registros quedarían:
            """)

            patients_to_remove_from_valid = list(overlap_train_valid.union(overlap_valid_test))
            patients_to_remove_from_test = list(overlap_train_test.union(overlap_valid_test))

            # Simular la limpieza (sin modificar los dataframes originales en la app)
            valid_df_cleaned_count = len(valid_df[~valid_df[PATIENT_ID_COL].astype(str).isin(patients_to_remove_from_valid)])
            test_df_cleaned_count = len(test_df[~test_df[PATIENT_ID_COL].astype(str).isin(patients_to_remove_from_test)])

            col_cln1, col_cln2 = st.columns(2)
            with col_cln1:
                st.metric("Registros Validación (Original)", valid_df.shape[0])
                st.metric("Registros Validación (Limpio)", valid_df_cleaned_count,
                          delta=f"{valid_df_cleaned_count - valid_df.shape[0]}", delta_color="inverse")
            with col_cln2:
                st.metric("Registros Prueba (Original)", test_df.shape[0])
                st.metric("Registros Prueba (Limpio)", test_df_cleaned_count,
                          delta=f"{test_df_cleaned_count - test_df.shape[0]}", delta_color="inverse")

            if valid_df_cleaned_count < 20 or test_df_cleaned_count < 20: # Umbral arbitrario
                 st.error("ADVERTENCIA: Después de la limpieza, uno o ambos conjuntos de validación/prueba quedarían muy pequeños, lo que haría la evaluación poco fiable. Esto sugiere que la división inicial de datos debería hacerse a nivel de paciente.", icon="🚨")

        else:
            st.success("¡No se detectó superposición de pacientes entre los conjuntos! Los conjuntos parecen estar correctamente separados a nivel de paciente.", icon="✅")

    else:
         st.warning("No se pudieron extraer los IDs de paciente de todos los dataframes.")

else:
    st.warning("Carga todos los conjuntos de datos (entrenamiento, validación, prueba) para realizar el análisis de superposición.")

st.markdown("---") # Separador final de página
