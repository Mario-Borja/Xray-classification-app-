# -----------------------------------------------------------------------------
# Archivo: app.py (Login MySQL con Logging a BD y Notificación por Email)
# -----------------------------------------------------------------------------
import streamlit as st
import mysql.connector
import bcrypt
import pandas as pd
import datetime
import smtplib # Para enviar correos
from email.message import EmailMessage # Para construir el mensaje de correo

# --- Configuración de Página ---
st.set_page_config(
    page_title="Mi Aplicación Segura",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuración de Conexión a Base de Datos MySQL ---
# (Sin cambios aquí, lee desde st.secrets["mysql"])
def init_connection():
    """Inicializa la conexión a la base de datos MySQL usando st.secrets."""
    try:
        return mysql.connector.connect(**st.secrets["mysql"])
    except mysql.connector.Error as e:
        st.error(f"Error al conectar a MySQL: {e}")
        return None
    except Exception as e:
         st.error(f"Error inesperado en la conexión: {e}")
         if "secrets" in str(e).lower():
              st.error("Asegúrate de haber configurado correctamente el archivo .streamlit/secrets.toml con la sección [mysql]")
         return None

# --- Funciones de Autenticación y Logging ---

def check_credentials(conn, username, password):
    """
    Verifica credenciales contra BD MySQL.
    Devuelve: (bool: success, str: role | None, str: email | None)
    ¡¡MODIFICADO!!: Ya NO cierra la conexión. Devuelve el email.
    """
    if conn is None:
        st.error("Intento de verificar credenciales sin conexión a BD.")
        return False, None, None

    email = None # Inicializa email
    try:
        with conn.cursor(dictionary=True) as cur:
            # ¡MODIFICADO! Selecciona también el email
            cur.execute(
                "SELECT password_hash, role, email FROM users WHERE username = %s",
                (username,)
            )
            result = cur.fetchone()

            if result:
                stored_hash = result['password_hash'].encode('utf-8')
                user_role = result['role']
                email = result['email'] # Obtiene el email
                if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
                    return True, user_role, email # Devuelve éxito, rol y email
                else:
                    return False, None, None # Contraseña incorrecta
            else:
                return False, None, None # Usuario no encontrado

    except mysql.connector.Error as e:
        st.error(f"Error en la consulta a la base de datos MySQL: {e}")
        return False, None, None
    except Exception as e:
        st.error(f"Error inesperado al verificar credenciales: {e}")
        return False, None, None
    # ¡¡MODIFICADO!!: Quitamos el conn.close() del finally aquí para reusar la conexión

def log_login_attempt(conn, username, role):
    """Registra un intento de login exitoso en la tabla login_logs."""
    if conn is None:
        st.warning("No se pudo registrar el login: Sin conexión a BD.")
        return False

    now = datetime.datetime.now()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO login_logs (log_timestamp, username, role)
                VALUES (%s, %s, %s)
                """,
                (now, username, role)
            )
        conn.commit() # ¡Importante! Confirma la transacción
        return True
    except mysql.connector.Error as e:
        st.warning(f"No se pudo registrar el login en la BD: {e}")
        conn.rollback() # Deshace en caso de error
        return False
    except Exception as e:
        st.warning(f"Error inesperado al registrar login: {e}")
        return False

# --- Función para Enviar Email ---

def send_login_email(recipient_email, username, role):
    """Envía una notificación por email sobre el inicio de sesión."""
    if not recipient_email:
        st.info(f"No se envió email para {username}: No hay dirección de correo registrada.")
        return False

    try:
        # Lee credenciales de email desde st.secrets
        email_config = st.secrets["email"]
        sender = email_config["sender_email"]
        password = email_config["sender_password"]
        smtp_server = email_config["smtp_server"]
        smtp_port = email_config["smtp_port"]

        # Crea el mensaje
        msg = EmailMessage()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
        msg['Subject'] = f"Alerta de Acceso: Inicio de sesión detectado ({username})"
        msg['From'] = sender
        msg['To'] = recipient_email
        msg.set_content(
            f"""Hola {username},

Se ha detectado un inicio de sesión en tu cuenta.

Detalles del Acceso:
- Usuario: {username}
- Rol: {role}
- Fecha y Hora: {now_str}

Si no reconoces esta actividad, por favor contacta al administrador.

Saludos,
Tu Aplicación Streamlit
"""
        )

        # Envía el correo usando SMTP_SSL
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender, password)
            server.send_message(msg)
        # st.sidebar.info(f"Email de notificación enviado a {recipient_email}") # Info opcional en UI
        return True

    except KeyError as e:
        st.warning(f"No se pudo enviar email: Falta configuración en st.secrets[email]: {e}")
        return False
    except smtplib.SMTPAuthenticationError:
        st.warning("No se pudo enviar email: Fallo de autenticación SMTP. Verifica usuario/contraseña de email en st.secrets.")
        return False
    except smtplib.SMTPServerDisconnected:
         st.warning("No se pudo enviar email: Desconectado del servidor SMTP.")
         return False
    except smtplib.SMTPException as e:
        st.warning(f"No se pudo enviar email: Error SMTP: {e}")
        return False
    except Exception as e:
        st.warning(f"Error inesperado al enviar email: {e}")
        return False


# --- Inicialización del Estado de Sesión (Sin cambios) ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    st.session_state['role'] = None
    st.session_state['email'] = None # Añadimos email al estado

# --- Lógica de Logout (Sin cambios, pero resetea email) ---
def logout():
    """Resetea el estado de sesión para cerrar sesión."""
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    st.session_state['role'] = None
    st.session_state['email'] = None # Resetea email también
    st.info("Has cerrado sesión.")
    st.rerun()


# --- Interfaz de Usuario ---

# Si el usuario NO está logueado, muestra el formulario de login
if not st.session_state['logged_in']:
    st.title("Inicio de Sesión")
    st.info("Por favor, introduce tus credenciales para acceder a la aplicación.")

    try:
        test_conn = init_connection()
        can_connect = test_conn is not None
        if can_connect:
            test_conn.close()
    except Exception:
        can_connect = False

    if can_connect:
        with st.form("login_form_mysql"):
            username = st.text_input("Usuario", key="login_username_mysql")
            password = st.text_input("Contraseña", type="password", key="login_password_mysql")
            submitted = st.form_submit_button("Iniciar Sesión")

            if submitted:
                login_conn = None # Inicializa la conexión
                try:
                    login_conn = init_connection()
                    if login_conn:
                        # ¡MODIFICADO! check_credentials ahora devuelve 3 valores y NO cierra la conexión
                        is_correct, user_role, user_email = check_credentials(login_conn, username, password)

                        if is_correct:
                            # 1. Registrar el login en la BD (usa la misma conexión)
                            log_success = log_login_attempt(login_conn, username, user_role)
                            if not log_success:
                                st.warning("El inicio de sesión fue exitoso, pero no se pudo registrar en la base de datos.")
                                # Decide si continuar o no. Por ahora continuamos.

                            # 2. Enviar email de notificación
                            email_success = send_login_email(user_email, username, user_role)
                            if not email_success:
                                # El warning ya se muestra dentro de send_login_email
                                pass # Continuar aunque el email falle

                            # 3. Actualizar estado de sesión
                            st.session_state['logged_in'] = True
                            st.session_state['username'] = username
                            st.session_state['role'] = user_role
                            st.session_state['email'] = user_email # Guarda el email en sesión

                            # 4. Rerun para mostrar la app principal
                            st.rerun()
                        else:
                            st.error("Usuario o contraseña incorrectos.")
                    else:
                         st.warning("No se pudo verificar credenciales debido a un problema de conexión.")

                finally:
                    # ¡¡IMPORTANTE!! Cerrar la conexión después de usarla para check y log
                    if login_conn and login_conn.is_connected():
                        login_conn.close()

    else:
        st.error("Fallo en la conexión inicial a la base de datos. Verifica la configuración en .streamlit/secrets.toml y que el servidor MySQL esté accesible.")

# Si el usuario SÍ está logueado, muestra la interfaz principal de la app
else:
    st.sidebar.success(f"Logueado como: {st.session_state['username']} ({st.session_state['role']})")
    if st.sidebar.button("Cerrar Sesión"):
        logout()

    st.title("Bienvenido a la Aplicación Principal")
    st.markdown(
        """
        Has iniciado sesión correctamente. Tu acceso ha sido registrado.

        **👈 Selecciona una de las páginas disponibles en la barra lateral**
        para continuar.
        """
    )
    st.info(f"Usuario: **{st.session_state['username']}** | Rol: **{st.session_state['role']}** | Email: **{st.session_state.get('email', 'N/D')}**")

