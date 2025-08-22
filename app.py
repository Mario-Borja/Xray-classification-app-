# -----------------------------------------------------------------------------
# Archivo: app.py y Notificación por Email)
# -----------------------------------------------------------------------------
import streamlit as st
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

# --- Funciones de Autenticación y Logging ---

def check_credentials(username, password):
    """
    Verifica credenciales contra BD MySQL.
    Devuelve: (bool: success, str: role | None, str: email | None)
    ¡¡MODIFICADO!!: Ya NO cierra la conexión. Devuelve el email.
    """
    

    email = None # Inicializa email
    try:
        
        result = next((obj for obj in st.secrets["login"].logins if obj.get ("username") ==  username), None)

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

    except Exception as e:
        st.error(f"Error inesperado al verificar credenciales: {e}")
        return False, None, None
    # ¡¡MODIFICADO!!: Quitamos el conn.close() del finally aquí para reusar la conexión

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

    with st.form("login_form"): # Cambié el nombre del form por simplicidad
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Iniciar Sesión")

        if submitted:
            is_correct, user_role, user_email = check_credentials(username, password)

            if is_correct:
                # 1. Actualizar estado de sesión
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.session_state['role'] = user_role
                st.session_state['email'] = user_email

                # 2. Enviar email de notificación
                send_login_email(user_email, username, user_role)

                # 3. NO es necesario llamar a st.rerun() aquí.
                # Streamlit lo hará automáticamente al final de esta ejecución.
                # Para dar feedback inmediato, podemos mostrar un mensaje de éxito.
                st.success("Inicio de sesión exitoso. Redirigiendo...")
                # Una pequeña pausa para que el usuario vea el mensaje
                import time
                time.sleep(1)
                st.experimental_rerun() # Usamos rerun aquí DESPUÉS del feedback para forzar la recarga a la página principal.

            else:
                st.error("Usuario o contraseña incorrectos.")

# Si el usuario SÍ está logueado, muestra la interfaz principal de la app
else:
    # ... (el resto de tu código para el usuario logueado no necesita cambios) ...
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
