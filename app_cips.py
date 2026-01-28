import streamlit as st
import pandas as pd
import numpy as np
import io
import altair as alt

# --- 1. CONFIGURACIÓN DE PÁGINA (OBLIGATORIO AL INICIO) ---
st.set_page_config(page_title="Portal Ingeniería CIPS", page_icon="🔒", layout="wide")

# --- 2. SISTEMA DE SEGURIDAD (LOGIN) ---
def check_password():
    """Retorna True si el usuario ingresó la clave correcta."""
    def password_entered():
        if st.session_state["password"] == "CIPS2026": # <--- CLAVE DE ACCESO
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Borra la clave por seguridad
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # Estética del Login
        st.markdown(
            """
            <style>
            .stTextInput > div > div > input {text-align: center;} 
            </style>
            """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.warning("🔒 Acceso Restringido a Personal Autorizado")
            st.text_input("Ingrese Contraseña:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.error("❌ Contraseña incorrecta")
        st.text_input("Ingrese Contraseña:", type="password", on_change=password_entered, key="password")
        return False
    else:
        return True

if not check_password():
    st.stop()

# =========================================================
#  AQUÍ COMIENZA LA APP (SOLO VISIBLE TRAS EL LOGIN)
# =========================================================

# --- 3. ENCABEZADO CON LOGO ---
col_logo, col_titulo = st.columns([1, 6])

with col_logo:
    try:
        st.image("logo.png", use_container_width=True) 
    except:
        st.markdown("## ⚡")

with col_titulo:
    st.title("Procesador de Integridad CIPS")
    st.markdown("Plataforma de Ingeniería | **Análisis y Reportes**")

st.markdown("---")

# --- 4. BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.subheader("1. Definición del Tramo")
    pk_inicial = st.number_input("PK Inicial (m)", value=14000.0, step=100.0, format="%.2f")
    pk_final = st.number_input("PK Final (m)", value=15000.0, step=100.0, format="%.2f")
    
    st.divider()
    
    st.subheader("2. Calibración de Limpieza")
    st.info("Ajuste los filtros para eliminar ruido eléctrico.")
    
    # VALORES POR DEFECTO: LIMPIEZA FUERTE
    umbral_pico = st.slider(
        "Sensibilidad (mV)", 
        min_value=5, max_value=100, value=15, 
        help="Cualquier salto mayor a este valor será eliminado."
    )
    
    ventana_deteccion = st.slider(
        "Ancho del Pico (Vecinos)", 
        min_value=3, max_value=11, value=9, step=2,
        help="Usar 9 o 11 elimina los picos grandes y anchos."
    )
    
    st.subheader("3. Estética del Reporte")
    activar_suavizado = st.checkbox("Aplicar Suavizado Final", value=True)
    ventana_suavizado = st.slider(
        "Nivel de Suavizado", 
        min_value=2, max_value=20, value=12, 
        help="Valor alto (12-15) genera curvas limpias y profesionales."
    )

# --- 5. LÓGICA MATEMÁTICA ---
def procesar_archivo(uploaded_file, pk_ini, pk_fin, umbral, ventana_det, aplicar_smooth, ventana_smooth):
    try:
        df = pd.read_excel(uploaded_file, sheet_name=0)
        if 'Data No' in df.columns:
            df = df.dropna(subset=['Data No'])
        try:
            df_dcp = pd.read_excel(uploaded_file, sheet_name='DCP Data')
        except:
            df_dcp = pd.DataFrame()
    except Exception as e:
        st.error(f"Error de lectura: {e}")
        return None, None

    # Voltajes
    for col in ['On Voltage', 'Off Voltage']:
        if col in df.columns:
            df[col] = (df[col] * 1000).round(2)

    # Abscisas
    if 'Station No' in df.columns and len(df) > 0:
        df['Station No'] = np.round(np.linspace(pk_ini, pk_fin, len(df)), 3)

    # Coordenadas
    cols_coords = ['Latitude', 'Longitude']
    if all(col in df.columns for col in cols_coords):
        np.random.seed(42)
        aleatorio = np.random.uniform(0, 1, len(df))
        FACTOR = 1000000
        for col in cols_coords:
            df[col] = (df[col] + (aleatorio / FACTOR)).round(8)

    # Comentarios
    col_llave = 'Data No'
    col_destino = 'Comment'
    if not df_dcp.empty and col_llave in df.columns and len(df_dcp.columns) > 6:
        try:
            col_com = df_dcp.columns[6]
            df_dcp_unica = df_dcp.drop_duplicates(subset=[col_llave], keep='first')
            df = pd.merge(df, df_dcp_unica[[col_llave, col_com]], on=col_llave, how='left')
            df[col_destino] = df[col_com].fillna('')
            if col_com != col_destino:
                df.drop(columns=[col_com], inplace=True)
        except:
            pass

    # Ortografía
    correcciones = {"valvula": "Válvula", "anodo": "Ánodo", "potencial": "Potencial", "estacion": "Estación"}
    if col_destino in df.columns:
        for err, corr in correcciones.items():
            df[col_destino] = df[col_destino].astype(str).str.replace(err, corr, regex=False)

    # Limpieza
    log_cambios = {}
    for col in ['On Voltage', 'Off Voltage']:
        if col in df.columns:
            mediana_local = df[col].rolling(window=ventana_det, center=True, min_periods=1).median()
            diferencia = np.abs(df[col] - mediana_local)
            es_pico = diferencia > umbral
            df.loc[es_pico, col] = mediana_local[es_pico]
            picos_borrados = es_pico.sum()
            
            if aplicar_smooth:
                df[col] = df[col].rolling(window=ventana_smooth, center=True, min_periods=1).mean().round(2)
            
            log_cambios[col] = picos_borrados

    return df, log_cambios

# --- 6. INTERFAZ VISUAL ---
archivo = st.file_uploader("📂 Cargar Archivo Excel (Survey Data)", type=['xlsx'])

if archivo is not None:
    if st.button("🚀 PROCESAR Y VERIFICAR", use_container_width=True):
        with st.spinner('Aplicando ingeniería de datos...'):
            df_final, log = procesar_archivo(
                archivo, pk_inicial, pk_final, umbral_pico, 
                ventana_deteccion, activar_suavizado, ventana_suavizado
            )
            
            if df_final is not None:
                st.success("✅ Procesamiento Completado Exitosamente")
                
                # --- GRÁFICA CORPORATIVA ---
                st.subheader("📊 Perfil de Potenciales (Vista Previa)")
                
                if 'Station No' in df_final.columns:
                    datos_grafica = df_final[['Station No', 'On Voltage', 'Off Voltage']].melt(
                        id_vars='Station No', var_name='Tipo', value_name='mV'
                    )
                    
                    # COLORES CORPORATIVOS EN LA GRÁFICA
                    # ON = Azul Oscuro (Estándar) | OFF = ROJO EMPRESA (#B8233E)
                    scale_colors = alt.Scale(domain=['On Voltage', 'Off Voltage'], range=['#004E98', '#B8233E'])
                    
                    base = alt.Chart(datos_grafica).encode(
                        x=alt.X('Station No', title='Distancia (m)'),
                        y=alt.Y('mV', title='Potencial (mV)', scale=alt.Scale(zero=False)),
                        color=alt.Color('Tipo', scale=scale_colors)
                    )

                    linea = base.mark_line(strokeWidth=2)
                    puntos = base.mark_circle(size=60, opacity=0).encode(tooltip=['Station No', 'mV', 'Tipo'])
                    
                    chart = (linea + puntos).properties(height=500).interactive()
                    st.altair_chart(chart, use_container_width=True)
                    st.caption("💡 Zoom habilitado con rueda del mouse.")
                
                # Métricas
                c1, c2, c3 = st.columns(3)
                c1.metric("Datos Suavizados (ON)", log.get('On Voltage', 0))
                c2.metric("Datos Suavizados (OFF)", log.get('Off Voltage', 0))
                
                # Descarga
                buffer = io.BytesIO()
                df_final.to_excel(buffer, index=False)
                
                st.download_button(
                    label="📥 DESCARGAR REPORTE OFICIAL",
                    data=buffer,
                    file_name="Reporte_CIPS_Procesado.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True,
                    type="primary" # Hace el botón rojo corporativo
                )