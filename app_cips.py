import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import altair as alt

# Librerías Geoespaciales
import geopandas as gpd
from shapely.ops import linemerge
from shapely.geometry import LineString, MultiLineString
from pyproj import Transformer
from sklearn.linear_model import LinearRegression

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Portal Ingeniería CIPS", page_icon="🔒", layout="wide")

# --- 2. SISTEMA DE SEGURIDAD (LOGIN) ---
def check_password():
    """Retorna True si el usuario ingresó la clave correcta."""
    def password_entered():
        if st.session_state["password"] == "CIPS2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
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
#  LÓGICA DE INFRAESTRUCTURA (CARGA DINÁMICA DESDE 'DUCTOS')
# =========================================================

@st.cache_data
def cargar_mapa_activos():
    """
    Lee el CSV de infraestructura dentro de la carpeta 'ductos'.
    Ruta esperada: ductos/nombres.csv
    """
    carpeta_base = "ductos"
    archivo_infra = os.path.join(carpeta_base, "nombres.csv")
    
    mapa = {}

    if not os.path.exists(archivo_infra):
        return {"Error": {"Archivo 'ductos/nombres.csv' no encontrado": ""}}

    try:
        # Intentar leer con UTF-8 primero (Estándar moderno)
        try:
            df_infra = pd.read_csv(archivo_infra, sep=';', encoding='utf-8')
        except:
            # Si falla (ej. Excel antiguo), intentar con Latin-1
            df_infra = pd.read_csv(archivo_infra, sep=';', encoding='latin-1')

        # Iteramos por cada fila del CSV
        for _, row in df_infra.iterrows():
            # Limpieza de datos (strip para quitar espacios extra)
            raw_dist = str(row['DISTRITO']).strip().upper()  
            nombre_tramo = str(row['TRAMO']).strip()         
            id_tramo = str(row['ID TRAMO']).strip()          
            
            # Formatear nombre del Distrito
            num_dist = raw_dist.replace('D', '').strip().zfill(2)
            nombre_distrito = f"Distrito {num_dist}"
            
            # RUTA DEL ARCHIVO GPKG (dentro de carpeta ductos)
            ruta_gpkg = os.path.join(carpeta_base, f"{id_tramo}")
            
            if nombre_distrito not in mapa:
                mapa[nombre_distrito] = {}
            
            mapa[nombre_distrito][nombre_tramo] = ruta_gpkg
            
        return dict(sorted(mapa.items()))
        
    except Exception as e:
        st.error(f"Error leyendo 'nombres.csv': {e}")
        return {}

# Cargamos el mapa al iniciar la app
MAPA_DE_ACTIVOS = cargar_mapa_activos()

# =========================================================
#  LÓGICA GEOESPACIAL (PROCESAMIENTO)
# =========================================================

def procesar_geometria_lrs(df, ruta_activo):
    """
    Realiza el snapping, cálculo de PK geométrico y corrección de coordenadas.
    """
    try:
        # 1. Normalización de Nombres
        df = df.rename(columns={
            "Dist From Start": "PK_equipo",
            "Latitude": "Lat",
            "Longitude": "Long"
        })

        # 2. Interpolación de Coordenadas Faltantes
        for coord in ["Lat", "Long"]:
            mask = df[coord].isna()
            if mask.any() and df.loc[~mask].shape[0] > 2:
                modelo = LinearRegression()
                modelo.fit(df.loc[~mask, ["PK_equipo"]], df.loc[~mask, coord])
                df.loc[mask, coord] = modelo.predict(df.loc[mask, ["PK_equipo"]])

        if df["Lat"].isna().all():
            return df, "Error: No hay coordenadas GPS válidas en el archivo."
            
        # 3. Conversión a Métrico (Web Mercator)
        t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        df["X"], df["Y"] = t.transform(df["Long"].values, df["Lat"].values)

        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs=3857)

        # 4. Carga del Ducto (Referencia)
        try:
            ducto = gpd.read_file(ruta_activo)
        except Exception as e:
            return df, f"Error cargando archivo geo: {str(e)}"

        if ducto.crs is None:
            ducto = ducto.set_crs(epsg=4326)
        ducto = ducto.to_crs(3857)

        # Unificar geometrías
        lineas_simples = []
        for geom in ducto.geometry:
            if isinstance(geom, LineString):
                lineas_simples.append(geom)
            elif isinstance(geom, MultiLineString):
                for parte in geom.geoms:
                    if isinstance(parte, LineString):
                        lineas_simples.append(parte)
        
        if not lineas_simples:
            return df, "Error: El archivo de referencia no tiene líneas válidas."

        merged = linemerge(lineas_simples)
        if isinstance(merged, MultiLineString):
            linea = max(merged.geoms, key=lambda x: x.length) 
        else:
            linea = merged

        # 5. Snap y PK Geométrico
        gdf["geom_snap"] = gdf.geometry.apply(lambda p: linea.interpolate(linea.project(p)))
        gdf["PK_geom_m"] = gdf.geometry.apply(lambda p: linea.project(p))

        # 6. Detección de Sentido
        df_pk = gdf[["PK_equipo", "PK_geom_m"]].dropna()
        if len(df_pk) > 5:
            corr = df_pk["PK_equipo"].corr(df_pk["PK_geom_m"])
            if corr < 0:
                gdf["PK_geom_m"] = linea.length - gdf["PK_geom_m"]

        # 7. Lat/Long Corregidas (Retorno a WGS84)
        t_back = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        gdf["Longitude"], gdf["Latitude"] = t_back.transform(
            gdf["geom_snap"].x.values, gdf["geom_snap"].y.values
        )

        # 8. Asignar Station No oficial y limpiar
        gdf["Station No"] = gdf["PK_geom_m"].round(2)
        
        cols_drop = ["X", "Y", "geometry", "geom_snap", "PK_equipo", "Lat", "Long"]
        gdf = gdf.drop(columns=[c for c in cols_drop if c in gdf.columns], errors='ignore')

        return pd.DataFrame(gdf), None 

    except Exception as e:
        return df, f"Error interno geoespacial: {str(e)}"

# =========================================================
#  APP PRINCIPAL (UI)
# =========================================================

# --- 3. ENCABEZADO ---
col_logo, col_titulo = st.columns([1, 6])
with col_logo:
    try:
        st.image("logo.png", use_container_width=True) 
    except:
        st.markdown("## ⚡") 

with col_titulo:
    st.title("Procesador de Integridad CIPS")
    st.markdown("Plataforma de Ingeniería | **Análisis Geoespacial y Reportes**")

st.markdown("---")

# --- 4. BARRA LATERAL (MODIFICADA CON BÚSQUEDA INTELIGENTE) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.subheader("1. Selección de Activo")
    
    if "Error" in MAPA_DE_ACTIVOS:
        st.error("❌ No se encontró 'ductos/nombres.csv'")
        distrito_sel = None
        ruta_geo = ""
    else:
        # Ordenamos los distritos
        lista_distritos = sorted(list(MAPA_DE_ACTIVOS.keys()))
        distrito_sel = st.selectbox("Distrito", lista_distritos)
        
        ruta_geo = ""
        if distrito_sel:
            tramos_dict = MAPA_DE_ACTIVOS[distrito_sel]
            ramal_sel = st.selectbox("Ramal / Sector", list(tramos_dict.keys()))
            
            # ID exacto del tramo que buscamos (Ej: T_OBTU)
            id_buscado = os.path.basename(tramos_dict[ramal_sel])
            carpeta_ductos = "ductos"
            
            # BÚSQUEDA INTELIGENTE: Ignora mayúsculas/minúsculas y busca la extensión
            archivo_encontrado = None
            if os.path.exists(carpeta_ductos):
                for archivo in os.listdir(carpeta_ductos):
                    # Comparamos el nombre del archivo en minúsculas
                    if archivo.lower().startswith(id_buscado.lower()) and archivo.lower().endswith(('.gpkg', '.shp')):
                        archivo_encontrado = os.path.join(carpeta_ductos, archivo)
                        break
            
            if archivo_encontrado:
                ruta_geo = archivo_encontrado
                ext = os.path.splitext(archivo_encontrado)[1]
                st.caption(f"✅ Archivo Geo detectado ({ext})")
            else:
                # Ruta dummy para que falle controlado
                ruta_geo = ""
                st.caption(f"❌ Archivo no encontrado para el ID: {id_buscado}")
                st.info(f"Asegúrate de que exista un archivo llamado '{id_buscado}.gpkg' en la carpeta 'ductos'.")

    st.divider()
    
    st.subheader("2. Calibración de Limpieza")
    st.info("Ajuste los filtros para eliminar ruido eléctrico.")
    umbral_pico = st.slider("Sensibilidad (mV)", 5, 100, 15)
    ventana_deteccion = st.slider("Ancho del Pico (Vecinos)", 3, 11, 9, step=2)
    
    st.subheader("3. Estética del Reporte")
    activar_suavizado = st.checkbox("Aplicar Suavizado Final", value=True)
    ventana_suavizado = st.slider("Nivel de Suavizado", 2, 20, 12)

    with st.expander("Opciones Manuales (Si falla Geo)"):
        pk_inicial = st.number_input("PK Inicial (m)", value=0.0, step=100.0)
        pk_final = st.number_input("PK Final (m)", value=1000.0, step=100.0)

# --- 5. LÓGICA DE PROCESAMIENTO UNIFICADO ---
def procesar_archivo_completo(uploaded_file, ruta_geo, umbral, ventana_det, aplicar_smooth, ventana_smooth):
    log_errores_geo = []
    
    # A. Lectura de todas las hojas
    try:
        xls = pd.ExcelFile(uploaded_file)
        # Hoja principal (asumimos index 0)
        df_survey = pd.read_excel(xls, sheet_name=0) 
        
        # Leer hojas extra para preservarlas
        hojas_extra = {}
        for sheet in xls.sheet_names:
            if sheet != xls.sheet_names[0]:
                hojas_extra[sheet] = pd.read_excel(xls, sheet_name=sheet)
        
        df_dcp = hojas_extra.get('DCP Data', pd.DataFrame())

    except Exception as e:
        st.error(f"Error crítico leyendo archivo Excel: {e}")
        return None, None, None

    # B. Procesamiento Geoespacial
    usar_metodo_manual = False
    
    if ruta_geo and os.path.exists(ruta_geo):
        with st.status("🗺️ Realizando alineación geoespacial...", expanded=True) as status:
            st.write(f"Procesando contra: {os.path.basename(ruta_geo)}")
            df_survey, error = procesar_geometria_lrs(df_survey, ruta_geo)
            
            if error:
                st.warning(f"⚠️ Fallo Geoespacial: {error}. Cambiando a modo manual.")
                usar_metodo_manual = True
                status.update(label="Usando método manual", state="error")
            else:
                st.write("✅ Snapping completado.")
                st.write("✅ Coordenadas corregidas.")
                status.update(label="Procesamiento Geo Exitoso", state="complete")
    else:
        st.warning(f"⚠️ No se encontró el archivo de referencia geográfico. Usando modo manual.")
        usar_metodo_manual = True

    # C. Procesamiento Manual (Fallback)
    if usar_metodo_manual:
        # Voltajes
        for col in ['On Voltage', 'Off Voltage']:
            if col in df_survey.columns:
                df_survey[col] = (df_survey[col] * 1000).round(2)
        
        # PK Manual
        if len(df_survey) > 0:
            df_survey['Station No'] = np.round(np.linspace(pk_inicial, pk_final, len(df_survey)), 3)
        
        # Coords Aleatorias (Simulación visual)
        cols_coords = ['Latitude', 'Longitude']
        if all(col in df_survey.columns for col in cols_coords):
            np.random.seed(42)
            aleatorio = np.random.uniform(0, 1, len(df_survey))
            for col in cols_coords:
                df_survey[col] = (df_survey[col] + (aleatorio / 1000000)).round(8)
    else:
        # Si fue geoespacial, asegurarnos que voltajes estén en mV
        for col in ['On Voltage', 'Off Voltage']:
            if col in df_survey.columns:
                # Si promedio es pequeño (<100), asumimos Voltios y convertimos a mV
                if df_survey[col].abs().mean() < 100: 
                    df_survey[col] = (df_survey[col] * 1000).round(2)

    # D. Integración de Comentarios (DCP)
    col_llave = 'Data No'
    col_destino = 'Comment'
    if not df_dcp.empty and col_llave in df_survey.columns:
        try:
            # Intentar encontrar columna de comentario (usualmente col 6)
            col_com = df_dcp.columns[6] if len(df_dcp.columns) > 6 else None
            
            if col_com:
                df_dcp_unica = df_dcp.drop_duplicates(subset=[col_llave], keep='first')
                df_survey = pd.merge(df_survey, df_dcp_unica[[col_llave, col_com]], on=col_llave, how='left')
                
                # Si ya existía columna Comment, rellenar nulos, si no crearla
                if col_destino in df_survey.columns:
                    df_survey[col_destino] = df_survey[col_destino].fillna(df_survey[col_com])
                else:
                    df_survey[col_destino] = df_survey[col_com].fillna('')
                
                if col_com != col_destino:
                    df_survey.drop(columns=[col_com], inplace=True)
        except:
            pass
    
    # E. Corrección Ortográfica
    correcciones = {"valvula": "Válvula", "anodo": "Ánodo", "potencial": "Potencial", "estacion": "Estación"}
    if col_destino in df_survey.columns:
        for err, corr in correcciones.items():
            df_survey[col_destino] = df_survey[col_destino].astype(str).str.replace(err, corr, regex=False)

    # F. Limpieza de Señal (Picos y Suavizado)
    log_cambios = {}
    for col in ['On Voltage', 'Off Voltage']:
        if col in df_survey.columns:
            # Detección de Picos
            mediana_local = df_survey[col].rolling(window=ventana_det, center=True, min_periods=1).median()
            diferencia = np.abs(df_survey[col] - mediana_local)
            es_pico = diferencia > umbral
            
            # Reemplazar picos
            df_survey.loc[es_pico, col] = mediana_local[es_pico]
            picos_borrados = es_pico.sum()
            
            # Suavizado Estético
            if aplicar_smooth:
                df_survey[col] = df_survey[col].rolling(window=ventana_smooth, center=True, min_periods=1).mean().round(2)
            
            log_cambios[col] = picos_borrados

    return df_survey, hojas_extra, log_cambios

# --- 6. INTERFAZ DE CARGA Y RESULTADOS ---
archivo = st.file_uploader("📂 Cargar Archivo Excel (Survey Data)", type=['xlsx'])

if archivo is not None:
    if st.button("🚀 PROCESAR Y VERIFICAR", use_container_width=True):
        
        with st.spinner('Procesando datos e infraestructura...'):
            df_final, hojas_guardadas, log = procesar_archivo_completo(
                archivo, ruta_geo, umbral_pico, 
                ventana_deteccion, activar_suavizado, ventana_suavizado
            )
        
        if df_final is not None:
            st.success("✅ Procesamiento Completado Exitosamente")
            
            # --- GRÁFICA ---
            st.subheader("📊 Perfil de Potenciales (Vista Previa)")
            
            if 'Station No' in df_final.columns:
                datos_grafica = df_final[['Station No', 'On Voltage', 'Off Voltage']].melt(
                    id_vars='Station No', var_name='Tipo', value_name='mV'
                )
                
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
            
            # --- GENERACIÓN EXCEL ---
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                # 1. Hoja Procesada
                df_final.to_excel(writer, sheet_name="Survey Data Procesada", index=False)
                
                # 2. Restaurar hojas originales
                if hojas_guardadas:
                    for nombre_hoja, df_hoja in hojas_guardadas.items():
                        df_hoja.to_excel(writer, sheet_name=nombre_hoja, index=False)
            
            st.download_button(
                label="📥 DESCARGAR REPORTE OFICIAL",
                data=buffer.getvalue(),
                file_name="Reporte_CIPS_Procesado.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary"
            )
