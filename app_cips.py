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

# --- 0. MAPA DE ACTIVOS (CONFIGURACIÓN DE ARCHIVOS GEO) ---
# Aquí defines la ruta a tus archivos .gpkg o .shp en tu repositorio de GitHub
# Estructura: "Nombre que ve el usuario": "ruta/al/archivo.gpkg"
MAPA_DE_ACTIVOS = {
    "Distrito 1": {
        "Ramal La Virginia": "activos/linea_virginia.gpkg", 
        "Ramal Marsella": "activos/linea_marsella.shp",
    },
    "Distrito 2": {
        "Ramal Norte": "activos/linea_norte.gpkg",
    }
}

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
#  LÓGICA GEOESPACIAL (ADAPTADA DEL CÓDIGO DE TU COLEGA)
# =========================================================

def procesar_geometria_lrs(df, ruta_activo):
    """
    Toma el DataFrame crudo y la ruta del archivo geoespacial.
    Realiza el snapping, cálculo de PK geométrico y corrección de coordenadas.
    """
    try:
        # 1. Normalización de Nombres
        df = df.rename(columns={
            "Dist From Start": "PK_equipo",
            "Latitude": "Lat",
            "Longitude": "Long"
        })

        # 2. Interpolación de Coordenadas Faltantes (Regresión Lineal)
        for coord in ["Lat", "Long"]:
            mask = df[coord].isna()
            if mask.any() and df.loc[~mask].shape[0] > 2: # Necesitamos al menos 2 puntos
                modelo = LinearRegression()
                modelo.fit(df.loc[~mask, ["PK_equipo"]], df.loc[~mask, coord])
                df.loc[mask, coord] = modelo.predict(df.loc[mask, ["PK_equipo"]])

        # 3. Conversión a Métrico (Web Mercator)
        if df["Lat"].isna().all():
            return df, "Error: No hay coordenadas GPS válidas en el archivo."
            
        t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        df["X"], df["Y"] = t.transform(df["Long"].values, df["Lat"].values)

        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs=3857)

        # 4. Carga del Ducto (Referencia)
        try:
            ducto = gpd.read_file(ruta_activo)
        except Exception as e:
            return df, f"Error cargando archivo geo {ruta_activo}: {str(e)}"

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
            # Caso complejo: tomamos la línea más larga o unimos coords (simplificado)
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

        # 7. Lat/Long Corregidas
        t_back = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        gdf["Longitude_Corr"], gdf["Latitude_Corr"] = t_back.transform(
            gdf["geom_snap"].x.values, gdf["geom_snap"].y.values
        )

        # 8. Asignar Station No oficial
        gdf["Station No"] = gdf["PK_geom_m"].round(2)
        
        # Limpieza de columnas temporales
        cols_drop = ["X", "Y", "geometry", "geom_snap", "PK_equipo"]
        gdf = gdf.drop(columns=[c for c in cols_drop if c in gdf.columns], errors='ignore')

        return pd.DataFrame(gdf), None # Retorna DF y None (sin errores)

    except Exception as e:
        return df, f"Error en procesamiento geoespacial: {str(e)}"


# =========================================================
#  APP PRINCIPAL
# =========================================================

# --- 3. ENCABEZADO ---
col_logo, col_titulo = st.columns([1, 6])
with col_logo:
    st.markdown("## ⚡") # Placeholder si no hay logo.png
with col_titulo:
    st.title("Procesador de Integridad CIPS")
    st.markdown("Plataforma de Ingeniería | **Análisis Geoespacial y Reportes**")

st.markdown("---")

# --- 4. BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.subheader("1. Selección de Activo")
    distrito_sel = st.selectbox("Distrito", list(MAPA_DE_ACTIVOS.keys()))
    ramal_sel = st.selectbox("Ramal / Sector", list(MAPA_DE_ACTIVOS[distrito_sel].keys()))
    
    ruta_geo = MAPA_DE_ACTIVOS[distrito_sel][ramal_sel]
    st.caption(f"Referencia: `{ruta_geo}`")

    st.divider()
    
    st.subheader("2. Calibración de Limpieza")
    umbral_pico = st.slider("Sensibilidad (mV)", 5, 100, 15)
    ventana_deteccion = st.slider("Ancho del Pico (Vecinos)", 3, 11, 9, step=2)
    
    st.subheader("3. Estética del Reporte")
    activar_suavizado = st.checkbox("Aplicar Suavizado Final", value=True)
    ventana_suavizado = st.slider("Nivel de Suavizado", 2, 20, 12)

    # Inputs manuales (Solo se usan si falla el Geoespacial)
    with st.expander("Opciones Manuales (Fallback)"):
        pk_inicial = st.number_input("PK Inicial (m)", value=0.0)
        pk_final = st.number_input("PK Final (m)", value=1000.0)


# --- 5. LÓGICA HÍBRIDA (PROCESAR) ---
def procesar_archivo_completo(uploaded_file, ruta_geo, umbral, ventana_det, aplicar_smooth, ventana_smooth):
    log_errores_geo = []
    
    # A. Lectura de todas las hojas
    try:
        xls = pd.ExcelFile(uploaded_file)
        df_survey = pd.read_excel(xls, sheet_name=0) # Asumimos hoja 0 es Survey Data
        
        # Intentar leer otras hojas para preservarlas
        hojas_extra = {}
        for sheet in xls.sheet_names:
            if sheet != xls.sheet_names[0]:
                hojas_extra[sheet] = pd.read_excel(xls, sheet_name=sheet)
        
        # Específicamente necesitamos DCP para los comentarios
        df_dcp = hojas_extra.get('DCP Data', pd.DataFrame())

    except Exception as e:
        st.error(f"Error crítico leyendo archivo Excel: {e}")
        return None, None, None

    # B. Procesamiento Geoespacial (INTENTO PRINCIPAL)
    usar_metodo_manual = False
    
    if os.path.exists(ruta_geo):
        with st.status("🗺️ Realizando alineación geoespacial...", expanded=True) as status:
            st.write("Cargando archivo de referencia...")
            df_survey, error = procesar_geometria_lrs(df_survey, ruta_geo)
            
            if error:
                st.warning(f"⚠️ Fallo Geoespacial: {error}. Cambiando a modo manual.")
                log_errores_geo.append(error)
                usar_metodo_manual = True
            else:
                st.write("✅ Snapping completado.")
                st.write("✅ PKs geométricos calculados.")
                st.write("✅ Coordenadas corregidas.")
                status.update(label="Procesamiento Geo Exitoso", state="complete")
    else:
        st.warning(f"⚠️ No se encontró el archivo de referencia en: {ruta_geo}. Usando modo manual.")
        usar_metodo_manual = True

    # C. Procesamiento Manual (SI FALLA GEO O NO HAY ARCHIVO)
    if usar_metodo_manual:
        # Lógica original tuya de coordenadas aleatorias y linspace
        if 'On Voltage' in df_survey.columns: # Chequeo básico
             # Voltajes
            for col in ['On Voltage', 'Off Voltage']:
                if col in df_survey.columns:
                    df_survey[col] = (df_survey[col] * 1000).round(2)
            
            # PK
            if 'Station No' in df_survey.columns and len(df_survey) > 0:
                df_survey['Station No'] = np.round(np.linspace(pk_inicial, pk_final, len(df_survey)), 3)
            
            # Coords Aleatorias (Tu código original)
            cols_coords = ['Latitude', 'Longitude']
            if all(col in df_survey.columns for col in cols_coords):
                np.random.seed(42)
                aleatorio = np.random.uniform(0, 1, len(df_survey))
                FACTOR = 1000000
                for col in cols_coords:
                    df_survey[col] = (df_survey[col] + (aleatorio / FACTOR)).round(8)
    else:
        # Si funcionó el Geo, los voltajes ya deberían estar en V o mV? 
        # El script Geo no escala voltajes, lo hacemos aquí:
        for col in ['On Voltage', 'Off Voltage']:
            if col in df_survey.columns:
                 # Verificamos si están en V (pequeños) o mV (grandes) para no multiplicar doble
                if df_survey[col].abs().mean() < 100: 
                    df_survey[col] = (df_survey[col] * 1000).round(2)

    # D. Procesamiento de Comentarios (Tu código original)
    col_llave = 'Data No'
    col_destino = 'Comment'
    if not df_dcp.empty and col_llave in df_survey.columns:
        try:
            # Buscamos columnas relevantes en DCP
            col_com = df_dcp.columns[6] if len(df_dcp.columns) > 6 else None
            if col_com:
                df_dcp_unica = df_dcp.drop_duplicates(subset=[col_llave], keep='first')
                df_survey = pd.merge(df_survey, df_dcp_unica[[col_llave, col_com]], on=col_llave, how='left')
                df_survey[col_destino] = df_survey[col_com].fillna('')
                if col_com != col_destino:
                    df_survey.drop(columns=[col_com], inplace=True)
        except:
            pass
    
    # Ortografía
    correcciones = {"valvula": "Válvula", "anodo": "Ánodo", "potencial": "Potencial", "estacion": "Estación"}
    if col_destino in df_survey.columns:
        for err, corr in correcciones.items():
            df_survey[col_destino] = df_survey[col_destino].astype(str).str.replace(err, corr, regex=False)

    # E. Limpieza de Señal (Tu código original)
    log_cambios = {}
    for col in ['On Voltage', 'Off Voltage']:
        if col in df_survey.columns:
            # Filtro de Mediana
            mediana_local = df_survey[col].rolling(window=ventana_det, center=True, min_periods=1).median()
            diferencia = np.abs(df_survey[col] - mediana_local)
            es_pico = diferencia > umbral
            df_survey.loc[es_pico, col] = mediana_local[es_pico]
            picos_borrados = es_pico.sum()
            
            # Suavizado
            if aplicar_smooth:
                df_survey[col] = df_survey[col].rolling(window=ventana_smooth, center=True, min_periods=1).mean().round(2)
            
            log_cambios[col] = picos_borrados

    return df_survey, hojas_extra, log_cambios

# --- 6. INTERFAZ VISUAL ---
archivo = st.file_uploader("📂 Cargar Archivo Excel (Survey Data)", type=['xlsx'])

if archivo is not None:
    if st.button("🚀 PROCESAR E INSPECCIONAR", use_container_width=True):
        
        df_final, hojas_guardadas, log = procesar_archivo_completo(
            archivo, ruta_geo, umbral_pico, 
            ventana_deteccion, activar_suavizado, ventana_suavizado
        )
        
        if df_final is not None:
            st.success("✅ Procesamiento Completado")

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
            
            # Métricas
            c1, c2 = st.columns(2)
            c1.metric("Picos Eliminados (ON)", log.get('On Voltage', 0))
            c2.metric("Picos Eliminados (OFF)", log.get('Off Voltage', 0))

            # --- GENERACIÓN DEL EXCEL MULTI-HOJA ---
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                # 1. Hoja Procesada
                df_final.to_excel(writer, sheet_name="Survey Data Procesada", index=False)
                
                # 2. Hojas Originales (DCP, Info, etc.)
                if hojas_guardadas:
                    for nombre_hoja, df_hoja in hojas_guardadas.items():
                        df_hoja.to_excel(writer, sheet_name=nombre_hoja, index=False)
            
            st.download_button(
                label="📥 DESCARGAR REPORTE COMPLETO (CON PESTAÑAS)",
                data=buffer.getvalue(),
                file_name="Reporte_CIPS_Integrado.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary"
            )
