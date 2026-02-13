import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import altair as alt
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
from pyproj import Transformer
from sklearn.linear_model import LinearRegression

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Portal Ingeniería CIPS", page_icon="⚡", layout="wide")

# --- 2. SEGURIDAD ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "CIPS2026": 
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("<style>.stTextInput > div > div > input {text-align: center;}</style>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.warning("🔒 Acceso Restringido")
            st.text_input("Ingrese Contraseña:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.error("❌ Contraseña incorrecta")
        st.text_input("Ingrese Contraseña:", type="password", on_change=password_entered, key="password")
        return False
    return True

if not check_password():
    st.stop()

# --- 3. ENCABEZADO ---
col_logo, col_titulo = st.columns([1, 6])
with col_logo:
    try:
        st.image("logo.png", use_container_width=True) 
    except:
        st.markdown("## ⚡")
with col_titulo:
    st.title("Procesador CIPS + LRS")
    st.markdown("**Sistema de Integridad y Ajuste Espacial**")
st.markdown("---")

# --- 4. MOTOR LRS (GEOMÁTICA) ---
def procesar_geospacial(df_original, df_dcp, ruta_mapa, umbral_outlier):
    status_log = []
    try:
        df = df_original.copy()

        # Identificar columnas críticas
        lat_col = next((c for c in df.columns if "lat" in c.lower()), "Latitude")
        lon_col = next((c for c in df.columns if "long" in c.lower()), "Longitude")
        pk_col = next((c for c in df.columns if "dist" in c.lower() or "pk" in c.lower()), "Dist From Start")
        on_col = next((c for c in df.columns if "on" in c.lower() and "volt" in c.lower()), "On Voltage")
        off_col = next((c for c in df.columns if "off" in c.lower() and "volt" in c.lower()), "Off Voltage")
        com_col = next((c for c in df.columns if "comment" in c.lower()), "Comment")

        # A. CRUCE CON DCP DATA (BUSCARV)
        if not df_dcp.empty:
            # Buscamos la columna de anomalías en DCP
            anom_col = next((c for c in df_dcp.columns if "anomaly" in c.lower() or "feature" in c.lower()), None)
            pk_dcp_col = next((c for c in df_dcp.columns if "dist" in c.lower() or "pk" in c.lower()), None)
            
            if anom_col and pk_dcp_col:
                # Limpiamos para el cruce
                df_dcp_clean = df_dcp[[pk_dcp_col, anom_col]].dropna().copy()
                df_dcp_clean[pk_dcp_col] = pd.to_numeric(df_dcp_clean[pk_dcp_col])
                
                # Merge por aproximación (PK más cercano)
                df = pd.merge_asof(
                    df.sort_values(pk_col),
                    df_dcp_clean.sort_values(pk_dcp_col),
                    left_on=pk_col,
                    right_on=pk_dcp_col,
                    direction='nearest',
                    tolerance=1.0 # Máximo 1 metro de diferencia
                )
                
                # Concatenar comentario original con la anomalía encontrada
                df[com_col] = df[com_col].fillna("") + " | " + df[anom_col].fillna("")
                status_log.append("🔗 Cruce con DCP Data realizado exitosamente.")

        # B. PROYECCIÓN LRS
        t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x, y = t.transform(df[lon_col].values, df[lat_col].values)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x, y), crs=3857)

        ducto = gpd.read_file(ruta_mapa)
        if ducto.crs is None: ducto = ducto.set_crs(epsg=4326)
        ducto = ducto.to_crs(3857)

        lineas = [geom for geom in ducto.geometry if isinstance(geom, (LineString, MultiLineString))]
        linea_ref = linemerge(lineas)
        if isinstance(linea_ref, MultiLineString):
            linea_ref = max(linea_ref.geoms, key=lambda x: x.length)
        
        status_log.append(f"✅ Ducto cargado ({round(linea_ref.length/1000, 2)} km).")

        # C. CÁLCULOS
        gdf["geom_snap"] = gdf.geometry.apply(lambda p: linea_ref.interpolate(linea_ref.project(p)))
        gdf["Dist_Eje_m"] = np.round(gdf.geometry.distance(gdf["geom_snap"]), 2)
        
        pk_calculado = gdf["geom_snap"].apply(lambda p: linea_ref.project(p))
        if df[pk_col].corr(pk_calculado) < 0:
            pk_calculado = linea_ref.length - pk_calculado
            status_log.append("🔄 Sentido Contraflujo corregido.")

        gdf["Station No"] = np.round(pk_calculado, 2)
        gdf["On_mV"] = gdf[on_col] * 1000
        gdf["Off_mV"] = gdf[off_col] * 1000
        
        # Limpieza de Outliers
        for col in ["On_mV", "Off_mV"]:
            med = gdf[col].rolling(15, center=True, min_periods=1).median()
            delta = np.abs(gdf[col] - med)
            gdf.loc[delta > umbral_outlier, col] = med[delta > umbral_outlier]

        if 'geometry' in gdf.columns: del gdf['geometry']
        if 'geom_snap' in gdf.columns: del gdf['geom_snap']

        return gdf, status_log
    except Exception as e:
        return None, [f"❌ Error: {str(e)}"]

# --- 5. BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    modo = st.radio("Modo:", ["Básico (Manual)", "Avanzado (Con Ducto LRS)"])
    ruta_final_ducto = None
    
    if modo == "Avanzado (Con Ducto LRS)":
        carpeta = "ductos"
        archivo_nombres = os.path.join(carpeta, "nombres.csv")
        
        if os.path.exists(archivo_nombres):
            try:
                df_temp = pd.read_csv(archivo_nombres, sep=None, header=None, engine='python', encoding='utf-8-sig')
                if "ID" in str(df_temp.iloc[0,0]).upper(): df_temp = df_temp.iloc[1:]
                distritos = sorted(df_temp.iloc[:, 2].unique())
                dist_sel = st.selectbox("1. Distrito:", distritos)
                
                df_f = df_temp[df_temp.iloc[:, 2] == dist_sel]
                opciones = dict(zip(df_f.iloc[:, 1], df_f.iloc[:, 0]))
                nombre_sel = st.selectbox("2. Infraestructura:", sorted(opciones.keys()))
                
                id_buscado = opciones[nombre_sel].strip()
                opcion = os.path.join(carpeta, id_buscado if id_buscado.endswith(".gpkg") else f"{id_buscado}.gpkg")
                if os.path.exists(opcion):
                    ruta_final_ducto = opcion
                    st.success("✅ Mapa listo")
            except: pass

    umbral = st.slider("Umbral Limpieza (mV)", 10, 300, 100)

# --- 6. INTERFAZ ---
archivo = st.file_uploader("📂 Cargar Excel Original", type=['xlsx'])

if archivo and st.button("🚀 PROCESAR"):
    with st.spinner("Integrando datos de DCP y LRS..."):
        try:
            xls = pd.ExcelFile(archivo)
            df_survey = pd.read_excel(xls, sheet_name='Survey Data') if 'Survey Data' in xls.sheet_names else pd.read_excel(xls, sheet_name=0)
            df_dcp = pd.read_excel(xls, sheet_name='DCP Data') if 'DCP Data' in xls.sheet_names else pd.DataFrame()
            
            if modo == "Avanzado (Con Ducto LRS)":
                if ruta_final_ducto:
                    df_final, logs = procesar_geospacial(df_survey, df_dcp, ruta_final_ducto, umbral)
                    with st.expander("Ver Detalles del Proceso"):
                        for m in logs: st.write(m)
                else:
                    st.error("Seleccione un ducto."); st.stop()
            else:
                df_final = df_survey.copy()
                df_final["Station No"] = np.round(np.linspace(0, 1000, len(df_final)), 2)

            st.subheader("📊 Perfil de Potenciales")
            if "On_mV" in df_final.columns:
                st.line_chart(df_final.set_index('Station No')[['On_mV', 'Off_mV']])

            # EXPORTACIÓN MAESTRA
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                # 1. Survey Data Actualizada
                df_final.to_excel(writer, sheet_name="Survey Data", index=False)
                
                # 2. Re-escribir TODAS las otras hojas originales para no perder nada
                for sheet_name in xls.sheet_names:
                    if sheet_name != "Survey Data":
                        temp_df = pd.read_excel(xls, sheet_name=sheet_name)
                        temp_df.to_excel(writer, sheet_name=sheet_name, index=False)

            st.download_button(
                label="📥 DESCARGAR REPORTE CONSOLIDADO",
                data=output.getvalue(),
                file_name="CIPS_Final_Integrado.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
        except Exception as e:
            st.error(f"Error: {e}")
