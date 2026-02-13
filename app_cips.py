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

# --- 4. MOTOR LRS ---
def procesar_geospacial(df_original, df_dcp, ruta_mapa, umbral_outlier):
    status_log = []
    try:
        df = df_original.copy()

        # Identificar columnas críticas dinámicamente
        lat_col = next((c for c in df.columns if "lat" in c.lower()), "Latitude")
        lon_col = next((c for c in df.columns if "long" in c.lower()), "Longitude")
        pk_col = next((c for c in df.columns if "dist" in c.lower() or "pk" in c.lower()), "Dist From Start")
        on_col = next((c for c in df.columns if "on" in c.lower() and "volt" in c.lower()), "On Voltage")
        off_col = next((c for c in df.columns if "off" in c.lower() and "volt" in c.lower()), "Off Voltage")
        com_col = next((c for c in df.columns if "comment" in c.lower()), "Comment")

        # A. FUSIÓN CON DCP DATA (BUSCARV AUTOMÁTICO)
        if not df_dcp.empty:
            anom_col = next((c for c in df_dcp.columns if "anomaly" in c.lower() or "feature" in c.lower()), None)
            pk_dcp_col = next((c for c in df_dcp.columns if "dist" in c.lower() or "pk" in c.lower()), None)
            
            if anom_col and pk_dcp_col:
                df_dcp_sub = df_dcp[[pk_dcp_col, anom_col]].dropna().copy()
                df_dcp_sub[pk_dcp_col] = pd.to_numeric(df_dcp_sub[pk_dcp_col])
                
                # Merge por PK más cercano
                df = pd.merge_asof(
                    df.sort_values(pk_col),
                    df_dcp_sub.sort_values(pk_dcp_col),
                    left_on=pk_col, right_on=pk_dcp_col,
                    direction='nearest', tolerance=1.5
                )
                
                # Concatenar anomalía al comentario
                df[com_col] = df[com_col].fillna("") + " | " + df[anom_col].fillna("")
                status_log.append("🔗 Información de DCP integrada en los comentarios.")

        # B. AJUSTE GEOGRÁFICO
        t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x, y = t.transform(df[lon_col].values, df[lat_col].values)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x, y), crs=3857)

        ducto = gpd.read_file(ruta_mapa)
        if ducto.crs is None: ducto = ducto.set_crs(epsg=4326)
        ducto = ducto.to_crs(3857)

        lineas = [g for g in ducto.geometry if isinstance(g, (LineString, MultiLineString))]
        linea_ref = linemerge(lineas)
        if isinstance(linea_ref, MultiLineString):
            linea_ref = max(linea_ref.geoms, key=lambda x: x.length)
        
        status_log.append(f"✅ Mapa cargado ({round(linea_ref.length/1000, 2)} km).")

        # C. CÁLCULOS LRS
        gdf["geom_snap"] = gdf.geometry.apply(lambda p: linea_ref.interpolate(linea_ref.project(p)))
        gdf["Dist_Eje_m"] = np.round(gdf.geometry.distance(gdf["geom_snap"]), 2)
        
        pk_calc = gdf["geom_snap"].apply(lambda p: linea_ref.project(p))
        if df[pk_col].corr(pk_calc) < 0:
            pk_calc = linea_ref.length - pk_calc
            status_log.append("🔄 Sentido Contraflujo corregido.")

        gdf["Station No"] = np.round(pk_calc, 2)
        
        # Potenciales en mV
        if on_col in gdf.columns: gdf["On_mV"] = gdf[on_col] * 1000
        if off_col in gdf.columns: gdf["Off_mV"] = gdf[off_col] * 1000
        
        for col in ["On_mV", "Off_mV"]:
            if col in gdf.columns:
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
    modo = st.radio("Modo:", ["Básico", "Avanzado (LRS)"])
    ruta_final_ducto = None
    
    if modo == "Avanzado (LRS)":
        carpeta = "ductos"
        archivo_nombres = os.path.join(carpeta, "nombres.csv")
        
        if os.path.exists(archivo_nombres):
            try:
                # Lector robusto de encoding para evitar "BolÃvar"
                df_temp = pd.read_csv(archivo_nombres, sep=None, header=None, engine='python', encoding='utf-8-sig')
                if "Ã" in str(df_temp.iloc[0,1]):
                    df_temp = pd.read_csv(archivo_nombres, sep=None, header=None, engine='python', encoding='latin-1')
                
                if df_temp.shape[1] >= 3:
                    if "ID" in str(df_temp.iloc[0,0]).upper(): df_temp = df_temp.iloc[1:]
                    
                    distritos = sorted(df_temp.iloc[:, 2].unique())
                    dist_sel = st.selectbox("1. Distrito:", distritos)
                    
                    df_f = df_temp[df_temp.iloc[:, 2] == dist_sel]
                    opciones = dict(zip(df_f.iloc[:, 1], df_f.iloc[:, 0]))
                    nombre_sel = st.selectbox("2. Infraestructura:", sorted(opciones.keys()))
                    
                    id_arch = opciones[nombre_sel].strip()
                    opc = os.path.join(carpeta, id_arch if id_arch.endswith(".gpkg") else f"{id_arch}.gpkg")
                    if os.path.exists(opc):
                        ruta_final_ducto = opc
                        st.success("✅ Mapa cargado")
            except: pass

    umbral = st.slider("Limpieza (mV)", 10, 300, 100)

# --- 6. INTERFAZ ---
archivo = st.file_uploader("📂 Cargar Excel Original", type=['xlsx'])

if archivo and st.button("🚀 PROCESAR"):
    with st.spinner("Integrando DCP y LRS..."):
        try:
            xls_in = pd.ExcelFile(archivo)
            # Buscamos la hoja de datos principal
            nombre_hoja_datos = "Survey Data" if "Survey Data" in xls_in.sheet_names else xls_in.sheet_names[0]
            df_survey = pd.read_excel(xls_in, sheet_name=nombre_hoja_datos)
            df_dcp = pd.read_excel(xls_in, sheet_name='DCP Data') if 'DCP Data' in xls_in.sheet_names else pd.DataFrame()
            
            if modo == "Avanzado (LRS)" and ruta_final_ducto:
                df_final, logs = procesar_geospacial(df_survey, df_dcp, ruta_final_ducto, umbral)
                with st.expander("Ver Detalles"):
                    for m in logs: st.write(m)
            else:
                df_final = df_survey.copy()
                df_final["Station No"] = np.round(np.linspace(0, 1000, len(df_final)), 2)

            st.subheader("📊 Gráfica de Potenciales")
            if "On_mV" in df_final.columns:
                st.line_chart(df_final.set_index('Station No')[['On_mV', 'Off_mV']])

            # EXPORTACIÓN ESPEJO (CONSERVA TODO)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                # Escribimos la hoja procesada
                df_final.to_excel(writer, sheet_name=nombre_hoja_datos, index=False)
                # Escribimos el resto de hojas originales EXACTAMENTE IGUAL
                for s in xls_in.sheet_names:
                    if s != nombre_hoja_datos:
                        pd.read_excel(xls_in, sheet_name=s).to_excel(writer, sheet_name=s, index=False)

            st.download_button(
                label="📥 DESCARGAR EXCEL CONSOLIDADO",
                data=output.getvalue(),
                file_name="Reporte_CIPS_Integrado.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
        except Exception as e:
            st.error(f"Error: {e}")
