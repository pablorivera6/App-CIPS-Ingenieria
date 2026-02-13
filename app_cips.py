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
def procesar_geospacial(df, df_dcp, ruta_mapa, umbral_outlier):
    status_log = []
    try:
        df = df.rename(columns={
            "Dist From Start": "PK_equipo", "On Voltage": "On_V", "Off Voltage": "Off_V",
            "Latitude": "Lat", "Longitude": "Long", "Comment": "Comentario",
            "DCP/Feature/DCVG Anomaly": "Anomalia"
        })
        
        for coord in ["Lat", "Long"]:
            if df[coord].isna().any():
                validos = df.dropna(subset=[coord, "PK_equipo"])
                if not validos.empty:
                    modelo = LinearRegression()
                    modelo.fit(validos[["PK_equipo"]], validos[coord])
                    mask = df[coord].isna()
                    df.loc[mask, coord] = modelo.predict(df.loc[mask, ["PK_equipo"]])
                    status_log.append(f"ℹ️ Interpoladas {mask.sum()} coordenadas en {coord}.")

        t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        df["X"], df["Y"] = t.transform(df["Long"].values, df["Lat"].values)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs=3857)

        ducto = gpd.read_file(ruta_mapa)
        if ducto.crs is None: ducto = ducto.set_crs(epsg=4326)
        ducto = ducto.to_crs(3857)

        lineas = []
        for geom in ducto.geometry:
            if isinstance(geom, LineString): lineas.append(geom)
            elif isinstance(geom, MultiLineString): lineas.extend(geom.geoms)
        
        merged = linemerge(lineas)
        if isinstance(merged, MultiLineString):
            merged = max(merged.geoms, key=lambda x: x.length)
        
        linea_ref = merged
        status_log.append(f"✅ Ducto cargado ({round(linea_ref.length/1000, 2)} km).")

        gdf["geom_snap"] = gdf.geometry.apply(lambda p: linea_ref.interpolate(linea_ref.project(p)))
        gdf["Dist_Eje_m"] = gdf.geometry.distance(gdf["geom_snap"])
        gdf["geometry"] = gdf["geom_snap"]
        gdf["PK_geom_m"] = gdf.geometry.apply(lambda p: linea_ref.project(p))

        df_val = gdf[["PK_equipo", "PK_geom_m"]].dropna()
        if not df_val.empty:
            corr = df_val["PK_equipo"].corr(df_val["PK_geom_m"])
            if corr < 0:
                gdf["PK_geom_m"] = linea_ref.length - gdf["PK_geom_m"]
                status_log.append("🔄 Sentido Contraflujo corregido.")

        gdf["Station No"] = np.round(gdf["PK_geom_m"], 2)

        t_back = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        gdf["Longitude"], gdf["Latitude"] = t_back.transform(gdf.geometry.x.values, gdf.geometry.y.values)

        gdf["On_mV"] = gdf["On_V"] * 1000
        gdf["Off_mV"] = gdf["Off_V"] * 1000
        
        for col in ["On_mV", "Off_mV"]:
            med = gdf[col].rolling(15, center=True, min_periods=1).median()
            delta = np.abs(gdf[col] - med)
            gdf.loc[delta > umbral_outlier, col] = med[delta > umbral_outlier]

        cols = ["Station No", "Latitude", "Longitude", "On_mV", "Off_mV", "Dist_Eje_m", "Comentario", "Anomalia"]
        for c in cols: 
            if c not in gdf.columns: gdf[c] = ""
            
        return gdf[cols], status_log
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
        
        df_maestro = pd.DataFrame()
        if os.path.exists(archivo_nombres):
            try:
                # CAMBIO CLAVE: Usamos utf-8-sig para leer correctamente los nombres en español
                df_temp = pd.read_csv(archivo_nombres, sep=None, header=None, engine='python', encoding='utf-8-sig')
                
                # Fallback si el archivo está en Latin-1 (formato antiguo)
                if "Ã" in str(df_temp.iloc[0,1]):
                    df_temp = pd.read_csv(archivo_nombres, sep=None, header=None, engine='python', encoding='latin-1')

                if df_temp.shape[1] >= 3:
                    if "ID" in str(df_temp.iloc[0,0]).upper(): df_temp = df_temp.iloc[1:]
                    df_maestro["ID"] = df_temp.iloc[:, 0].astype(str).str.strip().str.replace('"', '')
                    df_maestro["Nombre"] = df_temp.iloc[:, 1].astype(str).str.strip().str.replace('"', '')
                    df_maestro["Distrito"] = df_temp.iloc[:, 2].astype(str).str.strip().str.replace('"', '')
                    st.success(f"CSV cargado: {len(df_maestro)} registros.")
            except: st.error("Error leyendo nombres.csv")

        if not df_maestro.empty:
            distritos = sorted(df_maestro["Distrito"].unique())
            dist_sel = st.selectbox("1. Distrito:", distritos)
            
            df_f = df_maestro[df_maestro["Distrito"] == dist_sel]
            opciones = dict(zip(df_f["Nombre"], df_f["ID"]))
            nombre_sel = st.selectbox("2. Infraestructura:", sorted(opciones.keys()))
            
            id_buscado = opciones[nombre_sel].strip()
            opcion1 = os.path.join(carpeta, id_buscado)
            opcion2 = os.path.join(carpeta, f"{id_buscado}.gpkg")
            
            if os.path.exists(opcion1): ruta_final_ducto = opcion1
            elif os.path.exists(opcion2): ruta_final_ducto = opcion2
            
            if ruta_final_ducto:
                st.success(f"✅ Mapa listo: {id_buscado}")
            else:
                st.error(f"❌ No se encuentra el archivo '{id_buscado}.gpkg' en la carpeta ductos.")

    umbral = st.slider("Umbral Limpieza (mV)", 10, 300, 100)

# --- 6. INTERFAZ ---
archivo = st.file_uploader("📂 Cargar Excel", type=['xlsx'])
if archivo and st.button("🚀 PROCESAR"):
    with st.spinner("Procesando..."):
        try:
            df_raw = pd.read_excel(archivo, sheet_name=0)
            df_dcp = pd.read_excel(archivo, sheet_name='DCP Data') if 'DCP Data' in pd.ExcelFile(archivo).sheet_names else pd.DataFrame()
            
            if modo == "Avanzado (Con Ducto LRS)":
                if ruta_final_ducto:
                    df_final, logs = procesar_geospacial(df_raw, df_dcp, ruta_final_ducto, umbral)
                    with st.expander("Detalles"):
                        for m in logs: st.write(m)
                    if df_final is None: st.stop()
                else:
                    st.error("Seleccione un ducto."); st.stop()
            else:
                df_final = df_raw.copy().rename(columns={"On Voltage": "On_V", "Off Voltage": "Off_V"})
                df_final["On_mV"], df_final["Off_mV"] = df_final["On_V"]*1000, df_final["Off_V"]*1000
                df_final["Station No"] = np.round(np.linspace(0, 1000, len(df_final)), 2)

            st.subheader("📊 Perfil de Potenciales")
            st.line_chart(df_final.set_index('Station No')[['On_mV', 'Off_mV']])
            
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="openpyxl") as w:
                df_final.to_excel(w, sheet_name="Survey Data", index=False)
            st.download_button("📥 DESCARGAR", out, "CIPS_Procesado.xlsx")
        except Exception as e:
            st.error(f"Error general: {e}")
