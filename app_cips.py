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
import zipfile

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
def procesar_geospacial(df, df_dcp, ruta_lectura_ducto, umbral_outlier):
    status_log = []
    
    try:
        # A. PREPARAR DATOS
        df = df.rename(columns={
            "Dist From Start": "PK_equipo", "On Voltage": "On_V", "Off Voltage": "Off_V",
            "Latitude": "Lat", "Longitude": "Long", "Comment": "Comentario",
            "DCP/Feature/DCVG Anomaly": "Anomalia"
        })
        
        # B. INTERPOLAR GPS FALTANTE
        for coord in ["Lat", "Long"]:
            if df[coord].isna().any():
                validos = df.dropna(subset=[coord, "PK_equipo"])
                if not validos.empty:
                    modelo = LinearRegression()
                    modelo.fit(validos[["PK_equipo"]], validos[coord])
                    mask = df[coord].isna()
                    df.loc[mask, coord] = modelo.predict(df.loc[mask, ["PK_equipo"]])
                    status_log.append(f"ℹ️ Interpoladas {mask.sum()} coordenadas en {coord}.")

        # C. PROYECTAR A METROS
        t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        df["X"], df["Y"] = t.transform(df["Long"].values, df["Lat"].values)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs=3857)

        # D. CARGAR DUCTO
        try:
            ducto = gpd.read_file(ruta_lectura_ducto)
        except Exception as e:
            return None, [f"❌ Error leyendo ducto: {str(e)}"]

        if ducto.crs is None: ducto = ducto.set_crs(epsg=4326)
        ducto = ducto.to_crs(3857)

        # E. UNIR TRAMOS
        lineas = []
        for geom in ducto.geometry:
            if isinstance(geom, LineString): lineas.append(geom)
            elif isinstance(geom, MultiLineString): lineas.extend(geom.geoms)
        
        merged = linemerge(lineas)
        if isinstance(merged, MultiLineString):
            merged = max(merged.geoms, key=lambda x: x.length)
            status_log.append("⚠️ Ducto discontinuo. Se usó el tramo más largo.")
        
        linea_ref = merged
        status_log.append(f"✅ Ducto cargado ({round(linea_ref.length/1000, 2)} km).")

        # F. SNAP & LRS
        gdf["geom_snap"] = gdf.geometry.apply(lambda p: linea_ref.interpolate(linea_ref.project(p)))
        gdf["Dist_Eje_m"] = gdf.geometry.distance(gdf["geom_snap"])
        gdf["geometry"] = gdf["geom_snap"]
        gdf["PK_geom_m"] = gdf.geometry.apply(lambda p: linea_ref.project(p))

        # G. SENTIDO AUTO
        df_val = gdf[["PK_equipo", "PK_geom_m"]].dropna()
        if not df_val.empty:
            corr = df_val["PK_equipo"].corr(df_val["PK_geom_m"])
            if corr < 0:
                gdf["PK_geom_m"] = linea_ref.length - gdf["PK_geom_m"]
                status_log.append("🔄 Sentido Contraflujo corregido.")

        gdf["Station No"] = np.round(gdf["PK_geom_m"], 2)

        # H. COORDENADAS FINALES
        t_back = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        gdf["Longitude"], gdf["Latitude"] = t_back.transform(gdf.geometry.x.values, gdf.geometry.y.values)

        # I. DATOS FINALES
        gdf["On_mV"] = gdf["On_V"] * 1000
        gdf["Off_mV"] = gdf["Off_V"] * 1000
        
        # Limpieza
        for col in ["On_mV", "Off_mV"]:
            med = gdf[col].rolling(15, center=True, min_periods=1).median()
            delta = np.abs(gdf[col] - med)
            gdf.loc[delta > umbral_outlier, col] = med[delta > umbral_outlier]

        cols = ["Station No", "Latitude", "Longitude", "On_mV", "Off_mV", "Dist_Eje_m", "Comentario", "Anomalia"]
        for c in cols: 
            if c not in gdf.columns: gdf[c] = ""
            
        return gdf[cols], status_log

    except Exception as e:
        return None, [f"❌ Error Crítico: {str(e)}"]

# --- 5. BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    modo = st.radio("Modo:", ["Básico (Manual)", "Avanzado (Con Ducto LRS)"])
    ruta_final_ducto = None
    
    if modo == "Avanzado (Con Ducto LRS)":
        st.info("Ajuste LRS Activo")
        
        carpeta = "ductos"
        archivo_zip = os.path.join(carpeta, "ductos.zip")
        archivo_nombres = os.path.join(carpeta, "nombres.csv")
        
        # --- LECTURA NUCLEAR DE CSV (POR POSICIÓN, SIN NOMBRES) ---
        df_maestro = pd.DataFrame()
        
        if os.path.exists(archivo_nombres):
            try:
                # 1. Leer SIN encabezados (header=None) para que Pandas no se confunda con títulos
                # Probar punto y coma (;)
                df_temp = pd.read_csv(archivo_nombres, sep=';', header=None, dtype=str, on_bad_lines='skip', encoding='latin-1')
                
                # Si falló, probar coma (,)
                if df_temp.shape[1] < 2:
                    df_temp = pd.read_csv(archivo_nombres, sep=',', header=None, dtype=str, on_bad_lines='skip', encoding='latin-1')

                # 2. Verificar si tenemos al menos 3 columnas
                if df_temp.shape[1] >= 3:
                    # 3. Eliminar la primera fila SI parece ser un título (Si dice "ID" o "TRAMO")
                    primer_valor = str(df_temp.iloc[0,0]).upper()
                    if "ID" in primer_valor or "TRAMO" in primer_valor or "ARCHIVO" in primer_valor:
                        df_temp = df_temp.iloc[1:] # Borramos la fila 0 (títulos)
                    
                    # 4. ASIGNACIÓN POR POSICIÓN (ESTO NO FALLA)
                    # Columna 0 -> Archivo
                    # Columna 1 -> Nombre Bonito
                    # Columna 2 -> Distrito
                    df_maestro["Archivo"] = df_temp.iloc[:, 0].astype(str).str.strip().str.replace('"', '')
                    df_maestro["Nombre"] = df_temp.iloc[:, 1].astype(str).str.strip().str.replace('"', '')
                    df_maestro["Distrito"] = df_temp.iloc[:, 2].astype(str).str.strip().str.replace('"', '')
                    
                    # Asegurar extensión .gpkg
                    df_maestro["Archivo"] = df_maestro["Archivo"].apply(lambda x: x if str(x).lower().endswith(".gpkg") else f"{x}.gpkg")
                
                else:
                    st.error(f"El archivo CSV tiene solo {df_temp.shape[1]} columnas. Se necesitan 3 (Archivo, Nombre, Distrito).")
                    st.write("Vista previa de lo que leyó Python:", df_temp.head()) # Ayuda visual si falla

            except Exception as e:
                st.error(f"Error leyendo CSV: {e}")

        # 2. MENU EN CASCADA
        if not df_maestro.empty and os.path.exists(archivo_zip):
            lista_distritos = sorted(df_maestro["Distrito"].dropna().unique())
            
            if lista_distritos:
                distrito_sel = st.selectbox("1. Distrito:", lista_distritos)
                
                df_filtrado = df_maestro[df_maestro["Distrito"] == distrito_sel]
                # Crear diccionario
                opciones = dict(zip(df_
