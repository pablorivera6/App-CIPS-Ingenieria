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
import shutil

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
            # GEOPANDAS LEE ARCHIVO YA EXTRAÍDO (LOCAL)
            ducto = gpd.read_file(ruta_lectura_ducto)
        except Exception as e:
            return None, [f"❌ Error leyendo mapa: {str(e)}"]

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
        
        # 1. LEER CSV DE NOMBRES (SIN NOMBRES DE COLUMNA)
        df_maestro = pd.DataFrame()
        
        if os.path.exists(archivo_nombres):
            try:
                # Intento leer ; o ,
                df_temp = pd.read_csv(archivo_nombres, sep=';', header=None, dtype=str, on_bad_lines='skip', encoding='latin-1')
                if df_temp.shape[1] < 2:
                    df_temp = pd.read_csv(archivo_nombres, sep=',', header=None, dtype=str, on_bad_lines='skip', encoding='latin-1')

                # Asignación de columnas por POSICIÓN
                if df_temp.shape[1] >= 3:
                    # Si la primera fila parece titulo, la borramos
                    primer_val = str(df_temp.iloc[0,0]).upper()
                    if "ID" in primer_val or "TRAMO" in primer_val or "ARCHIVO" in primer_val:
                        df_temp = df_temp.iloc[1:]

                    # Columna 0: Archivo, Columna 1: Nombre, Columna 2: Distrito
                    df_maestro["Archivo"] = df_temp.iloc[:, 0].astype(str).str.strip().str.replace('"', '')
                    df_maestro["Nombre"] = df_temp.iloc[:, 1].astype(str).str.strip().str.replace('"', '')
                    df_maestro["Distrito"] = df_temp.iloc[:, 2].astype(str).str.strip().str.replace('"', '')
                    
                    # Aseguramos extensión .gpkg
                    df_maestro["Archivo"] = df_maestro["Archivo"].apply(lambda x: x if str(x).lower().endswith(".gpkg") else f"{x}.gpkg")

            except Exception as e:
                st.error(f"Error CSV: {e}")

        # 2. MENU EN CASCADA CON EXTRACCIÓN REAL
        if not df_maestro.empty and os.path.exists(archivo_zip):
            lista_distritos = sorted(df_maestro["Distrito"].dropna().unique())
            
            if lista_distritos:
                distrito_sel = st.selectbox("1. Distrito:", lista_distritos)
                
                df_filtrado = df_maestro[df_maestro["Distrito"] == distrito_sel]
                opciones = dict(zip(df_filtrado["Nombre"], df_filtrado["Archivo"]))
                
                if opciones:
                    nombre_sel = st.selectbox("2. Infraestructura:", sorted(opciones.keys()))
                    archivo_objetivo = opciones[nombre_sel] 
                    
                    # --- ESTRATEGIA DE EXTRACCIÓN ---
                    if st.button("🗺️ Cargar Mapa Seleccionado"):
                        with st.spinner("Descomprimiendo mapa..."):
                            try:
                                with zipfile.ZipFile(archivo_zip, 'r') as z:
                                    # Buscar el archivo dentro del ZIP (no importa la carpeta)
                                    archivo_encontrado = None
                                    for f in z.namelist():
                                        if os.path.basename(f) == archivo_objetivo:
                                            archivo_encontrado = f
                                            break
                                    
                                    if archivo_encontrado:
                                        # Extraerlo temporalmente a la raíz
                                        z.extract(archivo_encontrado, ".")
                                        ruta_final_ducto = archivo_encontrado
                                        st.session_state['ruta_mapa_temp'] = ruta_final_ducto
                                        st.success(f"Mapa cargado: {archivo_objetivo}")
                                    else:
                                        st.error(f"No se encontró '{archivo_objetivo}' dentro del ZIP.")
                            except Exception as e:
                                st.error(f"Error extrayendo: {e}")

                    # Recuperar ruta si ya se extrajo
                    if 'ruta_mapa_temp' in st.session_state:
                         ruta_final_ducto = st.session_state['ruta_mapa_temp']
                         st.caption(f"📍 Usando mapa: `{os.path.basename(ruta_final_ducto)}`")

                else:
                    st.warning("Sin datos.")
        else:
            # FALLBACK
            st.warning("⚠️ Usando modo directo.")
            if os.path.exists(archivo_zip):
                try:
                    with zipfile.ZipFile(archivo_zip, 'r') as z:
                        lista = [f for f in z.namelist() if f.endswith('.gpkg') and not f.startswith('__MACOSX')]
                    sel = st.selectbox("Archivo:", sorted(lista))
                    
                    # Extracción directa
                    if st.button("Cargar Mapa"):
                        with zipfile.ZipFile(archivo_zip, 'r') as z:
                            z.extract(sel, ".")
                            st.session_state['ruta_mapa_temp'] = sel
                            st.rerun()
                    
                    if 'ruta_mapa_temp' in st.session_state:
                        ruta_final_ducto = st.session_state['ruta_mapa_temp']
                except: pass

    else:
        st.subheader("Tramo Manual")
        pk_a = st.number_input("PK 1", value=14000.0)
        pk_b = st.number_input("PK 2", value=15000.0)
        sentido = st.radio("Sentido", ["Ascendente", "Contraflujo"])

    st.divider()
    umbral = st.slider("Umbral Limpieza (mV)", 10, 300, 100)

# --- 6. INTERFAZ ---
archivo = st.file_uploader("📂 Cargar Excel", type=['xlsx'])

if archivo and st.button("🚀 PROCESAR"):
    with st.spinner("Procesando..."):
        try:
            df_raw = pd.read_excel(archivo, sheet_name=0)
            df_dcp = pd.read_excel(archivo, sheet_name='DCP Data') if 'DCP Data' in pd.ExcelFile(archivo).sheet_names else pd.DataFrame()
        except:
            st.error("Error en Excel."); st.stop()

        if modo == "Avanzado (Con Ducto LRS)":
            if ruta_final_ducto and os.path.exists(ruta_final_ducto):
                df_final, logs = procesar_geospacial(df_raw, df_dcp, ruta_final_ducto, umbral)
                with st.expander("Detalles", expanded=True):
                    for m in logs: st.write(m)
                
                # Limpieza (borrar mapa temporal después de usar)
                # try: os.remove(ruta_final_ducto) 
                # except: pass
                
                if df_final is None: st.stop()
            else:
                st.error("⚠️ Primero debe dar click en el botón 'Cargar Mapa Seleccionado' en el menú lateral."); st.stop()
        else:
            # Modo Básico
            df_final = df_raw.copy()
            df_final = df_final.rename(columns={"On Voltage": "On_V", "Off Voltage": "Off_V"})
            df_final["On_mV"] = df_final["On_V"] * 1000
            df_final["Off_mV"] = df_final["Off_V"] * 1000
            
            pks = np.linspace(min(pk_a, pk_b), max(pk_a, pk_b), len(df_final))
            if sentido == "Contraflujo": pks = pks[::-1]
            df_final["Station No"] = np.round(pks, 2)
            
            for c in ["On_mV", "Off_mV"]:
                m = df_final[c].rolling(15, center=True).median()
                df_final.loc[np.abs(df_final[c]-m)>umbral, c] = m[np.abs(df_final[c]-m)>umbral]

        # Gráfica
        st.subheader("📊 Perfil de Potenciales")
        data = df_final[['Station No', 'On_mV', 'Off_mV']].melt('Station No', var_name='Tipo', value_name='mV')
        chart = alt.Chart(data).mark_line().encode(
            x=alt.X('Station No', title='Distancia (m)'),
            y=alt.Y('mV', scale=alt.Scale(zero=False)),
            color=alt.Color('Tipo', scale=alt.Scale(range=['#004E98', '#B8233E'])),
            tooltip=['Station No', 'mV']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

        # Descarga
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            df_final.to_excel(w, sheet_name="Survey Data", index=False)
            if not df_dcp.empty:
                df_dcp.to_excel(w, sheet_name="DCP Data", index=False)
        st.download_button("📥 DESCARGAR", out, "CIPS_Procesado.xlsx", "application/vnd.ms-excel", type="primary")
