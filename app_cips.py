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

        # D. CARGAR DUCTO (MODO LOCAL)
        try:
            ducto = gpd.read_file(ruta_lectura_ducto)
        except Exception as e:
            return None, [f"❌ Error leyendo mapa extraído: {str(e)}"]

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
        
        # --- DIAGNÓSTICO (INVISIBLE SI TODO VA BIEN) ---
        debug_log = []
        
        # 1. LEER CSV DE NOMBRES (ROBUST V4.0)
        df_maestro = pd.DataFrame()
        
        if os.path.exists(archivo_nombres):
            try:
                # Intento A: Punto y Coma
                df_temp = pd.read_csv(archivo_nombres, sep=';', header=None, dtype=str, on_bad_lines='skip', encoding='latin-1')
                if df_temp.shape[1] < 2:
                    # Intento B: Coma
                    df_temp = pd.read_csv(archivo_nombres, sep=',', header=None, dtype=str, on_bad_lines='skip', encoding='latin-1')

                if df_temp.shape[1] >= 3:
                    # Detectar si hay titulos en la primera fila
                    fila0 = str(df_temp.iloc[0,0]).upper()
                    start_row = 1 if ("ID" in fila0 or "ARCHIVO" in fila0) else 0
                    
                    df_temp = df_temp.iloc[start_row:]
                    
                    df_maestro["Archivo"] = df_temp.iloc[:, 0].astype(str).str.strip().str.replace('"', '')
                    df_maestro["Nombre"] = df_temp.iloc[:, 1].astype(str).str.strip().str.replace('"', '')
                    df_maestro["Distrito"] = df_temp.iloc[:, 2].astype(str).str.strip().str.replace('"', '')
                    
                    # Asegurar extensión y limpiar
                    df_maestro["Archivo"] = df_maestro["Archivo"].apply(lambda x: x.split('.')[0] + ".gpkg")
                    debug_log.append(f"✅ CSV cargado: {len(df_maestro)} registros.")
                else:
                    debug_log.append(f"⚠️ CSV con formato extraño: {df_temp.shape[1]} columnas.")
            except Exception as e:
                debug_log.append(f"❌ Error CSV: {e}")
        else:
            debug_log.append("❌ No se encuentra nombres.csv")

        # 2. MENU EN CASCADA CON EXTRACCIÓN AUTOMÁTICA
        if not df_maestro.empty and os.path.exists(archivo_zip):
            lista_distritos = sorted(df_maestro["Distrito"].dropna().unique())
            
            if lista_distritos:
                distrito_sel = st.selectbox("1. Distrito:", lista_distritos)
                
                df_filtrado = df_maestro[df_maestro["Distrito"] == distrito_sel]
                opciones = dict(zip(df_filtrado["Nombre"], df_filtrado["Archivo"]))
                
                if opciones:
                    nombre_sel = st.selectbox("2. Infraestructura:", sorted(opciones.keys()))
                    archivo_objetivo = opciones[nombre_sel] 
                    
                    # --- EXTRACCIÓN AUTOMÁTICA (PLAN Z) ---
                    # 1. Buscamos el archivo en el ZIP
                    archivo_encontrado = None
                    try:
                        with zipfile.ZipFile(archivo_zip, 'r') as z:
                            todos_los_archivos = z.namelist()
                            for f in todos_los_archivos:
                                # Compara el nombre final ignorando carpetas
                                if os.path.basename(f) == archivo_objetivo:
                                    archivo_encontrado = f
                                    break
                            
                            if archivo_encontrado:
                                # 2. Lo extraemos a un archivo temporal local
                                nombre_temp = "temp_ducto_activo.gpkg"
                                with open(nombre_temp, 'wb') as f_out:
                                    f_out.write(z.read(archivo_encontrado))
                                
                                ruta_final_ducto = nombre_temp
                                st.caption(f"✅ Mapa listo: `{archivo_objetivo}`")
                            else:
                                st.error(f"❌ No se encontró '{archivo_objetivo}' en el ZIP.")
                                debug_log.append(f"Buscaba: {archivo_objetivo}")
                                debug_log.append(f"Archivos en ZIP (primeros 5): {todos_los_archivos[:5]}")
                    
                    except Exception as e:
                        st.error("Error al descomprimir.")
                        debug_log.append(f"Error ZIP: {e}")
                else:
                    st.warning("Sin datos.")
        
        # Muestra el log solo si hay errores o dudas
        with st.expander("🛠️ Diagnóstico (Abrir si falla)"):
            for l in debug_log: st.write(l)
            if os.path.exists(archivo_zip):
                 with zipfile.ZipFile(archivo_zip, 'r') as z:
                     st.write("Contenido ZIP:", z.namelist()[:10])

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
                if df_final is None: st.stop()
            else:
                st.error("⚠️ El mapa no se pudo cargar. Revise el Diagnóstico."); st.stop()
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
