import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Predictor PM2.5", page_icon="🌫️", layout="wide")

# Título de la aplicación
st.title("🌫️ Predictor de Calidad del Aire - PM2.5")
st.markdown("Modelo Random Forest entrenado con datos históricos de contaminación")

# Información de provincias simplificada
PROVINCIAS_INFO = {
    1: {"nombre": "Azuay"},
    2: {"nombre": "Pichincha"}
}

# Cargar el modelo
@st.cache_resource
def load_model():
    try:
        model = joblib.load('modelo_pm25.pkl')
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Cargar modelo
with st.spinner('Cargando modelo PM2.5...'):
    model = load_model()

if model is not None:
    st.success("Modelo PM2.5 cargado correctamente!")
    
    # Mostrar información del modelo
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.subheader("📊 Información del Modelo")
        st.write(f"**Algoritmo:** Random Forest Regressor")
        st.write(f"**Variable objetivo:** PM2.5")
        st.write(f"**Características:** 7 variables")
    
    with col_info2:
        st.subheader("🎯 Características usadas")
        st.write("""
        - Año 
        - Mes 
        - Estacionalidad (mes_sin/mes_cos)
        - Provincia
        - PM2.5 mes anterior 
        - PM2.5 hace 2 meses 
        """)
    
    st.markdown("---")
    
    # SECCIÓN DE PREDICCIÓN
    st.subheader(" Realizar Predicción de PM2.5")
    st.write("Ingresa los valores históricos para predecir el PM2.5 del mes actual:")
    
    # Dividir en columnas para mejor organización
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📅 Información Temporal y Geográfica**")
        anio = st.number_input("Año", value=2024, min_value=2000, max_value=2030)
        mes = st.number_input("Mes", value=6, min_value=1, max_value=12)
        
        # Selector de provincia simplificado
        provincia_seleccionada = st.selectbox(
            "Provincia",
            options=list(PROVINCIAS_INFO.keys()),
            format_func=lambda x: f"{PROVINCIAS_INFO[x]['nombre']}",
            index=0,
            help="Selecciona la provincia para la predicción"
        )
        
        cod_prov = provincia_seleccionada
        provincia_nombre = PROVINCIAS_INFO[cod_prov]['nombre']
        
        # Calcular estacionalidad circular automáticamente
        mes_sin = np.sin(2 * np.pi * mes / 12)
        mes_cos = np.cos(2 * np.pi * mes / 12)
    
    with col2:
        st.markdown("**📊 Valores Históricos de PM2.5**")
        pm25_lag1 = st.number_input("PM2.5 mes anterior (μg/m³)", 
                                   value=15.0, min_value=0.0, max_value=500.0, step=0.1,
                                   help="Valor de PM2.5 del mes inmediatamente anterior")
        pm25_lag2 = st.number_input("PM2.5 hace 2 meses (μg/m³)", 
                                   value=14.0, min_value=0.0, max_value=500.0, step=0.1,
                                   help="Valor de PM2.5 de hace dos meses")
        
        # Mostrar resumen de inputs
        st.markdown("**📋 Resumen de Entradas**")
        st.write(f"**Fecha:** {mes}/{anio}")
        st.write(f"**Provincia:** {provincia_nombre}")
        st.write(f"**PM2.5 histórico:** {pm25_lag1} → {pm25_lag2} μg/m³")
    
    # Botón de predicción
    if st.button("🎯 Predecir PM2.5", type="primary", use_container_width=True):
        try:
            # Crear array de entrada en el orden EXACTO que espera el modelo
            input_features = np.array([[
                anio,        # Año de la medición
                mes,         # Código del mes
                mes_sin,     # Estacionalidad seno
                mes_cos,     # Estacionalidad coseno
                cod_prov,    # Código de provincia
                pm25_lag1,   # PM2.5 del mes anterior
                pm25_lag2    # PM2.5 de hace 2 meses
            ]])
            
            # Mostrar los datos que se enviarán al modelo (para transparencia)
            st.markdown("---")
            st.subheader("📤 Datos Enviados al Modelo")
            
            feature_names = [
                'ANIO', 'CODMES', 'mes_sin', 'mes_cos', 'COD_PROV', 
                'PM2.5_lag1', 'PM2.5_lag2'
            ]
            
            input_df = pd.DataFrame(input_features, columns=feature_names)
            
            # Mostrar código de provincia en lugar de nombre
            display_df = input_df.copy()
            
            st.dataframe(display_df.style.format({
                'ANIO': '{:.0f}',
                'CODMES': '{:.0f}',
                'mes_sin': '{:.4f}',
                'mes_cos': '{:.4f}',
                'COD_PROV': '{:.0f}',
                'PM2.5_lag1': '{:.1f}',
                'PM2.5_lag2': '{:.1f}'
            }))
            
            # Realizar predicción
            prediction = model.predict(input_features)
            pm25_pred = prediction[0]
            
            # Mostrar resultado
            st.markdown("---")
            st.subheader("🎯 Resultado de la Predicción")
            
            # Métrica principal
            col_metric, col_interpret = st.columns([1, 2])
            
            with col_metric:
                st.metric(
                    label="**Concentración Predicha de PM2.5**", 
                    value=f"{pm25_pred:.1f} μg/m³",
                    delta=f"{pm25_pred - pm25_lag1:+.1f} vs mes anterior"
                )
            
            with col_interpret:
                # Interpretación de la calidad del aire según TULSMA (Libro VI Anexo 4)
                if pm25_pred <= 15:
                    st.success("**✅ BUENA**")
                    st.write("Cumple con la normativa TULSMA (Libro VI Anexo 4)")
                else:
                    st.error("**🚨 MALA**")
                    st.write("**NO CUMPLE** con la normativa TULSMA (Libro VI Anexo 4) - Límite: 15 μg/m³")
            
            # Información adicional y análisis de tendencia
            st.info(f"""
            **📋 Análisis de la Predicción:**
            - **Período:** Mes {mes} del {anio}
            - **Provincia:** {provincia_nombre}
            - **PM2.5 mes anterior:** {pm25_lag1} μg/m³
            - **PM2.5 hace 2 meses:** {pm25_lag2} μg/m³
            - **Normativa TULSMA:** {'✅ CUMPLE' if pm25_pred <= 15 else '❌ NO CUMPLE'}
            """)
            
            
            # Alerta especial si no cumple la normativa
            if pm25_pred > 15:
                st.error(f"""
                **🚨 ALERTA - EXCEDE LÍMITE NORMATIVO**
                
                La predicción actual ({pm25_pred:.1f} μg/m³) **EXCEDE** el límite establecido en el 
                **Libro VI Anexo 4 TULSMA** (15 μg/m³).
                
                **Recomendaciones para la población:**
                - 🚫 **Evite actividades prolongadas al aire libre**
                - 😷 **Use mascarilla si debe salir**
                - 🏠 **Mantenga ventanas cerradas**
                - 👶 **Grupos sensibles (niños, ancianos, personas con enfermedades respiratorias) deben tomar precauciones extras**
                """)
                
        except Exception as e:
            st.error(f"❌ Error en la predicción: {e}")
            st.info("💡 Verifica que el modelo esté entrenado con estas 7 características exactas")

else:
    st.error("""
    ❌ No se pudo cargar el modelo. Verifica que:
    - El archivo 'modelo_pm25.pkl' esté en la carpeta correcta
    - El modelo use exactamente 7 características de entrada
    - Las versiones de scikit-learn sean compatibles
    """)

# Información adicional en la barra lateral
with st.sidebar:
    st.header("ℹ️ Información del Modelo")
    st.markdown("""
    **7 Características de Entrada:**
    
    1. **ANIO** - Año de la medición
    2. **CODMES** - Mes (1-12)
    3. **mes_sin** - Estacionalidad seno
    4. **mes_cos** - Estacionalidad coseno  
    5. **COD_PROV** - Provincia
    6. **PM2.5_lag1** - PM2.5 mes anterior
    7. **PM2.5_lag2** - PM2.5 hace 2 meses
    """)
    
    st.markdown("---")
    st.markdown("**📍 Provincias:**")
    for codigo, info in PROVINCIAS_INFO.items():
        st.write(f"• **{codigo}:** {info['nombre']}")
    
    st.markdown("---")
    st.markdown("**📊 Normativa TULSMA (Libro VI Anexo 4):**")
    st.write("• **✅ BUENA:** ≤ 15 μg/m³")
    st.write("• **🚨 MALA:** > 15 μg/m³")
    
    st.markdown("---")
    st.markdown("**🛡️ Recomendaciones cuando es MALA:**")
    st.write("• 🚫 Evitar actividades al aire libre")
    st.write("• 😷 Usar mascarilla si debe salir")
    st.write("• 🏠 Mantener ventanas cerradas")
    st.write("• 👶 Proteger grupos sensibles")
    
    st.markdown("---")
    st.markdown("**🎯 Objetivo del Modelo:**")
    st.write("Predecir el valor de PM2.5 para el mes actual usando datos históricos y estacionalidad")

# Pie de página
st.markdown("---")
st.caption("Modelo Random Forest entrenado con datos históricos de calidad del aire | 7 características de entrada | Normativa: Libro VI Anexo 4 TULSMA")