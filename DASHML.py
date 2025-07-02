import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import joblib
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Monitoreo Generador",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tabla de referencias para información básica de fallas (solo información descriptiva)
FAULT_INFO = {
    'F01': {'parameter': 'Presión de Aceite', 'condition': '<2 psi', 'description': 'Bomba en mal estado o falta de aceite', 'type': 'Mecánica'},
    'F02': {'parameter': 'Presión de Aceite', 'condition': '>7 psi', 'description': 'Bomba en mal estado o exceso de aceite', 'type': 'Mecánica'},
    'F03': {'parameter': 'Voltaje de Batería', 'condition': '<10V', 'description': 'Batería descargada o en mal estado', 'type': 'Eléctrica'},
    'F04': {'parameter': 'Voltaje de Batería', 'condition': '>14V', 'description': 'Falla en cargador de batería o alternador', 'type': 'Eléctrica'},
    'F05': {'parameter': 'Voltaje Alternador', 'condition': '<12V', 'description': 'Bobinas abiertas o carbones desgastados', 'type': 'Eléctrica'},
    'F06': {'parameter': 'Voltaje Alternador', 'condition': '>16V', 'description': 'Daño en tarjeta reguladora o cortocircuito', 'type': 'Eléctrica'},
    'F07': {'parameter': 'Temperatura (Vacío)', 'condition': '<50°C', 'description': 'Termostato atascado o calentador de camisa dañado', 'type': 'Térmica'},
    'F08': {'parameter': 'Temperatura (Vacío)', 'condition': '>76°C', 'description': 'Radiador obstruido o falla de termostatos', 'type': 'Térmica'},
    'F09': {'parameter': 'Temperatura (Carga)', 'condition': '<70°C', 'description': 'Sensor de temperatura defectuoso', 'type': 'Térmica'},
    'F10': {'parameter': 'Temperatura (Carga)', 'condition': '>90°C', 'description': 'Obstrucción en radiador o bomba de agua defectuosa', 'type': 'Térmica'},
    'F11': {'parameter': 'Nivel de Refrigerante', 'condition': 'BAJO', 'description': 'Fuga en sellos o radiador', 'type': 'Térmica'}
}

# Matriz de decisión para urgencia basada en parámetros del sensor
def determine_urgency_and_actions(fault_code, sensor_values):
    """Determina la urgencia y acciones basado en los valores de sensores y recursos adicionales"""
    
    # Parámetros críticos para cada tipo de falla
    urgency_matrix = {
        'F01': {'critical_threshold': 1.5, 'immediate_threshold': 2.5},
        'F02': {'critical_threshold': 8.0, 'immediate_threshold': 7.5},
        'F03': {'critical_threshold': 9.0, 'immediate_threshold': 10.5},
        'F04': {'critical_threshold': 15.0, 'immediate_threshold': 14.5},
        'F05': {'critical_threshold': 11.0, 'immediate_threshold': 12.5},
        'F06': {'critical_threshold': 17.0, 'immediate_threshold': 16.5},
        'F07': {'critical_threshold': 45.0, 'immediate_threshold': 50.0},
        'F08': {'critical_threshold': 80.0, 'immediate_threshold': 76.0},
        'F09': {'critical_threshold': 65.0, 'immediate_threshold': 70.0},
        'F10': {'critical_threshold': 95.0, 'immediate_threshold': 90.0},
        'F11': {'critical_threshold': 0.0, 'immediate_threshold': 0.5}
    }
    
    # Mapeo de parámetros a valores de sensores
    param_mapping = {
        'F01': 'presion_aceite', 'F02': 'presion_aceite',
        'F03': 'voltaje_bateria', 'F04': 'voltaje_bateria',
        'F05': 'voltaje_alternador', 'F06': 'voltaje_alternador',
        'F07': 'temp_vacio', 'F08': 'temp_vacio',
        'F09': 'temp_carga', 'F10': 'temp_carga',
        'F11': 'nivel_refrigerante'
    }
    
    param_name = param_mapping.get(fault_code)
    if not param_name or param_name not in sensor_values:
        return 'Preventiva', 'Verificar sensor y realizar inspección general'
    
    current_value = sensor_values[param_name]
    thresholds = urgency_matrix.get(fault_code, {})
    
    # Determinar urgencia basada en los valores actuales
    if fault_code in ['F01', 'F03', 'F05', 'F07', 'F09', 'F11']:  # Valores bajos críticos
        if current_value <= thresholds.get('critical_threshold', 0):
            urgency = 'Crítica'
        elif current_value <= thresholds.get('immediate_threshold', 0):
            urgency = 'Inmediata'
        else:
            urgency = 'Preventiva'
    else:  # Valores altos críticos
        if current_value >= thresholds.get('critical_threshold', 999):
            urgency = 'Crítica'
        elif current_value >= thresholds.get('immediate_threshold', 999):
            urgency = 'Inmediata'
        else:
            urgency = 'Preventiva'
    
    # Determinar acciones basadas en la urgencia y el tipo de falla
    actions = get_maintenance_actions(fault_code, urgency, current_value)
    
    return urgency, actions

def get_maintenance_actions(fault_code, urgency, current_value):
    """Determina las acciones de mantenimiento basadas en recursos externos"""
    
    base_actions = {
        'F01': {
            'Crítica': 'PARAR INMEDIATAMENTE - Verificar nivel de aceite, reemplazar filtro, revisar bomba de aceite',
            'Inmediata': 'Agregar aceite, verificar fugas, programar cambio de filtro en 24h',
            'Preventiva': 'Monitorear presión, verificar nivel de aceite semanalmente'
        },
        'F02': {
            'Crítica': 'PARAR INMEDIATAMENTE - Drenar exceso de aceite, revisar válvula reguladora',
            'Inmediata': 'Verificar viscosidad del aceite, ajustar presión del sistema',
            'Preventiva': 'Monitorear presión, revisar calibración de sensores'
        },
        'F03': {
            'Crítica': 'PARAR INMEDIATAMENTE - Cargar batería, verificar alternador, revisar conexiones',
            'Inmediata': 'Cargar batería, limpiar bornes, verificar carga del alternador',
            'Preventiva': 'Monitorear voltaje, limpiar bornes mensualmente'
        },
        'F04': {
            'Crítica': 'PARAR INMEDIATAMENTE - Desconectar cargador, revisar regulador de voltaje',
            'Inmediata': 'Verificar regulador de voltaje, revisar alternador',
            'Preventiva': 'Monitorear voltaje, calibrar sistema de carga'
        },
        'F05': {
            'Crítica': 'PARAR INMEDIATAMENTE - Reemplazar alternador, revisar sistema eléctrico',
            'Inmediata': 'Cambiar carbones del alternador, verificar diodos',
            'Preventiva': 'Inspeccionar alternador, limpiar conexiones'
        },
        'F06': {
            'Crítica': 'PARAR INMEDIATAMENTE - Reemplazar regulador, revisar cortocircuitos',
            'Inmediata': 'Verificar regulador de voltaje, inspeccionar cableado',
            'Preventiva': 'Monitorear voltaje del alternador regularmente'
        },
        'F07': {
            'Crítica': 'Reemplazar termostato inmediatamente, verificar calentador',
            'Inmediata': 'Revisar termostato, probar calentador de camisa',
            'Preventiva': 'Monitorear temperatura, inspeccionar sistema de calefacción'
        },
        'F08': {
            'Crítica': 'PARAR INMEDIATAMENTE - Limpiar radiador, reemplazar termostatos',
            'Inmediata': 'Verificar ventilación, limpiar radiador',
            'Preventiva': 'Monitorear temperatura, mantenimiento preventivo del radiador'
        },
        'F09': {
            'Crítica': 'Reemplazar sensor de temperatura, calibrar sistema',
            'Inmediata': 'Calibrar sensor, verificar conexiones eléctricas',
            'Preventiva': 'Monitorear sensor, verificar calibración mensual'
        },
        'F10': {
            'Crítica': 'PARAR INMEDIATAMENTE - Limpieza completa del sistema, cambiar refrigerante',
            'Inmediata': 'Revisar bomba de agua, verificar ventilador',
            'Preventiva': 'Monitorear temperatura, mantenimiento del sistema de enfriamiento'
        },
        'F11': {
            'Crítica': 'PARAR INMEDIATAMENTE - Reponer refrigerante, reparar fugas urgentemente',
            'Inmediata': 'Reponer refrigerante, inspeccionar fugas menores',
            'Preventiva': 'Monitorear nivel, inspeccionar sistema regularmente'
        }
    }
    
    return base_actions.get(fault_code, {}).get(urgency, 'Consultar manual de mantenimiento')

# Colores por urgencia
URGENCY_COLORS = {
    'Crítica': '#FF4444',
    'Inmediata': '#FF8C00',
    'Preventiva': '#FFD700',
    'Ninguna': '#28A745'
}

@st.cache_data
def load_data():
    """Carga los datos del CSV"""
    try:
        df = pd.read_csv("Dataset_de_prueba__50_registros_ - Dataset_de_prueba__50_registros_.csv")
        return df
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo Dataset_de_prueba__50_registros_ - Dataset_de_prueba__50_registros_.csv")
        return None

def load_model():
    """Carga el modelo entrenado"""
    try:
        model_data = joblib.load("modelo_fallas.pkl")
        return model_data['model'], model_data['feature_columns'], model_data['target_columns']
    except FileNotFoundError:
        st.warning("⚠️ No se encontró el modelo entrenado. Ejecuta train_model.py primero.")
        return None, None, None

def predict_faults_with_model(model, feature_columns, target_columns, sample_data):
    """Predice fallas usando el modelo ML entrenado"""
    if model is None or feature_columns is None or target_columns is None:
        return [], []
    
    try:
        # Preparar los datos para el modelo
        df_sample = pd.DataFrame([sample_data], columns=feature_columns)
        
        # Realizar predicción
        predictions = model.predict(df_sample)[0]
        probabilities = model.predict_proba(df_sample)
        
        # Identificar fallas detectadas
        detected_faults = []
        fault_probabilities = []
        
        for i, fault_code in enumerate(target_columns):
            if predictions[i] == 1:
                detected_faults.append(fault_code)
                # Obtener probabilidad de la falla
                prob = probabilities[i][0][1] if len(probabilities[i][0]) > 1 else probabilities[i][0][0]
                fault_probabilities.append(prob)
        
        return detected_faults, fault_probabilities
        
    except Exception as e:
        st.error(f"Error en la predicción del modelo: {str(e)}")
        return [], []

def get_parameter_status(value, param_name):
    """Determina el estado de un parámetro"""
    ranges = {
        'presion_aceite': {'min': 2, 'max': 7, 'ideal': 5},
        'voltaje_bateria': {'min': 10, 'max': 14, 'ideal': 13},
        'voltaje_alternador': {'min': 12, 'max': 16, 'ideal': 14},
        'temp_vacio': {'min': 50, 'max': 76, 'ideal': 70},
        'temp_carga': {'min': 70, 'max': 90, 'ideal': 80},
        'nivel_refrigerante': {'min': 1, 'max': 1, 'ideal': 1}
    }

    if param_name not in ranges:
        return 'Normal', '#28A745'

    r = ranges[param_name]

    if value < r['min'] or value > r['max']:
        return 'Crítico', '#FF4444'
    elif abs(value - r['ideal']) > (r['max'] - r['min']) * 0.3:
        return 'Advertencia', '#FF8C00'
    else:
        return 'Normal', '#28A745'

def main():
    st.title("DASHBOARD MONITOREO GENERADOR - ML PREDICTIVO")
    st.markdown("---")

    # Cargar datos y modelo
    df = load_data()
    model, feature_columns, target_columns = load_model()

    if df is None:
        st.stop()

    # Sidebar para configuración
    st.sidebar.title("⚙️ Configuración")

    # Mostrar estado del modelo
    if model is not None:
        st.sidebar.success("🤖 Modelo ML Cargado")
    else:
        st.sidebar.error("❌ Modelo ML No Disponible")

    # Simulación de tiempo real
    auto_refresh = st.sidebar.checkbox("🔄 Actualización Automática", value=False)
    refresh_interval = st.sidebar.slider("Intervalo (segundos)", 1, 60, 10)

    # Selector de muestra actual
    if 'current_sample' not in st.session_state:
        st.session_state.current_sample = 0

    max_samples = len(df) - 1
    st.session_state.current_sample = st.sidebar.number_input(
        "Muestra Actual", 0, max_samples, st.session_state.current_sample
    )

    # Botones de control
    col1, col2 = st.sidebar.columns(2)
    if col1.button("⏮️ Anterior"):
        if st.session_state.current_sample > 0:
            st.session_state.current_sample -= 1
            st.rerun()

    if col2.button("⏭️ Siguiente"):
        if st.session_state.current_sample < max_samples:
            st.session_state.current_sample += 1
            st.rerun()

    # Obtener muestra actual
    current_row = df.iloc[st.session_state.current_sample]

    # Crear pestañas
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Monitoreo en Tiempo Real",
        "📈 Análisis Histórico",
        "⚠️ Gestión de Fallas",
        "🔧 Recomendaciones Inteligentes"
    ])

    with tab1:
        show_real_time_monitoring(current_row, model, feature_columns, target_columns)

    with tab2:
        show_historical_analysis(df)

    with tab3:
        show_fault_management_ml(current_row, model, feature_columns, target_columns)

    with tab4:
        show_recommendations_ml(current_row, model, feature_columns, target_columns)

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        if st.session_state.current_sample < max_samples:
            st.session_state.current_sample += 1
        else:
            st.session_state.current_sample = 0
        st.rerun()

def show_real_time_monitoring(current_row, model, feature_columns, target_columns):
    """Muestra el monitoreo en tiempo real"""
    st.header("📊 Estado Actual del Generador")

    # Timestamp simulado
    timestamp = datetime.now() - timedelta(minutes=st.session_state.current_sample)
    st.info(f"🕐 Última lectura: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    # Predicción en tiempo real
    if model is not None:
        sensor_data = [current_row[col] for col in feature_columns]
        detected_faults, probabilities = predict_faults_with_model(model, feature_columns, target_columns, sensor_data)
        
        if detected_faults:
            st.error(f"🚨 SE DETECTARON {len(detected_faults)} FALLA(S)")
        else:
            st.success("✅SIN FALLAS DETECTADAS")

    # Métricas principales (resto del código igual)
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    params = [
        ('presion_aceite', 'Presión Aceite', 'psi'),
        ('voltaje_bateria', 'Voltaje Batería', 'V'),
        ('voltaje_alternador', 'Voltaje Alternador', 'V'),
        ('temp_vacio', 'Temp. Vacío', '°C'),
        ('temp_carga', 'Temp. Carga', '°C'),
        ('nivel_refrigerante', 'Refrigerante', '')
    ]

    cols = [col1, col2, col3, col4, col5, col6]

    for i, (param, label, unit) in enumerate(params):
        value = current_row[param]
        status, color = get_parameter_status(value, param)

        with cols[i]:
            st.metric(
                label=label,
                value=f"{value:.1f} {unit}",
                delta=status
            )
            st.markdown(f"<div style='color: {color}; text-align: center; font-weight: bold;'>{status}</div>",
                       unsafe_allow_html=True)

    # Crear gráficos de gauge
    fig_gauges = make_subplots(
        rows=2, cols=3,
        subplot_titles=[p[1] for p in params],
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
    )
    
    ranges = {
        'presion_aceite': [0, 10],
        'voltaje_bateria': [8, 16],
        'voltaje_alternador': [10, 18],
        'temp_vacio': [30, 100],
        'temp_carga': [50, 120],
        'nivel_refrigerante': [0, 1]
    }
    
    for i, (param, label, unit) in enumerate(params):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        value = current_row[param]
        param_range = ranges[param]
        
        fig_gauges.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': f"{label} ({unit})"},
                gauge={
                    'axis': {'range': param_range},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [param_range[0], param_range[1] * 0.5], 'color': "lightgray"},
                        {'range': [param_range[1] * 0.5, param_range[1] * 0.8], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': param_range[1] * 0.9
                    }
                }
            ),
            row=row, col=col
        )
    
    fig_gauges.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_gauges, use_container_width=True)

def show_historical_analysis(df):
    """Muestra el análisis histórico"""
    st.header("📈 Análisis Histórico de Parámetros")
    
    # Selector de parámetros
    params = ['presion_aceite', 'voltaje_bateria', 'voltaje_alternador', 
              'temp_vacio', 'temp_carga', 'nivel_refrigerante']
    selected_params = st.multiselect("Seleccionar parámetros", params, default=params[:3])
    
    if selected_params:
        # Gráfico de líneas temporales
        fig_lines = go.Figure()
        
        # Simular timestamp
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(len(df)-1, -1, -1)]
        
        for param in selected_params:
            fig_lines.add_trace(go.Scatter(
                x=timestamps[:1000],  # Mostrar últimas 1000 muestras
                y=df[param].head(1000),
                mode='lines',
                name=param.replace('_', ' ').title(),
                line=dict(width=2)
            ))
        
        fig_lines.update_layout(
            title="Evolución Temporal de Parámetros",
            xaxis_title="Tiempo",
            yaxis_title="Valor",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_lines, use_container_width=True)
        
        # Estadísticas por parámetro
        st.subheader("📊 Estadísticas por Parámetro")
        
        stats_df = df[selected_params].describe().round(2)
        st.dataframe(stats_df, use_container_width=True)
        
        # Histogramas
        st.subheader("📊 Distribución de Valores")
        
        cols = st.columns(len(selected_params))
        for i, param in enumerate(selected_params):
            with cols[i]:
                fig_hist = px.histogram(
                    df, x=param, 
                    title=param.replace('_', ' ').title(),
                    nbins=30
                )
                fig_hist.update_layout(height=300)
                st.plotly_chart(fig_hist, use_container_width=True)

def show_fault_management_ml(current_row, model, feature_columns, target_columns):
    """Muestra la gestión de fallas"""
    st.header("⚠️ Gestión de Fallas")

    # Mapeo de parámetros a claves de sensor_values
    PARAM_MAP = {
        "Presión de Aceite": "presion_aceite",
        "Voltaje de Batería": "voltaje_bateria",
        "Voltaje Alternador": "voltaje_alternador",
        "Temperatura (Vacío)": "temp_vacio",
        "Temperatura (Carga)": "temp_carga",
        "Nivel de Refrigerante": "nivel_refrigerante"
    }

    # Preparar datos de sensores para análisis
    sensor_values = {
        'presion_aceite': current_row['presion_aceite'],
        'voltaje_bateria': current_row['voltaje_bateria'],
        'voltaje_alternador': current_row['voltaje_alternador'],
        'temp_vacio': current_row['temp_vacio'],
        'temp_carga': current_row['temp_carga'],
        'nivel_refrigerante': current_row['nivel_refrigerante']
    }

    # Predicción de fallas
    if model is not None and feature_columns is not None:
        sensor_data = [current_row[col] for col in feature_columns]
        detected_faults, fault_probabilities = predict_faults_with_model(model, feature_columns, target_columns, sensor_data)

        st.subheader("Fallas detectadas")

        if detected_faults:
            st.error(f"🚨 Se ha detectado {len(detected_faults)} FALLA(S)")
            
            for i, fault_code in enumerate(detected_faults):
                fault_info = FAULT_INFO[fault_code]
                
                # Determinar urgencia y acciones dinámicamente
                urgency, actions = determine_urgency_and_actions(fault_code, sensor_values)
                urgency_color = URGENCY_COLORS[urgency]
                
                # Probabilidad de la falla
                probability = fault_probabilities[i] if i < len(fault_probabilities) else 0.0
                
                # Obtener el valor actual del parámetro
                param_key = PARAM_MAP.get(fault_info['parameter'])
                current_value = sensor_values.get(param_key, 'N/A')

                st.markdown(f"""
                <div style='border: 2px solid {urgency_color}; border-radius: 10px; padding: 15px; margin: 10px 0;'>
                    <h4 style='color: {urgency_color}; margin: 0;'>🤖 {fault_code} - {urgency}</h4>
                    <p><strong>Confianza del Modelo:</strong> {probability:.2%}</p>
                    <p><strong>Parámetro:</strong> {fault_info['parameter']}</p>
                    <p><strong>Valor Actual:</strong> {current_value}</p>
                    <p><strong>Descripción:</strong> {fault_info['description']}</p>
                    <p><strong>Tipo:</strong> {fault_info['type']}</p>
                    <hr>
                    <p><strong>🔧 Acción Recomendada:</strong></p>
                    <p style='background-color: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;'>{actions}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No se detectaron fallas en la lectura actual")
    else:
        st.warning("⚠️ Modelo ML no disponible. Usando análisis basado en reglas como respaldo.")

    # Resumen de urgencias basado en ML
    if model is not None and detected_faults:
        st.subheader("📊 Resumen por Nivel de Urgencia")

        urgency_counts = {'Crítica': 0, 'Inmediata': 0, 'Preventiva': 0, 'Ninguna': 0}

        for fault in detected_faults:
            urgency, _ = determine_urgency_and_actions(fault, sensor_values)
            urgency_counts[urgency] += 1

        if sum(urgency_counts.values()) == 0:
            urgency_counts['Ninguna'] = 1

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🔴 Crítica", urgency_counts['Crítica'])
        with col2:
            st.metric("🟠 Inmediata", urgency_counts['Inmediata'])
        with col3:
            st.metric("🟡 Preventiva", urgency_counts['Preventiva'])
        with col4:
            st.metric("🟢 Ninguna", urgency_counts['Ninguna'])

def show_recommendations_ml(current_row, model, feature_columns, target_columns):
    """Muestra las recomendaciones de mantenimiento"""
    st.header("🔧 Recomendaciones Inteligentes")

    # Preparar datos de sensores
    sensor_values = {
        'presion_aceite': current_row['presion_aceite'],
        'voltaje_bateria': current_row['voltaje_bateria'],
        'voltaje_alternador': current_row['voltaje_alternador'],
        'temp_vacio': current_row['temp_vacio'],
        'temp_carga': current_row['temp_carga'],
        'nivel_refrigerante': current_row['nivel_refrigerante']
    }

    # Obtener fallas detectadas por ML
    if model is not None and feature_columns is not None:
        sensor_data = [current_row[col] for col in feature_columns]
        detected_faults, fault_probabilities = predict_faults_with_model(model, feature_columns, target_columns, sensor_data)

        if detected_faults:
            # Agrupar por tipo de falla
            fault_types = {}
            for fault_code in detected_faults:
                fault_info = FAULT_INFO[fault_code]
                fault_type = fault_info['type']
                if fault_type not in fault_types:
                    fault_types[fault_type] = []
                fault_types[fault_type].append(fault_code)

       # Plan de mantenimiento consolidado
            st.markdown("---")
            st.subheader("📋 Plan de Mantenimiento Inteligente")

            # Priorizar por urgencia
            critical_faults = []
            immediate_faults = []
            preventive_faults = []

            for fault in detected_faults:
                urgency, _ = determine_urgency_and_actions(fault, sensor_values)
                if urgency == 'Crítica':
                    critical_faults.append(fault)
                elif urgency == 'Inmediata':
                    immediate_faults.append(fault)
                else:
                    preventive_faults.append(fault)

            if critical_faults:
                st.error("🔴 **ACCIÓN CRÍTICA REQUERIDA - PARAR EQUIPO**")
                for fault in critical_faults:
                    _, actions = determine_urgency_and_actions(fault, sensor_values)
                    st.write(f"• {fault}: {actions}")

            if immediate_faults:
                st.warning("🟠 **PROGRAMAR MANTENIMIENTO URGENTE (24-48H)**")
                for fault in immediate_faults:
                    _, actions = determine_urgency_and_actions(fault, sensor_values)
                    st.write(f"• {fault}: {actions}")

            if preventive_faults:
                st.info("🟡 **MANTENIMIENTO PREVENTIVO (1-2 SEMANAS)**")
                for fault in preventive_faults:
                    _, actions = determine_urgency_and_actions(fault, sensor_values)
                    st.write(f"• {fault}: {actions}")

            # Estimación de costos dinámico
            st.subheader("💰 Estimación de Costos Inteligente")

            cost_estimates = {
                'Crítica': 800,
                'Inmediata': 400,
                'Preventiva': 200
            }

            total_cost = 0
            for fault in detected_faults:
                urgency, _ = determine_urgency_and_actions(fault, sensor_values)
                total_cost += cost_estimates.get(urgency, 150)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Costo Estimado", f"${total_cost:,}")
            with col2:
                st.metric("Tiempo Estimado", f"{len(detected_faults) * 3} horas")
            with col3:
                st.metric("Técnicos Requeridos", max(1, len(detected_faults) // 2))

        else:
            st.success("✅ **GENERADOR EN ÓPTIMAS CONDICIONES**")
            st.info("El modelo de Machine Learning no detectó fallas. Continuar con mantenimiento preventivo.")

            # Recomendaciones preventivas inteligentes
            st.subheader("🤖 Mantenimiento Preventivo Inteligente")

            # Tabla mejorada con información más específica
            data = {
                "Componente": ["Filtros de Aire/Aceite", "Sistema de Inyección", "Turbocompresor", "Válvulas", "Sistema de Lubricación", "Sistema de Refrigeración"],
                "Intervalo Tiempo": ["6 meses", "12-24 meses", "5 años", "5 años", "6 meses", "2 años"],
                "Intervalo Horas": ["250 h", "1000 h", "2000 h", "2000 h", "250 h", "1000 h"],
                "Riesgo si no se Mantiene": [
                    "Reducción de eficiencia, daño del motor",
                    "Aumento consumo, daño cámara combustión",
                    "Pérdida potencia, daño completo del motor",
                    "Pérdida compresión, sobrecalentamiento",
                    "Daño catastrófico del motor",
                    "Sobrecalentamiento, daño del motor"
                ],
                "Costo Preventivo": ["$150", "$400", "$1,500", "$2,000", "$200", "$300"]
            }

            df_maintenance = pd.DataFrame(data)
            st.dataframe(df_maintenance, use_container_width=True)

    else:
        st.warning("⚠️ Modelo ML no disponible para generar recomendaciones inteligentes.")

if __name__ == "__main__":
    main()
