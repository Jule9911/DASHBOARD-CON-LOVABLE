
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import joblib
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Monitoreo Generador",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tabla de referencias para fallas
FAULT_REFERENCE = {
    'F01': {
        'parameter': 'Presión de Aceite',
        'condition': '<2 psi',
        'description': 'Bomba en mal estado o falta de aceite',
        'action': 'Verificar nivel y filtro de aceite; revisar bomba y sensores',
        'urgency': 'Crítica',
        'type': 'Mecánica'
    },
    'F02': {
        'parameter': 'Presión de Aceite',
        'condition': '>7 psi',
        'description': 'Bomba en mal estado o exceso de aceite',
        'action': 'Verificar viscosidad del aceite; revisar válvula reguladora de presión',
        'urgency': 'Inmediata',
        'type': 'Mecánica'
    },
    'F03': {
        'parameter': 'Voltaje de Batería',
        'condition': '<10V',
        'description': 'Batería descargada o en mal estado',
        'action': 'Cargar batería; limpiar bornes; reemplazar si es necesario',
        'urgency': 'Crítica',
        'type': 'Eléctrica'
    },
    'F04': {
        'parameter': 'Voltaje de Batería',
        'condition': '>14V',
        'description': 'Falla en cargador de batería o alternador',
        'action': 'Verificar regulador de voltaje; revisar fusibles y conexiones',
        'urgency': 'Inmediata',
        'type': 'Eléctrica'
    },
    'F05': {
        'parameter': 'Voltaje Alternador',
        'condition': '<12V',
        'description': 'Bobinas abiertas o carbones desgastados',
        'action': 'Cambiar carbones; revisar diodos y bobinados',
        'urgency': 'Inmediata',
        'type': 'Eléctrica'
    },
    'F06': {
        'parameter': 'Voltaje Alternador',
        'condition': '>16V',
        'description': 'Daño en tarjeta reguladora o cortocircuito',
        'action': 'Reemplazar regulador; inspeccionar cortocircuitos',
        'urgency': 'Crítica',
        'type': 'Eléctrica'
    },
    'F07': {
        'parameter': 'Temperatura (Vacío)',
        'condition': '<50°C',
        'description': 'Termostato atascado o calentador de camisa dañado',
        'action': 'Revisar termostato; probar calentador de camisa',
        'urgency': 'Preventiva',
        'type': 'Térmica'
    },
    'F08': {
        'parameter': 'Temperatura (Vacío)',
        'condition': '>76°C',
        'description': 'Radiador obstruido o falla de termostatos',
        'action': 'Limpiar radiador; reemplazar termostatos; verificar ventilación',
        'urgency': 'Crítica',
        'type': 'Térmica'
    },
    'F09': {
        'parameter': 'Temperatura (Carga)',
        'condition': '<70°C',
        'description': 'Sensor de temperatura defectuoso',
        'action': 'Calibrar o reemplazar sensor; verificar conexiones',
        'urgency': 'Preventiva',
        'type': 'Térmica'
    },
    'F10': {
        'parameter': 'Temperatura (Carga)',
        'condition': '>90°C',
        'description': 'Obstrucción en radiador o bomba de agua defectuosa',
        'action': 'Limpieza profunda del sistema; cambiar refrigerante; revisar bomba y ventilador',
        'urgency': 'Crítica',
        'type': 'Térmica'
    },
    'F11': {
        'parameter': 'Nivel de Refrigerante',
        'condition': 'BAJO',
        'description': 'Fuga en sellos o radiador',
        'action': 'Reponer refrigerante; inspeccionar fugas (bomba, mangueras, radiador)',
        'urgency': 'Inmediata',
        'type': 'Térmica'
    }
}

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
        df = pd.read_csv("dataset_entrenamiento_corregido.csv")
        return df
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo dataset_entrenamiento_corregido.cvs")
        return None

def load_model():
    """Carga el modelo entrenado"""
    try:
        model_data = joblib.load("modelo_fallas.pkl")
        return model_data['model'], model_data['feature_columns'], model_data['target_columns']
    except FileNotFoundError:
        st.warning("⚠️ No se encontró el modelo entrenado. Ejecuta train_model.py primero.")
        return None, None, None

def predict_faults(model, feature_columns, sample_data):
    """Predice fallas usando el modelo"""
    if model is None:
        return None, None
    
    df_sample = pd.DataFrame([sample_data], columns=feature_columns)
    prediction = model.predict(df_sample)[0]
    probabilities = model.predict_proba(df_sample)
    
    return prediction, probabilities

def get_parameter_status(value, param_name):
    """Determina el estado de un parámetro"""
    ranges = {
        'presion_aceite': {'min': 2, 'max': 7, 'ideal': 5},
        'voltaje_bateria': {'min': 10, 'max': 14, 'ideal': 13},
        'voltaje_alternador': {'min': 12, 'max': 16, 'ideal': 14},
        'temp_vacio': {'min': 50, 'max': 76, 'ideal': 70},
        'temp_carga': {'min': 70, 'max': 90, 'ideal': 80},
        'nivel_refrigerante': {'min': 0, 'max': 1, 'ideal': 1}
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
    st.title("🚗 Dashboard de Monitoreo Generador")
    st.markdown("---")
    
    # Cargar datos y modelo
    df = load_data()
    model, feature_columns, target_columns = load_model()
    
    if df is None:
        st.stop()
    
    # Sidebar para configuración
    st.sidebar.title("⚙️ Configuración")
    
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
        "🔧 Recomendaciones"
    ])
    
    with tab1:
        show_real_time_monitoring(current_row, model, feature_columns, target_columns)
    
    with tab2:
        show_historical_analysis(df)
    
    with tab3:
        show_fault_management(current_row, model, feature_columns, target_columns)
    
    with tab4:
        show_recommendations(current_row, model, feature_columns, target_columns)
    
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
    st.header("📊 Estado Actual del Vehículo")
    
    # Timestamp simulado
    timestamp = datetime.now() - timedelta(minutes=st.session_state.current_sample)
    st.info(f"🕐 Última lectura: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Métricas principales
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
    
    # Gráficos de medidores
    st.markdown("---")
    st.subheader("📈 Medidores en Tiempo Real")
    
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

def show_fault_management(current_row, model, feature_columns, target_columns):
    """Muestra la gestión de fallas"""
    st.header("⚠️ Gestión de Fallas")
    
    # Predicción de fallas usando el modelo
    if model is not None and feature_columns is not None:
        sample_data = [current_row[col] for col in feature_columns]
        predictions, probabilities = predict_faults(model, feature_columns, sample_data)
        
        if predictions is not None:
            # Mostrar fallas detectadas
            st.subheader("🔍 Fallas Detectadas")
            
            detected_faults = []
            for i, fault_code in enumerate(target_columns):
                if predictions[i] == 1:
                    detected_faults.append(fault_code)
            
            if detected_faults:
                for fault_code in detected_faults:
                    fault_info = FAULT_REFERENCE[fault_code]
                    urgency_color = URGENCY_COLORS[fault_info['urgency']]
                    
                    st.markdown(f"""
                    <div style='border: 2px solid {urgency_color}; border-radius: 10px; padding: 15px; margin: 10px 0;'>
                        <h4 style='color: {urgency_color}; margin: 0;'>🚨 {fault_code} - {fault_info['urgency']}</h4>
                        <p><strong>Parámetro:</strong> {fault_info['parameter']}</p>
                        <p><strong>Condición:</strong> {fault_info['condition']}</p>
                        <p><strong>Descripción:</strong> {fault_info['description']}</p>
                        <p><strong>Tipo:</strong> {fault_info['type']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No se detectaron fallas en la lectura actual")
    
    # Mostrar fallas reales del dataset
    st.subheader("📋 Fallas Registradas en Dataset")
    
    fault_columns = ['F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11']
    active_faults = []
    
    for fault in fault_columns:
        if current_row[fault] == 1:
            active_faults.append(fault)
    
    if active_faults:
        for fault_code in active_faults:
            fault_info = FAULT_REFERENCE[fault_code]
            urgency_color = URGENCY_COLORS[fault_info['urgency']]
            
            st.markdown(f"""
            <div style='border: 2px solid {urgency_color}; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: rgba(255,255,255,0.1);'>
                <h4 style='color: {urgency_color}; margin: 0;'>⚠️ {fault_code} - {fault_info['urgency']}</h4>
                <p><strong>Parámetro:</strong> {fault_info['parameter']}</p>
                <p><strong>Condición:</strong> {fault_info['condition']}</p>
                <p><strong>Descripción:</strong> {fault_info['description']}</p>
                <p><strong>Tipo:</strong> {fault_info['type']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No hay fallas registradas en el dataset para esta muestra")
    
    # Resumen de urgencias
    st.subheader("📊 Resumen por Nivel de Urgencia")
    
    urgency_counts = {'Crítica': 0, 'Inmediata': 0, 'Preventiva': 0, 'Ninguna': 0}
    
    for fault in active_faults:
        urgency = FAULT_REFERENCE[fault]['urgency']
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

def show_recommendations(current_row, model, feature_columns, target_columns):
    """Muestra las recomendaciones de mantenimiento"""
    st.header("🔧 Recomendaciones de Mantenimiento")
    
    # Obtener fallas activas
    fault_columns = ['F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11']
    active_faults = [fault for fault in fault_columns if current_row[fault] == 1]
    
    if active_faults:
        # Agrupar por tipo de falla
        fault_types = {}
        for fault_code in active_faults:
            fault_info = FAULT_REFERENCE[fault_code]
            fault_type = fault_info['type']
            if fault_type not in fault_types:
                fault_types[fault_type] = []
            fault_types[fault_type].append(fault_code)
        
        # Mostrar recomendaciones por tipo
        for fault_type, faults in fault_types.items():
            st.subheader(f"🔧 Mantenimiento {fault_type}")
            
            for fault_code in faults:
                fault_info = FAULT_REFERENCE[fault_code]
                urgency_color = URGENCY_COLORS[fault_info['urgency']]
                
                with st.expander(f"{fault_code} - {fault_info['description']}", expanded=True):
                    st.markdown(f"""
                    **Nivel de Urgencia:** <span style='color: {urgency_color}; font-weight: bold;'>{fault_info['urgency']}</span>
                    
                    **Parámetro Afectado:** {fault_info['parameter']}
                    
                    **Condición Detectada:** {fault_info['condition']}
                    
                    **Acción Recomendada:**
                    {fault_info['action']}
                    """, unsafe_allow_html=True)
        
        # Plan de mantenimiento consolidado
        st.markdown("---")
        st.subheader("📋 Plan de Mantenimiento Consolidado")
        
        # Priorizar por urgencia
        critical_faults = [f for f in active_faults if FAULT_REFERENCE[f]['urgency'] == 'Crítica']
        immediate_faults = [f for f in active_faults if FAULT_REFERENCE[f]['urgency'] == 'Inmediata']
        preventive_faults = [f for f in active_faults if FAULT_REFERENCE[f]['urgency'] == 'Preventiva']
        
        if critical_faults:
            st.error("🔴 **ACCIÓN INMEDIATA REQUERIDA**")
            for fault in critical_faults:
                st.write(f"• {fault}: {FAULT_REFERENCE[fault]['action']}")
        
        if immediate_faults:
            st.warning("🟠 **PROGRAMAR MANTENIMIENTO URGENTE**")
            for fault in immediate_faults:
                st.write(f"• {fault}: {FAULT_REFERENCE[fault]['action']}")
        
        if preventive_faults:
            st.info("🟡 **MANTENIMIENTO PREVENTIVO**")
            for fault in preventive_faults:
                st.write(f"• {fault}: {FAULT_REFERENCE[fault]['action']}")
        
        # Estimación de costos (simulada)
        st.subheader("💰 Estimación de Costos")
        
        cost_estimates = {
            'Crítica': 500,
            'Inmediata': 300,
            'Preventiva': 150
        }
        
        total_cost = 0
        for fault in active_faults:
            urgency = FAULT_REFERENCE[fault]['urgency']
            total_cost += cost_estimates.get(urgency, 100)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Costo Estimado", f"${total_cost:,}")
        with col2:
            st.metric("Tiempo Estimado", f"{len(active_faults) * 2} horas")
        with col3:
            st.metric("Técnicos Requeridos", max(1, len(active_faults) // 2))
    
    else:
        st.success("✅ **VEHÍCULO EN CONDICIONES ÓPTIMAS**")
        st.info("No se requieren acciones de mantenimiento en este momento.")
        
        # Recomendaciones preventivas generales
        st.subheader("🔧 Mantenimiento Preventivo Recomendado")
        
        preventive_actions = [
            "Filtros: 6 meses o 250 h - Falla de filtros o falla de motor",
            "Inyectores: 1 o 2 años o 1000 h - Consumo de combustible, daño de cámara, daño de sellos, falla de inyectores, daño de válvulas",
            "Turbo: 5 años o 2000 h - Pérdida de aceite, potencia, daño de máquina",
            "Válvulas: 5 años o 2000 h - Pérdida de compresión, recalentamiento de escape, exceso de consumo de combustible",
            "Aceite: 6 meses o 250 h - Ruptura de biela, daño de cilindro, ruptura de cigüeñal, ruptura de árbol de levas, daño de retenedor delantero y posterior",
            "Refrigerante: 2 años o 1000 h - Daño de radiador, deterioro de bandas, obstrucción del sistema de enfriamiento, incremento de temperatura del motor, daño de sello de válvula, daño de empaquetadura de cárter"
        ]
        
        for action in preventive_actions:
            st.write(f"• {action}")

if __name__ == "__main__":
    main()
