import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import joblib
from plotly.subplots import make_subplots
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Monitoreo Generador",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tabla de referencias para información básica de fallas
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

# Costos estimados por falla
FAULT_COSTS = {
    'F01': {'repair_cost': (150, 400), 'preventive_cost': (50, 100), 'time': '2-4h'},
    'F02': {'repair_cost': (200, 500), 'preventive_cost': (80, 150), 'time': '3-5h'},
    'F03': {'repair_cost': (100, 300), 'preventive_cost': (30, 80), 'time': '1-2h'},
    'F04': {'repair_cost': (250, 600), 'preventive_cost': (100, 200), 'time': '4-6h'},
    'F05': {'repair_cost': (300, 700), 'preventive_cost': (120, 250), 'time': '4-7h'},
    'F06': {'repair_cost': (400, 900), 'preventive_cost': (150, 300), 'time': '5-8h'},
    'F07': {'repair_cost': (80, 200), 'preventive_cost': (40, 100), 'time': '2-3h'},
    'F08': {'repair_cost': (350, 800), 'preventive_cost': (200, 400), 'time': '6-10h'},
    'F09': {'repair_cost': (50, 150), 'preventive_cost': (20, 60), 'time': '1-2h'},
    'F10': {'repair_cost': (500, 1200), 'preventive_cost': (300, 600), 'time': '8-12h'},
    'F11': {'repair_cost': (200, 600), 'preventive_cost': (100, 300), 'time': '3-6h'}
}

# Colores por urgencia
URGENCY_COLORS = {
    'Crítica': '#FF4444',
    'Inmediata': '#FF8C00',
    'Preventiva': '#FFD700',
    'Ninguna': '#28A745'
}

class MLDataPreprocessor:
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns
        self.preprocessor = None
    
    def load_preprocessor(self):
        """Carga un preprocesador neutral que no modifica los datos"""
        try:
            # Crea un preprocesador que no hace transformaciones
            self.preprocessor = Pipeline([
                ('no_op', FunctionTransformer(lambda x: x, validate=False))
            ])
            return True
        except Exception as e:
            st.warning(f"Preprocesador no cargado: {str(e)}")
            return False
    
    def preprocess(self, raw_data):
        """Pasa los datos sin modificaciones (para mantener compatibilidad)"""
        try:
            # Solo asegura que los datos tengan el formato correcto
            processed_data = {k: float(raw_data[k]) for k in self.feature_columns}
            return processed_data
        except Exception as e:
            st.error(f"Error en preprocesamiento: {str(e)}")
            return None

def determine_urgency_and_actions(fault_code, sensor_values):
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
        return 'Preventiva', 'Parámetro no encontrado en sensores.'

    current_value = sensor_values[param_name]

    # Condiciones específicas por falla
    if fault_code == 'F01':  # Presión de Aceite < 2
        if current_value < 1.0:
            return 'Crítica', 'Falla grave en bomba o sin aceite. Detener inmediatamente.'
        elif current_value < 2.0:
            return 'Inmediata', 'Verificar nivel, filtro y bomba de aceite.'
        else:
            return 'Preventiva', 'Presión cercana al mínimo. Monitorear.'

    elif fault_code == 'F02':  # Presión de Aceite > 7
        if current_value > 8.0:
            return 'Crítica', 'Posible obstrucción o exceso de aceite. Parar.'
        elif current_value > 7.5:
            return 'Inmediata', 'Verificar viscosidad y válvula reguladora.'
        else:
            return 'Preventiva', 'Presión alta, inspección sugerida.'

    elif fault_code == 'F03':  # Voltaje Batería < 10
        if current_value < 9.0:
            return 'Crítica', 'Batería descargada o dañada. Parar equipo.'
        elif current_value < 10.0:
            return 'Inmediata', 'Cargar batería y revisar conexiones.'
        else:
            return 'Preventiva', 'Voltaje bajo, programar mantenimiento.'

    elif fault_code == 'F04':  # Voltaje Batería > 14
        if current_value > 15.0:
            return 'Crítica', 'Sobrevoltaje. Revisar regulador y alternador.'
        elif current_value > 14.5:
            return 'Inmediata', 'Verificar regulador de voltaje.'
        else:
            return 'Preventiva', 'Voltaje ligeramente alto.'

    elif fault_code == 'F05':  # Voltaje Alternador < 12
        if current_value < 10.0:
            return 'Crítica', 'Alternador sin carga. Parar.'
        elif current_value < 12.0:
            return 'Inmediata', 'Revisar carbones o diodos.'
        else:
            return 'Preventiva', 'Carga inestable, monitorear.'

    elif fault_code == 'F06':  # Voltaje Alternador > 16
        if current_value > 17.0:
            return 'Crítica', 'Cortocircuito o falla en tarjeta.'
        elif current_value > 16.0:
            return 'Inmediata', 'Verificar regulador.'
        else:
            return 'Preventiva', 'Voltaje alto, revisar cableado.'

    elif fault_code == 'F07':  # Temp Vacío < 50
        if current_value < 40.0:
            return 'Crítica', 'Fallo térmico. Revisar termostato y calentador.'
        elif current_value < 50.0:
            return 'Inmediata', 'Probar calentador de camisa.'
        else:
            return 'Preventiva', 'Temperatura baja inusual.'

    elif fault_code == 'F08':  # Temp Vacío > 76
        if current_value > 85.0:
            return 'Crítica', 'Sobrecalentamiento grave. Detener.'
        elif current_value > 76.0:
            return 'Inmediata', 'Verificar radiador y termostatos.'
        else:
            return 'Preventiva', 'Temp. ligeramente elevada.'

    elif fault_code == 'F09':  # Temp Carga < 70
        if current_value < 60.0:
            return 'Crítica', 'Sensor o calentamiento deficiente.'
        elif current_value < 70.0:
            return 'Inmediata', 'Calibrar sensor de temperatura.'
        else:
            return 'Preventiva', 'Verificar calibración.'

    elif fault_code == 'F10':  # Temp Carga > 90
        if current_value > 100.0:
            return 'Crítica', 'Sobrecalentamiento. Parar y revisar sistema.'
        elif current_value > 90.0:
            return 'Inmediata', 'Limpiar radiador y revisar bomba.'
        else:
            return 'Preventiva', 'Temperatura de carga alta.'

    elif fault_code == 'F11':  # Nivel Refrigerante BAJO
        if current_value == 0:
            return 'Crítica', 'Sin refrigerante. Detener equipo.'
        elif current_value < 0.5:
            return 'Inmediata', 'Fuga menor, rellenar y revisar.'
        else:
            return 'Preventiva', 'Inspección recomendada.'

    return 'Preventiva', 'Consultar manual de mantenimiento.'
def get_maintenance_actions(fault_code, urgency, current_value):
    """Determina las acciones de mantenimiento"""
    base_actions = {
        'F01': {
            'Crítica': 'Detener el generador inmediatamente. Verificar nivel de aceite, reemplazar filtro, inspeccionar bomba de aceite y revisar sensores de presión.',
            'Inmediata': 'Agregar aceite al nivel recomendado. Revisar posibles fugas. Programar inspección de bomba y sensores en 24 horas.',
            'Preventiva': 'Verificar nivel y presión de aceite semanalmente. Limpiar o cambiar el filtro si es necesario.'
        },
        'F02': {
            'Crítica': 'Parar el generador. Drenar exceso de aceite si aplica. Inspeccionar válvula reguladora por atasco u obstrucción.',
            'Inmediata': 'Verificar viscosidad del aceite. Sustituir si está fuera de especificación. Revisar presión del sistema.',
            'Preventiva': 'Monitorear presión de aceite regularmente. Calibrar sensores y revisar válvula durante mantenimiento mensual.'
        },
        'F03': {
            'Crítica': 'Parar el equipo. Cargar batería, revisar alternador, limpiar bornes y reemplazar la batería si está defectuosa.',
            'Inmediata': 'Cargar batería. Limpiar bornes. Medir carga del alternador para prevenir fallas futuras.',
            'Preventiva': 'Medir voltaje semanalmente. Realizar mantenimiento mensual a bornes y revisar carga en vacío.'
        },
        'F04': {
            'Crítica': 'Desconectar cargador y detener operación. Inspeccionar regulador de voltaje y revisar fusibles por sobrecarga.',
            'Inmediata': 'Verificar funcionamiento del regulador de voltaje. Revisar alternador y conexiones.',
            'Preventiva': 'Controlar voltaje de carga regularmente. Calibrar sistema eléctrico si se detectan desviaciones.'
        },
        'F05': {
            'Crítica': 'Detener operación. Reemplazar alternador y revisar sistema eléctrico completo: bobinas, diodos, fusibles.',
            'Inmediata': 'Cambiar carbones del alternador. Inspeccionar bobinados y puentes de diodos.',
            'Preventiva': 'Inspeccionar alternador trimestralmente. Limpiar conexiones eléctricas.'
        },
        'F06': {
            'Crítica': 'Detener equipo por riesgo eléctrico. Reemplazar regulador de voltaje. Verificar posibles cortocircuitos en el sistema.',
            'Inmediata': 'Inspeccionar cableado. Comprobar regulador con multímetro. Revisar aislamiento eléctrico.',
            'Preventiva': 'Medir voltaje del alternador cada 100 horas. Calibrar regulador si se detectan fluctuaciones.'
        },
        'F07': {
            'Crítica': 'Verificar termostato dañado y reemplazar de inmediato. Inspeccionar el calentador de camisa.',
            'Inmediata': 'Probar funcionamiento del calentador. Reajustar o sustituir termostato si aplica.',
            'Preventiva': 'Revisar sistema térmico en frío semanalmente. Confirmar activación del calentador durante arranque.'
        },
        'F08': {
            'Crítica': 'Generador sobrecalentado. Parar y limpiar radiador. Cambiar termostatos si están atascados.',
            'Inmediata': 'Verificar ventilación, nivel de refrigerante y flujo del sistema de enfriamiento.',
            'Preventiva': 'Mantener radiador libre de obstrucciones. Realizar limpieza profunda trimestralmente.'
        },
        'F09': {
            'Crítica': 'Sensor de temperatura probablemente defectuoso. Reemplazar sensor y recalibrar el sistema.',
            'Inmediata': 'Calibrar sensor. Verificar cableado y conexiones. Validar lectura con instrumento externo.',
            'Preventiva': 'Inspección mensual de sensores de temperatura. Registrar lecturas anómalas para seguimiento.'
        },
        'F10': {
            'Crítica': 'Sobrecalentamiento severo. Detener generador. Realizar limpieza profunda del sistema, cambiar refrigerante y revisar bomba de agua y ventilador.',
            'Inmediata': 'Revisar bomba de agua. Verificar tensión de correas y estado del ventilador.',
            'Preventiva': 'Ejecutar mantenimiento del sistema de enfriamiento cada 150 horas o según manual del fabricante.'
        },
        'F11': {
            'Crítica': 'Nivel de refrigerante extremadamente bajo. Reponer de inmediato. Inspeccionar fugas en radiador, bomba y mangueras.',
            'Inmediata': 'Reponer refrigerante. Inspeccionar conexiones y sellos. Observar variaciones durante operación.',
            'Preventiva': 'Verificar nivel semanalmente. Observar signos de evaporación o microfugas en el sistema.'
        }
    }

    return base_actions.get(fault_code, {}).get(urgency, 'Consultar manual de mantenimiento.')

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

def load_data():
    """Carga datos desde CSV local O archivo subido por el usuario"""
    uploaded_file = st.file_uploader(
        "📤 Sube tu propio archivo (opcional)", 
        type=['csv', 'xlsx', 'parquet', 'json']
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            st.success(f"Archivo {uploaded_file.name} cargado correctamente!")
            return df
        except Exception as e:
            st.error(f"❌ Error al cargar archivo: {str(e)}")
            return None
    
    try:
        df = pd.read_csv("Dataset_de_prueba__50_registros_ - Dataset_de_prueba__50_registros_t.csv")
        st.info("✅ Usando dataset local por defecto")
        return df
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo local Dataset_de_prueba__50_registros_.csv")
        return None

def load_model():
    """Carga el modelo entrenado y el preprocesador"""
    try:
        model_data = joblib.load("modelo_fallas.pkl")
        preprocessor = MLDataPreprocessor(model_data['feature_columns'])
        preprocessor.load_preprocessor()
        return model_data['model'], preprocessor, model_data['feature_columns'], model_data['target_columns']
    except FileNotFoundError:
        st.warning("⚠️ No se encontró el modelo entrenado. Ejecuta train_model.py primero.")
        return None, None, None, None

def predict_faults_with_model(model, preprocessor, target_columns, sample_data):
    """Predice fallas usando el modelo con preprocesamiento"""
    if model is None or preprocessor is None:
        return [], []
    
    try:
        processed_data = preprocessor.preprocess(sample_data)
        if processed_data is None:
            return [], []
            
        df_sample = pd.DataFrame([processed_data], columns=preprocessor.feature_columns)
        
        predictions = model.predict(df_sample)[0]
        probabilities = model.predict_proba(df_sample)
        
        detected_faults = []
        fault_probabilities = []
        
        for i, fault_code in enumerate(target_columns):
            if predictions[i] == 1:
                detected_faults.append(fault_code)
                prob = probabilities[i][0][1] if len(probabilities[i][0]) > 1 else probabilities[i][0][0]
                fault_probabilities.append(prob)
        
        return detected_faults, fault_probabilities
        
    except Exception as e:
        st.error(f"Error en predicción: {str(e)}")
        return [], []

def show_real_time_monitoring(current_row, model, preprocessor, feature_columns, target_columns):
    """Muestra el monitoreo en tiempo real"""
    st.header("📊 Estado Actual del Generador")

    # Timestamp simulado
    timestamp = datetime.now() - timedelta(minutes=st.session_state.current_sample)
    st.info(f"🕐 Última lectura: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    # Predicción en tiempo real
    if model is not None:
        sensor_data = {
            'presion_aceite': current_row['presion_aceite'],
            'voltaje_bateria': current_row['voltaje_bateria'],
            'voltaje_alternador': current_row['voltaje_alternador'],
            'temp_vacio': current_row['temp_vacio'],
            'temp_carga': current_row['temp_carga'],
            'nivel_refrigerante': current_row['nivel_refrigerante']
        }
        
        detected_faults, probabilities = predict_faults_with_model(
            model,
            preprocessor,
            target_columns,
            sensor_data
        )
        
        if detected_faults:
            st.error(f"🚨 SE DETECTARON {len(detected_faults)} FALLA(S)")
        else:
            st.success("✅ SIN FALLAS DETECTADAS")

    # Métricas principales
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    params = [
        ('presion_aceite', 'Presión Aceite', 'psi'),
        ('voltaje_bateria', 'Voltaje Batería', 'V'),
        ('voltaje_alternador', 'Voltaje Alternador', 'V'),
        ('temp_vacio', 'Temp. Vacío', '°C'),
        ('temp_carga', 'Temp. Carga', '°C'),
        ('nivel_refrigerante', 'Refrigerante')
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

def show_real_time_monitoring(current_row, model, preprocessor, feature_columns, target_columns):
    """Muestra el monitoreo en tiempo real"""
    st.header("📊 Estado Actual del Generador")

    # Timestamp simulado
    timestamp = datetime.now() - timedelta(minutes=st.session_state.current_sample)
    st.info(f"🕐 Última lectura: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    # Predicción en tiempo real
    if model is not None:
        sensor_data = {
            'presion_aceite': current_row['presion_aceite'],
            'voltaje_bateria': current_row['voltaje_bateria'],
            'voltaje_alternador': current_row['voltaje_alternador'],
            'temp_vacio': current_row['temp_vacio'],
            'temp_carga': current_row['temp_carga'],
            'nivel_refrigerante': current_row['nivel_refrigerante']
        }
        
        detected_faults, probabilities = predict_faults_with_model(
            model,
            preprocessor,
            target_columns,
            sensor_data
        )
        
        if detected_faults:
            st.error(f"🚨 SE DETECTARON {len(detected_faults)} FALLA(S)")
        else:
            st.success("✅ SIN FALLAS DETECTADAS")

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
    """Muestra el análisis histórico en una pestaña separada"""
    st.header("📈 Análisis Histórico")
    
    # Definir parámetros específicos del dashboard
    PARAMETROS_DASHBOARD = [
        'presion_aceite', 'voltaje_bateria', 'voltaje_alternador',
        'temp_vacio', 'temp_carga', 'nivel_refrigerante'
    ]
    
    # Verificar qué parámetros existen realmente en los datos
    parametros_disponibles = [p for p in PARAMETROS_DASHBOARD if p in df.columns]
    
    if not parametros_disponibles:
        st.warning("No se encontraron los parámetros esperados en los datos")
        return
    
    # Crear pestañas para cada parámetro
    tabs = st.tabs([p.replace('_', ' ').title() for p in parametros_disponibles])
    
    for i, param in enumerate(parametros_disponibles):
        with tabs[i]:
            show_single_parameter_analysis(df, param)
def show_single_parameter_analysis(df, parametro):
    """Muestra el análisis para un solo parámetro"""
    # Configuración de límites según el parámetro
    LIMITES = {
        'presion_aceite': {'ideal': 5, 'ruptura': 7},
        'voltaje_bateria': {'ideal': 13, 'ruptura': 14},
        'voltaje_alternador': {'ideal': 14, 'ruptura': 16},
        'temp_vacio': {'ideal': 70, 'ruptura': 76},
        'temp_carga': {'ideal': 80, 'ruptura': 90},
        'nivel_refrigerante': {'ideal': 1, 'ruptura': 1}
    }

    # Obtener límites para este parámetro
    limites = LIMITES.get(parametro, {'ideal': None, 'ruptura': None})
    
    # Sidebar con controles para este parámetro
    with st.sidebar.expander(f"⚙️ Configuración {parametro.replace('_', ' ')}", expanded=False):
        ideal = st.number_input(
            f"Límite ideal ({parametro})",
            value=limites['ideal'] if limites['ideal'] is not None else df[parametro].mean(),
            key=f"ideal_{parametro}"
        )
        
        ruptura = st.number_input(
            f"Umbral de ruptura ({parametro})",
            value=limites['ruptura'] if limites['ruptura'] is not None else df[parametro].max(),
            key=f"ruptura_{parametro}"
        )

  # Verificar si existe columna de tiempo y es de tipo datetime
        tiene_timestamp = 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp'])
        
  # Selector de rango de fechas solo si existe timestamp válido
        if tiene_timestamp:
            # Convertir a datetime y extraer solo la parte de fecha
            min_date = pd.to_datetime(df['timestamp'].min()).date()
            max_date = pd.to_datetime(df['timestamp'].max()).date()
            
            try:
                rango_fechas = st.date_input(
                    f"Rango de fechas ({parametro})",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key=f"rango_{parametro}"
                )
            except Exception as e:
                st.warning(f"No se pudo crear el selector de fechas: {str(e)}")
                rango_fechas = (min_date, max_date)
                tiene_timestamp = False

    # Filtrar datos por rango de fechas si existe columna de tiempo válida
    if tiene_timestamp and len(rango_fechas) == 2:
        # Convertir las fechas seleccionadas a datetime para comparación
        fecha_inicio = pd.to_datetime(rango_fechas[0])
        fecha_fin = pd.to_datetime(rango_fechas[1])
        
        df_filtrado = df[
            (pd.to_datetime(df['timestamp']) >= fecha_inicio) & 
            (pd.to_datetime(df['timestamp']) <= fecha_fin)
        ]
    else:
        df_filtrado = df


    # --- Sección 1: Gráfico de histórico ---
    st.subheader(f"Histórico de {parametro.replace('_', ' ').title()}")

    # Mostrar métricas clave
    with st.container():
        cols = st.columns(3)
        current_val = df_filtrado[parametro].iloc[-1]
        max_val = df_filtrado[parametro].max()
        
        with cols[0]:
            delta = f"{(current_val - ideal):.2f} sobre ideal" if current_val > ideal else ""
            st.metric("Valor Actual", f"{current_val:.2f}", delta=delta)
        with cols[1]:
            st.metric("Máximo Registrado", f"{max_val:.2f}")
        with cols[2]:
            excedencias = (df_filtrado[parametro] > ruptura).sum()
            st.metric("Excedencias", excedencias)

    # Crear gráfico de histórico
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtrado['timestamp'] if 'timestamp' in df_filtrado.columns else df_filtrado.index,
        y=df_filtrado[parametro],
        mode='lines+markers',
        name=parametro.replace('_', ' ').title(),
        line=dict(color='#FF5733', width=2),
        marker=dict(size=4),
        hovertemplate="%{y:.2f}<extra></extra>"
    ))

    # Líneas de referencia
    if ideal is not None:
        fig.add_hline(y=ideal, line=dict(color='blue', dash='dash'),
                     annotation_text=f"Límite Ideal ({ideal:.2f})")
    
    if ruptura is not None:
        fig.add_hline(y=ruptura, line=dict(color='red', dash='dot'),
                     annotation_text=f"Umbral Peligroso ({ruptura:.2f})")

    # Configuración del gráfico
    fig.update_layout(
        height=500,
        xaxis_title="Tiempo",
        yaxis_title=parametro.replace('_', ' ').title(),
        hovermode="x unified",
        margin=dict(t=40, b=40),
 xaxis=dict(
        showticklabels=False  # ⛔️ Esto oculta las etiquetas de tiempo en el eje
    )
)
    st.plotly_chart(fig, use_container_width=True)

    # --- Sección 2: Estadísticas ---
    st.subheader("📊 Estadísticas del Parámetro")
    stats_df = df_filtrado[parametro].describe().to_frame().T.round(2)

    # Aplicar formato de 2 decimales a todas las columnas numéricas
    st.dataframe(
        stats_df.style.format("{:.2f}").highlight_max(axis=1, color='#FFA07A'), 
        use_container_width=True
    )

    # --- Sección 3: Distribución ---
    st.subheader("📊 Distribución de Valores")
    
    col1, col2 = st.columns(2)
    with col1:
        # Histograma
        fig_hist = px.histogram(
            df_filtrado, 
            x=parametro,
            title=f"Distribución de {parametro}",
            nbins=30,
            color_discrete_sequence=['#FF5733']
        )
        if ideal is not None:
            fig_hist.add_vline(x=ideal, line_dash="dash", line_color="blue")
        if ruptura is not None:
            fig_hist.add_vline(x=ruptura, line_dash="dot", line_color="red")
        fig_hist.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot
        fig_box = px.box(
            df_filtrado,
            y=parametro,
            title=f"Box Plot de {parametro}",
            color_discrete_sequence=['#FF5733']
        )
        if ideal is not None:
            fig_box.add_hline(y=ideal, line_dash="dash", line_color="blue")
        if ruptura is not None:
            fig_box.add_hline(y=ruptura, line_dash="dot", line_color="red")
        fig_box.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # --- Sección 4: Datos que exceden el límite ---
    if ruptura is not None and excedencias > 0:
        st.subheader(f"⚠️ Registros que exceden el umbral ({excedencias})")
        st.dataframe(
            df_filtrado[df_filtrado[parametro] > ruptura][['timestamp', parametro]].sort_values(
                parametro, ascending=False),
            use_container_width=True
        )

def show_fault_management_ml(current_row, model, preprocessor, target_columns):
    """Muestra la gestión de fallas"""
    st.header("⚠️ Gestión de Fallas")

    # Mapeo de parámetros
    PARAM_MAP = {
        "Presión de Aceite": "presion_aceite",
        "Voltaje de Batería": "voltaje_bateria",
        "Voltaje Alternador": "voltaje_alternador",
        "Temperatura (Vacío)": "temp_vacio",
        "Temperatura (Carga)": "temp_carga",
        "Nivel de Refrigerante": "nivel_refrigerante"
    }

    # Preparar datos de sensores
    sensor_values = {
        'presion_aceite': current_row['presion_aceite'],
        'voltaje_bateria': current_row['voltaje_bateria'],
        'voltaje_alternador': current_row['voltaje_alternador'],
        'temp_vacio': current_row['temp_vacio'],
        'temp_carga': current_row['temp_carga'],
        'nivel_refrigerante': current_row['nivel_refrigerante']
    }

    # Predicción de fallas
    if model is not None and preprocessor is not None:
        detected_faults, fault_probabilities = predict_faults_with_model(
            model,
            preprocessor,
            target_columns,
            sensor_values
        )

        st.subheader("Fallas detectadas")

        if detected_faults:
            st.error(f"🚨 Se ha detectado {len(detected_faults)} FALLA(S)")
            
            for i, fault_code in enumerate(detected_faults):
                fault_info = FAULT_INFO[fault_code]
                
                # Determinar urgencia y acciones
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

    # Resumen de urgencias
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

def show_recommendations_ml(current_row, model, preprocessor, feature_columns, target_columns, auto_refresh=False, refresh_interval=10, max_samples=0):
    """Muestra las recomendaciones de mantenimiento con análisis de costos"""
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

    # Función para parsear tiempos
    def parse_time(time_str):
        try:
            clean_str = time_str.replace('h', '').strip()
            if '-' in clean_str:
                parts = list(map(int, clean_str.split('-')))
                return sum(parts) / len(parts)
            return int(clean_str)
        except:
            return 0

    # Obtener fallas detectadas por ML
    if model is not None and preprocessor is not None:
        detected_faults, fault_probabilities = predict_faults_with_model(
            model,
            preprocessor,
            target_columns,
            sensor_values
        )

        if detected_faults:
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

            # Mostrar fallas críticas
            if critical_faults:
                with st.expander("🔴 **ACCIÓN CRÍTICA REQUERIDA - PARAR EQUIPO**", expanded=True):
                    for fault in critical_faults:
                        urgency, actions = determine_urgency_and_actions(fault, sensor_values)
                        cost = FAULT_COSTS.get(fault, {})
                        time_estimate = FAULT_COSTS.get(fault, {}).get('time', '0h')
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{fault}**: {actions}")
                        with col2:
                            st.metric("Costo reparación", f"${cost.get('repair_cost', (0,0))[1]} USD")
                        
                        st.caption(f"⏱️ Tiempo estimado: {time_estimate} (promedio: {parse_time(time_estimate):.1f} horas)")

            # Mostrar fallas inmediatas
            if immediate_faults:
                with st.expander("🟠 **MANTENIMIENTO URGENTE (24-48H)**", expanded=True):
                    for fault in immediate_faults:
                        urgency, actions = determine_urgency_and_actions(fault, sensor_values)
                        cost = FAULT_COSTS.get(fault, {})
                        time_estimate = FAULT_COSTS.get(fault, {}).get('time', '0h')
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{fault}**: {actions}")
                        with col2:
                            st.metric("Costo reparación", f"${cost.get('repair_cost', (0,0))[1]} USD")
                        st.caption(f"⏱️ Tiempo estimado: {time_estimate} (promedio: {parse_time(time_estimate):.1f} horas)")

            # Mostrar fallas preventivas
            if preventive_faults:
                with st.expander("🟡 **MANTENIMIENTO PREVENTIVO (1-2 SEMANAS)**", expanded=True):
                    for fault in preventive_faults:
                        urgency, actions = determine_urgency_and_actions(fault, sensor_values)
                        cost = FAULT_COSTS.get(fault, {})
                        time_estimate = FAULT_COSTS.get(fault, {}).get('time', '0h')
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{fault}**: {actions}")
                        with col2:
                            st.metric("Costo preventivo", f"${cost.get('preventive_cost', (0,0))[1]} USD")
                        
                        ahorro = cost.get('repair_cost', (0,0))[1] - cost.get('preventive_cost', (0,0))[1]
                        st.caption(f"💵 Ahorro potencial: ${ahorro} USD | ⏱️ Tiempo: {time_estimate}")

            # Resumen financiero
            st.markdown("---")
            st.subheader("💰 Resumen Financiero")
            
            total_repair = sum(FAULT_COSTS.get(f, {}).get('repair_cost', (0,0))[1] for f in detected_faults)
            total_preventive = sum(FAULT_COSTS.get(f, {}).get('preventive_cost', (0,0))[1] for f in detected_faults)
            total_time = sum(parse_time(FAULT_COSTS.get(f, {}).get('time', '0h')) for f in detected_faults)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Costo total reparación", f"${total_repair} USD")
            with col2:
                st.metric("Costo total preventivo", f"${total_preventive} USD")
            with col3:
                st.metric("Tiempo total estimado", f"{round(total_time)} horas (promedio)")

            # Gráfico comparativo
            st.markdown("### Comparativo Costo Reparación vs Preventivo")
            cost_comparison = pd.DataFrame({
                'Falla': detected_faults,
                'Reparación': [FAULT_COSTS.get(f, {}).get('repair_cost', (0,0))[1] for f in detected_faults],
                'Preventivo': [FAULT_COSTS.get(f, {}).get('preventive_cost', (0,0))[1] for f in detected_faults]
            })

            fig = px.bar(
                cost_comparison.melt(id_vars='Falla', var_name='Tipo', value_name='Costo'),
                x='Falla', 
                y='Costo',
                color='Tipo',
                barmode='group',
                color_discrete_map={'Reparación': '#FF4444', 'Preventivo': '#28A745'},
                title="Comparación de costos por tipo de mantenimiento"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.success("✅ **GENERADOR EN ÓPTIMAS CONDICIONES**")
            st.info("El modelo de Machine Learning no detectó fallas. Continuar con mantenimiento preventivo.")

            # Mostrar tabla de mantenimientos preestablecidos
            st.subheader("📅 Mantenimientos Programados Según Fabricante")
            
            maintenance_schedule = [
                {
                    "Componente": "Filtros",
                    "Frecuencia Temporal": "6 meses",
                    "Frecuencia Uso": "250 h",
                    "Fallas Potenciales": "Falla de filtros o falla de motor",
                    "Acción": "Reemplazar filtros"
                },
                {
                    "Componente": "Inyectores",
                    "Frecuencia Temporal": "1-2 años",
                    "Frecuencia Uso": "1000 h",
                    "Fallas Potenciales": "Consumo excesivo, daño de cámara, falla de inyectores",
                    "Acción": "Limpieza o reemplazo"
                },
                {
                    "Componente": "Turbo",
                    "Frecuencia Temporal": "5 años",
                    "Frecuencia Uso": "2000 h",
                    "Fallas Potenciales": "Pérdida de aceite, reducción de potencia",
                    "Acción": "Revisión de sellos y ejes"
                },
                {
                    "Componente": "Válvulas",
                    "Frecuencia Temporal": "5 años",
                    "Frecuencia Uso": "2000 h",
                    "Fallas Potenciales": "Pérdida de compresión, sobrecalentamiento",
                    "Acción": "Ajuste o reemplazo"
                },
                {
                    "Componente": "Aceite",
                    "Frecuencia Temporal": "6 meses",
                    "Frecuencia Uso": "250 h",
                    "Fallas Potenciales": "Daño en componentes internos del motor",
                    "Acción": "Cambio de aceite y filtro"
                },
                {
                    "Componente": "Refrigerante",
                    "Frecuencia Temporal": "2 años",
                    "Frecuencia Uso": "1000 h",
                    "Fallas Potenciales": "Sobrecalentamiento, daño al motor",
                    "Acción": "Reemplazo de refrigerante"
                }
            ]

            st.dataframe(
                pd.DataFrame(maintenance_schedule),
                column_config={
                    "Frecuencia Temporal": st.column_config.TextColumn("Periodo Temporal"),
                    "Frecuencia Uso": st.column_config.TextColumn("Horas de Operación")
                },
                use_container_width=True,
                hide_index=True
            )

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        if st.session_state.current_sample < max_samples:
            st.session_state.current_sample += 1
        else:
            st.session_state.current_sample = 0
# ... (todo tu código anterior se mantiene igual hasta la función main())

def show_risk_analysis(df, current_row):
    """Nueva pestaña para análisis de riesgo de parada"""
    st.header("🛑 Predicción de Parada del Equipo", divider="red")

    # 1. Configuración de parámetros y umbrales
    PARAM_CONFIG = {
        'presion_aceite': {'limite_sup': 7, 'limite_inf': 2, 'peso': 0.25},
        'voltaje_bateria': {'limite_sup': 14, 'limite_inf': 10, 'peso': 0.15},
        'voltaje_alternador': {'limite_sup': 16, 'limite_inf': 12, 'peso': 0.20},
        'temp_vacio': {'limite_sup': 76, 'limite_inf': 50, 'peso': 0.15},
        'temp_carga': {'limite_sup': 90, 'limite_inf': 70, 'peso': 0.15},
        'nivel_refrigerante': {'limite_sup': 1, 'limite_inf': 0.5, 'peso': 0.10}
    }

    # 2. Asegurar que timestamp está en formato correcto
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['timestamp'])  # Elimina fechas inválidas
    else:
        df['timestamp'] = pd.to_datetime(df.index, dayfirst=True, errors='coerce')
        df = df.dropna(subset=['timestamp'])

    # 3. Filtrar últimos 30 días
    ultimos_30d = df[df['timestamp'] >= (datetime.now() - timedelta(days=30))]

    if len(ultimos_30d) == 0:
        st.warning("No hay datos recientes para análisis de riesgo")
        return

    # 4. Calcular fallas por parámetro
    risk_data = []
    for param, config in PARAM_CONFIG.items():
        if param not in df.columns:
            continue

        fallas = ((ultimos_30d[param] > config['limite_sup']) | (ultimos_30d[param] < config['limite_inf'])).sum()
        frecuencia = (fallas / len(ultimos_30d)) * 100 if len(ultimos_30d) > 0 else 0
        severidad = max(
            abs(current_row[param] - config['limite_sup']),
            abs(current_row[param] - config['limite_inf']),
        )

        risk_data.append({
            'Parámetro': param.replace('_', ' ').title(),
            'Fallas (30d)': fallas,
            'Frecuencia': f"{frecuencia:.1f}%",
            'Severidad': f"{severidad:.2f}",
            'Valor Actual': f"{current_row[param]:.2f}",
            'Estado': '🔴 Crítico' if frecuencia > 20 else '🟡 Advertencia' if frecuencia > 10 else '🟢 Normal'
        })

    risk_df = pd.DataFrame(risk_data)

    # 5. Cálculo de riesgo global
    riesgo_global = 0
    total_pesos = 0

    for _, row in risk_df.iterrows():
        param_key = row['Parámetro'].lower().replace(' ', '_')
        if param_key in PARAM_CONFIG:
            try:
                frecuencia = float(row['Frecuencia'].replace('%', ''))
                riesgo_global += frecuencia * PARAM_CONFIG[param_key]['peso']
                total_pesos += PARAM_CONFIG[param_key]['peso']
            except ValueError:
                continue

    if total_pesos > 0:
        riesgo_global = riesgo_global / 100  # Normalizar
    else:
        riesgo_global = 0

    # 6. Determinar nivel de riesgo
    if riesgo_global > 0.7:
        nivel_riesgo = "🔴 PARADA INMINENTE"
        accion = "Detener equipo inmediatamente"
        color = "#FF4444"
    elif riesgo_global > 0.5:
        nivel_riesgo = "🟠 ALTO RIESGO"
        accion = "Mantenimiento urgente (<24h)"
        color = "#FF8C00"
    elif riesgo_global > 0.3:
        nivel_riesgo = "🟡 RIESGO MODERADO"
        accion = "Programar mantenimiento"
        color = "#FFD700"
    else:
        nivel_riesgo = "🟢 BAJO RIESGO"
        accion = "Monitoreo continuo"
        color = "#28A745"

    # 7. Mostrar métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("Índice de Riesgo", f"{riesgo_global:.0%}", delta=nivel_riesgo)
    col2.metric("Fallas Totales (30d)", risk_df['Fallas (30d)'].sum())
    col3.metric("Acción Recomendada", accion)

    st.progress(min(riesgo_global, 1.0), text=f"Probabilidad de falla catastrófica: {riesgo_global:.0%}")

        # Radar
    if not risk_df.empty:
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[float(row['Frecuencia'].replace('%', '')) for _, row in risk_df.iterrows()],
            theta=risk_df['Parámetro'],
            fill='toself',
            name='Frecuencia de Fallas (%)',
            line_color=color
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Frecuencia de Fallas por Parámetro (Últimos 30 días)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Tabla detallada
    with st.expander("📊 Detalle por Parámetro", expanded=True):
        st.dataframe(risk_df, use_container_width=True, hide_index=True)

    # -------------------------------------------
    # HISTÓRICO DE RIESGO DIARIO
    # -------------------------------------------
    st.subheader("📈 Evolución Histórica del Riesgo")

    df['fecha'] = df['timestamp'].dt.date

    def calcular_riesgo_por_dia(grupo):
        total_riesgo = 0
        total_pesos = 0
        for param, cfg in PARAM_CONFIG.items():
            if param in grupo.columns:
                serie = grupo[param]
                fuera = (serie > cfg['limite_sup']) | (serie < cfg['limite_inf'])
                total_riesgo += fuera.sum() * cfg['peso']
                total_pesos += len(serie)
        return total_riesgo / total_pesos if total_pesos > 0 else 0

    riesgo_diario = df.groupby('fecha').apply(calcular_riesgo_por_dia).reset_index(name='riesgo')
    riesgo_diario = riesgo_diario.sort_values('fecha')

    st.write("🔍 Riesgo diario calculado:")
    st.dataframe(riesgo_diario)

    fig_tendencia = px.area(
        riesgo_diario,
        x='fecha',
        y='riesgo',
        labels={'riesgo': 'Índice de Riesgo', 'fecha': 'Fecha'},
        title="Tendencia Diaria del Riesgo de Parada",
        line_shape='spline'
    )

    fig_tendencia.update_layout(
        xaxis=dict(type='date', tickformat='%b %d'),
        yaxis=dict(range=[0, 1])
    )

    fig_tendencia.add_hrect(y0=0.7, y1=1, fillcolor="red", opacity=0.2, annotation_text="Riesgo Crítico")
    fig_tendencia.add_hrect(y0=0.5, y1=0.7, fillcolor="orange", opacity=0.2, annotation_text="Riesgo Alto")
    fig_tendencia.add_hrect(y0=0.3, y1=0.5, fillcolor="yellow", opacity=0.2, annotation_text="Riesgo Medio")
    fig_tendencia.add_hrect(y0=0.0, y1=0.3, fillcolor="green", opacity=0.2, annotation_text="Riesgo Bajo")

    st.plotly_chart(fig_tendencia, use_container_width=True)

def main():
    st.title("⚡DASHBOARD MONITOREO GENERADOR - ML PREDICTIVO")
    st.markdown("---")

    # Cargar datos y modelo
    df = load_data()
    model, preprocessor, feature_columns, target_columns = load_model()

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

   
    # Crear pestañas (MODIFICADO PARA AÑADIR NUEVA TAB)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Monitoreo en Tiempo Real",
        "📈 Análisis Histórico",
        "⚠️ Gestión de Fallas",
        "🔧 Recomendaciones Inteligentes",
        "🛑 Riesgo de Parada"  # Nueva pestaña
    ])

    with tab1:
        show_real_time_monitoring(current_row, model, preprocessor, feature_columns, target_columns)

    with tab2:
        show_historical_analysis(df)

    with tab3:
        show_fault_management_ml(current_row, model, preprocessor, target_columns)

    with tab4:
        show_recommendations_ml(
            current_row=current_row,
            model=model,
            preprocessor=preprocessor,
            feature_columns=feature_columns,
            target_columns=target_columns,
            auto_refresh=auto_refresh,
            refresh_interval=refresh_interval,
            max_samples=len(df)-1 if df is not None else 0
        )
    
    with tab5:  # Nueva pestaña
        show_risk_analysis(df, current_row)


    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        if st.session_state.current_sample < max_samples:
            st.session_state.current_sample += 1
        else:
            st.session_state.current_sample = 0
        st.rerun()

if __name__ == "__main__":
    main()