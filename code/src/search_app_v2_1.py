import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import datetime
import os
from contextlib import redirect_stdout
import datetime
import io
import sys
from itertools import combinations
from tqdm import tqdm

# Funciones auxiliares
def make_all_stationary_m(df):
    transformed_df = df.copy()
    transformed_df = transformed_df.apply(np.log)
    diff_counts = {}
    for column in transformed_df.columns:
        series = transformed_df[column].dropna()
        p_value = 1
        diff_count = 0
        while p_value > 0.05:
            result = adfuller(series)
            p_value = result[1]
            if p_value > 0.05:
                series = series.diff(3).dropna()
                diff_count += 1
        transformed_df[column] = series
        diff_counts[column] = diff_count
    return transformed_df, diff_counts

def make_all_stationary_q(df):
    transformed_df = df.copy()
    transformed_df = transformed_df.apply(np.log)
    diff_counts = {}
    for column in transformed_df.columns:
        series = transformed_df[column].dropna()
        p_value = 1
        diff_count = 0
        while p_value > 0.05:
            result = adfuller(series)
            p_value = result[1]
            if p_value > 0.05:
                series = series.diff().dropna()
                diff_count += 1
        transformed_df[column] = series
        diff_counts[column] = diff_count
    return transformed_df, diff_counts

def evaluar_modelo(endog_m, endog_q, comb_m, comb_q, num_factores, start_date, end_date, incluir_covid=True):
    if not comb_m:
        return None, -np.inf, np.inf, -np.inf
    
    try:
        endog_m_filtered = endog_m.loc[start_date:end_date]
        endog_q_filtered = endog_q.loc[start_date:end_date]

        num_factores = min(len(comb_m), num_factores)

        
        # Añadir las dummies solo si se seleccionó la opción
        if incluir_covid:
            comb_q = comb_q + ['D2020Q1', 'D2020Q2', 'D2020Q3', 'D2020Q4']

        mod = sm.tsa.DynamicFactorMQ(
            endog_m_filtered[comb_m], 
            endog_quarterly=endog_q_filtered[comb_q], 
            factor_orders=num_factores,
            factors=num_factores,
            factor_multiplicities=None
        )
        res = mod.fit(maxiter=2000)

        rsquared = res.get_coefficients_of_determination(method='individual')
        r2_pib = rsquared.loc['pib'].max()

        pib_predicho = res.predict()['pib']
        variabilidad = np.sum(np.abs(np.diff(pib_predicho)))
        pib_ultimos = pib_predicho['2021Q3':end_date].resample('Q').mean().shift(-1).dropna()
        pib_ultimos_orig = endog_q_filtered.loc['2021Q3':end_date]['pib']

        pib_ultimos_orig = pib_ultimos_orig[pib_ultimos.index]

        
        modelo1 = sm.OLS(pib_ultimos_orig, sm.add_constant(pib_ultimos)).fit()
        r2_reciente = modelo1.rsquared.round(4)
    
        return res, r2_pib, variabilidad, r2_reciente

    except Exception as e:
        st.error(f"Error al estimar el modelo: {e}")
        return None, -np.inf, np.inf, -np.inf

def calcular_metrica_combinada(r2, variabilidad, r2_reciente, peso_r2, peso_variabilidad, peso_r2_reciente):
    r2_normalizado = (r2 + 1) / 2
    r2_reciente_normalizado = (r2_reciente + 1) / 2
    variabilidad_normalizada = 1 / (1 + variabilidad)
    return peso_r2 * r2_normalizado + peso_variabilidad * variabilidad_normalizada + peso_r2_reciente * r2_reciente_normalizado

def busqueda_eficiente(endog_m, endog_q, todos_regresores, start_date, end_date, 
                      variables_base_m=[], variables_base_q=[], 
                      max_vars=10, paciencia=5, 
                      peso_r2=0.4, peso_variabilidad=0.3, peso_r2_reciente=0.3, 
                      factores=[1, 2, 3], incluir_covid=True):
    """
    Búsqueda eficiente de variables para el modelo, permitiendo especificar variables base
    que siempre estarán incluidas en el modelo.
    
    Parámetros:
    - variables_base_m: Lista de variables mensuales que siempre se incluirán en el modelo
    - variables_base_q: Lista de variables trimestrales que siempre se incluirán en el modelo
    """
    # Crear un contenedor para los mensajes de progreso
    progress_container = st.empty()
    metrics_container = st.empty()
    current_vars_container = st.empty()
    
    # Filtrar las variables base de la lista de regresores disponibles
    regresores_disponibles_m = [var for var in todos_regresores if var in endog_m.columns and var not in variables_base_m]
    regresores_disponibles_q = [var for var in todos_regresores if var in endog_q.columns and var not in variables_base_q]
    
    if 'pib' in regresores_disponibles_q:
        regresores_disponibles_q.remove('pib')
    
    # Iniciar con las variables base
    mejor_comb_m = variables_base_m.copy()
    mejor_comb_q = ['pib'] + variables_base_q.copy()
    
    # Evaluar el modelo base
    mejor_modelo, mejor_r2, mejor_variabilidad, mejor_r2_reciente = evaluar_modelo(
        endog_m, endog_q, mejor_comb_m, mejor_comb_q, 
        factores[0], start_date, end_date, incluir_covid
    )
    mejor_metrica = calcular_metrica_combinada(mejor_r2, mejor_variabilidad, mejor_r2_reciente, 
                                              peso_r2, peso_variabilidad, peso_r2_reciente)
    mejor_num_factores = factores[0]
    
    # Inicializar sin_mejora
    sin_mejora = 0
    
    # Si no hay variables base mensuales y hay regresores disponibles, añadir la primera
    if not mejor_comb_m and regresores_disponibles_m:
        mejor_comb_m = [regresores_disponibles_m[0]]
        regresores_disponibles_m = regresores_disponibles_m[1:]
        # Reevaluar el modelo con la primera variable mensual
        mejor_modelo, mejor_r2, mejor_variabilidad, mejor_r2_reciente = evaluar_modelo(
            endog_m, endog_q, mejor_comb_m, mejor_comb_q, 
            factores[0], start_date, end_date, incluir_covid
        )
        mejor_metrica = calcular_metrica_combinada(mejor_r2, mejor_variabilidad, mejor_r2_reciente, 
                                                 peso_r2, peso_variabilidad, peso_r2_reciente)
    
    progress_bar = st.progress(0)
    
    # Crear métricas para mostrar valores actuales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        r2_metric = st.metric(label="Mejor R²", value=f"{mejor_r2:.3f}")
    with col2:
        var_metric = st.metric(label="Mejor Variabilidad", value=f"{mejor_variabilidad:.3f}")
    with col3:
        r2_reciente_metric = st.metric(label="Mejor R² Reciente", value=f"{mejor_r2_reciente:.3f}")
    with col4:
        metric_metric = st.metric(label="Mejor Métrica", value=f"{mejor_metrica:.3f}")
    
    # Mostrar variables base al inicio
    st.write("**Variables base del modelo:**")
    st.write(f"- Variables mensuales base: {variables_base_m}")
    st.write(f"- Variables trimestrales base: {variables_base_q}")
    
    for i in range(max_vars):
        mejor_nueva_var = None
        mejor_nueva_metrica = mejor_metrica
        mejor_nuevo_r2 = mejor_r2
        mejor_nueva_variabilidad = mejor_variabilidad
        mejor_nuevo_num_factores = mejor_num_factores
        
        progress_container.text(f"Iteración {i+1}/{max_vars}")
        
        for num_factores in factores:
            for var in regresores_disponibles_m:
                if var not in mejor_comb_m:
                    nueva_comb_m = mejor_comb_m + [var]
                    current_vars_container.text(
                        f"Probando:\n"
                        f"Variable mensual: {var}\n"
                        f"Factores: {num_factores}\n"
                        f"Variables actuales mensuales: {len(nueva_comb_m)}\n"
                        f"Variables actuales trimestrales: {len(mejor_comb_q)}"
                    )
                    
                    modelo, r2, variabilidad, r2_reciente = evaluar_modelo(
                        endog_m, endog_q, nueva_comb_m, mejor_comb_q, 
                        num_factores, start_date, end_date, incluir_covid
                    )
                    metrica = calcular_metrica_combinada(r2, variabilidad, r2_reciente, 
                                                       peso_r2, peso_variabilidad, peso_r2_reciente)
                    
                    metrics_container.text(
                        f"Resultados actuales:\n"
                        f"R²: {r2:.3f}\n"
                        f"Variabilidad: {variabilidad:.3f}\n"
                        f"R² Reciente: {r2_reciente:.3f}\n"
                        f"Métrica combinada: {metrica:.3f}"
                    )
                    
                    if metrica > mejor_nueva_metrica:
                        mejor_nueva_metrica = metrica
                        mejor_nuevo_r2 = r2
                        mejor_nueva_variabilidad = variabilidad
                        mejor_nuevo_r2_reciente = r2_reciente
                        mejor_nueva_var = ('m', var)
                        mejor_nuevo_num_factores = num_factores
            
            for var in regresores_disponibles_q:
                if var not in mejor_comb_q:
                    nueva_comb_q = mejor_comb_q + [var]
                    current_vars_container.text(
                        f"Probando:\n"
                        f"Variable trimestral: {var}\n"
                        f"Factores: {num_factores}\n"
                        f"Variables actuales mensuales: {len(mejor_comb_m)}\n"
                        f"Variables actuales trimestrales: {len(nueva_comb_q)}"
                    )
                    
                    modelo, r2, variabilidad, r2_reciente = evaluar_modelo(
                        endog_m, endog_q, mejor_comb_m, nueva_comb_q, 
                        num_factores, start_date, end_date, incluir_covid
                    )
                    metrica = calcular_metrica_combinada(r2, variabilidad, r2_reciente, 
                                                       peso_r2, peso_variabilidad, peso_r2_reciente)
                    
                    metrics_container.text(
                        f"Resultados actuales:\n"
                        f"R²: {r2:.3f}\n"
                        f"Variabilidad: {variabilidad:.3f}\n"
                        f"R² Reciente: {r2_reciente:.3f}\n"
                        f"Métrica combinada: {metrica:.3f}"
                    )
                    
                    if metrica > mejor_nueva_metrica:
                        mejor_nueva_metrica = metrica
                        mejor_nuevo_r2 = r2
                        mejor_nueva_variabilidad = variabilidad
                        mejor_nuevo_r2_reciente = r2_reciente
                        mejor_nueva_var = ('q', var)
                        mejor_nuevo_num_factores = num_factores
        
        if mejor_nueva_var:
            if mejor_nueva_var[0] == 'm':
                mejor_comb_m.append(mejor_nueva_var[1])
                regresores_disponibles_m.remove(mejor_nueva_var[1])
            else:
                mejor_comb_q.append(mejor_nueva_var[1])
                regresores_disponibles_q.remove(mejor_nueva_var[1])
            
            if mejor_nueva_metrica > mejor_metrica:
                mejor_metrica = mejor_nueva_metrica
                mejor_r2 = mejor_nuevo_r2
                mejor_r2_reciente = mejor_nuevo_r2_reciente
                mejor_variabilidad = mejor_nueva_variabilidad
                mejor_num_factores = mejor_nuevo_num_factores
                mejor_modelo, _, _, _ = evaluar_modelo(
                    endog_m, endog_q, mejor_comb_m, mejor_comb_q, 
                    mejor_num_factores, start_date, end_date, incluir_covid
                )
                sin_mejora = 0
                
                # Actualizar métricas en pantalla
                r2_metric.metric(label="Mejor R²", value=f"{mejor_r2:.3f}")
                var_metric.metric(label="Mejor Variabilidad", value=f"{mejor_variabilidad:.3f}")
                r2_reciente_metric.metric(label="Mejor R² Reciente", value=f"{mejor_r2_reciente:.3f}")
                metric_metric.metric(label="Mejor Métrica", value=f"{mejor_metrica:.3f}")
                
                st.write(f"""
                **Nueva mejor combinación encontrada:**
                - Métrica: {mejor_metrica:.3f}
                - R²: {mejor_r2:.3f}
                - Variabilidad: {mejor_variabilidad:.3f}
                - R² reciente: {mejor_r2_reciente:.3f}
                - Variables mensuales: {len(mejor_comb_m)}
                - Variables trimestrales: {len(mejor_comb_q)}
                - Factores: {mejor_num_factores}
                """)
            else:
                sin_mejora += 1
        else:
            break
        
        progress_bar.progress((i + 1) / max_vars)
        if sin_mejora >= paciencia:
            progress_container.text("Búsqueda terminada por criterio de paciencia")
            break
    
    # Limpiar contenedores temporales
    progress_container.empty()
    metrics_container.empty()
    current_vars_container.empty()
    
    return mejor_modelo, mejor_comb_m, mejor_comb_q, mejor_r2, mejor_variabilidad, mejor_r2_reciente, mejor_num_factores

# Aplicación Streamlit
st.title('Análisis de Series Temporales para Predicción del PIB')

# 1. Cargar archivo Excel
uploaded_file = st.file_uploader("Cargar archivo Excel", type="xlsx")

if uploaded_file is not None:
    # 2. Selector de fechas
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha de inicio", datetime.date(2002, 1, 1))
    with col2:
        end_date = st.date_input("Fecha de fin", datetime.date(2025, 3, 31))

    # 3. Cargar todas las variables
    df_mens = pd.read_excel(uploaded_file, sheet_name='Series_mens_vol_y_desest')
    df_trims = pd.read_excel(uploaded_file, sheet_name='Serie trim_vol_desest_Índice')
    
    variables_mensuales = df_mens.columns.tolist()[1:]  # Excluyendo la columna de fecha
    variables_trimestrales = df_trims.columns.tolist()[1:]  # Excluyendo la columna de fecha

    # Opción para descartar variables
    st.subheader("Variables seleccionadas")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Variables mensuales:")
        variables_mensuales_descartadas = st.multiselect('Seleccionar variables mensuales a descartar', variables_mensuales)
        selected_mensual = [var for var in variables_mensuales if var not in variables_mensuales_descartadas]
        st.write(f"Variables mensuales seleccionadas: {len(selected_mensual)}")
    with col2:
        st.write("Variables trimestrales:")
        variables_trimestrales_descartadas = st.multiselect('Seleccionar variables trimestrales a descartar', variables_trimestrales)
        selected_trimestral = [var for var in variables_trimestrales if var not in variables_trimestrales_descartadas]
        st.write(f"Variables trimestrales seleccionadas: {len(selected_trimestral)}")
    st.subheader("Variables base (siempre incluidas en el modelo)")
    col1, col2 = st.columns(2)
    with col1:
        variables_base_mensual = st.multiselect(
            'Seleccionar variables mensuales base', 
            selected_mensual,
            help="Estas variables siempre estarán incluidas en el modelo"
        )
    with col2:
        variables_base_trimestral = st.multiselect(
            'Seleccionar variables trimestrales base', 
            selected_trimestral,
            help="Estas variables siempre estarán incluidas en el modelo"
        )
    # En la sección de controles de la aplicación, añadir un checkbox:
    st.subheader("Configuración del modelo")
    incluir_covid = st.checkbox('Incluir variables dummy para COVID-19 (2020)', value=True)

    # 4. Controles para parámetros
    num_factores = st.slider('Número máximo de factores', 1, 5, 3)
    
    # Añadir aquí los nuevos controles
    col1, col2, col3 = st.columns(3)
    with col1:
        peso_r2 = st.slider('Peso del R²', 0.0, 1.0, 0.4)
    with col2:
        peso_variabilidad = st.slider('Peso de la variabilidad', 0.0, 1.0, 0.3)
    with col3:
        peso_r2_reciente = st.slider('Peso del R² reciente', 0.0, 1.0, 0.3)
        
    st.sidebar.subheader("Configuración de Salida (Guardado Directo)")
    # Usar st.session_state para recordar el valor entre ejecuciones
    if 'output_directory' not in st.session_state:
        # Intentar obtener un directorio por defecto razonable (ej: directorio actual)
        try:
            # Usar os.path.dirname para obtener el directorio del script si es posible
            # O simplemente el directorio de trabajo actual como fallback
            default_path = os.path.dirname(os.path.abspath(__file__)) 
        except NameError: # __file__ no está definido si se ejecuta de forma interactiva/diferente
             default_path = os.path.abspath(".")
        except Exception:
            default_path = "." # Fallback muy genérico
        st.session_state.output_directory = default_path

    output_dir_input = st.sidebar.text_input(
        "Directorio para guardado directo",
        value=st.session_state.output_directory,
        key="output_directory_input", # Clave única para el widget
        help="Ruta absoluta donde guardar 'resultados_modelo.txt' directamente (funciona mejor si ejecutas la app localmente)."
    )
    
    
    # Actualizar el estado de la sesión cuando cambie el input
    st.session_state.output_directory = output_dir_input

    if st.button('Ejecutar análisis'):
        # Preparar los datos exactamente igual que en la versión original
        # 1. Convertir fechas y establecer índices
        df_mens['fecha'] = pd.to_datetime(df_mens['fecha']).dt.strftime('%m-%y')
        df_mens.set_index('fecha', inplace=True)
        df_trims['fecha'] = pd.to_datetime(df_trims['fecha']).dt.strftime('%m-%y')
        df_trims.set_index('fecha', inplace=True)

        # 2. Aplicar media móvil a datos mensuales
        df_mens = df_mens.rolling(window=3).mean()
        df_mens = df_mens[2:]

        # 3. Hacer series estacionarias
        df_estacionario_mensual, _ = make_all_stationary_m(df_mens[selected_mensual])
        df_estacionario_trimestral, _ = make_all_stationary_q(df_trims[selected_trimestral])

        # 4. Aplicar diferenciaciones específicas
        if 'Pernoctaciones - Andalucía' in df_estacionario_mensual.columns:
            df_estacionario_mensual['Pernoctaciones - Andalucía'] = df_estacionario_mensual['Pernoctaciones - Andalucía'].diff(3)
        if 'Total afiliados SS Total - Andalucía' in df_estacionario_mensual.columns:
            df_estacionario_mensual['Total afiliados SS Total - Andalucía'] = df_mens['Total afiliados SS Total - Andalucía'].diff(3)

        # 5. Preparar series endógenas
        endog_m = df_estacionario_mensual
        endog_q = df_estacionario_trimestral  # Escalar por 100 las series trimestrales

        # 6. Convertir índices a PeriodIndex
        endog_m.index = pd.to_datetime(endog_m.index, format='%m-%y').to_period('M')
        endog_q.index = pd.to_datetime(endog_q.index, format='%m-%y').to_period('Q')

        # 7. Crear dummies 2020
        if incluir_covid:
            endog_q['D2020Q1'] = 0
            endog_q['D2020Q2'] = 0 
            endog_q['D2020Q3'] = 0
            endog_q['D2020Q4'] = 0

            endog_q.loc['2020Q1', 'D2020Q1'] = 1
            endog_q.loc['2020Q2', 'D2020Q2'] = 1
            endog_q.loc['2020Q3', 'D2020Q3'] = 1 
            endog_q.loc['2020Q4', 'D2020Q4'] = 1

        # 8. Convertir índices a PeriodIndex
        if not isinstance(endog_m.index, pd.PeriodIndex):
            endog_m.index = pd.to_datetime(endog_m.index, format='%m-%y').to_period('M')
        if not isinstance(endog_q.index, pd.PeriodIndex):
            endog_q.index = pd.to_datetime(endog_q.index, format='%m-%y').to_period('Q')

        # 9. Convertir fechas de inicio y fin al formato correcto
        start_date_str = f"{start_date.year}Q{(start_date.month-1)//3 + 1}"
        end_date_str = f"{end_date.year}Q{(end_date.month-1)//3 + 1}"

        # Ejecutar búsqueda eficiente
        with st.spinner('Ejecutando análisis...'):
            mejor_modelo, mejor_comb_m, mejor_comb_q, max_r2_pib, min_variabilidad, max_r2_reciente, mejor_num_factores = busqueda_eficiente(
                endog_m, endog_q, selected_mensual + selected_trimestral, 
                start_date_str, end_date_str,
                variables_base_m=variables_base_mensual,
                variables_base_q=variables_base_trimestral,
                factores=list(range(1, num_factores + 1)), 
                peso_r2=peso_r2,
                peso_variabilidad=peso_variabilidad,
                peso_r2_reciente=peso_r2_reciente,
                incluir_covid=incluir_covid
            )

        # 5. Visualización de resultados
        st.subheader("Resultados del modelo:")
        st.write(f"R² máximo PIB: {max_r2_pib:.3f}")
        st.write(f"Variabilidad mínima: {min_variabilidad:.3f}")
        st.write(f"R² máximo PIB reciente: {max_r2_reciente:.3f}")
        st.write(f"Número de factores: {mejor_num_factores}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Variables mensuales seleccionadas:")
            st.write(mejor_comb_m)
        with col2:
            st.write("Variables trimestrales seleccionadas:")
            st.write(mejor_comb_q)

        if mejor_modelo:
            # Obtener las predicciones del modelo
            predicciones = mejor_modelo.predict()
            
        
            # Crear un gráfico de las predicciones del PIB vs valores originales
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Filtrar y graficar predicciones desde 2021
            predicciones_filtradas = predicciones['pib']['2022Q1':].resample('Q').mean()
            pib_original_filtrado = endog_q['pib']['2022Q1':]
            predicciones_filtradas.plot(ax=ax, label='Predicción del modelo', color='blue', linewidth=2)
            pib_original_filtrado.plot(ax=ax, label='PIB observado', color='red', linestyle='--', linewidth=2)
            
            # Mejorar la visualización
            ax.set_title('Predicciones vs Valores Originales mensuales del PIB (desde 2022)')
            ax.set_xlabel('Fecha')
            ax.set_ylabel('PIB')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Mostrar algunos estadísticos del modelo
            st.subheader("Estadísticas del modelo:")
            st.write(mejor_modelo.summary())
        else:
            st.error("No se pudo encontrar un modelo válido.")
        
        
        # Preparar el contenido del archivo de resultados
        resultados_contenido = f"""Variables mensuales seleccionadas: {mejor_comb_m}
        Variables trimestrales seleccionadas: {mejor_comb_q}
        R² máximo PIB: {max_r2_pib:.3f}
        Variabilidad mínima: {min_variabilidad:.3f}
        R² máximo PIB reciente: {max_r2_reciente:.3f}
        Número de factores: {mejor_num_factores}"""

        # Guardar automáticamente en el directorio especificado
        try:
            # Verificar que el directorio existe
            if not os.path.exists(st.session_state.output_directory):
                st.warning(f"El directorio {st.session_state.output_directory} no existe. Intentando crearlo...")
                os.makedirs(st.session_state.output_directory, exist_ok=True)
                st.success(f"Directorio {st.session_state.output_directory} creado exitosamente.")
            
            # Añadir timestamp al nombre del archivo para evitar sobrescrituras
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(st.session_state.output_directory, f"resultados_modelo_{timestamp}.txt")
            
            # Guardar el archivo
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(resultados_contenido)
            
            st.success(f"✅ Resultados guardados automáticamente en: {file_path}")
        except Exception as e:
            st.error(f"Error al guardar el archivo: {e}")
            st.info("Se intentará usar el botón de descarga como alternativa.")

        # Mantener el botón de descarga como alternativa
        st.download_button(
            label="También puede descargar una copia",
            data=resultados_contenido,
            file_name="resultados_modelo.txt",
            mime="text/plain"
        )