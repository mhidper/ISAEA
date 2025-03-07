import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
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


def evaluar_modelo(endog_m, endog_q, comb_m, comb_q, num_factores, start_date, end_date, incluir_covid=False):
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




def busqueda_eficiente(endog_m, endog_q, todos_regresores, start_date, end_date, max_vars=10, paciencia=5, 
                      peso_r2=0.4, peso_variabilidad=0.3, peso_r2_reciente=0.3, factores=[1, 2, 3], incluir_covid=False):
    # Crear un contenedor para los mensajes de progreso
    progress_container = st.empty()
    metrics_container = st.empty()
    current_vars_container = st.empty()
    
    # Mantener exactamente la misma lógica que en la versión original
    regresores_disponibles_m = [var for var in todos_regresores if var in endog_m.columns]
    regresores_disponibles_q = [var for var in todos_regresores if var in endog_q.columns]
    
    if 'pib' in regresores_disponibles_q:
        regresores_disponibles_q.remove('pib')
    
    mejor_modelo = None
    mejor_metrica = -np.inf
    mejor_r2 = -np.inf
    mejor_variabilidad = np.inf
    mejor_r2_reciente = -np.inf 
    mejor_comb_m = []
    mejor_comb_q = ['pib']
    mejor_num_factores = 1
    sin_mejora = 0
    
    # Esta es la parte crítica que debe ser idéntica al original
    if regresores_disponibles_m:
        mejor_comb_m = [regresores_disponibles_m[0]]
        regresores_disponibles_m = regresores_disponibles_m[1:]
    
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
                    metrica = calcular_metrica_combinada(r2, variabilidad, r2_reciente, peso_r2, peso_variabilidad, peso_r2_reciente)
                    
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
                        mejor_nuevo_r2_reciente = r2_reciente  # Actualizar aquí también
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
                    
                    modelo, r2, variabilidad, r2_final = evaluar_modelo(
                        endog_m, endog_q, nueva_comb_m, mejor_comb_q, 
                        num_factores, start_date, end_date, incluir_covid
                    )
                    metrica = calcular_metrica_combinada(r2, variabilidad, r2_reciente, peso_r2, peso_variabilidad, peso_r2_reciente)
                    
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
                        mejor_nuevo_r2_reciente = r2_reciente  # Actualizar aquí también
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
                    mejor_num_factores, start_date, end_date
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
        end_date = st.date_input("Fecha de fin", datetime.date(2024, 9, 30))

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
    
    # En la sección de controles de la aplicación, añadir un checkbox:
    st.subheader("Configuración del modelo")
    incluir_covid = st.checkbox('Incluir variables dummy para COVID-19 (2020)', value=False)

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
        endog_q = df_estacionario_trimestral * 100  # Escalar por 100 las series trimestrales

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
            
            # Crear un gráfico de las predicciones del PIB
            fig, ax = plt.subplots(figsize=(12, 6))
            predicciones['pib'].plot(ax=ax)
            ax.set_title('Predicciones del PIB')
            ax.set_xlabel('Fecha')
            ax.set_ylabel('PIB')
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

        # Crear botón de descarga
        st.download_button(
            label="Descargar resultados",
            data=resultados_contenido,
            file_name="resultados_modelo.txt",
            mime="text/plain"
        )

        st.success("Haga clic en el botón de arriba para descargar los resultados")