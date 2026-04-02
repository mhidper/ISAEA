import streamlit as st
import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
import io
import sys
from contextlib import redirect_stdout 
from utils import *


# Configuración de la página
st.set_page_config(
    page_title="Procesador de Datos IECA - V3.2 (Experimental)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("ISAEA - Procesamiento y Verificación (V3.2)")

# Configuración en el sidebar
st.sidebar.header("Configuración")

# Entrada para la ruta del proyecto
default_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = st.sidebar.text_input(
    "Directorio raíz del proyecto",
    value=default_project_root,
    help="Ruta absoluta al directorio raíz del proyecto"
)

# Creación de directorios basados en la raíz del proyecto
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
OUTPUT_MONTHLY_DIR = os.path.join(OUTPUT_DIR, 'monthly')
OUTPUT_QUARTERLY_DIR = os.path.join(OUTPUT_DIR, 'quarterly')
OUTPUT_NOWCASTING_DIR = os.path.join(OUTPUT_DIR, 'nowcasting')
OUTPUT_RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
OUTPUT_FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')


for directory in [OUTPUT_DIR, OUTPUT_MONTHLY_DIR, OUTPUT_QUARTERLY_DIR, 
                 OUTPUT_NOWCASTING_DIR, OUTPUT_RESULTS_DIR, OUTPUT_FIGURES_DIR]:
    os.makedirs(directory, exist_ok=True)

def denton_method(y, x):
    """
    Implementa el método de Denton para mensualizar series trimestrales.
    
    :param y: Serie trimestral a desagregar
    :param x: Serie mensual relacionada (indicador)
    :return: Serie mensual desagregada
    """
    # Asegurar que las longitudes sean correctas
    assert len(y) * 3 == len(x), "La longitud de x debe ser 3 veces la longitud de y"
    
    n = len(x)
    m = len(y)
    
    # Crear la matriz de agregación
    C = np.zeros((m, n))
    for i in range(m):
        C[i, i*3:(i+1)*3] = 1
    
    # Crear la matriz de diferencias
    D = np.eye(n) - np.eye(n, k=-1)
    D = D[1:, :]
    
    # Calcular la solución
    z = np.linalg.inv(C.T @ C + D.T @ D) @ (C.T @ y + D.T @ D @ x)
    
    return z

def sanitize_excel_string(s):
    """Elimina caracteres de control que rompen el formato XML de Excel/openpyxl."""
    if not isinstance(s, str):
        return s
    # Eliminar caracteres de control excepto \n, \r, \t
    return "".join(ch for ch in s if ch == '\n' or ch == '\r' or ch == '\t' or 32 <= ord(ch) <= 1114111)

def generar_excel_impactos(news_results, ficheros):
    """
    Genera un DataFrame con el Impacto Total de cada transición de vintage 
    para el Excel consolidado.
    """
    if not news_results:
        return pd.DataFrame()

    all_impacts = []
    
    for i in range(1, len(ficheros)):
        vintage = ficheros[i]
        prev_vintage = ficheros[i-1]
        
        if vintage in news_results:
            result = news_results[vintage]
            key = f"{prev_vintage} -> {vintage}"
            
            impact_df = result.details_by_impact.reset_index()
            
            # Solo Variable, Fecha e Impacto Total (el impacto real en el PIB)
            temp_data = {
                'Variable': impact_df['updated variable'].apply(sanitize_excel_string),
                'Fecha': impact_df['update date'],
                f'Impact Total ({key})': impact_df['impact']
            }
            
            temp_df = pd.DataFrame(temp_data)
            all_impacts.append(temp_df)
    
    if not all_impacts:
        return pd.DataFrame()
    
    df_final = all_impacts[0]
    for i in range(1, len(all_impacts)):
        df_final = pd.merge(
            df_final, 
            all_impacts[i], 
            on=['Variable', 'Fecha'], 
            how='outer'
        )
    
    return df_final.sort_values(['Variable', 'Fecha'])


# Opción para cargar archivos
uploaded_files = st.sidebar.file_uploader(
    "Seleccionar archivos Excel",
    type=['xlsx'],
    accept_multiple_files=True
)

# Nombres de las hojas
nombre_hoja_mensual = 'Series_mens_vol_y_desest'
nombre_hoja_trimestre = 'Serie trim_vol_desest_Índice'

# Patrón para extraer la fecha
patron_fecha = r'Envío_(\d{2})_(\d{2})_(\d{4})'

if uploaded_files:
    # Contenedor para mostrar el progreso
    progress_container = st.container()
    
    with progress_container:
        st.write("Procesando archivos...")
        progress_bar = st.progress(0)
        
        # Diccionarios para almacenar los DataFrames
        dfs_m = {}
        dfs_q = {}
        
        # Procesar cada archivo
        for i, uploaded_file in enumerate(uploaded_files):
            # Actualizar barra de progreso
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            
            try:
                # Extraer la fecha del nombre del archivo
                match = re.search(patron_fecha, uploaded_file.name)
                if match:
                    dia, mes, anio = match.groups()
                    fecha = datetime(int(anio), int(mes), int(dia)).strftime('%Y-%m-%d')
                    
                    # Crear un expander para cada archivo
                    with st.expander(f"Procesando {uploaded_file.name}"):
                        # Leer datos mensuales
                        st.write("Leyendo datos mensuales...")
                        df_mens = pd.read_excel(uploaded_file, nombre_hoja_mensual)
                        df_mens['fecha'] = pd.to_datetime(df_mens['fecha']).dt.strftime('%m-%y')
                        df_mens.set_index('fecha', inplace=True)
                        
                        
                        # Leer datos trimestrales
                        st.write("Leyendo datos trimestrales...")
                        df_trims = pd.read_excel(uploaded_file, nombre_hoja_trimestre)
                        df_trims['fecha'] = pd.to_datetime(df_trims['fecha']).dt.strftime('%m-%y')
                        df_trims.set_index('fecha', inplace=True)
                        
                        # Procesar estacionaridad
                        st.write("Procesando estacionaridad...")
                        # Convert the monthly data to a stationary series
                        df_estacionario_mensual, _ = make_all_stationary_m(df_mens)
                        df_estacionario_trimestral, _ = make_all_stationary_q(df_trims)
                        

                                                
                        # Procesamiento adicional
                        #df_estacionario_mensual['Pernoctaciones - Andalucía'] = df_estacionario_mensual['Pernoctaciones - Andalucía'].diff(3)
                        df_estacionario_mensual['Total afiliados SS Total - Andalucía'] = df_mens['Total afiliados SS Total - Andalucía'].diff(3)
                        
                        endog_m = df_estacionario_mensual
                        endog_q = df_estacionario_trimestral*100
                        
                        # Convertir índices
                        endog_m.index = pd.to_datetime(endog_m.index, format='%m-%y')
                        endog_q.index = pd.to_datetime(endog_q.index, format='%m-%y')
                        
                        endog_q.index = endog_q.index.to_period('Q')
                        endog_m.index = endog_m.index.to_period('M')
                        
                        # Almacenar DataFrames
                        dfs_m[fecha] = endog_m
                        dfs_q[fecha] = endog_q
                        
                        st.success("Archivo procesado correctamente")
                        
                else:
                    st.error(f"No se pudo extraer la fecha del archivo {uploaded_file.name}")
                    
            except Exception as e:
                st.error(f"Error procesando {uploaded_file.name}: {str(e)}")
    

    # Ordenar ficheros por fecha
    if dfs_m:
        ficheros = list(dfs_m.keys())
        ficheros.sort(key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
        dfs_m = {fecha: dfs_m[fecha] for fecha in ficheros}
        
        # Mostrar resumen
        st.success(f"Procesamiento completado. {len(ficheros)} archivos procesados.")
        
        # Guardar en session_state para uso posterior
        st.session_state['dfs_m'] = dfs_m
        st.session_state['dfs_q'] = dfs_q
        st.session_state['ficheros'] = ficheros

        # Separador en el sidebar
        st.sidebar.markdown("---")
        st.sidebar.header("Configuración del Modelo DFMQ")

        # Parámetros del modelo en el sidebar
        factor_orders = st.sidebar.number_input("Factor Orders", 
            min_value=1, 
            max_value=10, 
            value=2,
            help="Número de rezagos en el modelo de factores"
        )
        
        factors = st.sidebar.number_input("Factors", 
            min_value=1, 
            max_value=5, 
            value=1,
            help="Número de factores a estimar"
        )
        
        # Selección de variables
        st.sidebar.markdown("---")
        st.sidebar.header("Selección de Variables")

        # Obtener todas las variables disponibles
        available_vars_m = list(dfs_m[ficheros[0]].columns)
        available_vars_q = list(dfs_q[ficheros[0]].columns)

        # Variables por defecto
        default_m = ['Consumo aparente de cemento - Andalucía', 
                        'Total afiliados SS Total - Andalucía',
                        'Matriculación de turismos - Andalucía'
                        ]

        
        default_q = ['Ocupados EPA  total - Andalucía',
                    'Índice de producción agrícola - Andalucía',
                    'pib']

        # Verificar que los valores por defecto existen en las opciones disponibles
        default_m = [var for var in default_m if var in available_vars_m]
        default_q = [var for var in default_q if var in available_vars_q]

        # Selección de variables mensuales
        mejores_regresores_m = st.sidebar.multiselect(
            "Variables Mensuales",
            options=available_vars_m,
            default=default_m,
            help="Selecciona las variables mensuales para el modelo"
        )

        # Selección de variables trimestrales
        mejores_regresores_q = st.sidebar.multiselect(
            "Variables Trimestrales",
            options=available_vars_q,  # Usamos las variables trimestrales disponibles
            default=default_q,
            help="Selecciona las variables trimestrales para el modelo"
        )

        # Añadir opción para seleccionar el final del periodo de predicción
        st.sidebar.markdown("---")
        st.sidebar.header("Configuración de Predicción")
        end_date_dt = st.sidebar.date_input(
            "Fecha final de predicción",
            value=datetime(2025, 12, 31),
            min_value=datetime(2000, 1, 1),
            max_value=datetime(2030, 12, 31),
            help="Selecciona la fecha final para las predicciones. Indica siempre el último día del trimestre."
        )
        end_date = end_date_dt.strftime('%Y-%m')

        # Punto 5: Selector de fecha de impacto para Nowcasting
        st.sidebar.markdown("---")
        st.sidebar.header("Configuración Nowcasting")
        impact_date_dt = st.sidebar.date_input(
            "Fecha de impacto (Trimestre a vigilar)",
            value=end_date_dt,
            min_value=datetime(2000, 1, 1),
            max_value=datetime(2030, 12, 31),
            help="Selecciona el trimestre sobre el que quieres ver los impactos de las novedades."
        )
        impact_date = impact_date_dt.strftime('%Y-%m')

        # Opción de incluir dummies
        st.sidebar.markdown("---")
        include_dummies = st.sidebar.checkbox(
            "Incluir dummies COVID-19 (2020)",
            value=False,
            help="Incluir variables dummy para los trimestres de 2020. Una para cada trimestre."
        )
        
        # Añadir selector de directorio en el sidebar
        st.sidebar.markdown("---")
        st.sidebar.header("Configuración de salida")
        output_dir = st.sidebar.text_input(
            "Directorio de salida",
            value=OUTPUT_DIR,  # Usar la constante definida al inicio
            help="Directorio base donde se guardarán los resultados. Los distintos tipos de resultados se guardarán en subdirectorios específicos."
        )

        # Botón para estimar el modelo
        if st.sidebar.button("Estimar Modelo DFMQ"):
            st.session_state['estimar_dfmq'] = True
            st.session_state['recalcular_dfmq'] = True

        if st.session_state.get('estimar_dfmq', False):
            recalcular = st.session_state.get('recalcular_dfmq', True)
            try:
                if not recalcular and 'modelo_dfmq' in st.session_state:
                    res = st.session_state['modelo_dfmq']
                else:
                    with st.spinner("Estimando modelo DFMQ..."):
                        # Preparar datos para el modelo
                        endog_m_c = dfs_m[ficheros[0]][mejores_regresores_m]
                        endog_q = dfs_q[ficheros[0]][mejores_regresores_q]
    
                        # Añadir dummies si la opción está seleccionada
                        if include_dummies:
                            # Crear las columnas dummy
                            for q in range(1, 5):
                                col_name = f'D2020Q{q}'
                                endog_q[col_name] = 0
                                if f'2020Q{q}' in endog_q.index:
                                    endog_q.loc[f'2020Q{q}', col_name] = 1
                                    
                        # Crear y ajustar el modelo
                        mod = sm.tsa.DynamicFactorMQ(
                            endog_m_c,
                            endog_quarterly=endog_q,
                            factor_orders=factor_orders,
                            factors=factors,
                            factor_multiplicities=None
                        )
                        
                        res = mod.fit(maxiter=2000)
    
                        # Guardar resultados en session_state
                        st.session_state['modelo_dfmq'] = res
                        
                        # Mostrar resultados
                        st.success("Modelo estimado correctamente")


                    # Generar predicciones
                    #--------------------------------------------------------------------------------------------
                    st.write("Generando predicciones...")
                    
                    if not recalcular and 'dfmq_vintage' in st.session_state:
                        vintage_results = st.session_state['dfmq_vintage']
                        forecast_mensuales = st.session_state['dfmq_forecasts']
                        forecast_trimestrales = st.session_state['dfmq_forecasts_t']
                        indices_trimestrales = st.session_state['dfmq_indices_t']
                        resultados_mensuales = st.session_state['dfmq_resultados_m']
                    else:
                        # Archivo vintages
                        vintage_results = {ficheros[0]: res}
                    
                        start = '2000'
                    
                        # Apply results to remaining vintages
                        for vintage in ficheros[1:]:
                            st.write(f"Procesando vintage: {vintage}")
                        
                            # Get updated data for the vintage
                            updated_endog_m = dfs_m[vintage][mejores_regresores_m].loc[start:, :]
                            updated_endog_q = dfs_q[vintage][mejores_regresores_q].loc[start:, :]

                            # Añadir dummies si la opción está seleccionada
                            if include_dummies:
                                # Crear las columnas dummy
                                for q in range(1, 5):
                                    col_name = f'D2020Q{q}'
                                    updated_endog_q[col_name] = 0
                                    if f'2020Q{q}' in updated_endog_q.index:
                                        updated_endog_q.loc[f'2020Q{q}', col_name] = 1
                        
                            # Get updated results for the vintage
                            vintage_results[vintage] = res.apply(
                                updated_endog_m, endog_quarterly=updated_endog_q, retain_standardization=False)
                    
                        # Create forecasts results objects
                        prediction_results = res.get_prediction(start='2006-01', end=end_date)
                    
                        variables = ['pib']
                    
                        # Get point predictions
                        point_predictions = prediction_results.predicted_mean[variables]
                        predictionGDP = point_predictions.resample('Q').last()
                    
                        # Predicción mejores regresores
                        point_predictions_mejores = prediction_results.predicted_mean[mejores_regresores_m + mejores_regresores_q]
                    
                        # Compute forecasts for each vintage
                        forecasts_m = {vintage: res.get_prediction(start='2006-01', end=end_date).predicted_mean['pib']
                                    for vintage, res in vintage_results.items()}
                    
                        forecast_mensuales = pd.DataFrame(forecasts_m)
                    
                        # Compute forecasts for each vintage (quarterly)
                        forecasts_q = {vintage: res.get_prediction(start='2006-01', end=end_date).predicted_mean['pib']
                                    .resample('Q').mean()
                                    for vintage, res in vintage_results.items()}
                    
                        forecast_trimestrales = pd.DataFrame(forecasts_q)
                    
                        # Convert indices to datetime
                        forecast_mensuales.index = pd.to_datetime(forecast_mensuales.index.to_timestamp())
                        forecast_trimestrales.index = pd.to_datetime(forecast_trimestrales.index.to_timestamp())
                    
                        # Crear índices trimestrales
                        indices_trimestrales = pd.DataFrame(index=forecast_trimestrales.index)
                    
                        for columna in forecast_trimestrales.columns:
                            serie_indice = pd.Series(index=forecast_trimestrales.index, dtype=float)
                            serie_indice.iloc[0] = 100
                        
                            for i in range(1, len(serie_indice)):
                                tasa_crecimiento = forecast_trimestrales[columna].iloc[i]
                                serie_indice.iloc[i] = serie_indice.iloc[i-1] * (1 + tasa_crecimiento/100)
                        
                            indices_trimestrales[columna] = serie_indice
                    
                        # Código a reemplazar en app_v2_4.py (líneas ~455-497)
                        # Sección: Mensualizar índices trimestrales
                        #-------------------------------------------------------------------------------------------------- 
                        st.write("Calculando índices mensuales...")

                        # Calcular la media móvil de tres meses
                        df_mens_moving_average = forecast_mensuales.rolling(window=3, min_periods=1).mean()
                        df_mensualizada = 100*((1 + df_mens_moving_average/100) ** (1/3)-1)

                        # Crear DataFrame para crecimiento
                        df_growth_based = pd.DataFrame(index=df_mensualizada.index, columns=df_mensualizada.columns)
                        df_growth_based.iloc[0] = 100

                        # Calcular crecimiento
                        for i in range(1, len(df_mensualizada)):
                            df_growth_based.iloc[i] = df_growth_based.iloc[i-1] * (1 + df_mensualizada.iloc[i]/100)

                        # ============================================================================
                        # NUEVA FUNCIÓN: Alinear series para Denton
                        # ============================================================================
                        def alinear_series_para_denton(serie_mensual, serie_trimestral):
                            """
                            Alinea las series mensual y trimestral para cumplir el requisito
                            de que len(mensual) == len(trimestral) * 3.
                            Prioriza los datos más recientes (alinea desde el final).
                            """
                            # Convertir a series numéricas
                            mensual = pd.to_numeric(pd.Series(serie_mensual), errors='coerce')
                            trimestral = pd.to_numeric(pd.Series(serie_trimestral), errors='coerce')
                        
                            # Encontrar el rango válido (sin NaN) para cada serie
                            mensual_valid = mensual.dropna()
                            trimestral_valid = trimestral.dropna()
                        
                            if len(mensual_valid) == 0 or len(trimestral_valid) == 0:
                                return None, None
                        
                            # Calcular cuántos trimestres completos podemos usar
                            n_trimestres = len(trimestral_valid)
                            n_meses_disponibles = len(mensual_valid)
                        
                            # Ajustar para que los meses sean exactamente 3x trimestres
                            n_trimestres_final = min(n_trimestres, n_meses_disponibles // 3)
                            n_meses_final = n_trimestres_final * 3
                        
                            if n_trimestres_final == 0:
                                return None, None
                        
                            # Alinear desde el FINAL para conservar los datos más recientes
                            mensual_alineada = mensual_valid.iloc[-n_meses_final:].to_numpy()
                            trimestral_alineada = trimestral_valid.iloc[-n_trimestres_final:].to_numpy()
                        
                            return mensual_alineada, trimestral_alineada

                        # Crear DataFrame para resultados mensuales
                        resultados_mensuales = pd.DataFrame(index=df_growth_based.index)

                        # Procesar cada columna
                        for columna in df_growth_based.columns:
                            X_mensual = df_growth_based[columna].to_numpy()
                        
                            if columna in indices_trimestrales.columns:
                                y_trimestral = indices_trimestrales[columna].to_numpy()
                            else:
                                continue

                            # Alinear las series antes de aplicar Denton
                            X_mensual_clean, y_trimestral_clean = alinear_series_para_denton(X_mensual, y_trimestral)
                        
                            if X_mensual_clean is None or y_trimestral_clean is None:
                                continue
                        
                            try:
                                z_mensual = denton_method(y_trimestral_clean, X_mensual_clean) * 3
                            
                                # Crear serie con el índice original, alineando desde el final
                                resultado = pd.Series(index=df_growth_based.index, dtype=float)
                                resultado.iloc[-len(z_mensual):] = z_mensual
                            
                                resultados_mensuales[columna] = resultado
                            
                            except:
                                continue


                        st.session_state['dfmq_vintage'] = vintage_results
                        st.session_state['dfmq_forecasts'] = forecast_mensuales
                        st.session_state['dfmq_forecasts_t'] = forecast_trimestrales
                        st.session_state['dfmq_indices_t'] = indices_trimestrales
                        st.session_state['dfmq_resultados_m'] = resultados_mensuales
                    
                    # Guardar resultados
                    forecast_mensuales.to_csv(os.path.join(OUTPUT_MONTHLY_DIR, "indicador_mensual.csv"), sep=";")
                    forecast_trimestrales.to_csv(os.path.join(OUTPUT_QUARTERLY_DIR, "indicador_trimestral.csv"), sep=";")
                    indices_trimestrales.to_csv(os.path.join(OUTPUT_QUARTERLY_DIR, "indices_trimestrales.csv"), sep=";")
                    resultados_mensuales.to_csv(os.path.join(OUTPUT_MONTHLY_DIR, "indices_mensuales.csv"), sep=";")

                    # Verificar que el directorio existe
                    output_file = os.path.join(OUTPUT_RESULTS_DIR, "resultados_estimaciones.txt")

                    with open(output_file, 'w') as f:
                        f.write("Resultados de las estimaciones\n")
                        f.write("=============================\n\n")
                        
                        old_stdout = sys.stdout
                        new_stdout = io.StringIO()
                        sys.stdout = new_stdout
                        print(res.summary())
                        sys.stdout = old_stdout
                        summary_output = new_stdout.getvalue()

                        f.write(summary_output)
                        f.write("\nInformación adicional:\n")
                        f.write(f"AIC: {res.aic}\n")
                        f.write(f"BIC: {res.bic}\n")

                        # Añadir los coeficientes de determinación
                        f.write("\nCoeficientes de determinación (R^2):\n")
                        f.write("===================================\n\n")
                        rsquared = res.get_coefficients_of_determination(method='individual')
                        
                        top_ten = []
                        for factor_name in rsquared.columns[:6]:
                            top_factor = (rsquared[factor_name].sort_values(ascending=False)
                                                         .iloc[:30].round(2).reset_index())
                            top_factor.columns = ['Variable', 'R^2']
                            top_factor.columns = pd.MultiIndex.from_product([
                                [f'Top ten variables explained by {factor_name}'],
                                ['Variable', r'$R^2$']])
                            top_ten.append(top_factor)

                        combined_df = pd.concat(top_ten, axis=1)
                        f.write(combined_df.to_string())

                    # Mostrar los resultados
                    st.text(summary_output)
                    st.write("\nCoeficientes de determinación (R²):")
                    st.dataframe(combined_df)
                    st.success(f"Los resultados han sido guardados en: {output_file}")
                    
                    st.session_state['recalcular_dfmq'] = False

            except Exception as e:
                st.error(f"Error al estimar el modelo: {str(e)}")
                st.session_state['recalcular_dfmq'] = False

            # Visualización de resultados
            st.write("Generando visualización...")

            # Filtrar datos desde 2022
            forecast_mensuales_f = forecast_mensuales.loc['2022-01':]
            forecast_trimestrales_f = forecast_trimestrales.loc['2022-01':]

            # Crear paletas de colores
            n_columns_mensual = forecast_mensuales_f.shape[1]
            palette_mensual = sns.color_palette("Blues_r", n_colors=n_columns_mensual)

            n_columns_trimestral = forecast_trimestrales_f.shape[1]
            palette_trimestral = sns.color_palette("Blues_r", n_colors=n_columns_trimestral)

            # Crear figura
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

            # Subplot mensual
            for i, col in enumerate(forecast_mensuales_f.columns):
                sns.lineplot(data=forecast_mensuales_f, x=forecast_mensuales_f.index, y=col, 
                            color=palette_mensual[i], label=col, ax=ax1)

            ax1.set_title('Factor Mensual (ISAEA)')
            ax1.set_ylabel('Valor')
            ax1.legend(title='Vintage')

            # Subplot trimestral
            for i, col in enumerate(forecast_trimestrales_f.columns):
                sns.lineplot(data=forecast_trimestrales_f, x=forecast_trimestrales_f.index, y=col, 
                            color=palette_trimestral[i], label=col, ax=ax2)

            ax2.set_title('Factor Trimestral (ISAEA)')
            ax2.set_ylabel('Valor')

            plt.suptitle('Comparación de ISAEA por ola de indicadores', fontsize=16, y=1.02)
            plt.tight_layout()
            st.pyplot(fig)

            # Guardar figura
            fig_path = os.path.join(OUTPUT_FIGURES_DIR, "ISAEA_comparison.png")
            fig.savefig(fig_path, format='png', bbox_inches='tight', dpi=300)
            plt.close(fig)

            # Análisis de Nowcasting
            st.write("Realizando análisis de nowcasting...")
            nowcast_dir = os.path.join(OUTPUT_DIR, "Nowcasting")
            if not os.path.exists(nowcast_dir):
                os.makedirs(nowcast_dir)

            if not recalcular and 'dfmq_news' in st.session_state:
                news_results = st.session_state['dfmq_news']
            else:
                news_results = {}
                progress_bar = st.progress(0)

                for i in range(1, len(ficheros)):
                    vintage = ficheros[i]
                    prev_vintage = ficheros[i - 1]
                    progress = (i) / (len(ficheros) - 1)
                    progress_bar.progress(progress)
                
                    idx_current = vintage_results[vintage].model._index
                    idx_previous = vintage_results[prev_vintage].model._index
                
                    if not idx_previous.isin(idx_current).all():
                        idx_common = idx_current.intersection(idx_previous)
                        start_common = idx_common.min()
                        try:
                            endog_m_current = dfs_m[vintage][mejores_regresores_m].loc[start_common:, :]
                            q_start = start_common.to_timestamp().to_period('Q')
                            endog_q_current = dfs_q[vintage][mejores_regresores_q].loc[q_start:, :]
                            endog_m_previous = dfs_m[prev_vintage][mejores_regresores_m].loc[start_common:, :]
                            endog_q_previous = dfs_q[prev_vintage][mejores_regresores_q].loc[q_start:, :]
                        
                            if include_dummies:
                                for q in range(1, 4): # Fix: dummies COVID usually end at Q4, adjusted loop
                                    col_name = f'D2020Q{q}'
                                    for df_q_tmp in [endog_q_current, endog_q_previous]:
                                        df_q_tmp[col_name] = 0
                                        if f'2020Q{q}' in df_q_tmp.index:
                                            df_q_tmp.loc[f'2020Q{q}', col_name] = 1
                        
                            vintage_aligned = res.apply(endog_m_current, endog_quarterly=endog_q_current, retain_standardization=False)
                            prev_vintage_aligned = res.apply(endog_m_previous, endog_quarterly=endog_q_previous, retain_standardization=False)
                        
                            news_results[vintage] = vintage_aligned.news(prev_vintage_aligned, impact_date=impact_date, impacted_variable='pib', comparison_type='previous')
                        except: continue
                    else:
                        try:
                            news_results[vintage] = vintage_results[vintage].news(vintage_results[prev_vintage], impact_date=impact_date, impacted_variable='pib', comparison_type='previous')
                        except: continue

                st.session_state['dfmq_news'] = news_results
                st.session_state['recalcular_dfmq'] = False

            # Guardar y mostrar News
            # [Lógica simplificada para brevedad, igual a v3_1]
            for i, (name, result) in enumerate(news_results.items()):
                prev_vintage = ficheros[i]
                current_vintage = name
                key = f"{prev_vintage} -> {current_vintage}"
                impact_df = result.details_by_impact.reset_index()
                summary_text = result.summary().as_text()
                
                detailed_df = pd.DataFrame({
                    'Comparison': [key] * len(impact_df),
                    'Updated Variable': impact_df['updated variable'].apply(sanitize_excel_string),
                    'Update Date': impact_df['update date'],
                    'Impact Total': impact_df['impact'],
                    'Previous Forecast': impact_df.get('forecast (prev)', 0.0)
                })
                # Lógica Revised combinada
                num_cols = impact_df.select_dtypes(include=[np.number]).columns
                candidates = [c for c in num_cols if 'forecast' in str(c).lower() and 'prev' not in str(c).lower()]
                detailed_df['Revised Forecast'] = impact_df[candidates[0]] if candidates else (detailed_df['Previous Forecast'] + impact_df.get('news', 0.0) + impact_df.get('revision', 0.0))
                
                excel_path = os.path.join(nowcast_dir, f'nowcasting_{prev_vintage}_to_{current_vintage}.xlsx')
                with pd.ExcelWriter(excel_path) as writer:
                    detailed_df.to_excel(writer, sheet_name='Impacts', index=False)
                    pd.DataFrame({'Summary': [sanitize_excel_string(l) for l in summary_text.split('\n')]}).to_excel(writer, sheet_name='Summary', index=False)
                
                with open(os.path.join(nowcast_dir, f'nowcasting_{prev_vintage}_to_{current_vintage}_summary.txt'), 'w', encoding='utf-8') as f:
                    f.write(summary_text)

                with st.expander(f"Preview Nowcasting {key}"):
                    st.dataframe(detailed_df)

            # Consolidado e Impactos (Gráfico final)
            if news_results:
                df_final = generar_excel_impactos(news_results, ficheros)
                if not df_final.empty:
                    df_final.to_excel(os.path.join(nowcast_dir, "impactos_consolidado.xlsx"), index=False)
                    st.success("Impactos consolidados guardados.")

                st.write("### Evolución de estimaciones e impactos")
                try:
                    # 1. Construir DataFrame de estimaciones (t6)
                    point_estimates = []
                    vintages_list = list(news_results.keys())
                    
                    # Primera estimación (scalar)
                    p_vintage = ficheros[0]
                    p_est = float(vintage_results[p_vintage].get_prediction(
                        start='2006-01', end=end_date
                    ).predicted_mean['pib'].resample('Q').mean().iloc[-1])
                    
                    point_estimates.append({'vintage': p_vintage, 'point estimate': p_est})
                    
                    # Estimaciones de cada vintage posterior
                    for vintage in vintages_list:
                        try:
                            # Asegurar que sacamos un escalar
                            impacto = float(news_results[vintage].total_impacts.iloc[-1])
                        except:
                            impacto = float(news_results[vintage].details_by_impact['impact'].sum())
                        
                        nueva_estimacion = impacto + point_estimates[-1]['point estimate']
                        point_estimates.append({'vintage': vintage, 'point estimate': nueva_estimacion})
                    
                    t6 = pd.DataFrame(point_estimates).set_index('vintage')
                    
                    # 2. Construir DataFrame de impactos por variable (t8)
                    impacts_by_v = {p_vintage: {}}
                    for vintage in vintages_list:
                        details = news_results[vintage].details_by_impact.reset_index()
                        # Asegurar que el impacto es numérico
                        details['impact'] = pd.to_numeric(details['impact'], errors='coerce')
                        impacts = details.groupby('updated variable')['impact'].sum()
                        impacts_by_v[vintage] = impacts.to_dict()
                    
                    all_v = set().union(*(d.keys() for d in impacts_by_v.values()))
                    t8_data = []
                    for vintage in [p_vintage] + vintages_list:
                        row = {'vintage': vintage}
                        for var in all_v:
                            row[var] = float(impacts_by_v[vintage].get(var, 0))
                        t8_data.append(row)
                    
                    t8 = pd.DataFrame(t8_data).set_index('vintage')
                    # Asegurar tipos numéricos para el gráfico
                    for col in t8.columns:
                        t8[col] = pd.to_numeric(t8[col], errors='coerce').fillna(0)
                    t8['Total'] = t8.sum(axis=1)

                    # 3. Función de visualización completa (mejorada)
                    def plot_nowcasting_summary_robust(data_impacts, data_estimates):
                        data = data_impacts.drop('Total', axis=1, errors='ignore')
                        
                        fig, ax1 = plt.subplots(figsize=(14, 10))
                        fig.patch.set_facecolor('#f8f9fa')
                        ax1.set_facecolor('#f8f9fa')
                        ax2 = ax1.twinx()
                        
                        variables = data.columns.tolist()
                        colors = sns.color_palette("deep", n_colors=max(len(variables), 1))
                        color_map = dict(zip(variables, colors))
                        x = np.arange(len(data))
                        x_labels = [str(label).split(' ')[0] for label in data.index]
                        
                        bases_pos = np.zeros(len(data))
                        bases_neg = np.zeros(len(data))
                        
                        for column in data.columns:
                            valores = data[column].values
                            pos = np.where(valores > 0, valores, 0)
                            neg = np.where(valores < 0, valores, 0)
                            
                            if np.any(pos != 0):
                                ax1.bar(x, pos, bottom=bases_pos, color=color_map[column], label=column, alpha=0.8, edgecolor='white')
                                bases_pos += pos
                            if np.any(neg != 0):
                                ax1.bar(x, neg, bottom=bases_neg, color=color_map[column], alpha=0.8, edgecolor='white')
                                bases_neg += neg

                        estimates = data_estimates['point estimate'].values
                        ax2.plot(x, estimates, 'o-', color='darkred', linewidth=3, markersize=12, label='Estimación PIB', zorder=5)
                        for i, val in enumerate(estimates):
                            ax2.text(x[i], val + 0.01, f'{val:.2f}', ha='center', va='bottom', color='darkred', fontweight='bold')
                        
                        ax1.set_xticks(x)
                        ax1.set_xticklabels(x_labels, rotation=45, ha='right')
                        ax1.set_ylabel('Impacto por variable')
                        ax2.set_ylabel('Estimación PIB (%)', color='darkred')
                        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
                        plt.tight_layout()
                        plt.subplots_adjust(bottom=0.25)
                        return fig

                    fig_nowcast = plot_nowcasting_summary_robust(t8, t6)
                    st.pyplot(fig_nowcast)
                    plt.close(fig_nowcast)
                    
                    with st.expander("📋 Ver datos de la visualización"):
                        st.dataframe(t8)
                except Exception as e:
                    st.warning(f"Error en gráfico final: {str(e)}")

