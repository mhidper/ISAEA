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
    page_title="Procesador de Datos IECA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("Procesamiento de Datos IECA")

# Configuración en el sidebar
st.sidebar.header("Configuración")

# Entrada para la ruta del proyecto
default_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


for directory in [OUTPUT_DIR, OUTPUT_MONTHLY_DIR, OUTPUT_QUARTERLY_DIR, 
                 OUTPUT_NOWCASTING_DIR, OUTPUT_RESULTS_DIR]:
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

def generar_excel_impactos(news_results, ficheros):
    """
    Genera un DataFrame con todos los impactos de cada vintage para guardar en Excel.
    """
    if not news_results:
        return pd.DataFrame()

    print(f"Procesando {len(news_results)} resultados de news")
    print(f"Ficheros disponibles: {ficheros}")
    print(f"News_results keys: {list(news_results.keys())}")

    all_impacts = []
    
    # Iteramos desde el segundo fichero para tener siempre un 'prev_vintage'
    for i in range(1, len(ficheros)):
        vintage = ficheros[i]
        prev_vintage = ficheros[i-1]
        
        if vintage in news_results:
            result = news_results[vintage]
            key = f"{prev_vintage} -> {vintage}"
            
            print(f"\nProcesando {key}")
            
            # Obtener detalles de impacto
            impact_df = result.details_by_impact.reset_index()
            print(f"Shape de impact_df para {key}: {impact_df.shape}")
            print(f"Columnas: {impact_df.columns.tolist()}")
            print(f"Fechas de actualización únicas: {impact_df['update date'].unique()}")
            
            # Preparar datos para concatenar - CLAVE: mantener tanto variable como fecha
            temp_data = {
                'updated_variable': impact_df['updated variable'],
                'update_date': impact_df['update date'],
                f'impacto_{key}': impact_df['impact']
            }
            
            temp_df = pd.DataFrame(temp_data)
            all_impacts.append(temp_df)
            
            print(f"Datos preparados para {key}:")
            print(temp_df.head())
    
    if not all_impacts:
        return pd.DataFrame()
    
    # CORRECCIÓN CLAVE: Combinar usando AMBAS claves: variable Y fecha
    df_final = all_impacts[0]
    
    for i in range(1, len(all_impacts)):
        print(f"\nHaciendo merge {i}")
        print(f"df_final antes del merge: {df_final.shape}")
        print(f"all_impacts[{i}] shape: {all_impacts[i].shape}")
        
        # Merge por AMBAS claves: updated_variable Y update_date
        df_final = pd.merge(
            df_final, 
            all_impacts[i], 
            on=['updated_variable', 'update_date'],  # ←← ESTO ES LO CLAVE
            how='outer'
        )
        
        print(f"df_final después del merge: {df_final.shape}")
    
    # Renombrar columnas para que sean más claras
    df_final.rename(columns={'updated_variable': 'updated variable', 
                            'update_date': 'update date'}, inplace=True)
    
    # Ordenar por variable y fecha para mejor legibilidad
    df_final = df_final.sort_values(['updated variable', 'update date'])
    
    print(f"\nDataFrame final shape: {df_final.shape}")
    print(f"Columnas finales: {df_final.columns.tolist()}")
    print(f"Preview del resultado:")
    print(df_final.head(10))
    
    return df_final


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
        end_date = st.sidebar.date_input(
            "Fecha final de predicción",
            value=datetime(2025, 12, 31),
            min_value=datetime(2000, 1, 1),
            max_value=datetime(2030, 12, 31),
            help="Selecciona la fecha final para las predicciones. Indica siempre el último día del trimestre."
        ).strftime('%Y-%m')

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
            try:
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
                                .resample('Q').first()
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
                    
                    # Mensualizar índices trimestrales
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

                    # Función para conversión segura a float
                    def safe_convert_to_float(arr):
                        series = pd.Series(arr)
                        numeric_series = pd.to_numeric(series, errors='coerce')
                        cleaned_series = numeric_series.dropna()
                        return cleaned_series.to_numpy()

                    # Crear DataFrame para resultados mensuales
                    resultados_mensuales = pd.DataFrame(index=df_growth_based.index)

                    # Procesar cada columna
                    for columna in df_growth_based.columns:
                        X_mensual = df_growth_based[columna].to_numpy()
                        
                        if columna in indices_trimestrales.columns:
                            y_trimestral = indices_trimestrales[columna].to_numpy()
                        else:
                            st.write(f"La columna {columna} no está en indices_trimestrales. Se omite.")
                            continue

                        X_mensual_clean = safe_convert_to_float(X_mensual)
                        y_trimestral_clean = safe_convert_to_float(y_trimestral)

                        z_mensual = denton_method(y_trimestral_clean, X_mensual_clean) * 3
                        z_mensual = pd.Series(z_mensual, index=df_growth_based.index)

                        resultados_mensuales[columna] = z_mensual

                    # Guardar todos los resultados
                    forecast_mensuales.to_csv(os.path.join(OUTPUT_MONTHLY_DIR, "indicador_mensual.csv"), sep=";")
                    forecast_trimestrales.to_csv(os.path.join(OUTPUT_QUARTERLY_DIR, "indicador_trimestral.csv"), sep=";")
                    indices_trimestrales.to_csv(os.path.join(OUTPUT_QUARTERLY_DIR, "indices_trimestrales.csv"), sep=";")
                    resultados_mensuales.to_csv(os.path.join(OUTPUT_MONTHLY_DIR, "indices_mensuales.csv"), sep=";")

                    # Mostrar resultados en la interfaz
                    st.write("Resultados calculados correctamente")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Preview de índices trimestrales:")
                        st.dataframe(indices_trimestrales.tail())
                    
                    with col2:
                        st.write("Preview de índices mensuales:")
                        st.dataframe(resultados_mensuales.tail())

                    # Verificar que el directorio existe
                    output_file = os.path.join(OUTPUT_RESULTS_DIR, "resultados_estimaciones.txt")

                    with open(output_file, 'w') as f:
                        # Escribir una cabecera
                        f.write("Resultados de las estimaciones\n")
                        f.write("=============================\n\n")
                        
                        f.write("Estimación del modelo DFMQ\n\n")
                        f.write("=============================\n\n")

                        # Capturar el output del summary() en una variable
                        old_stdout = sys.stdout
                        new_stdout = io.StringIO()
                        sys.stdout = new_stdout

                        print(res.summary())

                        sys.stdout = old_stdout
                        summary_output = new_stdout.getvalue()

                        # Escribir el summary en el archivo
                        f.write(summary_output)

                        # Añadir información adicional
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
                            
                            # Guardar para el DataFrame combinado
                            top_factor.columns = pd.MultiIndex.from_product([
                                [f'Top ten variables explained by {factor_name}'],
                                ['Variable', r'$R^2$']])
                            top_ten.append(top_factor)

                        # Crear y escribir el DataFrame combinado
                        f.write("\nTabla combinada de top variables por factor:\n")
                        combined_df = pd.concat(top_ten, axis=1)
                        f.write(combined_df.to_string())

                    # Mostrar los resultados en la pantalla
                    st.text(summary_output)
                    st.write("Información adicional:")
                    st.write(f"AIC: {res.aic}")
                    st.write(f"BIC: {res.bic}")
                    st.write("\nCoeficientes de determinación (R^2):")
                    st.dataframe(combined_df)
                    
                    st.success(f"Los resultados han sido guardados en: {output_file}")

            except Exception as e:
                st.error(f"Error al estimar el modelo: {str(e)}")

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

            # Configurar el estilo
            sns.set(style="white")
            sns.set_context("notebook", font_scale=0.8)

            # Crear figura
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

            # Subplot mensual
            for i, col in enumerate(forecast_mensuales_f.columns):
                sns.lineplot(data=forecast_mensuales_f, 
                            x=forecast_mensuales_f.index, 
                            y=col, 
                            color=palette_mensual[i], 
                            label=col, 
                            ax=ax1)

            ax1.set_title('Mensual')
            ax1.set_ylabel('Valor')
            ax1.legend(title='Columnas')

            # Subplot trimestral
            for i, col in enumerate(forecast_trimestrales_f.columns):
                sns.lineplot(data=forecast_trimestrales_f, 
                            x=forecast_trimestrales_f.index, 
                            y=col, 
                            color=palette_trimestral[i], 
                            label=col, 
                            ax=ax2)

            ax2.set_title('Trimestral')
            ax2.set_ylabel('Valor')

            # Ajustar layout
            plt.suptitle('Comparación de ISAEA por ola de indicadores', 
                        fontsize=16, 
                        y=1.02)
            plt.tight_layout()

            # Mostrar figura en Streamlit
            st.pyplot(fig)

            # Opción para descargar la figura
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
            buf.seek(0)
            
            st.download_button(
                label="Descargar figura",
                data=buf,
                file_name="ISAEA_comparison.png",
                mime="image/png"
            )

            # Cerrar la figura para liberar memoria
            plt.close(fig)

            # Código mejorado para la sección de nowcasting (reemplaza desde la línea del análisis de nowcasting)
            # Análisis de Nowcasting
            st.write("Realizando análisis de nowcasting...")

            # Crear directorio para nowcasting si no existe
            nowcast_dir = os.path.join(OUTPUT_DIR, "Nowcasting")
            if not os.path.exists(nowcast_dir):
                os.makedirs(nowcast_dir)

            # Calcular resultados de nowcasting
            news_results = {}
            impact_date = end_date

            # Progress bar para el proceso
            progress_bar = st.progress(0)

            for i in range(1, len(ficheros)):
                vintage = ficheros[i]
                prev_vintage = ficheros[i - 1]
                
                # Actualizar progress bar
                progress = (i) / (len(ficheros) - 1)
                progress_bar.progress(progress)
                
                st.write(f"Procesando vintage: {vintage}")
                
                # Calcular news
                news_results[vintage] = vintage_results[vintage].news(
                    vintage_results[prev_vintage],
                    impact_date=impact_date,
                    impacted_variable='pib',
                    comparison_type='previous'
                )

            # Guardar resultados individuales y crear descargas
            st.write("Guardando resultados del nowcasting...")

            # Crear contenedor para descargas
            download_container = st.container()

            with download_container:
                st.subheader("Descargas individuales de nowcasting")
                
                # Crear archivos individuales para cada resultado
                individual_files = {}
                
                for i, (name, result) in enumerate(news_results.items()):
                    prev_vintage = ficheros[i]  # i porque empezamos desde 1
                    current_vintage = name
                    key = f"{prev_vintage} -> {current_vintage}"
                    
                    # Crear DataFrame detallado para este nowcasting
                    impact_df = result.details_by_impact.reset_index()
                    
                    # Información del summary
                    summary_text = result.summary().as_text()
                    
                    # Crear DataFrame con información completa
                    detailed_df = pd.DataFrame({
                        'Comparison': [key] * len(impact_df),
                        'Updated Variable': impact_df['updated variable'],
                        'Update Date': impact_df['update date'],
                        'Impact': impact_df['impact'],
                        'Previous Forecast': impact_df.get('previous forecast', 'N/A'),
                        'Revised Forecast': impact_df.get('revised forecast', 'N/A')
                    })
                    
                    # Guardar en Excel
                    excel_filename = f'nowcasting_{prev_vintage}_to_{current_vintage}.xlsx'
                    excel_path = os.path.join(nowcast_dir, excel_filename)
                    
                    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                        # Hoja con los impactos detallados
                        detailed_df.to_excel(writer, sheet_name='Impacts', index=False)
                        
                        # Hoja con el summary como texto (convertido a DataFrame)
                        summary_lines = summary_text.split('\n')
                        summary_df = pd.DataFrame({'Summary': summary_lines})
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Almacenar para descargas en Streamlit
                    individual_files[key] = excel_path
                    
                    # Crear botón de descarga para cada archivo
                    with open(excel_path, 'rb') as f:
                        st.download_button(
                            label=f"📊 Descargar {key}",
                            data=f.read(),
                            file_name=excel_filename,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    # Mostrar preview en un expander
                    with st.expander(f"Ver preview de {key}"):
                        st.dataframe(detailed_df)
                        st.text_area(f"Summary para {key}", summary_text, height=200)

            # Generar archivo consolidado de impactos
            st.write("### Generando archivo consolidado de impactos")

            if news_results:
                # Generar el DataFrame final
                df_final = generar_excel_impactos(news_results, ficheros)
                
                if not df_final.empty:
                    # Mostrar preview del archivo consolidado
                    st.write("Preview del archivo consolidado:")
                    st.dataframe(df_final)
                    
                    # Guardar archivo consolidado
                    consolidated_path = os.path.join(nowcast_dir, "impactos_consolidado.xlsx")
                    df_final.to_excel(consolidated_path, index=False, sheet_name='Impactos_Consolidados')
                    
                    # Botón de descarga para el archivo consolidado
                    with open(consolidated_path, 'rb') as f:
                        st.download_button(
                            label="📈 Descargar Impactos Consolidados",
                            data=f.read(),
                            file_name="impactos_consolidado.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    st.success(f"Archivo consolidado guardado en: {consolidated_path}")
                else:
                    st.warning("No se generó información de impactos consolidados.")
                    
                # Información de debug
                with st.expander("🔍 Información de debug"):
                    st.write("**Archivos procesados:**")
                    for i, fichero in enumerate(ficheros):
                        st.write(f"{i}: {fichero}")
                    
                    st.write("**Resultados de news disponibles:**")
                    for key, result in news_results.items():
                        impact_shape = result.details_by_impact.shape
                        st.write(f"- {key}: {impact_shape[0]} variables con impacto")

            st.success(f"Todos los resultados de nowcasting guardados en: {nowcast_dir}")

