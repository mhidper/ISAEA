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

#Direcciones para guardar ficheros

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    Genera un archivo Excel con todos los impactos de cada vintage.
    
    Args:
        news_results (dict): Diccionario con los resultados de news entre diferentes vintages
        ficheros (list): Lista de fechas de los vintages en orden cronológico
    
    Returns:
        BytesIO: Buffer con el archivo Excel generado
    """
    import pandas as pd
    import io
    
    # Creamos una lista vacía para almacenar los DataFrames de cada período
    dfs = []
    
    # Diccionario para almacenar los detalles por impacto
    impactos = {}
    
    # Extraer los detalles de impacto de cada resultado de news
    for i in range(1, len(ficheros)):
        vintage = ficheros[i]
        prev_vintage = ficheros[i-1]
        key = f"{prev_vintage} -> {vintage}"
        
        # Obtenemos los detalles de impacto y los guardamos
        impactos[key] = news_results[vintage].details_by_impact.reset_index()
    
    # Procesamos cada período
    for key in impactos.keys():
        # Creamos un DataFrame temporal para este período
        temp_df = impactos[key][['update date', 'updated variable', 'impact']]
        temp_df = temp_df.rename(columns={'impact': f'impacto {key}'})
        dfs.append(temp_df)
    
    # Si no hay datos, devolvemos un DataFrame vacío
    if not dfs:
        return pd.DataFrame()
    
    # Hacemos el merge de todos los DataFrames
    df_final = dfs[0]
    for df in dfs[1:]:
        df_final = pd.merge(
            df_final,
            df,
            on='updated variable',
            how='outer',
            suffixes=('', f'_{len(dfs)}')
        )
    
    # Renombramos las columnas para hacerlas únicas
    df_final.columns = pd.Index([f"{col}_{i}" if df_final.columns[:i].str.contains(col).any() else col 
                                for i, col in enumerate(df_final.columns)])
    
    # Ahora identificamos las columnas de update date
    update_date_cols = [col for col in df_final.columns if 'update date' in col.lower()]
    
    # Creamos una nueva columna combinando todas las columnas de update date
    df_final['update date_final'] = None  # Inicializamos la nueva columna
    
    # Iteramos por cada columna de update date
    for col in update_date_cols:
        current_mask = df_final['update date_final'].isna()
        df_final.loc[current_mask, 'update date_final'] = df_final.loc[current_mask, col]
    
    # Eliminamos las columnas originales de update date
    df_final = df_final.drop(columns=update_date_cols)
    
    # Renombramos la columna final
    df_final = df_final.rename(columns={'update date_final': 'update date'})
    
    # Crear un buffer para guardar el Excel en memoria
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_final.to_excel(writer, index=False, sheet_name='Impactos')
    buffer.seek(0)
    
    return buffer, df_final

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
            value=datetime(2024, 12, 31),
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
            # Diagnóstico simple
            st.write("Diagnóstico de news_results:")
            st.write(f"¿news_results existe y tiene elementos? {bool(news_results)}")
            st.write(f"Número de elementos: {len(news_results) if news_results else 0}")

            # Calcular impactos
                
            
            # Guardar resultados
            st.write("Guardando resultados del nowcasting...")
            
            # Crear un expander para mostrar los resultados
            with st.expander("Ver resultados del nowcasting"):
                for name, result in news_results.items():
                    output_file = os.path.join(nowcast_dir, f'summary_{name}.txt')
                    
                    # Guardar en archivo
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result.summary().as_text())
                    
                    # Mostrar en la interfaz
                    st.subheader(f"Resultados para {name}")
                    st.text(result.summary().as_text())
        
        
            # Código para descargar los ficheros de impacto:
            if news_results:
                st.write("### Descargar información de impactos")
                st.write("Puede descargar un archivo Excel con todos los impactos detallados por vintage.")
                
                # Añadir selector de directorio para guardar el archivo
                output_impactos_dir = st.text_input(
                    "Directorio para guardar impactos (opcional)",
                    value=OUTPUT_RESULTS_DIR,  # Usar directorio de resultados por defecto
                    help="Si se especifica, se guardará una copia del archivo Excel en este directorio"
                )
                
                # Botón para generar y descargar el archivo Excel
                if st.button("Generar Excel de impactos"):
                    # Mostrar spinner mientras se genera el archivo
                    with st.spinner("Generando archivo Excel..."):
                        try:
                            # Generar el Excel
                            excel_buffer, df_impactos = generar_excel_impactos(news_results, ficheros)
                            
                            # Guardar en el directorio especificado
                            if output_impactos_dir:
                                # Asegurar que el directorio existe
                                os.makedirs(output_impactos_dir, exist_ok=True)
                                
                                # Ruta completa del archivo
                                output_file = os.path.join(output_impactos_dir, "impactos.xlsx")
                                
                                # Guardar archivo en disco
                                df_impactos.to_excel(output_file, index=False, sheet_name='Impactos')
                                st.success(f"Archivo guardado en: {output_file}")
                            
                            # Ofrecer descarga
                            st.download_button(
                                label="Descargar Excel de impactos",
                                data=excel_buffer,
                                file_name="impactos.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                            st.success("Excel generado correctamente. Haga clic en el botón de descarga.")
                        except Exception as e:
                            st.error(f"Error al generar el Excel: {e}")
                            
            st.success(f"Resultados de nowcasting guardados en: {nowcast_dir}")

