import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from utils import *
import io
import contextlib
import sys



# LECTURA DE FICHEROS



# Directorio donde están los archivos Excel
directorio = r"C:\Users\Usuario\Documents\Github\IECA\Datos"  # Reemplaza esto con la ruta real

# Patrón para extraer la fecha del nombre del archivo
patron_fecha = r'Envío_(\d{2})_(\d{2})_(\d{4})'

nombre_hoja_mensual = 'Series_mens_vol_y_desest'  # nombre de la hoja en el archivo Excel
nombre_hoja_trimestre = 'Serie trim_vol_desest_Índice'  # nombre de la hoja en el archivo Excel

# Diccionario para almacenar los DataFrames
dfs_m = {}
dfs_q = {}

# Iterar sobre los archivos en el directorio
for archivo in os.listdir(directorio):
    if archivo.endswith('.xlsx'):  # Asumiendo que son archivos Excel
        ruta_completa = os.path.join(directorio, archivo)
        
        # Extraer la fecha del nombre del archivo
        match = re.search(patron_fecha, archivo)
        if match:
            dia, mes, anio = match.groups()
            fecha = datetime(int(anio), int(mes), int(dia)).strftime('%Y-%m-%d')
            
            # Leer el archivo Excel
            df_mens = pd.read_excel(ruta_completa, nombre_hoja_mensual)
            df_mens['fecha'] = pd.to_datetime(df_mens['fecha']).dt.strftime('%m-%y')
            # Establece la columna 'fecha' como el índice del DataFrame
            df_mens.set_index('fecha', inplace=True)
            df_mens = df_mens.rolling(window=3).mean()
            df_mens = df_mens[2:]
            df_trims = pd.read_excel(ruta_completa, nombre_hoja_trimestre)
            df_trims['fecha'] = pd.to_datetime(df_trims['fecha']).dt.strftime('%m-%y')
            # Establece la columna 'Fecha' como el índice del DataFrame
            df_trims.set_index('fecha', inplace=True)

            #Estacionaridad de series 

            df_estacionario_mensual, diferencias_por_columna = make_all_stationary_m(df_mens)
            df_estacionario_trimestral, diferencias_por_columna = make_all_stationary_q(df_trims)

            df_estacionario_mensual['Pernoctaciones - Andalucía'] = df_estacionario_mensual['Pernoctaciones - Andalucía'].diff(3)
            df_estacionario_mensual['Total afiliados SS Total - Andalucía'] = df_mens['Total afiliados SS Total - Andalucía'].diff(3)

            endog_m = df_estacionario_mensual
            endog_q = df_estacionario_trimestral*100

                        
            #Hago merge con datos mensuales "trimestralizados" y el trimestral

            # Convertir el índice del DataFrame mensual a formato de fecha y hora
            endog_m.index = pd.to_datetime(endog_m.index, format='%m-%y')
            # Convertir el índice del DataFrame mensual a formato de fecha y hora
            endog_q.index = pd.to_datetime(endog_q.index, format='%m-%y')

            endog_q.index = endog_q.index.to_period('Q')
            endog_m.index = endog_m.index.to_period('M')
         
            # Almacenar el DataFrame en el diccionario con la fecha como clave
            dfs_m[fecha] = endog_m
            dfs_q[fecha] = endog_q

            
ficheros = list(dfs_m.keys())

# ESTIMACIÓN DE MODELO

mejores_regresores = ['Importaciones de bienes - Andalucía', 
                        'Liquidación de presupuestos de la Junta de Andalucía. Capítulo 1 - Andalucía', 
                        'Total afiliados SS Total - Andalucía', 
                        'Matriculación de turismos - Andalucía', 
                        'Consumo de gasolina y gasóleo', 
                        'Cifra negocios del sector servicios - Andalucía'
                        ]


endog_m_c = dfs_m[ficheros[0]][mejores_regresores]



endog_q=dfs_q[ficheros[0]][['pib']]

mod = sm.tsa.DynamicFactorMQ(endog_m_c, 
                            endog_quarterly=endog_q, 
                            factor_orders = 2, 
                            factors=2,
                            factor_multiplicities=1)
res = mod.fit(maxiter=1000)

