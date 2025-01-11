import numpy as np
from statsmodels.tsa.stattools import adfuller
import pandas as pd


#FUNCIONES
#----------------------------
#Estacionariedad

def make_all_stationary_m(df):
    transformed_df = df.copy()
    transformed_df = transformed_df.apply(np.log)
    
    diff_counts = {}  # Diccionario para almacenar el número de diferenciaciones por columna

    for column in transformed_df.columns:
        series = transformed_df[column]
        
        # Imputa valores faltantes con la media de la serie
        series = series.dropna()
        
        p_value = 1  # Inicializa el p-valor con un valor alto para entrar al bucle
        diff_count = 0  # Contador de veces que se ha diferenciado la serie

        while p_value > 0.05:  # Hasta que la serie sea estacionaria (p-valor menor que 0.05)
            result = adfuller(series)
            p_value = result[1]  # Obtiene el p-valor del test ADF
            
            if p_value > 0.05:  # Si la serie no es estacionaria
                series = series.diff(1).dropna()  # Aplica una diferenciación
                diff_count += 1  # Incrementa el contador de diferenciaciones
        
        transformed_df[column] = series  # Actualiza la columna en el DataFrame transformado
        diff_counts[column] = diff_count  # Almacena el número de diferenciaciones en el diccionario
    
    return transformed_df, diff_counts

def make_all_stationary_q(df):
    transformed_df = df.copy()
    transformed_df = transformed_df.apply(np.log)
    
    diff_counts = {}  # Diccionario para almacenar el número de diferenciaciones por columna
    
    for column in transformed_df.columns:
        series = transformed_df[column].dropna()
        
        # Imputa valores faltantes con la media de la serie
        #series = series.fillna(series.mean())
        
        p_value = 1  # Inicializa el p-valor con un valor alto para entrar al bucle
        diff_count = 0  # Contador de veces que se ha diferenciado la serie

        while p_value > 0.05:  # Hasta que la serie sea estacionaria (p-valor menor que 0.05)
            result = adfuller(series)
            p_value = result[1]  # Obtiene el p-valor del test ADF
            
            if p_value > 0.05:  # Si la serie no es estacionaria
                series = series.diff().dropna()  # Aplica una diferenciación
                diff_count += 1  # Incrementa el contador de diferenciaciones
        
        transformed_df[column] = series  # Actualiza la columna en el DataFrame transformado
        diff_counts[column] = diff_count  # Almacena el número de diferenciaciones en el diccionario
    
    return transformed_df, diff_counts

# Normalización
def standardize_df(df):
    # Copia el DataFrame para evitar modificar el original
    standardized_df = df.copy()

    
    # Estandariza cada columna del DataFrame
    for column in standardized_df.columns:
        col_mean = standardized_df[column].mean()
        col_std = standardized_df[column].std()
        
        # Aplica la fórmula de estandarización (normalización con media y desviación estándar)
        standardized_df[column] = (standardized_df[column] - col_mean) / col_std
    
    return standardized_df

# ------------------------------
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