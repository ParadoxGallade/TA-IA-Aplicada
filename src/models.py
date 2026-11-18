"""Implementación de modelos.

Predicción del número mensual de siniestros de tránsito fatales por departamento (ONSV 2021-2023).

Baseline:           regresión lineal y árbol de regresión (vistos en clase; Özden y Acı, 2018)
Vistos en clase:    KNN y MLP
De la literatura:   SVR (Özden y Acı, 2018) y Random Forest (Herrera, 2021)
Referencias:        dos pronósticos ingenuos que no entrenan nada (promedio móvil y mes anterior)

Este archivo solo define las piezas (datos, partición, modelos, búsqueda de hiperparámetros).
El flujo completo, con resultados y gráficos, está en notebooks/02_modelos.ipynb.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# El MLP avisa cuando no termina de converger en algunas combinaciones de la búsqueda;
# se oculta ese aviso para que la salida se lea mejor.
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Semilla fija para que los resultados se repitan (árbol, MLP y Random Forest usan azar)
SEMILLA = 42

# Entradas de los modelos finales: solo información del pasado + mes del año.
# El departamento se agrega aparte con One-Hot Encoding (ver preparar_xy).
ENTRADAS = ['lag_1', 'lag_2', 'lag_3', 'prom_movil_3', 'mes']

# Entradas de la versión anterior (con rezagos de 6 y 12 meses). Solo se usan en la prueba
# de rezagos: como lag_12 necesita un año de historia, obliga a descartar todo 2021.
ENTRADAS_CON_LAG_6_12 = ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'prom_movil_3', 'mes']

# Partición temporal (NO aleatoria, para no usar información del futuro)
FIN_SUBTRAIN = '2022-12'   # sub-train: hasta dic-2022
FIN_TRAIN = '2023-06'      # validación: ene-jun 2023 | test: jul-dic 2023


# =============================================================================
# 1. Datos: X, y y partición temporal
# =============================================================================

def preparar_xy(df, entradas=ENTRADAS, usar_departamento=True):
    """Devuelve X (entradas) e y (siniestros del mes).

    El departamento es categórico (25 valores), así que se convierte con One-Hot Encoding.
    """
    if usar_departamento:
        X = pd.get_dummies(df[entradas + ['departamento']], columns=['departamento'], dtype=int)
    else:
        X = df[entradas].copy()
    y = df['siniestros']
    return X, y


def dividir_temporal(df):
    """Devuelve las máscaras (True/False por fila) de cada conjunto.

    sub-train  : inicio a dic-2022   -> para probar hiperparámetros
    validación : ene-2023 a jun-2023 -> para elegir hiperparámetros
    train      : sub-train + validación -> para el ajuste final
    test       : jul-2023 a dic-2023 -> solo para la evaluación final
    """
    periodo = df['periodo']
    return {
        'subtrain': periodo <= FIN_SUBTRAIN,
        'val': (periodo > FIN_SUBTRAIN) & (periodo <= FIN_TRAIN),
        'train': periodo <= FIN_TRAIN,
        'test': periodo > FIN_TRAIN,
    }


# =============================================================================
# 2. Modelos e hiperparámetros a probar
# =============================================================================

def crear_modelos():
    """Devuelve los 6 modelos con los valores de hiperparámetros que se van a probar.

    'escalar' = True para los modelos que usan distancias o pesos (KNN, MLP, SVR):
    necesitan que todas las entradas estén en la misma escala.
    """
    return {
        'Regresión lineal': {
            'modelo': LinearRegression(),
            'grid': {},
            'escalar': False},
        'Árbol de regresión': {
            'modelo': DecisionTreeRegressor(random_state=SEMILLA),
            'grid': {'max_depth': [3, 4, 5, 6, 8],
                     'min_samples_leaf': [1, 5, 10]},
            'escalar': False},
        'KNN': {
            'modelo': KNeighborsRegressor(),
            'grid': {'n_neighbors': [3, 5, 7, 9, 11, 15],
                     'weights': ['uniform', 'distance']},
            'escalar': True},
        'MLP': {
            'modelo': MLPRegressor(max_iter=3000, random_state=SEMILLA),
            'grid': {'hidden_layer_sizes': [(16,), (32,), (32, 16)],
                     'alpha': [0.0001, 0.01, 1.0],
                     'learning_rate_init': [0.001, 0.01]},
            'escalar': True},
        'SVR': {
            'modelo': SVR(),
            'grid': {'kernel': ['linear', 'rbf'],
                     'C': [1, 10, 100],
                     'epsilon': [0.1, 0.5, 1.0]},
            'escalar': True},
        'Random Forest': {
            'modelo': RandomForestRegressor(random_state=SEMILLA),
            'grid': {'n_estimators': [100, 300],
                     'max_depth': [None, 5, 10],
                     'min_samples_leaf': [1, 5]},
            'escalar': False},
    }


# =============================================================================
# 3. Búsqueda de hiperparámetros y entrenamiento
# =============================================================================

def _escalar(X_ajuste, X_otro):
    """Escala con StandardScaler ajustado SOLO con los datos de ajuste (evita data leakage)."""
    scaler = StandardScaler().fit(X_ajuste)
    return scaler.transform(X_ajuste), scaler.transform(X_otro)


def buscar_hiperparametros(cfg, X_subtrain, y_subtrain, X_val, y_val):
    """Búsqueda en grilla: prueba todas las combinaciones del grid.

    Entrena con sub-train, mide el MAE en validación y se queda con la combinación de
    menor MAE. Devuelve (mejores hiperparámetros, MAE en validación).
    """
    if cfg['escalar']:
        X_subtrain, X_val = _escalar(X_subtrain, X_val)

    mejor_mae, mejores = np.inf, {}
    for params in ParameterGrid(cfg['grid']):
        modelo = cfg['modelo'].set_params(**params)
        modelo.fit(X_subtrain, y_subtrain)
        mae_val = mean_absolute_error(y_val, modelo.predict(X_val))
        if mae_val < mejor_mae:
            mejor_mae, mejores = mae_val, params
    return mejores, mejor_mae


def entrenar_y_predecir(cfg, params, X_train, y_train, X_test):
    """Reentrena con TODO train usando los hiperparámetros elegidos y predice en test.

    Devuelve (modelo entrenado, predicciones en test).
    """
    if cfg['escalar']:
        X_train, X_test = _escalar(X_train, X_test)
    modelo = cfg['modelo'].set_params(**params)
    modelo.fit(X_train, y_train)
    return modelo, modelo.predict(X_test)


# =============================================================================
# 4. Referencias ingenuas (no entrenan nada)
# =============================================================================

def predicciones_ingenuas(df):
    """Dos pronósticos sin modelo, para saber si entrenar un modelo vale la pena.

    - Promedio móvil: "este mes habrá el promedio de los 3 meses anteriores".
    - Mes anterior:   "este mes habrá lo mismo que el mes pasado".
    """
    return {
        'Ref. promedio móvil (3 meses)': df['prom_movil_3'],
        'Ref. mes anterior': df['lag_1'],
    }
