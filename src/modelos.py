# Comparación de modelos - Siniestros fatales por departamento y mes (ONSV 2021-2023)
#
# Baseline: regresión lineal y árbol de regresión
# Modelos:  KNN, MLP (vistos en clase), SVR (Özden y Acı, 2018) y Random Forest (Herrera, 2021)
#
# Entrada: data/processed/dataset_modelo.csv  (generado por src/preprocessing.py)
#
# Ejecutar desde la carpeta principal del repositorio:
#   python src/modelos.py

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.exceptions import ConvergenceWarning

# El MLP avisa cuando no termina de converger en algunas combinaciones de la búsqueda;
# se oculta ese aviso para que la salida se lea mejor.
warnings.filterwarnings('ignore', category=ConvergenceWarning)


# =============================================================================
# 1. Cargar datos y armar X, y
# =============================================================================

df = pd.read_csv('data/processed/dataset_modelo.csv')
print('Dataset:', df.shape, '| periodos:', df['periodo'].min(), 'a', df['periodo'].max())

# Entradas: solo información del pasado (lags y promedio móvil) + mes del año + departamento.
# 'anio' no se usa: en el dataset del modelo solo hay 2022 y 2023.
features = ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'prom_movil_3', 'mes']

# El departamento es categórico (25 valores) -> One-Hot Encoding
X = pd.get_dummies(df[features + ['departamento']], columns=['departamento'], dtype=int)
y = df['siniestros']
print('Número de variables de entrada:', X.shape[1])


# =============================================================================
# 2. División temporal (NO aleatoria, para evitar data leakage)
# =============================================================================

# Train: ene-2022 a jun-2023 (18 meses) | Test: jul-2023 a dic-2023 (6 meses)
# Para elegir hiperparámetros se separa una validación dentro de train:
#   sub-train: 2022 completo | validación: ene-2023 a jun-2023
es_train = df['periodo'] <= '2023-06'
es_test = df['periodo'] > '2023-06'
es_subtrain = df['periodo'] <= '2022-12'
es_val = (df['periodo'] > '2022-12') & (df['periodo'] <= '2023-06')

X_train, y_train = X[es_train], y[es_train]
X_test, y_test = X[es_test], y[es_test]
X_subtrain, y_subtrain = X[es_subtrain], y[es_subtrain]
X_val, y_val = X[es_val], y[es_val]

print('Train:', len(X_train), '| Test:', len(X_test),
      '| Sub-train:', len(X_subtrain), '| Validación:', len(X_val))


# =============================================================================
# 3. Modelos e hiperparámetros a probar
# =============================================================================

# escalar = True para los modelos que usan distancias o pesos (KNN, MLP, SVR).
# El scaler se ajusta SOLO con los datos de entrenamiento.
modelos = {
    'Regresión lineal': {'modelo': LinearRegression(), 'grid': {}, 'escalar': False},
    'Árbol de regresión': {'modelo': DecisionTreeRegressor(random_state=42),
                           'grid': {'max_depth': [3, 4, 5, 6, 8],
                                    'min_samples_leaf': [1, 5, 10]},
                           'escalar': False},
    'KNN': {'modelo': KNeighborsRegressor(),
            'grid': {'n_neighbors': [3, 5, 7, 9, 11, 15],
                     'weights': ['uniform', 'distance']},
            'escalar': True},
    'MLP': {'modelo': MLPRegressor(max_iter=3000, random_state=42),
            'grid': {'hidden_layer_sizes': [(16,), (32,), (32, 16)],
                     'alpha': [0.0001, 0.01, 1.0],
                     'learning_rate_init': [0.001, 0.01]},
            'escalar': True},
    'SVR': {'modelo': SVR(),
            'grid': {'kernel': ['linear', 'rbf'],
                     'C': [1, 10, 100],
                     'epsilon': [0.1, 0.5, 1.0]},
            'escalar': True},
    'Random Forest': {'modelo': RandomForestRegressor(random_state=42),
                      'grid': {'n_estimators': [100, 300],
                               'max_depth': [None, 5, 10],
                               'min_samples_leaf': [1, 5]},
                      'escalar': False},
}


# =============================================================================
# 4. Elegir hiperparámetros con validación y evaluar en test
# =============================================================================

resultados = []
mejores_params = {}
predicciones_test = {}

for nombre, cfg in modelos.items():
    # 4.1 Validación: entrenar con sub-train, medir MAE en validación
    if cfg['escalar']:
        scaler = StandardScaler().fit(X_subtrain)
        Xs_sub, Xs_val = scaler.transform(X_subtrain), scaler.transform(X_val)
    else:
        Xs_sub, Xs_val = X_subtrain, X_val

    mejor_mae, mejor = np.inf, {}
    for params in ParameterGrid(cfg['grid']):
        m = cfg['modelo'].set_params(**params)
        m.fit(Xs_sub, y_subtrain)
        mae_val = mean_absolute_error(y_val, m.predict(Xs_val))
        if mae_val < mejor_mae:
            mejor_mae, mejor = mae_val, params
    mejores_params[nombre] = mejor

    # 4.2 Test: reentrenar con TODO train usando los mejores hiperparámetros
    if cfg['escalar']:
        scaler = StandardScaler().fit(X_train)
        Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)
    else:
        Xs_train, Xs_test = X_train, X_test

    m = cfg['modelo'].set_params(**mejor)
    m.fit(Xs_train, y_train)
    pred = m.predict(Xs_test)
    predicciones_test[nombre] = pred

    resultados.append({
        'Modelo': nombre,
        'MAE': mean_absolute_error(y_test, pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
        'R2': r2_score(y_test, pred),
    })

print('\nMejores hiperparámetros (elegidos con validación):')
for nombre, p in mejores_params.items():
    print(f'  {nombre}: {p}')

tabla = pd.DataFrame(resultados)
print('\nResultados en test (jul-2023 a dic-2023):')
print(tabla.round(2).to_string(index=False))


# =============================================================================
# 5. ¿Sirven lag_6 y lag_12? Mismo modelo con y sin esas variables
# =============================================================================

# Mejor modelo: menor MAE; si empatan, menor RMSE
mejor_modelo = tabla.sort_values(['MAE', 'RMSE']).iloc[0]['Modelo']
cfg = modelos[mejor_modelo]
print(f'\nPrueba de lags con el mejor modelo ({mejor_modelo}):')

for etiqueta, cols in [('lags 1, 2, 3', ['lag_1', 'lag_2', 'lag_3', 'prom_movil_3', 'mes']),
                       ('lags 1, 2, 3, 6, 12', features)]:
    X2 = pd.get_dummies(df[cols + ['departamento']], columns=['departamento'], dtype=int)
    X2_train, X2_test = X2[es_train], X2[es_test]
    if cfg['escalar']:
        scaler = StandardScaler().fit(X2_train)
        X2_train, X2_test = scaler.transform(X2_train), scaler.transform(X2_test)
    m = cfg['modelo'].set_params(**mejores_params[mejor_modelo])
    m.fit(X2_train, y_train)
    p = m.predict(X2_test)
    print(f'  {etiqueta:20s} MAE = {mean_absolute_error(y_test, p):.2f} | '
          f'R2 = {r2_score(y_test, p):.2f}')


# =============================================================================
# 6. Total nacional por mes: real vs predicho (test)
# =============================================================================

nacional = df.loc[es_test, ['periodo']].copy()
nacional['real'] = y_test.values
nacional['predicho'] = predicciones_test[mejor_modelo]
nacional = nacional.groupby('periodo').sum().round(0)
print(f'\nTotal nacional en test - real vs {mejor_modelo}:')
print(nacional.to_string())
