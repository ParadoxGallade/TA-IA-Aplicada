# Modelo del estado del arte: Gradient Boosting (Cansiz et al., 2025)
#
# Predice el número mensual de siniestros fatales por departamento.
# Compara el modelo elegido de la literatura con un baseline y con regresión lineal.
#
# Entrada: data/processed/dataset_modelo.csv  (generado por src/preprocessing.py)
#
# Ejecutar desde la carpeta principal del repositorio:
#   python src/modelo_gradient_boosting.py

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# 1. Cargar datos y armar X, y
# =============================================================================

df = pd.read_csv('data/processed/dataset_modelo.csv')
print('Dataset:', df.shape, '| periodos:', df['periodo'].min(), 'a', df['periodo'].max())

# Variables de entrada: solo información del pasado (lags y promedio móvil) + mes del año.
# 'anio' no se usa: en el dataset del modelo solo hay 2022 y 2023, así que casi no aporta
# y un árbol no puede extrapolar a un año que no vio.
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

print('Train:', X_train.shape[0], 'filas | Test:', X_test.shape[0], 'filas')
print('Sub-train:', X_subtrain.shape[0], 'filas | Validación:', X_val.shape[0], 'filas')


# =============================================================================
# 3. Baseline: promedio de los 3 meses anteriores
# =============================================================================

# Sin entrenar nada: "este mes habrá lo mismo que el promedio de los últimos 3 meses"
pred_baseline = df.loc[es_test, 'prom_movil_3']


# =============================================================================
# 4. Regresión lineal (modelo visto en clase)
# =============================================================================

lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)


# =============================================================================
# 5. Gradient Boosting: elección de hiperparámetros con la validación
# =============================================================================

# Gradient Boosting = muchos árboles de regresión pequeños en secuencia;
# cada árbol nuevo se entrena para corregir el error que dejaron los anteriores.
#   n_estimators : cantidad de árboles
#   learning_rate: cuánto aporta cada árbol (más bajo = más lento pero más estable)
#   max_depth    : profundidad de cada árbol (más profundo = más riesgo de overfitting)

resultados_val = []
for n in [100, 200, 300]:
    for lr_rate in [0.03, 0.05, 0.1]:
        for depth in [2, 3, 4]:
            gb = GradientBoostingRegressor(n_estimators=n, learning_rate=lr_rate,
                                           max_depth=depth, random_state=42)
            gb.fit(X_subtrain, y_subtrain)
            mae_val = mean_absolute_error(y_val, gb.predict(X_val))
            resultados_val.append({'n_estimators': n, 'learning_rate': lr_rate,
                                   'max_depth': depth, 'MAE_val': mae_val})

resultados_val = pd.DataFrame(resultados_val).sort_values('MAE_val')
print('\nMejores combinaciones en validación:')
print(resultados_val.head(5).round(2).to_string(index=False))

mejor = resultados_val.iloc[0]
print('\nHiperparámetros elegidos:', mejor[['n_estimators', 'learning_rate', 'max_depth']].to_dict())

# Con los hiperparámetros elegidos, se reentrena con TODO el train y se evalúa en test
gb = GradientBoostingRegressor(n_estimators=int(mejor['n_estimators']),
                               learning_rate=mejor['learning_rate'],
                               max_depth=int(mejor['max_depth']),
                               random_state=42)
gb.fit(X_train, y_train)
pred_gb = gb.predict(X_test)


# =============================================================================
# 6. Comparación en test (MAE, RMSE, R²)
# =============================================================================

# No se usa MAPE: hay meses con 0 siniestros y MAPE divide entre cero.
predicciones = {
    'Baseline (promedio móvil 3 meses)': pred_baseline,
    'Regresión lineal': pred_lr,
    'Gradient Boosting': pred_gb,
}

tabla = []
for nombre, pred in predicciones.items():
    tabla.append({
        'Modelo': nombre,
        'MAE': mean_absolute_error(y_test, pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
        'R2': r2_score(y_test, pred),
    })
tabla = pd.DataFrame(tabla).round(2)

print('\nResultados en test (jul-2023 a dic-2023):')
print(tabla.to_string(index=False))

# Error de entrenamiento vs test del Gradient Boosting (para revisar overfitting)
mae_train_gb = mean_absolute_error(y_train, gb.predict(X_train))
print('\nGradient Boosting -> MAE train:', round(mae_train_gb, 2),
      '| MAE test:', round(mean_absolute_error(y_test, pred_gb), 2))


# =============================================================================
# 7. Importancia de variables del Gradient Boosting
# =============================================================================

importancias = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False)
print('\nTop 10 variables más importantes:')
print(importancias.head(10).round(3).to_string())
