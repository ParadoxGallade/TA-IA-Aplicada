"""Métricas y evaluación."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calcular_mae(y_real, y_pred):
    return mean_absolute_error(y_real, y_pred)


def calcular_rmse(y_real, y_pred):
    return np.sqrt(mean_squared_error(y_real, y_pred))


def evaluar_modelo(nombre, y_real, y_pred):
    return {
        'modelo': nombre,
        'MAE': calcular_mae(y_real, y_pred),
        'RMSE': calcular_rmse(y_real, y_pred),
    }
