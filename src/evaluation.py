"""Métricas y evaluación.

No se usa MAPE: hay meses con 0 siniestros y el MAPE divide entre el valor real.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calcular_mae(y_real, y_pred):
    """Error absoluto medio: en promedio, por cuántos siniestros se equivoca el modelo."""
    return mean_absolute_error(y_real, y_pred)


def calcular_rmse(y_real, y_pred):
    """Raíz del error cuadrático medio: castiga más los errores grandes."""
    return np.sqrt(mean_squared_error(y_real, y_pred))


def calcular_r2(y_real, y_pred):
    """Proporción de la variación de y que explica el modelo (1 = perfecto)."""
    return r2_score(y_real, y_pred)


def calcular_sesgo(y_real, y_pred):
    """Error medio con signo: positivo = el modelo predice de más; negativo = de menos."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_real)))


def evaluar_modelo(nombre, y_real, y_pred):
    return {
        'modelo': nombre,
        'MAE': calcular_mae(y_real, y_pred),
        'RMSE': calcular_rmse(y_real, y_pred),
        'R2': calcular_r2(y_real, y_pred),
        'sesgo': calcular_sesgo(y_real, y_pred),
    }


def diferencia_mae_bootstrap(y_real, pred_a, pred_b, repeticiones=5000, semilla=42):
    """¿La diferencia de MAE entre dos predictores es real o puede ser azar?

    Bootstrap pareado: se remuestrean las filas de test (con reemplazo) muchas veces y se
    recalcula MAE(a) - MAE(b). Devuelve (diferencia, límite inferior, límite superior) del
    intervalo al 95 %. Si el intervalo incluye el 0, la diferencia no es concluyente.
    """
    y_real, pred_a, pred_b = np.asarray(y_real), np.asarray(pred_a), np.asarray(pred_b)
    dif = np.abs(pred_a - y_real) - np.abs(pred_b - y_real)   # negativo = 'a' es mejor
    rng = np.random.default_rng(semilla)
    n = len(dif)
    muestras = [dif[rng.integers(0, n, n)].mean() for _ in range(repeticiones)]
    li, ls = np.percentile(muestras, [2.5, 97.5])
    return float(dif.mean()), float(li), float(ls)


def error_por_departamento(departamentos, y_real, y_pred):
    """MAE y promedio mensual de siniestros de cada departamento (para ver dónde falla el modelo)."""
    tabla = pd.DataFrame({
        'departamento': np.asarray(departamentos),
        'real': np.asarray(y_real),
        'error_abs': np.abs(np.asarray(y_pred) - np.asarray(y_real)),
    })
    resumen = tabla.groupby('departamento').agg(promedio_real=('real', 'mean'),
                                                 MAE=('error_abs', 'mean'))
    return resumen.sort_values('promedio_real', ascending=False)


def mae_minimo_poisson(promedios):
    """Error que tendría un modelo 'perfecto' si los conteos fueran puro azar de Poisson.

    Si cada departamento tuviera una tasa fija (su promedio) y el conteo de cada mes fuera
    aleatorio alrededor de esa tasa, ni conociendo la tasa exacta se podría acertar el mes.
    Devuelve el MAE esperado de predecir siempre la tasa, para cada promedio dado.

    Fórmula exacta de la desviación absoluta media de una Poisson de tasa t:
        E|X - t| = 2 * t * P(X = parte entera de t)
    """
    tasas = np.asarray(promedios, dtype=float)
    return 2 * tasas * poisson.pmf(np.floor(tasas), tasas)
