# Modelos del estado del arte y resultados

Resumen corto. El detalle está en `notebooks/02_modelos.ipynb` y `results/model_comparison.txt`.
Los modelos se definen en `src/models.py`; `python src/modelos.py` imprime la tabla de resultados.

## Modelos comparados

| Modelo | Fuente | Por qué |
|---|---|---|
| Regresión lineal y árbol de regresión (baseline) | Özden y Acı (2018) | Modelos simples que el paper también comparó para conteos mensuales de accidentes. |
| KNN | Visto en clase | Modelo de regresión basado en distancias. |
| MLP | Visto en clase | Red neuronal para regresión. |
| SVR | Özden y Acı (2018) | Fue el mejor modelo del paper más parecido al nuestro (conteo mensual de accidentes por territorio). |
| Random Forest | Herrera (2021) | Fue el modelo del paper; es un conjunto de árboles de regresión. |

Chango et al. (2026) y Gavrilov et al. (2026) usan SARIMA, Prophet e híbridos con redes recurrentes. No se incluyen porque necesitan un modelo por serie (25 departamentos con solo 36 meses cada uno). Además, en esos papers ningún modelo complejo superó claramente a SARIMA.

## Configuración

- **Datos:** 2021–2023, porque 2024 y 2025 son preliminares.
- **Entradas:**
  - `lag_1`, `lag_2` y `lag_3`
  - `prom_movil_3`
  - `mes`
  - departamento con One-Hot Encoding
- **División temporal:**
  - train: abr-2021 a jun-2023
  - test: jul–dic 2023
  - validación para hiperparámetros: ene–jun 2023, entrenando con abr-2021 a dic-2022
- **Escalado:** KNN, MLP y SVR usan `StandardScaler`, ajustado solo con train.

## Resultados en test (jul–dic 2023)

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Regresión lineal | 2.85 | 3.81 | 0.82 |
| Árbol de regresión (baseline) | 2.37 | 3.32 | 0.86 |
| KNN | 2.98 | 4.02 | 0.80 |
| MLP | 2.46 | 3.37 | 0.86 |
| SVR | 2.58 | 3.50 | 0.85 |
| Random Forest | **2.21** | **3.08** | **0.88** |
| Referencia: promedio móvil de 3 meses (sin entrenar) | 2.10 | 3.09 | 0.88 |

**Modelo elegido: Random Forest (Herrera, 2021).**

- Tiene el menor MAE y RMSE y el mayor R² entre los modelos entrenados.
- Casi no tiene sesgo (+0.07), a diferencia de la regresión lineal, KNN y SVR, que predicen de más.

**Referencia:** el promedio de los 3 meses previos obtiene MAE 2.10. Random Forest lo iguala (diferencia de 0.11, con intervalo al 95 % de −0.09 a 0.31), pero no lo supera. La explicación es que en 2023 los siniestros bajaron 19 % respecto a 2022 y la mejor señal disponible es el nivel reciente.

**Prueba de rezagos (MAE de Random Forest en validación):**

| Entradas | Filas de sub-train | MAE |
|---|---|---|
| lags 1, 2, 3, 6, 12 | 300 | 2.66 |
| lags 1, 2, 3 | 525 | 2.50 |

`lag_6` y `lag_12` no reducen el error y obligan a descartar todo 2021, por eso se quitaron.
