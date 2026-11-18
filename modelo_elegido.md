# Modelos del estado del arte y resultados

Script: `src/modelos.py` (requiere haber corrido antes `src/preprocessing.py`).

## Modelos comparados

| Modelo | Fuente | Por qué |
|---|---|---|
| Regresión lineal y árbol de regresión (baseline) | Özden y Acı (2018) | Modelos simples que el paper también comparó para conteos mensuales de accidentes. |
| KNN | Visto en clase | Modelo de regresión basado en distancias. |
| MLP | Visto en clase | Red neuronal para regresión. |
| SVR | Özden y Acı (2018) | Fue el mejor modelo del paper más parecido al nuestro (conteo mensual de accidentes por territorio). |
| Random Forest | Herrera (2021) | Fue el mejor modelo del paper; es un conjunto de árboles de regresión. |

Chango et al. (2026) y Gavrilov et al. (2026) usan SARIMA, Prophet e híbridos con redes recurrentes. No se incluyen porque necesitan un modelo por serie (25 departamentos con solo 36 meses cada uno). Además, en esos papers ningún modelo complejo superó claramente a SARIMA.

## Configuración

- **Datos:** 2021–2023, porque 2024 y 2025 son preliminares.
- **Entradas:**
  - `lag_1`, `lag_2`, `lag_3`, `lag_6` y `lag_12`
  - `prom_movil_3`
  - `mes`
  - departamento con One-Hot Encoding
- **División temporal:**
  - train: ene-2022 a jun-2023
  - test: jul–dic 2023
  - validación para hiperparámetros: ene–jun 2023, entrenando con 2022
- **Escalado:** KNN, MLP y SVR usan `StandardScaler`, ajustado solo con train.

## Resultados en test (jul–dic 2023)

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Regresión lineal | 2.96 | 4.01 | 0.80 |
| Árbol de regresión (baseline) | **2.37** | 3.62 | 0.84 |
| KNN | 2.94 | 4.01 | 0.80 |
| MLP | 3.30 | 4.34 | 0.77 |
| SVR | 2.71 | 3.72 | 0.83 |
| Random Forest | **2.37** | **3.42** | **0.85** |

**Modelo elegido: Random Forest (Herrera, 2021).**

- Empata con el árbol en MAE.
- Tiene el menor RMSE, así que comete menos errores grandes.
- Tiene el mayor R².

**Prueba de rezagos (Random Forest):**

| Entradas | MAE |
|---|---|
| lags 1, 2, 3 | 2.35 |
| lags 1, 2, 3, 6, 12 | 2.37 |

Agregar `lag_6` y `lag_12` no reduce el error.

**Referencia:** un pronóstico ingenuo, que usa como predicción el promedio de los 3 meses previos, obtiene MAE 2.10. Los modelos todavía no lo superan. Una explicación probable es que en 2023 los siniestros bajaron 19 % respecto a 2022.
