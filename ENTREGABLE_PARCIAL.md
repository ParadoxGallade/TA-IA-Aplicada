# Entregable Parcial (Semana 8)

## IV. Checklist de avance (SI/NO)

| Ítem | Completado |
|---|---|
| README completo | SI |
| Resumen de papers (mínimo 1 por integrante) | SI |
| Dataset | SI |
| Descripción del dataset (origen, características, ...) | SI |
| Notebooks con EDA | SI |
| Propuesta de modelos | SI |
| Notebooks con baseline implementado | SI |
| Breve descripción de lo realizado (ENTREGABLE_PARCIAL.md) | SI |

## Descripción de lo realizado

Se definió el dataset de siniestros de tránsito fatales del ONSV (2021-2024) y se documentó su origen y
características en `data/raw/README.md`. Se hizo el EDA en `notebooks/01_eda.ipynb` (valores faltantes,
evolución temporal, estacionalidad, comparación entre departamentos, distribución, outliers y correlación
de variables), con los gráficos en `results/plots/EDA/`. En el README se dejó la propuesta de modelos a
partir de los papers revisados. Como baseline se implementaron y evaluaron Regresión Lineal y Árbol de
Regresión en `notebooks/02_modelos.ipynb` (funciones en `src/models.py` y `src/evaluation.py`), con
resultados en `results/metrics.csv` y `results/model_comparison.txt`.
