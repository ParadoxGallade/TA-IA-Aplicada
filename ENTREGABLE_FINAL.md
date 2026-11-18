# Entregable Final (Semana 15)

## Checklist

| Ítem | Completado | Dónde |
|---|---|---|
| Código limpio, comentado y reproducible | SI | `src/`, `notebooks/` (semilla 42) |
| `requirements.txt` con las dependencias | SI | `requirements.txt` |
| Todos los modelos implementados (mínimo 2) en `src/models.py` | SI | 6 modelos: regresión lineal, árbol de regresión, KNN, MLP, SVR y Random Forest |
| Tabla comparativa de métricas | SI | `results/metrics.csv` |
| Gráficos de resultados | SI | `results/plots/` |
| Resumen de resultados y análisis detallado | SI | `results/model_comparison.txt` |
| Análisis de por qué un modelo funcionó mejor, limitaciones e insights | SI | `results/model_comparison.txt` y `report.pdf` (secciones VI y VII) |
| Paper/Reporte académico (3-5 páginas) | SI | `report.pdf` |
| Presentación | SI | `presentation.pdf` |
| Archivo `ENTREGABLE_FINAL.md` | SI | este archivo |

## Descripción de lo realizado

- **Datos.** Se cambió el periodo a 2021–2023 (36 meses), porque los registros de 2024 y 2025 son preliminares. El panel
  tiene 900 filas (25 departamentos × 36 meses) y el dataset de modelado, 825 (`src/preprocessing.py`).
- **EDA.** `notebooks/01_eda.ipynb` se actualizó a 2021–2023, con una sección nueva de meses sin siniestros por departamento.
  Gráficos en `results/plots/EDA/`.
- **Modelos.** `src/models.py` implementa los seis modelos, la partición temporal, la búsqueda de hiperparámetros en grilla y dos
  referencias ingenuas. `src/evaluation.py` calcula MAE, RMSE, R², sesgo, intervalos por bootstrap y el error por departamento.
- **Comparación.** `notebooks/02_modelos.ipynb` ejecuta todo y genera `results/metrics.csv`, `results/model_comparison.txt` y
  los gráficos de `results/plots/`.
- **Resultado.** Random Forest es el mejor modelo entrenado en test (MAE 2.21, RMSE 3.08, R² 0.88). Ningún modelo supera al
  promedio móvil de 3 meses (MAE 2.10); Random Forest lo iguala.
- **Cambio respecto a la versión anterior.** Se quitaron los rezagos de 6 y 12 meses: en validación no redujeron el error y
  obligaban a descartar todo 2021.
