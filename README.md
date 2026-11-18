# TA-IA-Aplicada
Repositorio para la TA del grupo

## Descripción del proyecto

**Título:** Predicción del número mensual de siniestros de tránsito fatales por departamento del Perú a partir de registros históricos del ONSV (2021-2023)

**Problema:** En el Perú se registran más de 2 000 siniestros de tránsito fatales al año, pero no existen estimaciones mensuales por departamento que permitan anticipar dónde se concentrará la siniestralidad. A partir de los registros históricos del ONSV se busca predecir el número mensual de siniestros fatales por departamento, para que el MTC y la PNP puedan anticipar la demanda de operativos y campañas por región.

**Objetivo:** Construir y comparar modelos de regresión que estimen el número de siniestros fatales por departamento y mes, evaluados con métricas que funcionan aunque haya meses con cero siniestros (MAE, RMSE y R²).

**Dataset:** Registros de personas involucradas en siniestros fatales del Observatorio Nacional de Seguridad Vial (ONSV - MTC), 2021-2025. Se procesan a un panel de 25 departamentos × 36 meses (2021-2023); 2024 y 2025 se descartan porque sus registros son preliminares. Ver [data/raw/README.md](data/raw/README.md) para el detalle de origen, permisos y procesamiento.

### Modelos

A partir de la literatura revisada (ver `papers/`) se implementaron seis modelos y dos referencias que no entrenan nada:

| Modelo | Origen |
|---|---|
| Regresión lineal (baseline) | Visto en clase; comparado por Özden y Acı (2018) |
| Árbol de regresión (baseline) | Visto en clase; comparado por Özden y Acı (2018) |
| KNN (regresión) | Visto en clase |
| MLP (regresión) | Visto en clase |
| SVR | Mejor modelo de Özden y Acı (2018) |
| Random Forest | Herrera (2021); conjunto de árboles de regresión |
| Promedio móvil de 3 meses y mes anterior | Referencias ingenuas: un modelo solo aporta si se equivoca menos que ellas |

SARIMA, Prophet y los híbridos con redes recurrentes (Chango et al., 2026; Gavrilov et al., 2026) no se implementaron porque necesitan un modelo por serie: serían 25 series de solo 36 meses.

### Metodología

- **Entradas:** siniestros de 1, 2 y 3 meses antes (`lag_1`, `lag_2`, `lag_3`), su promedio (`prom_movil_3`), el mes y el departamento (One-Hot Encoding). Los rezagos de 6 y 12 meses se probaron en validación y se descartaron: no reducen el error y obligan a perder todo 2021.
- **Partición temporal (no aleatoria):** sub-train abr-2021 a dic-2022 (525 filas), validación ene–jun 2023 (150) y test jul–dic 2023 (150).
- **Hiperparámetros:** búsqueda en grilla; se elige el menor MAE en validación. Semilla 42.
- **Escalado:** KNN, MLP y SVR usan `StandardScaler` ajustado solo con train.

### Resultados en test (jul–dic 2023)

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Regresión lineal | 2.85 | 3.81 | 0.82 |
| Árbol de regresión | 2.37 | 3.32 | 0.86 |
| KNN | 2.98 | 4.02 | 0.80 |
| MLP | 2.46 | 3.37 | 0.86 |
| SVR | 2.58 | 3.50 | 0.85 |
| **Random Forest** | **2.21** | **3.08** | **0.88** |
| Referencia: promedio móvil (3 meses) | 2.10 | 3.09 | 0.88 |
| Referencia: mes anterior | 2.53 | 3.76 | 0.82 |

- **Random Forest es el mejor modelo entrenado.** Supera de forma concluyente a la regresión lineal, KNN y SVR.
- **Ningún modelo supera al promedio móvil de 3 meses.** Random Forest lo iguala: la diferencia de MAE es 0.11, con un intervalo al 95 % de −0.09 a 0.31.
- Regresión lineal, KNN y SVR predicen de más porque 2023 tuvo 19 % menos siniestros que 2022.

El análisis completo está en [results/model_comparison.txt](results/model_comparison.txt) y en [report.pdf](report.pdf).

## Estructura del repositorio
```
├── README.md                 # Descripción del proyecto, cómo ejecutar
├── ENTREGABLE_PARCIAL.md     # Checklist de lo entregado en semana 8
├── ENTREGABLE_FINAL.md       # Checklist de lo entregado en semana 15
├── data/
│   ├── raw/                  # Datos originales
│   └── processed/            # Datos limpios/preparados
├── notebooks/
│   ├── 01_eda.ipynb          # Análisis exploratorio
│   └── 02_modelos.ipynb      # Implementación y comparación de modelos
├── src/
│   ├── preprocessing.py      # Funciones de preparación
│   ├── models.py             # Implementación de modelos
│   └── evaluation.py         # Métricas y evaluación
├── results/
│   ├── metrics.csv           # Tabla comparativa
│   ├── plots/                # Gráficos
│   └── model_comparison.txt  # Resumen de resultados
├── papers/                   # Referencias de papers revisados y sus resúmenes (papers/resume/)
├── report.pdf                # Paper/reporte final
├── presentation.pdf          # Slides de presentación
└── requirements.txt          # Dependencias
```

Archivos adicionales: `src/modelos.py` (ejecuta la comparación desde la terminal usando `src/models.py`) y `modelo_elegido.md` (resumen corto del modelo elegido).

## Cómo ejecutar
```bash
pip install -r requirements.txt
python src/preprocessing.py      # genera data/processed/ a partir de data/raw/
jupyter notebook                 # ejecutar notebooks/01_eda.ipynb y notebooks/02_modelos.ipynb
```

`notebooks/02_modelos.ipynb` entrena los modelos y vuelve a generar `results/metrics.csv`, `results/model_comparison.txt` y los gráficos de `results/plots/` (tarda unos 2 minutos). Para ver solo la tabla de resultados desde la terminal: `python src/modelos.py`.

Probado con Python 3.13, pandas 3.0, numpy 2.5 y scikit-learn 1.8.

## Integrantes

Esteban Palomino (20220577) · Flavio Ascama Rojo (20212426) · Rodrigo Aragon Vilca (20205966) · César Membrillo Acosta (20220438)
