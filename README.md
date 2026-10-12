# TA-IA-Aplicada
Repositorio para la TA del grupo

## Descripción del proyecto

**Título:** Predicción del número mensual de siniestros de tránsito fatales por departamento del Perú a partir de registros históricos del ONSV (2021-2024)

**Problema:** En el Perú se registran más de 2 000 siniestros de tránsito fatales al año, pero la asignación de recursos de fiscalización y prevención vial se hace de forma reactiva, sin anticipar dónde se concentrará la siniestralidad. A partir de los registros históricos del ONSV se busca predecir el número mensual de siniestros fatales por departamento, usando rezagos temporales y estacionalidad, para que el MTC y la PNP puedan anticipar la demanda de operativos y campañas por región.

**Objetivo:** Construir y comparar modelos de regresión que estimen el número de siniestros fatales por departamento y mes, evaluados con métricas robustas a los departamentos con pocos siniestros (MAE y RMSE).

**Dataset:** Registros de personas involucradas en siniestros fatales del Observatorio Nacional de Seguridad Vial (ONSV - MTC), 2021-2025. Se procesan a un panel de 25 departamentos × 48 meses (2021-2024), descartando 2025 por subregistro. Ver [data/raw/README.md](data/raw/README.md) para el detalle de origen, permisos y procesamiento.

### Propuesta de modelos

A partir de la literatura revisada (ver `papers/`), que compara regresión lineal, árboles de regresión, redes neuronales (MLP) y SVM para predecir conteos de siniestros de tránsito, se proponen las siguientes técnicas (dentro del temario del curso):

| Modelo | Estado |
|---|---|
| Regresión lineal | Baseline implementado (Entregable Parcial) |
| Árbol de regresión | Implementado como comparación (Entregable Parcial) |
| KNN (regresión) | Propuesto para el Entregable Final |
| MLP (regresión) | Propuesto para el Entregable Final |

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
├── papers/                   # Referencias de papers revisados (paper_integranteN.pdf)
├── report.pdf                # Paper/reporte final
├── presentation.pdf          # Slides de presentación
└── requirements.txt          # Dependencias
```

## Cómo ejecutar
```bash
pip install -r requirements.txt
jupyter notebook
```
