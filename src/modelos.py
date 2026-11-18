# Comparación de modelos desde la terminal (versión corta de notebooks/02_modelos.ipynb)
#
# Usa las funciones de src/models.py y src/evaluation.py: aquí no se define ningún modelo.
# Solo imprime la tabla de resultados; los archivos de results/ los genera el notebook.
#
# Entrada: data/processed/dataset_modelo.csv  (generado por src/preprocessing.py)
#
# Ejecutar desde la carpeta principal del repositorio:
#   python src/modelos.py

import pandas as pd

from evaluation import evaluar_modelo
from models import (ENTRADAS, buscar_hiperparametros, crear_modelos, dividir_temporal,
                    entrenar_y_predecir, predicciones_ingenuas, preparar_xy)

df = pd.read_csv('data/processed/dataset_modelo.csv')
X, y = preparar_xy(df, ENTRADAS)
m = dividir_temporal(df)
print('Train:', m['train'].sum(), '| Test:', m['test'].sum(),
      '| Sub-train:', m['subtrain'].sum(), '| Validación:', m['val'].sum())

resultados = []
print('\nMejores hiperparámetros (elegidos con validación):')
for nombre, cfg in crear_modelos().items():
    # 1) elegir hiperparámetros con validación  2) reentrenar con todo train  3) evaluar en test
    params, mae_val = buscar_hiperparametros(cfg, X[m['subtrain']], y[m['subtrain']],
                                             X[m['val']], y[m['val']])
    _, pred = entrenar_y_predecir(cfg, params, X[m['train']], y[m['train']], X[m['test']])
    resultados.append({**evaluar_modelo(nombre, y[m['test']], pred), 'MAE_validacion': mae_val})
    print(f'  {nombre}: {params}')

# Referencias ingenuas: no entrenan nada
for nombre, pred in predicciones_ingenuas(df).items():
    resultados.append(evaluar_modelo(nombre, y[m['test']], pred[m['test']]))

print('\nResultados en test (jul-2023 a dic-2023):')
print(pd.DataFrame(resultados).round(2).to_string(index=False))
