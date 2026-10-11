# Preprocesamiento de datos - Siniestros de tránsito fatales (ONSV 2021-2024)
#
# Entrada:  data/raw/BBDD_ONSV_PERSONAS_2021-2025_preliminar.xlsx  (1 fila por persona)
# Salidas:  data/processed/siniestros_depto_mes.csv  (conteo de siniestros por departamento y mes)
#           data/processed/dataset_modelo.csv        (conteo + variables para el modelo)
#
# Ejecutar desde la carpeta principal del repositorio:
#   python src/preprocessing.py

import pandas as pd

# Último mes que se usa. 2025 está incompleto (registro preliminar), por eso se corta en 2024.
FECHA_FIN = '2024-12'


# =============================================================================
# 1. Importar dataset
# =============================================================================

# Las 4 primeras filas del Excel son título y notas; los nombres de columna están en la fila 5
df = pd.read_excel('data/raw/BBDD_ONSV_PERSONAS_2021-2025_preliminar.xlsx', header=4)

# Algunos nombres de columna vienen con espacios al final (ej. 'CLASE DE SINIESTRO     ')
df.columns = df.columns.str.strip()

print('Dataset original (1 fila por persona):', df.shape)


# =============================================================================
# 2. Valores faltantes
# =============================================================================

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    'Columna': missing.index,
    'Faltantes': missing.values,
    'Porcentaje (%)': missing_pct.values.round(1)
})

# Mostrar solo columnas con faltantes
missing_df = missing_df[missing_df['Faltantes'] > 0].sort_values('Faltantes', ascending=False)
print('\nColumnas con valores faltantes:')
print(missing_df.to_string(index=False))

# Para el modelo solo se usan CÓDIGO SINIESTRO, FECHA y DEPARTAMENTO
columnas_modelo = ['CÓDIGO SINIESTRO', 'FECHA', 'DEPARTAMENTO']
print('\nFaltantes en las columnas que se usan:')
print(df[columnas_modelo].isnull().sum())
# Resultado: 0 en las tres -> no es necesario imputar


# =============================================================================
# 3. Convertir la fecha y filtrar el periodo de estudio
# =============================================================================

# FECHA viene como texto 'dd/mm/aaaa'; se convierte a fecha
df['FECHA'] = pd.to_datetime(df['FECHA'], format='%d/%m/%Y')

# Periodo año-mes (ej. 2021-01)
df['PERIODO'] = df['FECHA'].dt.to_period('M')

# Quedarse con enero 2021 - diciembre 2024
df = df[df['PERIODO'] <= FECHA_FIN]
print('\nPersonas en el periodo 2021-2024:', df.shape[0])


# =============================================================================
# 4. De personas a siniestros
# =============================================================================

# Un siniestro con 4 personas aparece 4 veces. Para contar siniestros (no personas),
# se deja una sola fila por CÓDIGO SINIESTRO.

# Verificación: cada siniestro debe tener un solo departamento y una sola fecha
deptos_por_siniestro = df.groupby('CÓDIGO SINIESTRO')['DEPARTAMENTO'].nunique()
fechas_por_siniestro = df.groupby('CÓDIGO SINIESTRO')['FECHA'].nunique()
print('\nSiniestros con más de un departamento:', (deptos_por_siniestro > 1).sum())
print('Siniestros con más de una fecha:', (fechas_por_siniestro > 1).sum())

siniestros = df.drop_duplicates(subset='CÓDIGO SINIESTRO')
print('Siniestros únicos:', siniestros.shape[0])


# =============================================================================
# 5. Conteo de siniestros por departamento y mes
# =============================================================================

# Tabla: filas = departamentos, columnas = meses, valores = número de siniestros.
# crosstab pone 0 en los meses en que un departamento no tuvo siniestros
# (un 0 aquí es un valor real, no un dato faltante).
tabla = pd.crosstab(siniestros['DEPARTAMENTO'], siniestros['PERIODO'])
print('\nTabla departamentos x meses:', tabla.shape)

# Pasar la tabla a formato largo: una fila por (departamento, mes)
conteo = tabla.reset_index().melt(id_vars='DEPARTAMENTO', var_name='periodo', value_name='siniestros')
conteo = conteo.rename(columns={'DEPARTAMENTO': 'departamento'})
conteo['periodo'] = conteo['periodo'].astype(str)  # texto 'aaaa-mm'
conteo = conteo.sort_values(['departamento', 'periodo']).reset_index(drop=True)

print('Filas con 0 siniestros (%):', round((conteo['siniestros'] == 0).mean() * 100, 1))


# =============================================================================
# 6. Variables para el modelo
# =============================================================================

dataset = conteo.copy()

# Siniestros del mismo departamento 1, 2 y 3 meses antes (shift mira hacia atrás)
dataset['lag_1'] = dataset.groupby('departamento')['siniestros'].shift(1)
dataset['lag_2'] = dataset.groupby('departamento')['siniestros'].shift(2)
dataset['lag_3'] = dataset.groupby('departamento')['siniestros'].shift(3)

# Promedio de los 3 meses anteriores (no incluye el mes que se quiere predecir)
dataset['prom_movil_3'] = (dataset['lag_1'] + dataset['lag_2'] + dataset['lag_3']) / 3

# Año y mes del año (para capturar estacionalidad)
dataset['anio'] = dataset['periodo'].str[:4].astype(int)   # 'aaaa-mm' -> aaaa
dataset['mes'] = dataset['periodo'].str[5:].astype(int)    # 'aaaa-mm' -> mm

# Los 3 primeros meses de cada departamento no tienen 3 meses previos -> se eliminan
dataset = dataset.dropna().reset_index(drop=True)


# =============================================================================
# 7. Guardar
# =============================================================================

conteo.to_csv('data/processed/siniestros_depto_mes.csv', index=False)
dataset.to_csv('data/processed/dataset_modelo.csv', index=False)

print('\nsiniestros_depto_mes.csv:', conteo.shape)
print('dataset_modelo.csv:', dataset.shape)
print(dataset.head())
