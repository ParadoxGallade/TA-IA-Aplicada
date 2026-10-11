# Dataset: personas involucradas en siniestros de tránsito fatales (ONSV, 2021–2025)

## Origen

| Campo | Detalle |
|---|---|
| Fuente | Observatorio Nacional de Seguridad Vial (ONSV) – Ministerio de Transportes y Comunicaciones del Perú |
| Registro original | Unidades de Prevención e Investigación de Accidentes de Tránsito de la PNP (UPIAT-PNP) |
| URL de descarga | https://www.onsv.gob.pe/estaticos/excel/BBDD%20ONSV%20-%20PERSONAS%202021-2025%20(preliminar).xlsx |
| Archivo en este repo | `BBDD_ONSV_PERSONAS_2021-2025_preliminar.xlsx` (sin modificar; solo se renombró el archivo) |
| Fecha de descarga | octubre 2026 |
| Permisos | Datos publicados de forma abierta y gratuita por una entidad pública en su portal web. Se usan con fines académicos y citando la fuente. No contienen nombres ni documentos de identidad de las personas. |

## Características

- **Unidad de registro:** una fila por **persona** involucrada en un siniestro fatal (conductores, pasajeros, peatones, etc.).
- **Tamaño:** 25 412 filas × 32 columnas.
- **Siniestros únicos:** 9 106 (columna `CÓDIGO SINIESTRO`). Cada siniestro tiene una sola fecha y un solo departamento.
- **Periodo:** 01/01/2021 – 30/12/2025.
- **Cobertura:** 25 departamentos (Lima y Callao por separado).
- **Estado:** según la Nota 2 del archivo, los registros de **2024 y 2025 son preliminares**.

### Grupos de columnas

| Grupo | Columnas |
|---|---|
| Identificadores | `CÓDIGO SINIESTRO`, `CÓDIGO VEHÍCULO`, `CÓDIGO PERSONA` |
| Ubicación | `DEPARTAMENTO`, `PROVINCIA`, `DISTRITO`, `TIPO DE VÍA`, `CÓDIGO DE CARRETERA`, `RED VIAL` |
| Tiempo | `FECHA`, `AÑO`, `MES`, `DIA`, `HORA` |
| Siniestro | `CLASE DE SINIESTRO`, `CAUSA`, `CAUSA ESPECIFICA`, `VEHÍCULO` |
| Persona | `TIPO PERSONA`, `GRAVEDAD`, `EDAD`, `SEXO`, `PAÍS DE NACIONALIDAD`, `SITUACIÓN DE PERSONA` |
| Licencia y alcohol | `POSEE LICENCIA`, `ESTADO LICENCIA`, `CLASE_LICENCIA`, columnas de dosaje etílico |
| Atención | `LUGAR ATENCIÓN LESIONADO`, `LUGAR DE DEFUNCIÓN` |

Para el modelo **solo se usan `CÓDIGO SINIESTRO`, `FECHA` y `DEPARTAMENTO`**. Las demás columnas describen lo que pasó *después* del siniestro (gravedad, dosaje, lugar de atención) y no se conocen antes del mes que se quiere predecir.

### Valores faltantes (columnas con más vacíos)

| Columna | Vacíos | % |
|---|---|---|
| RESULTADO DEL DOSAJE ETÍLICO CUALITATIVO | 21 457 | 84 % |
| LUGAR ATENCIÓN LESIONADO | 18 341 | 72 % |
| POSEE LICENCIA | 18 236 | 72 % |
| LUGAR DE DEFUNCIÓN | 15 341 | 60 % |
| ¿SE SOMETIÓ A DOSAJE ETÍLICO CUANTITATIVO? | 4 282 | 17 % |
| ¿SE SOMETIÓ A DOSAJE ETÍLICO CUALITATIVO? | 3 891 | 15 % |
| PAÍS DE NACIONALIDAD | 3 308 | 13 % |
| EDAD | 2 650 | 10 % |

Las tres columnas que usa el modelo (`CÓDIGO SINIESTRO`, `FECHA`, `DEPARTAMENTO`) **no tienen vacíos**, así que no hace falta imputar.

## Problema detectado: 2025 incompleto

Siniestros fatales únicos por mes a nivel nacional:

| Mes | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|
| Ene | 167 | 216 | 159 | 144 | 115 |
| Feb | 161 | 211 | 150 | 136 | 110 |
| Mar | 228 | 202 | 164 | 135 | 102 |
| Abr | 204 | 192 | 178 | 118 | **59** |
| May | 187 | 205 | 169 | 135 | **37** |
| Jun | 229 | 213 | 191 | 135 | **56** |
| Jul | 201 | 210 | 175 | 131 | **45** |
| Ago | 221 | 222 | 161 | 150 | **32** |
| Set | 191 | 212 | 159 | 116 | **44** |
| Oct | 219 | 188 | 174 | 119 | **29** |
| Nov | 189 | 198 | 153 | 110 | **24** |
| Dic | 195 | 211 | 168 | 126 | **25** |

Desde abril de 2025 el conteo cae a menos de la mitad de golpe y sigue bajando. Es lo esperable en datos **preliminares** (siniestros que todavía no se cargan al sistema), no una mejora real de la seguridad vial. Por eso el procesamiento usa **enero 2021 – diciembre 2024** (48 meses) y descarta 2025 completo.

2024 también es preliminar y está por debajo de 2021–2023, pero su nivel es estable dentro del año, así que se mantiene. Hay que tenerlo en cuenta al interpretar los resultados (ver EDA).

## Datos procesados (`data/processed/`)

Se generan con `python src/preprocessing.py`:

```
personas (25 412 filas)
   │  filtrar 2021–2024 (23 502 filas)
   │  dejar una fila por CÓDIGO SINIESTRO (drop_duplicates)
   ▼
siniestros únicos 2021–2024 (8 428)
   │  contar por departamento y mes, rellenar meses sin siniestros con 0
   ▼
siniestros_depto_mes.csv   → 1 200 filas (25 deptos × 48 meses)
   │  agregar lags y promedio móvil, quitar los 3 primeros meses de cada depto
   ▼
dataset_modelo.csv         → 1 125 filas
```

| Columna | Descripción |
|---|---|
| `departamento` | Departamento (25 categorías) |
| `periodo` | Año-mes (AAAA-MM) |
| `siniestros` | **Target:** número de siniestros fatales en ese departamento y mes |
| `lag_1`, `lag_2`, `lag_3` | Siniestros del mismo departamento 1, 2 y 3 meses antes |
| `prom_movil_3` | Promedio de los 3 meses anteriores |
| `anio`, `mes` | Año y mes del año (1–12) |

**Dato clave para el modelado:** el 21 % de las filas tiene `siniestros = 0` (departamentos como Moquegua, Ucayali o Amazonas promedian ~1 siniestro al mes). Por eso el MAPE no sirve (divide entre cero) y se usan **MAE y RMSE**.
