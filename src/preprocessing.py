"""
preprocessing.py
----------------
Convierte la base del ONSV (una fila por PERSONA involucrada en un siniestro fatal)
en una tabla de conteos mensuales de SINIESTROS por departamento, lista para modelar.

Flujo:
    personas (25 412 filas)
      -> siniestros únicos (1 fila por CÓDIGO SINIESTRO)
      -> conteo por departamento y mes (rellenando con 0 los meses sin siniestros)
      -> variables predictoras: lags t-1, t-2, t-3, promedio móvil de 3 meses y mes del año

Uso (desde la raíz del repositorio):
    python src/preprocessing.py
"""

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_RAW = RAIZ / "data" / "raw" / "BBDD_ONSV_PERSONAS_2021-2025_preliminar.xlsx"
DIR_PROCESSED = RAIZ / "data" / "processed"

# Último mes que se usa. Desde abril de 2025 el registro preliminar cae de forma abrupta
# (de ~100 a ~30 siniestros al mes a nivel nacional), lo que refleja registros aún no
# cargados y no una caída real. Ver data/raw/README.md.
FECHA_INICIO = "2021-01"
FECHA_FIN = "2025-03"


def cargar_personas(ruta=RUTA_RAW):
    """Lee el Excel original. Las 4 primeras filas son título y notas; el encabezado está en la fila 5."""
    df = pd.read_excel(ruta, header=4)
    df.columns = [c.strip() for c in df.columns]  # algunos nombres traen espacios al final
    df["FECHA"] = pd.to_datetime(df["FECHA"], format="%d/%m/%Y")
    return df


def personas_a_siniestros(df_personas):
    """
    Pasa de una fila por persona a una fila por siniestro.
    Se verificó que cada CÓDIGO SINIESTRO tiene una sola fecha y un solo departamento.
    """
    siniestros = (
        df_personas.groupby("CÓDIGO SINIESTRO")
        .agg(fecha=("FECHA", "first"), departamento=("DEPARTAMENTO", "first"))
        .reset_index()
    )
    siniestros["periodo"] = siniestros["fecha"].dt.to_period("M")
    return siniestros


def conteo_mensual(siniestros, inicio=FECHA_INICIO, fin=FECHA_FIN):
    """
    Cuenta siniestros por departamento y mes.
    Los meses sin siniestros no aparecen en la base, así que se rellenan con 0
    (un 0 aquí es un valor real, no un dato faltante).
    """
    tabla = siniestros.groupby(["departamento", "periodo"]).size().unstack(fill_value=0)
    meses = pd.period_range(inicio, fin, freq="M")
    tabla = tabla.reindex(columns=meses, fill_value=0)

    largo = tabla.stack().rename("siniestros").reset_index()
    largo.columns = ["departamento", "periodo", "siniestros"]
    return largo.sort_values(["departamento", "periodo"]).reset_index(drop=True)


def agregar_variables(conteo):
    """
    Crea las variables predictoras usando SOLO información de meses anteriores
    (para no filtrar información del mes que se quiere predecir).
    """
    df = conteo.copy()
    por_depto = df.groupby("departamento")["siniestros"]

    df["lag_1"] = por_depto.shift(1)  # siniestros del mes anterior
    df["lag_2"] = por_depto.shift(2)
    df["lag_3"] = por_depto.shift(3)
    # promedio de los 3 meses anteriores (shift(1) para no incluir el mes actual)
    df["prom_movil_3"] = por_depto.transform(lambda s: s.shift(1).rolling(3).mean())

    df["anio"] = df["periodo"].dt.year
    df["mes"] = df["periodo"].dt.month
    df["preliminar"] = (df["anio"] >= 2024).astype(int)  # según la Nota 2 del ONSV

    # Los 3 primeros meses de cada departamento no tienen historial completo
    df = df.dropna(subset=["lag_1", "lag_2", "lag_3", "prom_movil_3"]).reset_index(drop=True)
    for c in ["lag_1", "lag_2", "lag_3"]:
        df[c] = df[c].astype(int)
    return df


def main():
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

    personas = cargar_personas()
    siniestros = personas_a_siniestros(personas)
    conteo = conteo_mensual(siniestros)
    modelo = agregar_variables(conteo)

    conteo.to_csv(DIR_PROCESSED / "siniestros_depto_mes.csv", index=False)
    modelo.to_csv(DIR_PROCESSED / "dataset_modelo.csv", index=False)

    print(f"Personas: {len(personas):,} | Siniestros únicos: {len(siniestros):,}")
    print(f"Conteo mensual: {len(conteo):,} filas ({conteo['departamento'].nunique()} deptos x "
          f"{conteo['periodo'].nunique()} meses, {FECHA_INICIO} a {FECHA_FIN})")
    print(f"Dataset para modelar: {len(modelo):,} filas, columnas: {list(modelo.columns)}")


if __name__ == "__main__":
    main()
