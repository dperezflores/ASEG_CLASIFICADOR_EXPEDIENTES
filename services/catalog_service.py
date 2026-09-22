from pathlib import Path

import pandas as pd

from config.constants import CATALOGO_PATH, HOJAS_PROCEDIMIENTO


def _obtener_ruta_catalogo() -> Path:
    """
    Devuelve la ruta del archivo de codificación y valida que exista.
    """
    ruta = Path(CATALOGO_PATH)

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el catálogo de codificación en: {CATALOGO_PATH}"
        )

    return ruta


def obtener_hojas_catalogo() -> list[str]:
    """
    Devuelve las hojas válidas DIR, LPU y LSI encontradas en el catálogo.
    """
    ruta = _obtener_ruta_catalogo()
    excel = pd.ExcelFile(ruta)

    return [
        hoja
        for hoja in HOJAS_PROCEDIMIENTO
        if hoja in excel.sheet_names
    ]


def cargar_catalogo(procedimiento: str) -> pd.DataFrame:
    """
    Carga la hoja correspondiente al procedimiento seleccionado.

    El archivo Excel tiene:
    - fila 1: título de la hoja;
    - fila 2: encabezados "Código" y "Concepto";
    - fila 3 en adelante: registros del catálogo.

    Por eso se usa header=1: pandas toma la segunda fila de Excel
    como nombres de las columnas.
    """
    procedimiento = procedimiento.strip().upper()

    if procedimiento not in HOJAS_PROCEDIMIENTO:
        raise ValueError(
            f"Procedimiento no válido: {procedimiento}. "
            f"Valores permitidos: {', '.join(HOJAS_PROCEDIMIENTO)}"
        )

    ruta = _obtener_ruta_catalogo()

    df = pd.read_excel(
        ruta,
        sheet_name=procedimiento,
        header=1,
        engine="openpyxl",
    )

    # Normalizamos los nombres de las columnas.
    # Por ejemplo, "Concepto " pasa a ser "Concepto".
    df.columns = [str(columna).strip() for columna in df.columns]

    columnas_requeridas = {"Código", "Concepto"}
    faltantes = columnas_requeridas.difference(df.columns)

    if faltantes:
        raise ValueError(
            "El catálogo no contiene las columnas requeridas: "
            + ", ".join(sorted(faltantes))
        )

    # Conservamos solamente las columnas que necesita esta primera etapa.
    df = df[["Código", "Concepto"]].copy()

    # Eliminamos filas completamente vacías.
    df = df.dropna(how="all")

    # Eliminamos registros sin código o sin concepto.
    df = df.dropna(subset=["Código", "Concepto"])

    # Convertimos a texto y quitamos espacios al inicio y al final.
    df["Código"] = df["Código"].astype(str).str.strip()
    df["Concepto"] = df["Concepto"].astype(str).str.strip()

    # Eliminamos cualquier fila que haya quedado vacía después de limpiar.
    df = df[
        (df["Código"] != "")
        & (df["Concepto"] != "")
    ].reset_index(drop=True)

    return df
