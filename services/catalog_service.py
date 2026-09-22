from pathlib import Path

import pandas as pd

from config.constants import CATALOGO_PATH, HOJAS_PROCEDIMIENTO


def obtener_hojas_catalogo() -> list[str]:
    """
    Devuelve las hojas válidas encontradas en el archivo de codificación.

    En esta primera etapa la función únicamente comprueba si el archivo existe
    y qué hojas DIR, LPU y LSI están disponibles.
    """
    ruta = Path(CATALOGO_PATH)

    if not ruta.exists():
        return []

    excel = pd.ExcelFile(ruta)
    return [hoja for hoja in HOJAS_PROCEDIMIENTO if hoja in excel.sheet_names]
