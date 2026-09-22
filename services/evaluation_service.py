from __future__ import annotations

from io import BytesIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

import pandas as pd


def preparar_muestra_ocr(
    resultado_ocr: pd.DataFrame,
    catalogo: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrupa el OCR por documento y genera alias neutros para evaluación.

    El alias es el identificador que después verá el clasificador. La ruta
    original se conserva solo para control interno y comparación posterior.
    """
    if resultado_ocr.empty:
        return pd.DataFrame()

    catalogo_lookup = {
        str(fila["Código"]).strip().lower(): str(fila["Concepto"]).strip()
        for _, fila in catalogo.iterrows()
    }

    registros = []

    for indice, (ruta, grupo) in enumerate(
        resultado_ocr.groupby("Ruta original", sort=False),
        start=1,
    ):
        grupo = grupo.sort_values("Página")
        archivo = PurePosixPath(str(ruta)).name

        textos = [
            str(texto).strip()
            for texto in grupo["Texto OCR"].tolist()
            if str(texto).strip()
        ]

        confianza = pd.to_numeric(
            grupo["Confianza media (%)"],
            errors="coerce",
        ).dropna()

        codigo_real = archivo
        concepto_real = catalogo_lookup.get(
            codigo_real.lower(),
            "Pendiente de referencia",
        )

        registros.append(
            {
                "Documento": f"documento_{indice:03d}",
                "Ruta original": str(ruta),
                "Archivo real": archivo,
                "Páginas OCR": int((grupo["Página"] > 0).sum()),
                "Caracteres OCR": int(
                    pd.to_numeric(
                        grupo["Caracteres OCR"],
                        errors="coerce",
                    ).fillna(0).sum()
                ),
                "Confianza OCR media (%)": round(
                    float(confianza.mean()) if not confianza.empty else 0.0,
                    1,
                ),
                "Texto OCR": "\n\n".join(textos),
                "Código real": (
                    codigo_real
                    if concepto_real != "Pendiente de referencia"
                    else "Pendiente"
                ),
                "Concepto real": concepto_real,
            }
        )

    return pd.DataFrame(registros)


def extraer_pdf_del_zip(
    contenido_zip: bytes,
    ruta_pdf: str,
) -> bytes:
    """
    Recupera un PDF concreto del expediente ZIP sin escribirlo en disco.
    """
    try:
        with ZipFile(BytesIO(contenido_zip)) as archivo_zip:
            if ruta_pdf not in archivo_zip.namelist():
                raise ValueError(
                    f"No se encontró el documento dentro del ZIP: {ruta_pdf}"
                )
            return archivo_zip.read(ruta_pdf)
    except BadZipFile as error:
        raise ValueError(
            "El expediente cargado no es un ZIP válido."
        ) from error
