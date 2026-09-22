from __future__ import annotations

from io import BytesIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

import fitz
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
                "Páginas evaluadas": [
                    int(pagina)
                    for pagina in grupo["Página"].tolist()
                    if int(pagina) > 0
                ],
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



def extraer_paginas_pdf_del_zip(
    contenido_zip: bytes,
    ruta_pdf: str,
    paginas_1_based: list[int],
) -> bytes:
    """
    Construye un PDF temporal en memoria con exactamente las páginas usadas
    por la prueba OCR, para que la comparación multimodal sea equivalente.
    """
    pdf_original = extraer_pdf_del_zip(contenido_zip, ruta_pdf)

    if not paginas_1_based:
        raise ValueError("No se indicaron páginas para la evaluación multimodal.")

    origen = fitz.open(stream=pdf_original, filetype="pdf")
    destino = fitz.open()

    try:
        for numero in paginas_1_based:
            indice = int(numero) - 1
            if indice < 0 or indice >= origen.page_count:
                raise ValueError(
                    f"La página {numero} no existe en el documento evaluado."
                )
            destino.insert_pdf(
                origen,
                from_page=indice,
                to_page=indice,
            )

        return destino.tobytes(
            garbage=4,
            deflate=True,
        )
    finally:
        destino.close()
        origen.close()
