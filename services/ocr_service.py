from __future__ import annotations

from io import BytesIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

import fitz
import numpy as np
import pandas as pd
from rapidocr_onnxruntime import RapidOCR

from config.constants import CARPETAS_IGNORADAS


OCR_DPI = 180


def _ruta_ignorada(ruta: PurePosixPath) -> bool:
    return any(parte in CARPETAS_IGNORADAS for parte in ruta.parts)


def _renderizar_pagina(pagina: fitz.Page) -> np.ndarray:
    """
    Convierte una página PDF en imagen RGB para OCR.
    """
    escala = OCR_DPI / 72
    matriz = fitz.Matrix(escala, escala)
    pix = pagina.get_pixmap(
        matrix=matriz,
        alpha=False,
        colorspace=fitz.csRGB,
    )

    imagen = np.frombuffer(
        pix.samples,
        dtype=np.uint8,
    ).reshape(pix.height, pix.width, pix.n)

    return imagen


def _extraer_texto_ocr(motor: RapidOCR, imagen: np.ndarray) -> tuple[str, float]:
    """
    Ejecuta OCR sobre una imagen y devuelve texto + confianza media.
    """
    resultado, _ = motor(imagen)

    if not resultado:
        return "", 0.0

    textos = []
    confianzas = []

    for elemento in resultado:
        # RapidOCR devuelve: [coordenadas, texto, confianza]
        if len(elemento) < 3:
            continue

        texto = str(elemento[1]).strip()
        confianza = float(elemento[2])

        if texto:
            textos.append(texto)
            confianzas.append(confianza)

    confianza_media = (
        sum(confianzas) / len(confianzas)
        if confianzas
        else 0.0
    )

    return "\n".join(textos), confianza_media


def ejecutar_ocr_controlado(
    contenido_zip: bytes,
    rutas_pdf: list[str],
    max_paginas_por_pdf: int = 2,
) -> pd.DataFrame:
    """
    Ejecuta OCR solo sobre los PDF seleccionados por el usuario.

    Para controlar tiempo y recursos:
    - máximo 5 PDF por corrida;
    - máximo N primeras páginas por PDF;
    - no modifica el ZIP ni los documentos originales.
    """
    if not rutas_pdf:
        return pd.DataFrame()

    if len(rutas_pdf) > 5:
        raise ValueError(
            "El OCR controlado admite como máximo 5 PDF por ejecución."
        )

    if max_paginas_por_pdf < 1 or max_paginas_por_pdf > 5:
        raise ValueError(
            "El número de páginas por PDF debe estar entre 1 y 5."
        )

    seleccion = set(rutas_pdf)
    registros = []
    motor = RapidOCR()

    try:
        with ZipFile(BytesIO(contenido_zip)) as archivo_zip:
            nombres_zip = set(archivo_zip.namelist())

            for ruta_pdf in rutas_pdf:
                ruta = PurePosixPath(ruta_pdf)

                if (
                    ruta_pdf not in nombres_zip
                    or _ruta_ignorada(ruta)
                    or ruta.suffix.lower() != ".pdf"
                ):
                    registros.append(
                        {
                            "Ruta original": ruta_pdf,
                            "Archivo": ruta.name,
                            "Página": 0,
                            "Caracteres OCR": 0,
                            "Confianza media (%)": 0.0,
                            "Texto OCR": "",
                            "Estado OCR": "No disponible",
                        }
                    )
                    continue

                try:
                    contenido_pdf = archivo_zip.read(ruta_pdf)
                    documento = fitz.open(
                        stream=contenido_pdf,
                        filetype="pdf",
                    )

                    paginas_a_procesar = min(
                        documento.page_count,
                        max_paginas_por_pdf,
                    )

                    for indice in range(paginas_a_procesar):
                        pagina = documento.load_page(indice)
                        imagen = _renderizar_pagina(pagina)
                        texto, confianza = _extraer_texto_ocr(
                            motor,
                            imagen,
                        )

                        registros.append(
                            {
                                "Ruta original": ruta_pdf,
                                "Archivo": ruta.name,
                                "Página": indice + 1,
                                "Caracteres OCR": len(texto),
                                "Confianza media (%)": round(
                                    confianza * 100,
                                    1,
                                ),
                                "Texto OCR": texto,
                                "Estado OCR": (
                                    "Texto detectado"
                                    if texto.strip()
                                    else "Sin texto detectado"
                                ),
                            }
                        )

                    documento.close()

                except Exception as error:
                    registros.append(
                        {
                            "Ruta original": ruta_pdf,
                            "Archivo": ruta.name,
                            "Página": 0,
                            "Caracteres OCR": 0,
                            "Confianza media (%)": 0.0,
                            "Texto OCR": "",
                            "Estado OCR": f"Error: {error}",
                        }
                    )

    except BadZipFile as error:
        raise ValueError(
            "El archivo cargado no es un ZIP válido o está dañado."
        ) from error

    return pd.DataFrame(registros)
