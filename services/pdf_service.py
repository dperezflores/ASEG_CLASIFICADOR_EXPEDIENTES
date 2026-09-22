from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

import fitz
import pandas as pd

from config.constants import CARPETAS_IGNORADAS


UMBRAL_CARACTERES_POR_PAGINA = 30
PROPORCION_TEXTO_SUFCIENTE = 0.80


def _ruta_ignorada(ruta: PurePosixPath) -> bool:
    return any(parte in CARPETAS_IGNORADAS for parte in ruta.parts)


def _pagina_tiene_texto(pagina: fitz.Page) -> bool:
    """
    Considera que una página tiene texto útil cuando PyMuPDF puede extraer
    al menos un pequeño volumen de caracteres visibles.
    """
    texto = pagina.get_text("text")
    texto_limpio = "".join(texto.split())
    return len(texto_limpio) >= UMBRAL_CARACTERES_POR_PAGINA


def _clasificar_pdf(paginas: int, paginas_con_texto: int) -> str:
    if paginas == 0:
        return "Sin páginas"

    proporcion = paginas_con_texto / paginas

    if paginas_con_texto == 0:
        return "Requiere OCR"

    if proporcion >= PROPORCION_TEXTO_SUFCIENTE:
        return "Texto extraíble"

    return "Mixto"


def analizar_pdfs_zip(contenido_zip: bytes) -> tuple[pd.DataFrame, dict]:
    """
    Revisa todos los PDF del expediente sin aplicar OCR.

    Para cada PDF determina:
    - número de páginas;
    - páginas con texto extraíble;
    - proporción de páginas con texto;
    - estado preliminar: Texto extraíble, Mixto o Requiere OCR.

    Los PDF no se modifican.
    """
    registros = []

    try:
        with ZipFile(BytesIO(contenido_zip)) as archivo_zip:
            for info in archivo_zip.infolist():
                ruta = PurePosixPath(info.filename)

                if (
                    info.is_dir()
                    or not ruta.parts
                    or ruta.parts[0] == "__MACOSX"
                    or _ruta_ignorada(ruta)
                    or ruta.suffix.lower() != ".pdf"
                ):
                    continue

                try:
                    contenido_pdf = archivo_zip.read(info)
                    documento = fitz.open(
                        stream=contenido_pdf,
                        filetype="pdf",
                    )

                    paginas = documento.page_count
                    paginas_con_texto = 0

                    for pagina in documento:
                        if _pagina_tiene_texto(pagina):
                            paginas_con_texto += 1

                    documento.close()

                    proporcion = (
                        paginas_con_texto / paginas
                        if paginas
                        else 0.0
                    )

                    estado = _clasificar_pdf(
                        paginas,
                        paginas_con_texto,
                    )

                    registros.append(
                        {
                            "Ruta original": str(ruta),
                            "Archivo": ruta.name,
                            "Páginas": paginas,
                            "Páginas con texto": paginas_con_texto,
                            "Texto (%)": round(proporcion * 100, 1),
                            "Estado PDF": estado,
                            "Detalle": "",
                        }
                    )

                except Exception as error:
                    registros.append(
                        {
                            "Ruta original": str(ruta),
                            "Archivo": ruta.name,
                            "Páginas": 0,
                            "Páginas con texto": 0,
                            "Texto (%)": 0.0,
                            "Estado PDF": "Error de lectura",
                            "Detalle": str(error),
                        }
                    )

    except BadZipFile as error:
        raise ValueError(
            "El archivo cargado no es un ZIP válido o está dañado."
        ) from error

    analisis = pd.DataFrame(registros)

    if analisis.empty:
        return analisis, {
            "pdfs": 0,
            "paginas": 0,
            "estados": {},
        }

    estados = Counter(analisis["Estado PDF"])

    resumen = {
        "pdfs": len(analisis),
        "paginas": int(analisis["Páginas"].sum()),
        "estados": dict(sorted(estados.items())),
    }

    return analisis, resumen
