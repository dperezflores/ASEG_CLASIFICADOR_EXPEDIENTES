from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import fitz
import pandas as pd

from services.evaluation_service import extraer_paginas_pdf_del_zip
from services.jev_classifier_service import JevError, clasificar_texto_con_jev
from services.openai_multimodal_service import (
    OpenAIMultimodalError,
    clasificar_pdf_multimodal,
)


PAGINAS_INICIALES = 2
MIN_CARACTERES_TEXTO_NATIVO = 200


def preparar_muestra_hibrida(
    analisis_pdf: pd.DataFrame,
    catalogo: pd.DataFrame,
) -> pd.DataFrame:
    catalogo_lookup = {
        str(fila["Código"]).strip().lower(): str(fila["Concepto"]).strip()
        for _, fila in catalogo.iterrows()
    }

    registros = []

    for indice, fila in analisis_pdf.reset_index(drop=True).iterrows():
        archivo = str(fila["Archivo"]).strip()
        concepto_real = catalogo_lookup.get(
            archivo.lower(),
            "Sin referencia conocida",
        )

        registros.append(
            {
                "Documento": f"documento_{indice + 1:03d}",
                "Ruta original": str(fila["Ruta original"]),
                "Archivo real": archivo,
                "Estado PDF": str(fila["Estado PDF"]),
                "Páginas": int(fila["Páginas"]),
                "Concepto de referencia": concepto_real,
            }
        )

    return pd.DataFrame(registros)


def _extraer_texto_nativo(
    contenido_zip: bytes,
    ruta_pdf: str,
    max_paginas: int = PAGINAS_INICIALES,
) -> tuple[str, list[int]]:
    with ZipFile(BytesIO(contenido_zip)) as archivo_zip:
        contenido_pdf = archivo_zip.read(ruta_pdf)

    documento = fitz.open(stream=contenido_pdf, filetype="pdf")
    textos = []
    paginas = []

    try:
        limite = min(documento.page_count, max_paginas)
        for indice in range(limite):
            texto = documento.load_page(indice).get_text("text").strip()
            if texto:
                textos.append(texto)
            paginas.append(indice + 1)
    finally:
        documento.close()

    return "\n\n".join(textos).strip(), paginas


def _multimodal(
    fila: pd.Series,
    catalogo: pd.DataFrame,
    contenido_zip: bytes,
    modelo: str,
    motivo_ruta: str,
) -> dict:
    paginas = list(
        range(
            1,
            min(int(fila["Páginas"]), PAGINAS_INICIALES) + 1,
        )
    )

    pdf_reducido = extraer_paginas_pdf_del_zip(
        contenido_zip,
        str(fila["Ruta original"]),
        paginas,
    )

    resultado = clasificar_pdf_multimodal(
        documento_alias=str(fila["Documento"]),
        pdf_bytes=pdf_reducido,
        catalogo=catalogo,
        modelo=modelo,
    )

    return {
        "Documento": str(fila["Documento"]),
        "Ruta utilizada": "Multimodal",
        "Motivo de ruta": motivo_ruta,
        "Título detectado": resultado["Título detectado"],
        "Coincide catálogo": bool(resultado["Coincide catálogo"]),
        "Concepto propuesto": (
            resultado["Resultado Ruta B"]
            if resultado["Coincide catálogo"]
            else ""
        ),
        "Código propuesto": (
            resultado["Código Ruta B"]
            if resultado["Coincide catálogo"]
            else ""
        ),
        "Confianza (%)": float(resultado["Confianza B"]),
        "Evidencia": resultado["Evidencia B"],
        "Modelo": resultado["Modelo B"],
        "Tokens entrada": int(resultado["Tokens entrada B"]),
        "Tokens salida": int(resultado["Tokens salida B"]),
        "Costo (USD)": float(resultado["Costo B (USD)"]),
        "Tiempo (s)": float(resultado["Tiempo B (s)"]),
        "Páginas usadas": ",".join(str(p) for p in paginas),
    }


def clasificar_documento_hibrido(
    fila: pd.Series,
    catalogo: pd.DataFrame,
    contenido_zip: bytes,
    modelo_multimodal: str,
    umbral_jev: float,
) -> dict:
    estado = str(fila["Estado PDF"])

    if estado == "Texto extraíble":
        texto, paginas = _extraer_texto_nativo(
            contenido_zip,
            str(fila["Ruta original"]),
        )

        if len(texto) >= MIN_CARACTERES_TEXTO_NATIVO:
            try:
                jev = clasificar_texto_con_jev(
                    documento_alias=str(fila["Documento"]),
                    texto_ocr=texto,
                    catalogo=catalogo,
                )

                fuera = (
                    jev["Resultado Ruta A"]
                    == "Fuera de catálogo / no identificado"
                )
                confianza = float(jev["Confianza A"])

                if not fuera and confianza >= umbral_jev:
                    return {
                        "Documento": str(fila["Documento"]),
                        "Ruta utilizada": "Texto nativo + Jev",
                        "Motivo de ruta": "Texto nativo suficiente",
                        "Título detectado": jev["Resultado Ruta A"],
                        "Coincide catálogo": True,
                        "Concepto propuesto": jev["Resultado Ruta A"],
                        "Código propuesto": jev["Código Ruta A"],
                        "Confianza (%)": confianza,
                        "Evidencia": (
                            "Clasificación Jev sobre texto nativo. "
                            "No requirió OCR ni multimodal."
                        ),
                        "Modelo": jev["Modelo A"],
                        "Tokens entrada": int(jev["Tokens entrada A"]),
                        "Tokens salida": int(jev["Tokens salida A"]),
                        "Costo (USD)": float(jev["Costo A (USD)"]),
                        "Tiempo (s)": float(jev["Tiempo A (s)"]),
                        "Páginas usadas": ",".join(
                            str(p) for p in paginas
                        ),
                    }

                motivo = (
                    "Jev indicó fuera de catálogo"
                    if fuera
                    else (
                        f"Confianza Jev {confianza:.1f}% "
                        f"menor al umbral {umbral_jev:.1f}%"
                    )
                )

                mm = _multimodal(
                    fila,
                    catalogo,
                    contenido_zip,
                    modelo_multimodal,
                    motivo,
                )
                mm["Costo (USD)"] += float(jev["Costo A (USD)"])
                mm["Tiempo (s)"] += float(jev["Tiempo A (s)"])
                mm["Motivo de ruta"] = "Escalado: " + motivo
                return mm

            except JevError:
                return _multimodal(
                    fila,
                    catalogo,
                    contenido_zip,
                    modelo_multimodal,
                    "Fallo de Jev; escalado automático",
                )

        return _multimodal(
            fila,
            catalogo,
            contenido_zip,
            modelo_multimodal,
            "Texto nativo insuficiente",
        )

    if estado in ("Requiere OCR", "Mixto"):
        return _multimodal(
            fila,
            catalogo,
            contenido_zip,
            modelo_multimodal,
            f"{estado}: se evita OCR local",
        )

    return {
        "Documento": str(fila["Documento"]),
        "Ruta utilizada": "No procesado",
        "Motivo de ruta": f"Estado PDF no procesable: {estado}",
        "Título detectado": "",
        "Coincide catálogo": False,
        "Concepto propuesto": "",
        "Código propuesto": "",
        "Confianza (%)": 0.0,
        "Evidencia": "",
        "Modelo": "",
        "Tokens entrada": 0,
        "Tokens salida": 0,
        "Costo (USD)": 0.0,
        "Tiempo (s)": 0.0,
        "Páginas usadas": "",
    }


def clasificar_muestra_hibrida(
    muestra: pd.DataFrame,
    catalogo: pd.DataFrame,
    contenido_zip: bytes,
    modelo_multimodal: str,
    umbral_jev: float,
) -> pd.DataFrame:
    resultados = []

    for _, fila in muestra.iterrows():
        resultado = clasificar_documento_hibrido(
            fila=fila,
            catalogo=catalogo,
            contenido_zip=contenido_zip,
            modelo_multimodal=modelo_multimodal,
            umbral_jev=umbral_jev,
        )

        resultado["Concepto de referencia"] = str(
            fila["Concepto de referencia"]
        )

        referencia = str(fila["Concepto de referencia"])
        if referencia != "Sin referencia conocida":
            resultado["Acierto conocido"] = (
                bool(resultado["Coincide catálogo"])
                and str(resultado["Concepto propuesto"]).strip().casefold()
                == referencia.strip().casefold()
            )
        else:
            resultado["Acierto conocido"] = None

        resultados.append(resultado)

    return pd.DataFrame(resultados)
