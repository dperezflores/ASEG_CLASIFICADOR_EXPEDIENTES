from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import fitz
import pandas as pd

from services.openai_multimodal_service import (
    generar_ficha_documental_multimodal,
)


MAX_DOCUMENTOS_MUESTRA = 5


def _paginas_pdf(pdf_bytes: bytes) -> int:
    documento = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )
    try:
        return int(documento.page_count)
    finally:
        documento.close()


def analizar_muestra_fichas_documentales(
    contenido_zip: bytes,
    inventario: pd.DataFrame,
    rutas_pdf: list[str],
    modelo: str,
    on_progress=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Ejecuta una sola llamada multimodal por PDF seleccionado y construye una
    representación tabular de la ficha documental canónica.

    Esta función es experimental y deliberadamente no conecta todavía la ficha
    con catálogo, JEV ni el motor validado de estimaciones.
    """
    rutas = [
        str(ruta)
        for ruta in rutas_pdf
        if str(ruta).strip()
    ]

    if not rutas:
        raise ValueError("Selecciona al menos un PDF.")

    if len(rutas) > MAX_DOCUMENTOS_MUESTRA:
        raise ValueError(
            f"La prueba admite máximo {MAX_DOCUMENTOS_MUESTRA} PDFs."
        )

    inventario_pdf = inventario[
        inventario["Extensión"]
        .astype(str)
        .str.lower()
        .eq(".pdf")
    ].copy()

    mapa = (
        inventario_pdf.drop_duplicates(
            subset=["Ruta original"],
            keep="first",
        )
        .set_index("Ruta original")
        .to_dict("index")
    )

    archivos_registros = []
    documentos_registros = []
    marcadores_registros = []
    perfiles_crudos = {}

    costo_total = 0.0
    tiempo_total = 0.0
    tokens_entrada = 0
    tokens_salida = 0

    with ZipFile(BytesIO(contenido_zip)) as archivo_zip:
        total = len(rutas)

        for posicion, ruta in enumerate(rutas, start=1):
            if ruta not in mapa:
                raise ValueError(
                    f"La ruta seleccionada no corresponde a un PDF: {ruta}"
                )

            meta = mapa[ruta]
            pdf_bytes = archivo_zip.read(ruta)
            paginas_reales = _paginas_pdf(pdf_bytes)

            if on_progress is not None:
                on_progress(
                    posicion,
                    total,
                    str(meta.get("Archivo", ruta)),
                )

            alias = f"documento_muestra_{posicion:02d}"
            perfil = generar_ficha_documental_multimodal(
                documento_alias=alias,
                pdf_bytes=pdf_bytes,
                modelo=modelo,
            )
            perfiles_crudos[ruta] = perfil

            fisico = perfil["physical_document"]
            logicos = perfil["logical_documents"]

            costo = float(perfil["cost_usd"])
            duracion = float(perfil["duration_s"])
            entrada = int(perfil["input_tokens"])
            salida = int(perfil["output_tokens"])

            costo_total += costo
            tiempo_total += duracion
            tokens_entrada += entrada
            tokens_salida += salida

            paginas_observadas = int(
                fisico.get("page_count_observed") or 0
            )

            archivos_registros.append(
                {
                    "Archivo": str(meta.get("Archivo", "")),
                    "Ruta original": ruta,
                    "Páginas reales": paginas_reales,
                    "Páginas observadas IA": paginas_observadas,
                    "Conteo páginas coincide": (
                        paginas_observadas == paginas_reales
                    ),
                    "Documento compuesto": bool(
                        fisico.get("is_compound", False)
                    ),
                    "Documentos lógicos detectados": len(logicos),
                    "Calidad de lectura": str(
                        fisico.get("reading_quality", "")
                    ),
                    "Requiere OCR adicional": bool(
                        fisico.get("needs_additional_ocr", False)
                    ),
                    "Resumen general": str(
                        fisico.get("general_summary", "")
                    ),
                    "Incertidumbre segmentación": str(
                        fisico.get(
                            "segmentation_uncertainty",
                            "",
                        )
                    ),
                    "Modelo": str(perfil["model"]),
                    "Tokens entrada": entrada,
                    "Tokens salida": salida,
                    "Costo (USD)": costo,
                    "Tiempo (s)": duracion,
                }
            )

            for indice_logico, documento in enumerate(
                logicos,
                start=1,
            ):
                inicio = int(documento.get("page_start") or 0)
                fin = int(documento.get("page_end") or 0)
                rango_valido = (
                    1 <= inicio <= fin <= paginas_reales
                )

                confianza = float(
                    documento.get("confidence") or 0.0
                )
                if 0 <= confianza <= 1:
                    confianza *= 100

                logical_id = str(
                    documento.get("logical_id")
                    or f"logical_{indice_logico:02d}"
                )

                hechos = [
                    str(item).strip()
                    for item in documento.get("key_facts", [])
                    if str(item).strip()
                ]

                documentos_registros.append(
                    {
                        "Archivo": str(meta.get("Archivo", "")),
                        "Ruta original": ruta,
                        "ID lógico": logical_id,
                        "Página inicial": inicio,
                        "Página final": fin,
                        "Rango válido": rango_valido,
                        "Título detectado": str(
                            documento.get("detected_title", "")
                        ),
                        "Función formal": str(
                            documento.get("formal_function", "")
                        ),
                        "Acto documentado": str(
                            documento.get("documented_act", "")
                        ),
                        "Alcance documental": str(
                            documento.get("document_scope", "")
                        ),
                        "Texto representativo": str(
                            documento.get(
                                "representative_text",
                                "",
                            )
                        ),
                        "Datos clave": " | ".join(hechos),
                        "Confianza (%)": round(confianza, 2),
                        "Evidencia": str(
                            documento.get("evidence", "")
                        ),
                    }
                )

                for marcador in documento.get(
                    "page_markers",
                    [],
                ):
                    pagina = int(
                        marcador.get("page") or 0
                    )
                    marcadores_registros.append(
                        {
                            "Archivo": str(
                                meta.get("Archivo", "")
                            ),
                            "Ruta original": ruta,
                            "ID lógico": logical_id,
                            "Página": pagina,
                            "Página válida": (
                                1 <= pagina <= paginas_reales
                            ),
                            "Texto marcador": str(
                                marcador.get("text", "")
                            ),
                        }
                    )

    archivos = pd.DataFrame(archivos_registros)
    documentos = pd.DataFrame(documentos_registros)
    marcadores = pd.DataFrame(marcadores_registros)

    resumen = {
        "pdf_analizados": len(rutas),
        "llamadas_multimodales": len(rutas),
        "documentos_logicos": len(documentos),
        "documentos_compuestos": (
            int(archivos["Documento compuesto"].sum())
            if not archivos.empty
            else 0
        ),
        "requieren_ocr_adicional": (
            int(archivos["Requiere OCR adicional"].sum())
            if not archivos.empty
            else 0
        ),
        "rangos_invalidos": (
            int((~documentos["Rango válido"]).sum())
            if not documentos.empty
            else 0
        ),
        "conteos_pagina_incorrectos": (
            int((~archivos["Conteo páginas coincide"]).sum())
            if not archivos.empty
            else 0
        ),
        "tokens_entrada": tokens_entrada,
        "tokens_salida": tokens_salida,
        "costo_total_usd": costo_total,
        "tiempo_total_s": tiempo_total,
    }

    return (
        archivos,
        documentos,
        marcadores,
        resumen,
        perfiles_crudos,
    )
