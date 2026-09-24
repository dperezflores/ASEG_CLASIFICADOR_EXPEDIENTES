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
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
    dict,
]:
    """
    Ejecuta una sola llamada multimodal por PDF seleccionado y tabula la ficha
    documental canónica V2.

    Sigue siendo un experimento aislado: no alimenta catálogo, JEV, estimaciones
    ni codificación final.
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
    registros_internos = []
    relaciones_registros = []
    marcadores_registros = []
    perfiles_crudos = {}

    costo_total = 0.0
    tiempo_total = 0.0
    tokens_entrada = 0
    tokens_salida = 0
    relaciones_invalidas = 0
    registros_rango_invalido = 0

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
            relaciones = perfil.get(
                "logical_relationships",
                [],
            )

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
                    "Registros internos detectados": sum(
                        len(
                            documento.get(
                                "internal_records",
                                [],
                            )
                        )
                        for documento in logicos
                    ),
                    "Relaciones lógicas detectadas": len(relaciones),
                    "Calidad de lectura": str(
                        fisico.get("reading_quality", "")
                    ),
                    "Requiere OCR adicional": bool(
                        fisico.get(
                            "needs_additional_ocr",
                            False,
                        )
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

            ids_logicos = {
                str(
                    documento.get("logical_id")
                    or f"logical_{indice_logico:02d}"
                )
                for indice_logico, documento in enumerate(
                    logicos,
                    start=1,
                )
            }

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
                    for item in documento.get(
                        "key_facts",
                        [],
                    )
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
                            documento.get(
                                "detected_title",
                                "",
                            )
                        ),
                        "Función formal": str(
                            documento.get(
                                "formal_function",
                                "",
                            )
                        ),
                        "Acto documentado": str(
                            documento.get(
                                "documented_act",
                                "",
                            )
                        ),
                        "Alcance de identidad": str(
                            documento.get(
                                "identity_scope",
                                "",
                            )
                        ),
                        "Límites de identidad": str(
                            documento.get(
                                "identity_limits",
                                "",
                            )
                        ),
                        "Alcance documental": str(
                            documento.get(
                                "document_scope",
                                "",
                            )
                        ),
                        "Texto representativo": str(
                            documento.get(
                                "representative_text",
                                "",
                            )
                        ),
                        "Datos clave": " | ".join(hechos),
                        "Registros internos": len(
                            documento.get(
                                "internal_records",
                                [],
                            )
                        ),
                        "Confianza (%)": round(
                            confianza,
                            2,
                        ),
                        "Evidencia": str(
                            documento.get("evidence", "")
                        ),
                    }
                )

                for registro in documento.get(
                    "internal_records",
                    [],
                ):
                    inicio_registro = int(
                        registro.get("page_start") or 0
                    )
                    fin_registro = int(
                        registro.get("page_end") or 0
                    )
                    rango_registro_valido = (
                        1
                        <= inicio_registro
                        <= fin_registro
                        <= paginas_reales
                    )
                    if not rango_registro_valido:
                        registros_rango_invalido += 1

                    confianza_registro = float(
                        registro.get("confidence")
                        or 0.0
                    )
                    if 0 <= confianza_registro <= 1:
                        confianza_registro *= 100

                    hechos_registro = [
                        str(item).strip()
                        for item in registro.get(
                            "key_facts",
                            [],
                        )
                        if str(item).strip()
                    ]

                    registros_internos.append(
                        {
                            "Archivo": str(
                                meta.get("Archivo", "")
                            ),
                            "Ruta original": ruta,
                            "ID lógico": logical_id,
                            "ID registro": str(
                                registro.get(
                                    "record_id",
                                    "",
                                )
                            ),
                            "Tipo registro": str(
                                registro.get(
                                    "record_type",
                                    "",
                                )
                            ),
                            "Página inicial": (
                                inicio_registro
                            ),
                            "Página final": fin_registro,
                            "Rango válido": (
                                rango_registro_valido
                            ),
                            "Etiqueta": str(
                                registro.get(
                                    "label",
                                    "",
                                )
                            ),
                            "Acto documentado": str(
                                registro.get(
                                    "documented_act",
                                    "",
                                )
                            ),
                            "Texto representativo": str(
                                registro.get(
                                    "representative_text",
                                    "",
                                )
                            ),
                            "Datos clave": " | ".join(
                                hechos_registro
                            ),
                            "Confianza (%)": round(
                                confianza_registro,
                                2,
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
                                1
                                <= pagina
                                <= paginas_reales
                            ),
                            "Texto marcador": str(
                                marcador.get(
                                    "text",
                                    "",
                                )
                            ),
                        }
                    )

            for relacion in relaciones:
                source = str(
                    relacion.get(
                        "source_logical_id",
                        "",
                    )
                )
                target = str(
                    relacion.get(
                        "target_logical_id",
                        "",
                    )
                )
                ids_validos = (
                    source in ids_logicos
                    and target in ids_logicos
                    and source != target
                )
                if not ids_validos:
                    relaciones_invalidas += 1

                confianza_relacion = float(
                    relacion.get("confidence")
                    or 0.0
                )
                if 0 <= confianza_relacion <= 1:
                    confianza_relacion *= 100

                relaciones_registros.append(
                    {
                        "Archivo": str(
                            meta.get("Archivo", "")
                        ),
                        "Ruta original": ruta,
                        "Origen": source,
                        "Relación": str(
                            relacion.get(
                                "relation_type",
                                "",
                            )
                        ),
                        "Destino": target,
                        "IDs válidos": ids_validos,
                        "Confianza (%)": round(
                            confianza_relacion,
                            2,
                        ),
                        "Evidencia": str(
                            relacion.get(
                                "evidence",
                                "",
                            )
                        ),
                    }
                )

    archivos = pd.DataFrame(archivos_registros)
    documentos = pd.DataFrame(documentos_registros)
    registros = pd.DataFrame(registros_internos)
    relaciones = pd.DataFrame(relaciones_registros)
    marcadores = pd.DataFrame(marcadores_registros)

    resumen = {
        "pdf_analizados": len(rutas),
        "llamadas_multimodales": len(rutas),
        "documentos_logicos": len(documentos),
        "registros_internos": len(registros),
        "relaciones_logicas": len(relaciones),
        "documentos_compuestos": (
            int(
                archivos[
                    "Documento compuesto"
                ].sum()
            )
            if not archivos.empty
            else 0
        ),
        "requieren_ocr_adicional": (
            int(
                archivos[
                    "Requiere OCR adicional"
                ].sum()
            )
            if not archivos.empty
            else 0
        ),
        "rangos_invalidos": (
            int(
                (~documentos["Rango válido"]).sum()
            )
            if not documentos.empty
            else 0
        ),
        "registros_rango_invalido": (
            registros_rango_invalido
        ),
        "relaciones_invalidas": relaciones_invalidas,
        "conteos_pagina_incorrectos": (
            int(
                (~archivos[
                    "Conteo páginas coincide"
                ]).sum()
            )
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
        registros,
        relaciones,
        marcadores,
        resumen,
        perfiles_crudos,
    )
