from __future__ import annotations

import pandas as pd

from services.catalog_service import cargar_catalogo
from services.document_profile_service import (
    analizar_muestra_fichas_documentales,
)
from services.profile_jev_service import (
    clasificar_fichas_v2_con_jev,
)
from services.structural_analysis_service import (
    construir_mapa_estructural,
    obtener_archivos_unidad,
    obtener_estimaciones_detectadas,
)
from services.unit_content_analysis_service import (
    agrupar_resultados_unidad,
    consolidar_candidatos_unidad,
)


def listar_estimaciones_para_prueba(
    inventario: pd.DataFrame,
) -> pd.DataFrame:
    mapa = construir_mapa_estructural(inventario)
    estimaciones = obtener_estimaciones_detectadas(mapa)
    if estimaciones.empty:
        return estimaciones

    return estimaciones.sort_values(
        by=["Consecutivo", "Ruta carpeta"],
        kind="stable",
    ).reset_index(drop=True)


def _resultados_base_desde_jev(
    detalle_jev: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adapta Ficha V2 + JEV al contrato tabular del motor de comparación conjunta
    ya validado. No cambia la lógica de dicho motor.
    """
    filas = []

    for _, fila in detalle_jev.iterrows():
        decision = str(
            fila.get("Decisión provisional", "")
        )
        codigo_inicial = str(
            fila.get("Código JEV", "")
        ).strip()
        concepto_inicial = str(
            fila.get("Concepto JEV", "")
        ).strip()

        if decision == "Candidato al código de la unidad":
            relacion = "candidato_unidad"
            coincide = True
            codigo_final = codigo_inicial
            concepto_final = concepto_inicial
            rol = "Candidato a representar la unidad"

        elif decision == "Código propio candidato validado":
            relacion = "documento_independiente"
            coincide = True
            codigo_final = str(
                fila.get("Código provisional", "")
            ).strip()
            concepto_final = concepto_inicial
            rol = "Documento con código propio candidato"

        elif decision in (
            "Fuera de catálogo",
            "Coincidencia conceptual; extracto sin código completo",
            "Soporte por relación interna",
            "Candidato rechazado por equivalencia estricta",
        ):
            relacion = "soporte"
            coincide = False
            codigo_final = ""
            concepto_final = ""
            rol = "Soporte / fuera de catálogo"

        else:
            relacion = "indeterminado"
            coincide = False
            codigo_final = ""
            concepto_final = ""
            rol = "Revisión necesaria"

        confianza_jev = float(
            fila.get("Confianza JEV (%)", 0.0)
            or 0.0
        )
        confianza_estricta = float(
            fila.get(
                "Confianza estricta JEV (%)",
                0.0,
            )
            or 0.0
        )

        if (
            str(
                fila.get(
                    "Validación estricta JEV",
                    "",
                )
            )
            == "equivalente"
            and confianza_estricta > 0
        ):
            confianza = min(
                confianza_jev,
                confianza_estricta,
            )
        else:
            confianza = confianza_jev

        costo = float(
            fila.get(
                "Costo JEV clasificación (USD)",
                0.0,
            )
            or 0.0
        ) + float(
            fila.get(
                "Costo JEV validación (USD)",
                0.0,
            )
            or 0.0
        )

        tiempo = float(
            fila.get(
                "Tiempo JEV clasificación (s)",
                0.0,
            )
            or 0.0
        ) + float(
            fila.get(
                "Tiempo JEV validación (s)",
                0.0,
            )
            or 0.0
        )

        evidencia = (
            f"Ficha V2: {fila.get('Título detectado', '')}. "
            f"JEV: {concepto_inicial or 'fuera de catálogo'}. "
            f"Decisión: {decision}. "
            f"{fila.get('Motivo', '')}"
        )

        filas.append(
            {
                "Archivo": str(
                    fila.get("Archivo", "")
                ),
                "Ruta original": str(
                    fila.get("Ruta original", "")
                ),
                "ID lógico": str(
                    fila.get("ID lógico", "")
                ),
                "Ruta utilizada": (
                    "Ficha V2 + JEV experimental"
                ),
                "Motivo de ruta": (
                    "La identidad proviene de una Ficha V2 "
                    "reutilizable y el candidato de JEV."
                ),
                "Título detectado": str(
                    fila.get("Título detectado", "")
                ),
                "Clasificación inicial": concepto_inicial,
                "Código inicial": codigo_inicial,
                "Verificación identidad pura": (
                    "Sustituida por Ficha V2"
                ),
                "Título identidad pura": str(
                    fila.get("Título detectado", "")
                ),
                "Función identidad pura": str(
                    fila.get("Función formal", "")
                ),
                "Acto identidad pura": str(
                    fila.get("Acto documentado", "")
                ),
                "Confianza identidad pura (%)": confianza,
                "Evidencia identidad pura": str(
                    fila.get("Límites de identidad", "")
                ),
                "Equivalencia resuelta por": (
                    "JEV estricto sobre Ficha V2"
                ),
                "Equivalencia JEV": str(
                    fila.get(
                        "Validación estricta JEV",
                        "No requerida",
                    )
                ),
                "Decisión original JEV": str(
                    fila.get(
                        "Decisión original estricta JEV",
                        "",
                    )
                ),
                "Confianza JEV (%)": confianza_jev,
                "Probabilidad elegida JEV (%)": float(
                    fila.get(
                        "Probabilidad elegida JEV (%)",
                        0.0,
                    )
                    or 0.0
                ),
                "Margen JEV (%)": fila.get(
                    "Margen estricta JEV (%)",
                    None,
                ),
                "Control JEV": str(
                    fila.get(
                        "Control estricta JEV",
                        "",
                    )
                ),
                "Fallback multimodal JEV": "No",
                "Error JEV": str(
                    fila.get(
                        "Error validación estricta",
                        "",
                    )
                ),
                "Llamadas multimodales documento": 0,
                "Llamadas JEV documento": (
                    1
                    + int(
                        str(
                            fila.get(
                                "Validación estricta JEV",
                                "No requerida",
                            )
                        )
                        not in (
                            "",
                            "No requerida",
                        )
                    )
                ),
                "Equivalencia funcional": str(
                    fila.get(
                        "Validación estricta JEV",
                        "no_evaluada",
                    )
                ),
                "Confianza equivalencia (%)": (
                    confianza_estricta
                ),
                "Evidencia equivalencia": str(
                    fila.get("Motivo", "")
                ),
                "Validación secundaria": (
                    "Ficha V2 + JEV"
                ),
                "Alcance documental": str(
                    fila.get(
                        "Alcance documental",
                        "indeterminado",
                    )
                ),
                "Relación con la unidad": relacion,
                "Concepto relacionado": "",
                "Código relacionado": "",
                "Coincide catálogo": coincide,
                "Concepto propuesto": concepto_final,
                "Código de catálogo": codigo_final,
                "Confianza (%)": confianza,
                "Rol propuesto en la unidad": rol,
                "Evidencia": evidencia,
                "Modelo": "Ficha V2 + JEV",
                "Costo (USD)": costo,
                "Tiempo (s)": tiempo,
                "Páginas usadas": "",
                "Intentos de análisis": 1,
                "Reintento aplicado": False,
                "Error": "",
            }
        )

    return pd.DataFrame(filas)


def ejecutar_est_completa_ficha_v2(
    contenido_zip: bytes,
    inventario: pd.DataFrame,
    ruta_carpeta: str,
    procedimiento: str,
    consecutivo: int,
    modelo_multimodal: str,
    drive_folder_id: str | None,
    usar_cache: bool = True,
    forzar_reanalisis: bool = False,
    on_progress=None,
) -> dict:
    """
    Pipeline experimental EST completa.

    No reemplaza el flujo actual:
    estructura -> Ficha V2 cacheable -> JEV -> adaptador ->
    comparación conjunta EXISTENTE -> agrupación EXISTENTE.
    """
    archivos_unidad = obtener_archivos_unidad(
        inventario,
        ruta_carpeta,
    )
    pdfs = archivos_unidad[
        archivos_unidad["Extensión"]
        .astype(str)
        .str.lower()
        .eq(".pdf")
    ].copy()

    if pdfs.empty:
        raise ValueError(
            "La unidad seleccionada no contiene PDFs directos."
        )

    rutas_pdf = pdfs[
        "Ruta original"
    ].astype(str).tolist()

    def progreso_ficha(posicion, total, archivo):
        if on_progress is not None:
            on_progress(
                "Ficha V2",
                posicion,
                total,
                archivo,
            )

    (
        archivos_ficha,
        documentos,
        registros,
        relaciones,
        marcadores,
        resumen_ficha,
        perfiles_crudos,
    ) = analizar_muestra_fichas_documentales(
        contenido_zip=contenido_zip,
        inventario=inventario,
        rutas_pdf=rutas_pdf,
        modelo=modelo_multimodal,
        on_progress=progreso_ficha,
        drive_folder_id=drive_folder_id,
        usar_cache=usar_cache,
        forzar_reanalisis=forzar_reanalisis,
        max_documentos=None,
    )

    catalogo_base = cargar_catalogo(
        procedimiento
    )

    def progreso_jev(
        posicion,
        total,
        archivo,
        logical_id,
    ):
        if on_progress is not None:
            on_progress(
                "JEV",
                posicion,
                total,
                f"{archivo} · {logical_id}",
            )

    detalle_jev, resumen_jev = (
        clasificar_fichas_v2_con_jev(
            documentos=documentos,
            registros=registros,
            relaciones=relaciones,
            inventario=inventario,
            catalogo_base=catalogo_base,
            procedimiento=procedimiento,
            on_progress=progreso_jev,
        )
    )

    resultados_base = _resultados_base_desde_jev(
        detalle_jev
    )

    if on_progress is not None:
        on_progress(
            "Comparación conjunta",
            1,
            1,
            "Comparando candidatos al código de la estimación",
        )

    (
        resultados_comparados,
        detalle_comparacion,
        meta_comparacion,
    ) = consolidar_candidatos_unidad(
        contenido_zip=contenido_zip,
        resultados=resultados_base,
        procedimiento=procedimiento,
        consecutivo=int(consecutivo),
        modelo_multimodal=modelo_multimodal,
        tipo_unidad="Estimación",
    )

    resultados_finales, resumen_grupos = (
        agrupar_resultados_unidad(
            resultados=resultados_comparados,
            procedimiento=procedimiento,
            consecutivo=int(consecutivo),
        )
    )

    costo_comparacion = float(
        meta_comparacion.get(
            "Costo comparación (USD)",
            0.0,
        )
        or 0.0
    )
    tiempo_comparacion = float(
        meta_comparacion.get(
            "Tiempo comparación (s)",
            0.0,
        )
        or 0.0
    )
    llamada_comparacion = int(
        meta_comparacion.get("Estado")
        == "Comparación ejecutada"
    )

    resumen_pipeline = {
        "unidad": "Estimación",
        "consecutivo": int(consecutivo),
        "ruta_carpeta": ruta_carpeta,
        "pdfs_directos": len(pdfs),
        "documentos_logicos": len(documentos),
        "fichas_cache_reutilizadas": int(
            resumen_ficha.get(
                "fichas_cache_reutilizadas",
                0,
            )
        ),
        "multimodales_ficha_nuevas": int(
            resumen_ficha.get(
                "llamadas_multimodales",
                0,
            )
        ),
        "costo_fichas_nuevas_usd": float(
            resumen_ficha.get(
                "costo_total_usd",
                0.0,
            )
        ),
        "llamadas_jev": int(
            resumen_jev.get(
                "llamadas_jev",
                0,
            )
        ),
        "costo_jev_usd": float(
            resumen_jev.get(
                "costo_total_jev_usd",
                0.0,
            )
        ),
        "multimodales_comparacion_conjunta": (
            llamada_comparacion
        ),
        "costo_comparacion_conjunta_usd": (
            costo_comparacion
        ),
        "tiempo_comparacion_conjunta_s": (
            tiempo_comparacion
        ),
        "multimodales_totales_nuevas": (
            int(
                resumen_ficha.get(
                    "llamadas_multimodales",
                    0,
                )
            )
            + llamada_comparacion
        ),
        "costo_api_total_usd": (
            float(
                resumen_ficha.get(
                    "costo_total_usd",
                    0.0,
                )
            )
            + float(
                resumen_jev.get(
                    "costo_total_jev_usd",
                    0.0,
                )
            )
            + costo_comparacion
        ),
        "representante": str(
            meta_comparacion.get(
                "Representante",
                "",
            )
        ),
        "estado_comparacion": str(
            meta_comparacion.get(
                "Estado",
                "",
            )
        ),
    }

    return {
        "archivos_unidad": archivos_unidad,
        "archivos_ficha": archivos_ficha,
        "documentos": documentos,
        "registros": registros,
        "relaciones": relaciones,
        "marcadores": marcadores,
        "perfiles_crudos": perfiles_crudos,
        "resumen_ficha": resumen_ficha,
        "detalle_jev": detalle_jev,
        "resumen_jev": resumen_jev,
        "resultados_base": resultados_base,
        "resultados_finales": resultados_finales,
        "detalle_comparacion": detalle_comparacion,
        "meta_comparacion": meta_comparacion,
        "resumen_grupos": resumen_grupos,
        "resumen_pipeline": resumen_pipeline,
    }
