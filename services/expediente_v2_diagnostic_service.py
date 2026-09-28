from __future__ import annotations

from pathlib import PurePosixPath

import pandas as pd

from services.catalog_service import cargar_catalogo
from services.document_profile_service import (
    analizar_muestra_fichas_documentales,
)
from services.estimation_v2_experimental_service import (
    _estabilizar_roles_intrinsecos_soporte,
    _resultados_base_desde_jev,
    consolidar_salida_fisica,
)
from services.global_relationship_service import (
    analizar_duplicados_exactos,
)
from services.profile_jev_service import (
    PIPELINE_VERSION_FICHA_JEV,
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


DIAGNOSTICO_INTEGRAL_SCHEMA_VERSION = 1


def _pdfs_inventario(
    inventario: pd.DataFrame,
) -> pd.DataFrame:
    if inventario.empty:
        return inventario.copy()

    return inventario[
        inventario["Extensión"]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq(".pdf")
    ].copy()


def _agrupar_no_estimacion(
    resultados: pd.DataFrame,
) -> pd.DataFrame:
    """
    Añade las columnas de grupo/acción necesarias para consolidación física a
    resultados que no pertenecen a una unidad EST_n.

    En esta primera versión integral no se inventan unidades conjuntas nuevas:
    un código propio validado se conserva; soportes se mantienen; resultados
    indeterminados quedan en revisión.
    """
    salida = resultados.copy()

    if salida.empty:
        return salida

    for columna, valor in (
        ("Grupo lógico", ""),
        ("Relación consolidada", ""),
        ("Acción provisional", ""),
    ):
        if columna not in salida.columns:
            salida[columna] = valor

    for indice, fila in salida.iterrows():
        codigo = str(
            fila.get("Código de catálogo", "")
        ).strip()
        coincide = bool(
            fila.get("Coincide catálogo", False)
        )
        relacion = str(
            fila.get("Relación con la unidad", "")
        ).strip()

        if coincide and codigo:
            grupo = (
                codigo[:-4]
                if codigo.lower().endswith(".pdf")
                else codigo
            )
            salida.at[indice, "Grupo lógico"] = grupo
            salida.at[
                indice,
                "Relación consolidada",
            ] = "Documento con identidad propia"
            salida.at[
                indice,
                "Acción provisional",
            ] = f"Codificación propuesta: {codigo}"
        elif relacion == "indeterminado":
            salida.at[
                indice,
                "Grupo lógico",
            ] = "Revisión necesaria"
            salida.at[
                indice,
                "Relación consolidada",
            ] = "Relación indeterminada"
            salida.at[
                indice,
                "Acción provisional",
            ] = "Revisión manual"
        else:
            salida.at[
                indice,
                "Grupo lógico",
            ] = "Soporte / fuera de catálogo"
            salida.at[
                indice,
                "Relación consolidada",
            ] = "Componente de soporte"
            salida.at[
                indice,
                "Acción provisional",
            ] = "Conservar nombre original"

    return salida


def _rutas_estimacion(
    inventario: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    mapa = construir_mapa_estructural(
        inventario
    )
    estimaciones = obtener_estimaciones_detectadas(
        mapa
    )

    rutas = {}
    if estimaciones.empty:
        return estimaciones, rutas

    for _, fila in estimaciones.iterrows():
        carpeta = str(fila["Ruta carpeta"])
        consecutivo = int(fila["Consecutivo"])
        archivos = obtener_archivos_unidad(
            inventario,
            carpeta,
        )
        pdfs = _pdfs_inventario(archivos)
        for ruta in pdfs[
            "Ruta original"
        ].fillna("").astype(str).tolist():
            rutas[ruta] = {
                "carpeta": carpeta,
                "consecutivo": consecutivo,
            }

    return estimaciones, rutas


def _integrar_duplicados_exactos(
    decisiones_fisicas: pd.DataFrame,
    detalle_duplicados: pd.DataFrame,
) -> pd.DataFrame:
    salida = decisiones_fisicas.copy()

    if salida.empty:
        return salida

    salida["Grupo duplicado exacto"] = ""
    salida["Es duplicado exacto"] = False

    if detalle_duplicados.empty:
        return salida

    mapa = (
        detalle_duplicados[
            ["Ruta original", "Grupo duplicado"]
        ]
        .drop_duplicates(
            subset=["Ruta original"],
            keep="first",
        )
        .set_index("Ruta original")[
            "Grupo duplicado"
        ]
        .to_dict()
    )

    salida[
        "Grupo duplicado exacto"
    ] = (
        salida["Ruta original"]
        .astype(str)
        .map(mapa)
        .fillna("")
    )
    salida[
        "Es duplicado exacto"
    ] = salida[
        "Grupo duplicado exacto"
    ].astype(str).str.strip().ne("")

    return salida


def ejecutar_diagnostico_integral_v2(
    contenido_zip: bytes,
    inventario: pd.DataFrame,
    procedimiento: str,
    modelo_multimodal: str,
    drive_folder_id: str | None,
    usar_cache: bool = True,
    forzar_reanalisis: bool = False,
    on_progress=None,
) -> dict:
    """
    Diagnóstico integral del expediente, sin modificar archivos.

    Fases:
    1. Ficha V2 persistente para todos los PDF.
    2. JEV + reglas sobre todos los documentos lógicos.
    3. Comparación conjunta para cada EST_n detectada.
    4. Consolidación lógico -> archivo físico.
    5. Duplicados exactos globales.
    6. Tabla única de propuesta física.

    Las unidades no EST todavía no reciben una comparación conjunta especial;
    su resultado debe considerarse diagnóstico hasta validar esas familias.
    """
    pdfs = _pdfs_inventario(
        inventario
    )
    if pdfs.empty:
        raise ValueError(
            "El expediente activo no contiene archivos PDF."
        )

    rutas_pdf = pdfs[
        "Ruta original"
    ].fillna("").astype(str).tolist()

    def progreso_ficha(
        posicion,
        total,
        archivo,
    ):
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

    version_recibida = int(
        resumen_jev.get(
            "pipeline_version",
            0,
        )
        or 0
    )
    if (
        version_recibida
        != PIPELINE_VERSION_FICHA_JEV
    ):
        raise RuntimeError(
            "El servicio JEV activo no corresponde a la versión "
            "requerida por el diagnóstico integral. "
            f"Esperada: {PIPELINE_VERSION_FICHA_JEV}; "
            f"recibida: {version_recibida or 'sin versión'}."
        )

    resultados_base = (
        _resultados_base_desde_jev(
            detalle_jev
        )
    )

    estimaciones, rutas_est = (
        _rutas_estimacion(
            inventario
        )
    )
    rutas_est_set = set(
        rutas_est.keys()
    )

    partes = []
    comparaciones_est = []
    resumenes_grupos_est = []
    costo_comparaciones = 0.0
    tiempo_comparaciones = 0.0
    llamadas_comparacion = 0

    if not estimaciones.empty:
        estimaciones_ordenadas = (
            estimaciones.sort_values(
                by=[
                    "Consecutivo",
                    "Ruta carpeta",
                ],
                kind="stable",
            )
        )

        total_est = len(
            estimaciones_ordenadas
        )

        for posicion_est, (
            _,
            fila_est,
        ) in enumerate(
            estimaciones_ordenadas.iterrows(),
            start=1,
        ):
            carpeta = str(
                fila_est["Ruta carpeta"]
            )
            consecutivo = int(
                fila_est["Consecutivo"]
            )

            rutas_unidad = {
                ruta
                for ruta, info
                in rutas_est.items()
                if (
                    info["carpeta"]
                    == carpeta
                    and int(
                        info["consecutivo"]
                    )
                    == consecutivo
                )
            }

            bloque = resultados_base[
                resultados_base[
                    "Ruta original"
                ]
                .astype(str)
                .isin(rutas_unidad)
            ].copy()

            if bloque.empty:
                continue

            if on_progress is not None:
                on_progress(
                    "Comparaciones EST",
                    posicion_est,
                    total_est,
                    f"EST {consecutivo}",
                )

            (
                comparado,
                detalle_comparacion,
                meta_comparacion,
            ) = consolidar_candidatos_unidad(
                contenido_zip=contenido_zip,
                resultados=bloque,
                procedimiento=procedimiento,
                consecutivo=consecutivo,
                modelo_multimodal=modelo_multimodal,
                tipo_unidad="Estimación",
            )

            (
                comparado,
                detalle_comparacion,
                meta_comparacion,
            ) = _estabilizar_roles_intrinsecos_soporte(
                resultados=comparado,
                detalle_comparacion=detalle_comparacion,
                meta_comparacion=meta_comparacion,
            )

            agrupado, resumen_grupos = (
                agrupar_resultados_unidad(
                    resultados=comparado,
                    procedimiento=procedimiento,
                    consecutivo=consecutivo,
                )
            )
            partes.append(agrupado)

            costo_est = float(
                meta_comparacion.get(
                    "Costo comparación (USD)",
                    0.0,
                )
                or 0.0
            )
            tiempo_est = float(
                meta_comparacion.get(
                    "Tiempo comparación (s)",
                    0.0,
                )
                or 0.0
            )
            ejecutada = int(
                meta_comparacion.get(
                    "Estado"
                )
                == "Comparación ejecutada"
            )

            costo_comparaciones += costo_est
            tiempo_comparaciones += tiempo_est
            llamadas_comparacion += ejecutada

            comparaciones_est.append(
                {
                    "Estimación": consecutivo,
                    "Ruta carpeta": carpeta,
                    "PDF directos": len(
                        rutas_unidad
                    ),
                    "Estado": str(
                        meta_comparacion.get(
                            "Estado",
                            "",
                        )
                    ),
                    "Representante": str(
                        meta_comparacion.get(
                            "Representante",
                            "",
                        )
                    ),
                    "Costo comparación (USD)": costo_est,
                    "Tiempo comparación (s)": tiempo_est,
                    "Estabilizaciones soporte": int(
                        meta_comparacion.get(
                            "Estabilizaciones rol soporte",
                            0,
                        )
                        or 0
                    ),
                }
            )

            if not resumen_grupos.empty:
                rg = resumen_grupos.copy()
                rg.insert(
                    0,
                    "Estimación",
                    consecutivo,
                )
                resumenes_grupos_est.append(
                    rg
                )

    no_est = resultados_base[
        ~resultados_base[
            "Ruta original"
        ]
        .astype(str)
        .isin(rutas_est_set)
    ].copy()

    no_est = _agrupar_no_estimacion(
        no_est
    )
    if not no_est.empty:
        partes.append(no_est)

    if partes:
        resultados_logicos = pd.concat(
            partes,
            ignore_index=True,
            sort=False,
        )
    else:
        resultados_logicos = pd.DataFrame()

    orden = {
        (
            str(fila.get(
                "Ruta original",
                "",
            )),
            str(fila.get(
                "ID lógico",
                "",
            )),
        ): posicion
        for posicion, (
            _,
            fila,
        ) in enumerate(
            detalle_jev.iterrows()
        )
    }

    if not resultados_logicos.empty:
        resultados_logicos[
            "_orden_integral"
        ] = [
            orden.get(
                (
                    str(ruta),
                    str(logical_id),
                ),
                10**9,
            )
            for ruta, logical_id in zip(
                resultados_logicos[
                    "Ruta original"
                ],
                resultados_logicos[
                    "ID lógico"
                ],
            )
        ]
        resultados_logicos = (
            resultados_logicos
            .sort_values(
                by="_orden_integral",
                kind="stable",
            )
            .drop(
                columns=[
                    "_orden_integral"
                ]
            )
            .reset_index(drop=True)
        )

    (
        decisiones_fisicas,
        resumen_fisico,
    ) = consolidar_salida_fisica(
        resultados_finales=resultados_logicos,
        documentos=documentos,
        relaciones=relaciones,
    )

    if on_progress is not None:
        on_progress(
            "Duplicados exactos",
            1,
            1,
            "Calculando SHA-256 global",
        )

    (
        detalle_duplicados,
        grupos_duplicados,
        resumen_duplicados,
    ) = analizar_duplicados_exactos(
        contenido_zip=contenido_zip,
        inventario=inventario,
    )

    decisiones_fisicas = (
        _integrar_duplicados_exactos(
            decisiones_fisicas,
            detalle_duplicados,
        )
    )

    no_pdf = inventario[
        ~inventario["Extensión"]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq(".pdf")
    ].copy()

    if not no_pdf.empty:
        no_pdf = no_pdf[
            [
                columna
                for columna in (
                    "Archivo",
                    "Ruta original",
                    "Carpeta",
                    "Extensión",
                    "Tipo",
                    "Tamaño (bytes)",
                )
                if columna
                in no_pdf.columns
            ]
        ].copy()
        no_pdf[
            "Acción diagnóstica"
        ] = (
            "Conservar por ahora; motor de contenido no-PDF "
            "pendiente de esta fase."
        )

    comparaciones_est_df = pd.DataFrame(
        comparaciones_est
    )
    grupos_est_df = (
        pd.concat(
            resumenes_grupos_est,
            ignore_index=True,
            sort=False,
        )
        if resumenes_grupos_est
        else pd.DataFrame()
    )

    costo_fichas = float(
        resumen_ficha.get(
            "costo_total_usd",
            0.0,
        )
    )
    costo_jev = float(
        resumen_jev.get(
            "costo_total_jev_usd",
            0.0,
        )
    )
    costo_total = (
        costo_fichas
        + costo_jev
        + costo_comparaciones
    )

    resumen = {
        "schema_version": (
            DIAGNOSTICO_INTEGRAL_SCHEMA_VERSION
        ),
        "pipeline_jev_version": version_recibida,
        "archivos_totales": len(
            inventario
        ),
        "pdf_totales": len(pdfs),
        "no_pdf_totales": len(
            no_pdf
        ),
        "documentos_logicos": len(
            documentos
        ),
        "estimaciones_detectadas": len(
            estimaciones
        ),
        "fichas_cache_reutilizadas": int(
            resumen_ficha.get(
                "fichas_cache_reutilizadas",
                0,
            )
        ),
        "fichas_multimodales_nuevas": int(
            resumen_ficha.get(
                "llamadas_multimodales",
                0,
            )
        ),
        "llamadas_jev": int(
            resumen_jev.get(
                "llamadas_jev",
                0,
            )
        ),
        "comparaciones_est_multimodales": (
            llamadas_comparacion
        ),
        "archivos_fisicos_consolidados": int(
            resumen_fisico.get(
                "archivos_fisicos",
                0,
            )
        ),
        "salidas_fisicas_propuestas": int(
            resumen_fisico.get(
                "salidas_fisicas",
                0,
            )
        ),
        "archivos_revision": int(
            resumen_fisico.get(
                "archivos_revision",
                0,
            )
        ),
        "archivos_a_dividir": int(
            resumen_fisico.get(
                "archivos_a_dividir",
                0,
            )
        ),
        "grupos_duplicados_exactos": int(
            resumen_duplicados.get(
                "grupos_duplicados_exactos",
                0,
            )
        ),
        "copias_exactas_adicionales": int(
            resumen_duplicados.get(
                "copias_adicionales",
                0,
            )
        ),
        "costo_fichas_usd": costo_fichas,
        "costo_jev_usd": costo_jev,
        "costo_comparaciones_est_usd": (
            costo_comparaciones
        ),
        "costo_api_total_usd": costo_total,
        "tiempo_comparaciones_est_s": (
            tiempo_comparaciones
        ),
        "nota_alcance": (
            "Diagnóstico integral V1: EST_n usa comparación conjunta "
            "validada. Las demás familias documentales todavía no "
            "tienen comparación conjunta específica; no se modifican "
            "archivos."
        ),
    }

    return {
        "resumen": resumen,
        "archivos_ficha": archivos_ficha,
        "documentos": documentos,
        "registros": registros,
        "relaciones": relaciones,
        "marcadores": marcadores,
        "perfiles_crudos": perfiles_crudos,
        "detalle_jev": detalle_jev,
        "resumen_jev": resumen_jev,
        "resultados_logicos": resultados_logicos,
        "decisiones_fisicas": decisiones_fisicas,
        "resumen_fisico": resumen_fisico,
        "estimaciones": estimaciones,
        "comparaciones_est": comparaciones_est_df,
        "grupos_est": grupos_est_df,
        "detalle_duplicados": detalle_duplicados,
        "grupos_duplicados": grupos_duplicados,
        "resumen_duplicados": resumen_duplicados,
        "no_pdf": no_pdf,
    }
