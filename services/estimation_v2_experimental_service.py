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



def _nombre_codigo_pdf(codigo: str) -> str:
    valor = str(codigo).strip()
    if not valor:
        return ""
    return valor if valor.lower().endswith(".pdf") else f"{valor}.pdf"


def _rango_logico(
    mapa_paginas: dict[tuple[str, str], tuple[int, int, bool]],
    ruta: str,
    logical_id: str,
) -> tuple[int, int] | None:
    valor = mapa_paginas.get(
        (str(ruta), str(logical_id))
    )
    if not valor:
        return None

    inicio, fin, valido = valor
    if not valido or inicio <= 0 or fin < inicio:
        return None

    return int(inicio), int(fin)


def consolidar_salida_fisica(
    resultados_finales: pd.DataFrame,
    documentos: pd.DataFrame,
    relaciones: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Convierte decisiones por DOCUMENTO LÓGICO en decisiones por ARCHIVO FÍSICO.

    Reglas:
    - sin código principal -> conservar una sola vez el archivo original;
    - un único documento principal con un único código -> el archivo completo
      adopta ese código, aunque contenga piezas de soporte subordinadas;
    - varios códigos principales distintos -> proponer división por segmentos
      cuando las fronteras son válidas y no se traslapan;
    - varias piezas principales independientes con el mismo código -> revisión,
      para no colapsar silenciosamente instancias distintas.

    Esta etapa no modifica ni divide bytes; solo construye la decisión física.
    """
    if resultados_finales.empty:
        return pd.DataFrame(), {
            "archivos_fisicos": 0,
            "salidas_fisicas": 0,
            "archivos_codificados_completos": 0,
            "archivos_soporte": 0,
            "archivos_a_dividir": 0,
            "archivos_revision": 0,
        }

    mapa_paginas = {}
    if not documentos.empty:
        for _, fila in documentos.iterrows():
            ruta = str(fila.get("Ruta original", ""))
            logical_id = str(fila.get("ID lógico", ""))
            mapa_paginas[(ruta, logical_id)] = (
                int(fila.get("Página inicial", 0) or 0),
                int(fila.get("Página final", 0) or 0),
                bool(fila.get("Rango válido", False)),
            )

    relaciones_por_ruta = {}
    if not relaciones.empty:
        for _, fila in relaciones.iterrows():
            ruta = str(fila.get("Ruta original", ""))
            relaciones_por_ruta.setdefault(
                ruta,
                [],
            ).append(
                {
                    "origen": str(fila.get("Origen", "")),
                    "destino": str(fila.get("Destino", "")),
                    "tipo": str(fila.get("Relación", "")),
                    "valida": bool(fila.get("IDs válidos", True)),
                }
            )

    registros = []
    rutas_revision = set()
    rutas_division = set()
    rutas_codificadas = set()
    rutas_soporte = set()

    for ruta, bloque in resultados_finales.groupby(
        "Ruta original",
        sort=False,
        dropna=False,
    ):
        ruta = str(ruta)
        bloque = bloque.copy()
        archivo = str(
            bloque.iloc[0].get("Archivo", "")
        )

        bloque["_codigo_final"] = (
            bloque["Código de catálogo"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        bloque["_principal"] = (
            bloque["_codigo_final"].ne("")
            & bloque["Coincide catálogo"].fillna(False).astype(bool)
        )

        principales = bloque[
            bloque["_principal"]
        ].copy()

        codigos = []
        for codigo in principales["_codigo_final"].tolist():
            nombre = _nombre_codigo_pdf(codigo)
            if nombre and nombre not in codigos:
                codigos.append(nombre)

        ids_logicos = (
            bloque["ID lógico"]
            .fillna("")
            .astype(str)
            .tolist()
        )
        titulos = (
            bloque["Título detectado"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        if principales.empty:
            hay_revision = bloque[
                "Grupo lógico"
            ].astype(str).isin(
                ["Revisión necesaria", "Error de análisis"]
            ).any()

            if hay_revision:
                decision = "Revisión física necesaria"
                motivo = (
                    "Ningún documento lógico recibió código final y al menos "
                    "una pieza quedó en revisión; se conserva provisionalmente "
                    "el archivo original sin renombrar."
                )
                rutas_revision.add(ruta)
            else:
                decision = "Conservar archivo original"
                motivo = (
                    "Todos los documentos lógicos del PDF son soporte, "
                    "extractos o fuera de catálogo; se conserva una sola "
                    "instancia física con su nombre original."
                )
                rutas_soporte.add(ruta)

            registros.append(
                {
                    "Archivo original": archivo,
                    "Ruta original": ruta,
                    "Salida #": 1,
                    "Documentos lógicos": len(bloque),
                    "IDs lógicos": " | ".join(ids_logicos),
                    "Títulos detectados": " | ".join(titulos),
                    "Códigos principales": "",
                    "Decisión física": decision,
                    "Nombre de salida propuesto": archivo,
                    "Páginas de salida": "Archivo completo",
                    "Requiere división": False,
                    "Requiere revisión": hay_revision,
                    "Motivo": motivo,
                }
            )
            continue

        if len(codigos) == 1:
            # Un mismo código puede aparecer en varias piezas autónomas. No las
            # colapsamos salvo que el motor lógico haya dejado solo una como
            # principal y las demás como soporte.
            if len(principales) > 1:
                registros.append(
                    {
                        "Archivo original": archivo,
                        "Ruta original": ruta,
                        "Salida #": 1,
                        "Documentos lógicos": len(bloque),
                        "IDs lógicos": " | ".join(ids_logicos),
                        "Títulos detectados": " | ".join(titulos),
                        "Códigos principales": codigos[0],
                        "Decisión física": (
                            "Revisión: múltiples documentos principales "
                            "con el mismo código"
                        ),
                        "Nombre de salida propuesto": archivo,
                        "Páginas de salida": "Archivo completo",
                        "Requiere división": False,
                        "Requiere revisión": True,
                        "Motivo": (
                            "Más de un documento lógico autónomo quedó como "
                            "principal para el mismo código. No se renombra ni "
                            "fusiona automáticamente para evitar ocultar "
                            "instancias documentales distintas."
                        ),
                    }
                )
                rutas_revision.add(ruta)
                continue

            codigo = codigos[0]
            principal = principales.iloc[0]
            principal_id = str(
                principal.get("ID lógico", "")
            )

            subordinados = []
            for relacion in relaciones_por_ruta.get(
                ruta,
                [],
            ):
                if (
                    relacion["valida"]
                    and relacion["destino"] == principal_id
                    and relacion["tipo"] in {
                        "soporte_de",
                        "autentica_a",
                        "anexo_de",
                        "complementa_a",
                    }
                ):
                    subordinados.append(
                        relacion["origen"]
                    )

            registros.append(
                {
                    "Archivo original": archivo,
                    "Ruta original": ruta,
                    "Salida #": 1,
                    "Documentos lógicos": len(bloque),
                    "IDs lógicos": " | ".join(ids_logicos),
                    "Títulos detectados": " | ".join(titulos),
                    "Códigos principales": codigo,
                    "Decisión física": "Codificar archivo completo",
                    "Nombre de salida propuesto": codigo,
                    "Páginas de salida": "Archivo completo",
                    "Requiere división": False,
                    "Requiere revisión": False,
                    "Motivo": (
                        f"El documento lógico principal {principal_id} recibe "
                        f"{codigo}. Las demás piezas del mismo PDF no compiten "
                        "con otro código final"
                        + (
                            " y se reconocen como subordinadas: "
                            + ", ".join(subordinados)
                            if subordinados
                            else "."
                        )
                    ),
                }
            )
            rutas_codificadas.add(ruta)
            continue

        # --------------------------------------------------------------
        # Varios códigos principales distintos en el mismo PDF.
        # --------------------------------------------------------------
        principal_por_id = {}
        for _, fila in principales.iterrows():
            logical_id = str(fila.get("ID lógico", ""))
            codigo = _nombre_codigo_pdf(
                fila.get("Código de catálogo", "")
            )
            principal_por_id[logical_id] = codigo

        # Si un mismo código conserva varias piezas principales, la frontera
        # física no es suficientemente inequívoca para automatizar.
        conteo_por_codigo = {}
        for codigo in principal_por_id.values():
            conteo_por_codigo[codigo] = (
                conteo_por_codigo.get(codigo, 0) + 1
            )
        if any(
            cantidad > 1
            for cantidad in conteo_por_codigo.values()
        ):
            registros.append(
                {
                    "Archivo original": archivo,
                    "Ruta original": ruta,
                    "Salida #": 1,
                    "Documentos lógicos": len(bloque),
                    "IDs lógicos": " | ".join(ids_logicos),
                    "Títulos detectados": " | ".join(titulos),
                    "Códigos principales": " | ".join(codigos),
                    "Decisión física": (
                        "Revisión antes de dividir"
                    ),
                    "Nombre de salida propuesto": archivo,
                    "Páginas de salida": "Pendiente",
                    "Requiere división": True,
                    "Requiere revisión": True,
                    "Motivo": (
                        "Hay varios códigos principales y al menos uno aparece "
                        "en más de un documento lógico principal."
                    ),
                }
            )
            rutas_revision.add(ruta)
            rutas_division.add(ruta)
            continue

        paginas_por_codigo = {}
        ids_por_codigo = {}
        faltan_rangos = False

        for logical_id, codigo in principal_por_id.items():
            rango = _rango_logico(
                mapa_paginas,
                ruta,
                logical_id,
            )
            if rango is None:
                faltan_rangos = True
                break
            paginas_por_codigo.setdefault(
                codigo,
                [],
            ).append(rango)
            ids_por_codigo.setdefault(
                codigo,
                [],
            ).append(logical_id)

        # Extiende cada segmento con piezas explícitamente subordinadas.
        if not faltan_rangos:
            for relacion in relaciones_por_ruta.get(
                ruta,
                [],
            ):
                if (
                    not relacion["valida"]
                    or relacion["tipo"]
                    not in {
                        "soporte_de",
                        "autentica_a",
                        "anexo_de",
                        "complementa_a",
                    }
                ):
                    continue

                codigo_destino = principal_por_id.get(
                    relacion["destino"]
                )
                if not codigo_destino:
                    continue

                rango = _rango_logico(
                    mapa_paginas,
                    ruta,
                    relacion["origen"],
                )
                if rango is None:
                    faltan_rangos = True
                    break

                paginas_por_codigo[
                    codigo_destino
                ].append(rango)
                ids_por_codigo[
                    codigo_destino
                ].append(
                    relacion["origen"]
                )

        segmentos = []
        if not faltan_rangos:
            for codigo, rangos in paginas_por_codigo.items():
                inicio = min(
                    rango[0] for rango in rangos
                )
                fin = max(
                    rango[1] for rango in rangos
                )
                segmentos.append(
                    {
                        "codigo": codigo,
                        "inicio": inicio,
                        "fin": fin,
                        "ids": ids_por_codigo[codigo],
                    }
                )

            segmentos.sort(
                key=lambda item: (
                    item["inicio"],
                    item["fin"],
                )
            )

        traslape = False
        if segmentos:
            for anterior, siguiente in zip(
                segmentos,
                segmentos[1:],
            ):
                if siguiente["inicio"] <= anterior["fin"]:
                    traslape = True
                    break

        if faltan_rangos or not segmentos or traslape:
            registros.append(
                {
                    "Archivo original": archivo,
                    "Ruta original": ruta,
                    "Salida #": 1,
                    "Documentos lógicos": len(bloque),
                    "IDs lógicos": " | ".join(ids_logicos),
                    "Títulos detectados": " | ".join(titulos),
                    "Códigos principales": " | ".join(codigos),
                    "Decisión física": "Revisión antes de dividir",
                    "Nombre de salida propuesto": archivo,
                    "Páginas de salida": "Pendiente",
                    "Requiere división": True,
                    "Requiere revisión": True,
                    "Motivo": (
                        "El PDF contiene varios códigos principales, pero las "
                        "fronteras de páginas son incompletas o se traslapan."
                    ),
                }
            )
            rutas_revision.add(ruta)
            rutas_division.add(ruta)
            continue

        rutas_division.add(ruta)
        for numero, segmento in enumerate(
            segmentos,
            start=1,
        ):
            registros.append(
                {
                    "Archivo original": archivo,
                    "Ruta original": ruta,
                    "Salida #": numero,
                    "Documentos lógicos": len(bloque),
                    "IDs lógicos": " | ".join(
                        segmento["ids"]
                    ),
                    "Títulos detectados": " | ".join(titulos),
                    "Códigos principales": segmento["codigo"],
                    "Decisión física": (
                        "Dividir PDF y codificar segmento"
                    ),
                    "Nombre de salida propuesto": (
                        segmento["codigo"]
                    ),
                    "Páginas de salida": (
                        f"{segmento['inicio']}-"
                        f"{segmento['fin']}"
                    ),
                    "Requiere división": True,
                    "Requiere revisión": False,
                    "Motivo": (
                        "El PDF contiene varios documentos principales con "
                        "códigos distintos y rangos no traslapados. Las piezas "
                        "subordinadas relacionadas se incorporan al segmento "
                        "de su documento principal."
                    ),
                }
            )

    decisiones = pd.DataFrame(registros)

    resumen = {
        "archivos_fisicos": int(
            resultados_finales[
                "Ruta original"
            ].astype(str).nunique()
        ),
        "salidas_fisicas": len(decisiones),
        "archivos_codificados_completos": len(
            rutas_codificadas
        ),
        "archivos_soporte": len(rutas_soporte),
        "archivos_a_dividir": len(rutas_division),
        "archivos_revision": len(rutas_revision),
    }

    return decisiones, resumen


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

    decisiones_fisicas, resumen_fisico = (
        consolidar_salida_fisica(
            resultados_finales=resultados_finales,
            documentos=documentos,
            relaciones=relaciones,
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
        "archivos_a_dividir": int(
            resumen_fisico.get(
                "archivos_a_dividir",
                0,
            )
        ),
        "archivos_revision_fisica": int(
            resumen_fisico.get(
                "archivos_revision",
                0,
            )
        ),
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
        "decisiones_fisicas": decisiones_fisicas,
        "resumen_fisico": resumen_fisico,
        "resumen_pipeline": resumen_pipeline,
    }
