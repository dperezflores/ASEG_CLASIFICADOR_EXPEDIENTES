from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import fitz
import pandas as pd

from services.catalog_family_service import (
    construir_catalogo_operativo_estimacion,
)
from services.evaluation_service import extraer_paginas_pdf_del_zip
from services.jev_classifier_service import (
    jev_configurado,
    validar_equivalencia_identidad_con_jev,
)
from services.openai_multimodal_service import (
    clasificar_pdf_multimodal,
    comparar_candidatos_unidad_multimodal,
    identificar_documento_multimodal,
    validar_equivalencia_documental_multimodal,
    validar_integridad_documental_multimodal,
)


PAGINAS_INICIALES = 2
MAX_INTENTOS_ANALISIS = 2
MAX_PAGINAS_INTEGRIDAD_COMPLETA = 10
USAR_IDENTIDAD_PURA_SELECTIVA = True
CODIGOS_IDENTIDAD_PURA_SELECTIVA = frozenset(
    {
        "ETR_LSI_ETR",
        "ETR_LSI_FIN",
        "ETR_LSI_GVO",
        "CNT_LSI_PTC",
    }
)

# Primera versión controlada: solo estos códigos se consideran de
# representación única para resolver conflictos por confianza.
CODIGOS_REPRESENTACION_UNICA_CONFIANZA = frozenset(
    {
        "ETR_LSI_ETR",
        "ETR_LSI_FIN",
        "ETR_LSI_GVO",
        "CNT_LSI_PTC",
    }
)
CONFIANZA_MIN_GANADOR_CONFLICTO = 90.0
DIFERENCIA_MIN_CONFLICTO = 10.0


def diagnosticar_componente_pdf(
    contenido_zip: bytes,
    ruta_pdf: str,
) -> dict:
    """
    Diagnóstico técnico previo. No usa IA ni toma decisiones documentales.
    """
    with ZipFile(BytesIO(contenido_zip)) as archivo_zip:
        pdf_bytes = archivo_zip.read(ruta_pdf)

    documento = fitz.open(stream=pdf_bytes, filetype="pdf")
    textos = []
    paginas = []

    try:
        limite = min(documento.page_count, PAGINAS_INICIALES)
        for indice in range(limite):
            pagina = documento.load_page(indice)
            texto = pagina.get_text("text").strip()
            if texto:
                textos.append(texto)
            paginas.append(indice + 1)

        total_paginas = documento.page_count
    finally:
        documento.close()

    texto = "\n\n".join(textos).strip()
    caracteres = len("".join(texto.split()))

    return {
        "Páginas totales": total_paginas,
        "Páginas evaluadas": paginas,
        "Caracteres útiles": caracteres,
        "Ruta sugerida": "Multimodal en prototipo",
        "Motivo": (
            "En esta fase se prioriza precisión. La clasificación inicial "
            "se hace por contenido y las validaciones secundarias solo se "
            "ejecutan cuando el resultado realmente lo requiere."
        ),
        "Texto nativo": texto,
    }


def _codigo_sin_extension(codigo: str) -> str:
    valor = str(codigo).strip()
    if valor.lower().endswith(".pdf"):
        return valor[:-4]
    return valor


def _codigo_unidad(
    procedimiento: str,
    consecutivo: int,
) -> str:
    return f"EJE_{procedimiento.upper()}_EST_{int(consecutivo)}"


def _paginas_contexto(total_paginas: int) -> list[int]:
    """
    Usa pocas páginas, pero incorpora el final cuando el documento es largo.
    """
    total = int(total_paginas)

    if total <= 0:
        return []
    if total <= 4:
        return list(range(1, total + 1))

    return [1, 2, total - 1, total]


def _paginas_integridad(total_paginas: int) -> list[int]:
    """
    Para decidir completitud conviene observar el documento completo cuando
    es corto. En documentos largos se conserva una muestra de inicio y cierre.
    """
    total = int(total_paginas)

    if total <= 0:
        return []
    if total <= MAX_PAGINAS_INTEGRIDAD_COMPLETA:
        return list(range(1, total + 1))

    centro = max(3, (total + 1) // 2)
    paginas = [1, 2, centro, total - 1, total]
    return sorted(set(p for p in paginas if 1 <= p <= total))


def _extraer_pdf_paginas(
    contenido_zip: bytes,
    ruta_pdf: str,
    paginas: list[int],
) -> bytes:
    return extraer_paginas_pdf_del_zip(
        contenido_zip,
        ruta_pdf,
        paginas,
    )


def analizar_componente_unidad(
    contenido_zip: bytes,
    ruta_pdf: str,
    catalogo: pd.DataFrame,
    procedimiento: str,
    modelo_multimodal: str,
    tipo_unidad: str = "Estimación",
    consecutivo: int = 1,
) -> dict:
    """
    Flujo documental controlado.

    ETAPA 1
    Identifica el documento y propone un concepto candidato del catálogo,
    sin contexto de carpeta ni nombre real.

    ETAPA 2A
    Si el candidato tiene un código propio distinto al de la unidad, valida
    de forma independiente la equivalencia documental y funcional.

    ETAPA 2B
    Solo si la equivalencia fue confirmada, valida si el documento es completo
    o un parcial/extracto.

    Los candidatos al código de la propia unidad se reservan para la
    comparación conjunta posterior.
    """
    diagnostico = diagnosticar_componente_pdf(
        contenido_zip,
        ruta_pdf,
    )

    catalogo_operativo, _ = construir_catalogo_operativo_estimacion(
        catalogo=catalogo,
        procedimiento=procedimiento,
        consecutivo=int(consecutivo),
    )

    paginas_iniciales = list(
        range(
            1,
            min(
                int(diagnostico["Páginas totales"]),
                PAGINAS_INICIALES,
            )
            + 1,
        )
    )

    pdf_inicial = _extraer_pdf_paginas(
        contenido_zip,
        ruta_pdf,
        paginas_iniciales,
    )

    etapa1 = clasificar_pdf_multimodal(
        documento_alias="componente_unidad",
        pdf_bytes=pdf_inicial,
        catalogo=catalogo_operativo,
        modelo=modelo_multimodal,
    )

    titulo = str(etapa1["Título detectado"])
    concepto_inicial = str(etapa1["Resultado Ruta B"])
    codigo_inicial = str(etapa1["Código Ruta B"])
    coincide_inicial = bool(etapa1["Coincide catálogo"])
    confianza_inicial = float(etapa1["Confianza B"])
    evidencia_inicial = str(etapa1["Evidencia B"])

    costo = float(etapa1["Costo B (USD)"])
    tiempo = float(etapa1["Tiempo B (s)"])

    codigo_unidad = _codigo_unidad(
        procedimiento,
        consecutivo,
    )

    alcance = "no_evaluado"
    equivalencia = "no_evaluada"
    confianza_equivalencia = 0.0
    evidencia_equivalencia = ""
    verificacion_identidad_pura = "No aplicada"
    titulo_identidad_pura = ""
    funcion_identidad_pura = ""
    acto_identidad_pura = ""
    confianza_identidad_pura = 0.0
    evidencia_identidad_pura = ""
    equivalencia_resuelta_por = "No requerida"
    fallback_multimodal_jev = "No"
    equivalencia_jev = "no_evaluada"
    decision_original_jev = ""
    confianza_jev = 0.0
    probabilidad_jev = 0.0
    margen_jev = None
    control_jev = ""
    error_jev = ""
    llamadas_multimodales = 1
    llamadas_jev = 0
    relacion = "soporte"
    validacion_secundaria = "No requerida"
    evidencia_secundaria = ""
    confianza_final = confianza_inicial

    concepto_relacionado = (
        concepto_inicial if coincide_inicial else ""
    )
    codigo_relacionado = (
        codigo_inicial if coincide_inicial else ""
    )

    coincide_final = False
    concepto_final = ""
    codigo_final = ""

    if not coincide_inicial:
        rol = "Soporte / fuera de catálogo"
        relacion = "soporte"

    elif (
        _codigo_sin_extension(codigo_inicial).upper()
        == codigo_unidad.upper()
    ):
        relacion = "candidato_unidad"
        validacion_secundaria = "Pendiente comparación conjunta"
        rol = "Candidato a comparación de unidad"

    else:
        paginas_contexto = _paginas_integridad(
            diagnostico["Páginas totales"]
        )
        pdf_contexto = _extraer_pdf_paginas(
            contenido_zip,
            ruta_pdf,
            paginas_contexto,
        )

        codigo_normalizado = _codigo_sin_extension(
            codigo_inicial
        ).upper()
        identidad_para_equivalencia = titulo

        if (
            USAR_IDENTIDAD_PURA_SELECTIVA
            and codigo_normalizado
            in CODIGOS_IDENTIDAD_PURA_SELECTIVA
        ):
            identidad_pura = identificar_documento_multimodal(
                documento_alias="verificacion_identidad",
                pdf_bytes=pdf_contexto,
                modelo=modelo_multimodal,
            )

            verificacion_identidad_pura = "Aplicada"
            titulo_identidad_pura = str(
                identidad_pura["Título identidad pura"]
            )
            funcion_identidad_pura = str(
                identidad_pura["Función identidad pura"]
            )
            acto_identidad_pura = str(
                identidad_pura["Acto identidad pura"]
            )
            confianza_identidad_pura = float(
                identidad_pura[
                    "Confianza identidad pura (%)"
                ]
            )
            evidencia_identidad_pura = str(
                identidad_pura[
                    "Evidencia identidad pura"
                ]
            )

            costo += float(
                identidad_pura[
                    "Costo identidad pura (USD)"
                ]
            )
            tiempo += float(
                identidad_pura[
                    "Tiempo identidad pura (s)"
                ]
            )
            llamadas_multimodales += 1
            confianza_final = min(
                confianza_final,
                confianza_identidad_pura,
            )

            identidad_para_equivalencia = (
                f"Identidad documental: {titulo_identidad_pura}. "
                f"Función formal: {funcion_identidad_pura}. "
                f"Acto documentado: {acto_identidad_pura}."
            )

        usar_jev = (
            verificacion_identidad_pura == "Aplicada"
            and codigo_normalizado
            in CODIGOS_IDENTIDAD_PURA_SELECTIVA
        )

        resolver_con_multimodal = not usar_jev

        if usar_jev:
            if jev_configurado():
                try:
                    llamadas_jev += 1
                    etapa_jev = validar_equivalencia_identidad_con_jev(
                        documento_alias="equivalencia_selectiva",
                        identidad_documental=titulo_identidad_pura,
                        funcion_formal=funcion_identidad_pura,
                        acto_documentado=acto_identidad_pura,
                        concepto_objetivo=concepto_inicial,
                    )

                    equivalencia_jev = str(
                        etapa_jev["Equivalencia JEV"]
                    )
                    decision_original_jev = str(
                        etapa_jev["Decisión original JEV"]
                    )
                    confianza_jev = float(
                        etapa_jev["Confianza JEV (%)"]
                    )
                    probabilidad_jev = float(
                        etapa_jev[
                            "Probabilidad elegida JEV (%)"
                        ]
                    )
                    margen_jev = etapa_jev["Margen JEV (%)"]
                    control_jev = str(
                        etapa_jev["Control JEV"]
                    )

                    costo += float(
                        etapa_jev["Costo JEV (USD)"]
                    )
                    tiempo += float(
                        etapa_jev["Tiempo JEV (s)"]
                    )

                    if equivalencia_jev in (
                        "equivalente",
                        "no_equivalente",
                    ):
                        equivalencia = equivalencia_jev
                        confianza_equivalencia = max(
                            confianza_jev,
                            probabilidad_jev,
                        )
                        evidencia_equivalencia = (
                            "JEV comparó exclusivamente la identidad "
                            "documental pura contra el concepto candidato. "
                            f"Decisión: {equivalencia_jev}."
                        )
                        if control_jev:
                            evidencia_equivalencia += (
                                " Control: " + control_jev
                            )
                        equivalencia_resuelta_por = "JEV"
                        resolver_con_multimodal = False
                    else:
                        fallback_multimodal_jev = (
                            "Sí - JEV indeterminado"
                        )
                        resolver_con_multimodal = True

                except Exception as error:
                    error_jev = str(error)
                    fallback_multimodal_jev = (
                        "Sí - error JEV"
                    )
                    resolver_con_multimodal = True
            else:
                fallback_multimodal_jev = (
                    "Sí - JEV no configurado"
                )
                resolver_con_multimodal = True

        if resolver_con_multimodal:
            etapa_equivalencia = validar_equivalencia_documental_multimodal(
                documento_alias="componente_unidad",
                pdf_bytes=pdf_contexto,
                modelo=modelo_multimodal,
                concepto_objetivo=concepto_inicial,
                identidad_detectada=identidad_para_equivalencia,
            )
            llamadas_multimodales += 1

            equivalencia = str(
                etapa_equivalencia["Equivalencia funcional"]
            )
            confianza_equivalencia = float(
                etapa_equivalencia["Confianza equivalencia (%)"]
            )
            evidencia_equivalencia = str(
                etapa_equivalencia["Evidencia equivalencia"]
            )
            confianza_final = min(
                confianza_final,
                confianza_equivalencia,
            )
            costo += float(
                etapa_equivalencia["Costo equivalencia (USD)"]
            )
            tiempo += float(
                etapa_equivalencia["Tiempo equivalencia (s)"]
            )
            equivalencia_resuelta_por = "Multimodal"
        else:
            confianza_final = min(
                confianza_final,
                confianza_equivalencia,
            )

        if equivalencia == "equivalente":
            etapa_integridad = validar_integridad_documental_multimodal(
                documento_alias="componente_unidad",
                pdf_bytes=pdf_contexto,
                modelo=modelo_multimodal,
                concepto_objetivo=concepto_inicial,
                identidad_detectada=(
                    identidad_para_equivalencia
                    if verificacion_identidad_pura == "Aplicada"
                    else titulo
                ),
            )

            alcance = str(
                etapa_integridad["Alcance documental"]
            )
            confianza_alcance = float(
                etapa_integridad["Confianza alcance (%)"]
            )
            confianza_final = min(
                confianza_final,
                confianza_alcance,
            )
            evidencia_secundaria = str(
                etapa_integridad["Evidencia alcance"]
            )
            costo += float(
                etapa_integridad["Costo alcance (USD)"]
            )
            tiempo += float(
                etapa_integridad["Tiempo alcance (s)"]
            )
            llamadas_multimodales += 1
            validacion_secundaria = (
                "Equivalencia documental-funcional + integridad"
            )
            relacion = "documento_independiente"

            if alcance == "completo":
                coincide_final = True
                concepto_final = concepto_inicial
                codigo_final = codigo_inicial
                rol = "Posible documento con código propio"
            elif alcance == "parcial_extracto":
                rol = "Soporte / extracto de otro documento"
            else:
                relacion = "indeterminado"
                rol = "Revisión necesaria"

        elif equivalencia == "no_equivalente":
            validacion_secundaria = (
                "Equivalencia documental-funcional"
            )
            relacion = "soporte"
            rol = "Soporte / no equivalente al concepto candidato"

        else:
            validacion_secundaria = (
                "Equivalencia documental-funcional"
            )
            relacion = "indeterminado"
            rol = "Revisión necesaria"

    evidencias = [f"Etapa 1: {evidencia_inicial}"]
    if evidencia_identidad_pura:
        evidencias.append(
            "Identidad pura selectiva: "
            + evidencia_identidad_pura
        )
    if equivalencia_jev != "no_evaluada":
        evidencias.append(
            "JEV equivalencia: "
            + equivalencia_jev
            + (
                f" (decisión original: {decision_original_jev})."
                if decision_original_jev
                else "."
            )
        )
    if error_jev:
        evidencias.append(
            "JEV error/fallback: " + error_jev
        )
    if evidencia_equivalencia:
        evidencias.append(
            "Equivalencia: " + evidencia_equivalencia
        )
    if evidencia_secundaria:
        evidencias.append(
            "Integridad: " + evidencia_secundaria
        )
    evidencia = " ".join(evidencias)

    if equivalencia != "no_evaluada":
        paginas_reporte = _paginas_integridad(
            diagnostico["Páginas totales"]
        )
    else:
        paginas_reporte = _paginas_contexto(
            diagnostico["Páginas totales"]
        )

    paginas_usadas = ", ".join(
        str(p) for p in paginas_reporte
    )

    return {
        "Ruta utilizada": (
            "Multimodal + JEV selectivo con fallback controlado"
        ),
        "Motivo de ruta": (
            "El flujo validado se conserva. Para códigos sensibles, la "
            "identidad pura se compara por texto con JEV; solo si JEV queda "
            "indeterminado, no está configurado o falla, se usa equivalencia "
            "multimodal como respaldo."
        ),
        "Título detectado": titulo,
        "Clasificación inicial": concepto_inicial,
        "Código inicial": codigo_inicial,
        "Verificación identidad pura": verificacion_identidad_pura,
        "Título identidad pura": titulo_identidad_pura,
        "Función identidad pura": funcion_identidad_pura,
        "Acto identidad pura": acto_identidad_pura,
        "Confianza identidad pura (%)": confianza_identidad_pura,
        "Evidencia identidad pura": evidencia_identidad_pura,
        "Equivalencia resuelta por": equivalencia_resuelta_por,
        "Equivalencia JEV": equivalencia_jev,
        "Decisión original JEV": decision_original_jev,
        "Confianza JEV (%)": confianza_jev,
        "Probabilidad elegida JEV (%)": probabilidad_jev,
        "Margen JEV (%)": margen_jev,
        "Control JEV": control_jev,
        "Fallback multimodal JEV": fallback_multimodal_jev,
        "Error JEV": error_jev,
        "Llamadas multimodales documento": llamadas_multimodales,
        "Llamadas JEV documento": llamadas_jev,
        "Equivalencia funcional": equivalencia,
        "Confianza equivalencia (%)": confianza_equivalencia,
        "Evidencia equivalencia": evidencia_equivalencia,
        "Validación secundaria": validacion_secundaria,
        "Alcance documental": alcance,
        "Relación con la unidad": relacion,
        "Concepto relacionado": concepto_relacionado,
        "Código relacionado": codigo_relacionado,
        "Coincide catálogo": coincide_final,
        "Concepto propuesto": concepto_final,
        "Código de catálogo": codigo_final,
        "Confianza (%)": confianza_final,
        "Rol propuesto en la unidad": rol,
        "Evidencia": evidencia,
        "Modelo": str(etapa1["Modelo B"]),
        "Costo (USD)": costo,
        "Tiempo (s)": tiempo,
        "Páginas usadas": paginas_usadas,
    }


def analizar_unidad_completa(
    contenido_zip: bytes,
    archivos_pdf: pd.DataFrame,
    catalogo: pd.DataFrame,
    procedimiento: str,
    modelo_multimodal: str,
    tipo_unidad: str,
    consecutivo: int,
    on_progress=None,
) -> pd.DataFrame:
    """
    Analiza todos los PDF directos de una sola unidad documental.

    Si un archivo falla por una incidencia transitoria, se reintenta una vez
    antes de marcarlo como error definitivo.
    """
    resultados = []
    total = len(archivos_pdf)

    for posicion, (_, fila) in enumerate(
        archivos_pdf.reset_index(drop=True).iterrows(),
        start=1,
    ):
        archivo = str(fila["Archivo"])
        ruta_pdf = str(fila["Ruta original"])
        resultado = None
        errores_intentos = []

        for intento in range(1, MAX_INTENTOS_ANALISIS + 1):
            try:
                resultado = analizar_componente_unidad(
                    contenido_zip=contenido_zip,
                    ruta_pdf=ruta_pdf,
                    catalogo=catalogo,
                    procedimiento=procedimiento,
                    modelo_multimodal=modelo_multimodal,
                    tipo_unidad=tipo_unidad,
                    consecutivo=consecutivo,
                )
                break
            except Exception as error:
                errores_intentos.append(str(error))

        if resultado is not None:
            registro = {
                "Archivo": archivo,
                "Ruta original": ruta_pdf,
                **resultado,
                "Intentos de análisis": len(errores_intentos) + 1,
                "Reintento aplicado": bool(errores_intentos),
                "Error": "",
            }
        else:
            detalle_error = " | ".join(
                f"Intento {i + 1}: {mensaje}"
                for i, mensaje in enumerate(errores_intentos)
            )
            registro = {
                "Archivo": archivo,
                "Ruta original": ruta_pdf,
                "Ruta utilizada": "Error",
                "Motivo de ruta": "",
                "Título detectado": "",
                "Clasificación inicial": "",
                "Código inicial": "",
                "Verificación identidad pura": "No aplicada",
                "Título identidad pura": "",
                "Función identidad pura": "",
                "Acto identidad pura": "",
                "Confianza identidad pura (%)": 0.0,
                "Evidencia identidad pura": "",
                "Equivalencia resuelta por": "No requerida",
                "Equivalencia JEV": "no_evaluada",
                "Decisión original JEV": "",
                "Confianza JEV (%)": 0.0,
                "Probabilidad elegida JEV (%)": 0.0,
                "Margen JEV (%)": None,
                "Control JEV": "",
                "Fallback multimodal JEV": "No",
                "Error JEV": "",
                "Llamadas multimodales documento": 0,
                "Llamadas JEV documento": 0,
                "Equivalencia funcional": "indeterminado",
                "Confianza equivalencia (%)": 0.0,
                "Evidencia equivalencia": "",
                "Validación secundaria": "",
                "Alcance documental": "indeterminado",
                "Relación con la unidad": "indeterminado",
                "Concepto relacionado": "",
                "Código relacionado": "",
                "Coincide catálogo": False,
                "Concepto propuesto": "",
                "Código de catálogo": "",
                "Confianza (%)": 0.0,
                "Rol propuesto en la unidad": "No analizado",
                "Evidencia": "",
                "Modelo": "",
                "Costo (USD)": 0.0,
                "Tiempo (s)": 0.0,
                "Páginas usadas": "",
                "Intentos de análisis": MAX_INTENTOS_ANALISIS,
                "Reintento aplicado": MAX_INTENTOS_ANALISIS > 1,
                "Error": detalle_error,
            }

        resultados.append(registro)

        if on_progress is not None:
            on_progress(posicion, total, archivo)

    return pd.DataFrame(resultados)



def consolidar_candidatos_unidad(
    contenido_zip: bytes,
    resultados: pd.DataFrame,
    procedimiento: str,
    consecutivo: int,
    modelo_multimodal: str,
    tipo_unidad: str = "Estimación",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Tercera etapa: compara simultáneamente todos los archivos cuya
    clasificación inicial apunta al código de la propia unidad.

    No toca documentos con código propio distinto ni soportes ya descartados.
    """
    if resultados.empty:
        return resultados.copy(), pd.DataFrame(), {}

    salida = resultados.copy()

    for columna, valor in (
        ("Relación comparativa", ""),
        ("Confianza comparativa (%)", 0.0),
        ("Evidencia comparativa", ""),
    ):
        if columna not in salida.columns:
            salida[columna] = valor

    codigo_unidad = _codigo_unidad(
        procedimiento,
        consecutivo,
    )

    mascara = (
        salida["Código inicial"]
        .astype(str)
        .apply(_codigo_sin_extension)
        .str.upper()
        == codigo_unidad.upper()
    ) & (
        salida["Error"].astype(str).str.strip() == ""
    )

    candidatos_df = salida[mascara].copy()

    if candidatos_df.empty:
        return salida, pd.DataFrame(), {
            "Estado": "Sin candidatos a comparación",
            "Representante": "",
            "Evidencia unidad": "",
            "Costo comparación (USD)": 0.0,
            "Tiempo comparación (s)": 0.0,
            "Modelo comparación": "",
        }

    candidatos = []
    alias_a_indice = {}
    alias_a_archivo = {}

    for numero, (indice, fila) in enumerate(
        candidatos_df.iterrows(),
        start=1,
    ):
        alias = f"candidate_{numero:02d}"
        ruta_pdf = str(fila["Ruta original"])

        diagnostico = diagnosticar_componente_pdf(
            contenido_zip,
            ruta_pdf,
        )
        paginas = _paginas_contexto(
            diagnostico["Páginas totales"]
        )
        pdf_reducido = _extraer_pdf_paginas(
            contenido_zip,
            ruta_pdf,
            paginas,
        )

        candidatos.append(
            {
                "alias": alias,
                "pdf_bytes": pdf_reducido,
                "titulo": str(fila["Título detectado"]),
                "clasificacion": str(
                    fila["Clasificación inicial"]
                ),
                "evidencia": str(fila["Evidencia"]),
            }
        )
        alias_a_indice[alias] = indice
        alias_a_archivo[alias] = str(fila["Archivo"])

    comparacion = comparar_candidatos_unidad_multimodal(
        candidatos=candidatos,
        modelo=modelo_multimodal,
        tipo_unidad=tipo_unidad,
        consecutivo=int(consecutivo),
    )

    filas_comparacion = []

    for decision in comparacion["Decisiones"]:
        alias = str(decision["Alias"])
        indice = alias_a_indice[alias]
        relacion = str(decision["Relación comparativa"])
        confianza = float(
            decision["Confianza comparativa (%)"]
        )
        evidencia = str(decision["Evidencia comparativa"])

        salida.at[indice, "Relación comparativa"] = relacion
        salida.at[
            indice,
            "Confianza comparativa (%)",
        ] = confianza
        salida.at[
            indice,
            "Evidencia comparativa",
        ] = evidencia
        salida.at[
            indice,
            "Relación con la unidad",
        ] = relacion
        salida.at[
            indice,
            "Validación secundaria",
        ] = "Comparación conjunta de unidad"
        salida.at[
            indice,
            "Confianza (%)",
        ] = min(
            float(salida.at[indice, "Confianza (%)"]),
            confianza,
        )

        if relacion == "representante_unidad":
            salida.at[indice, "Coincide catálogo"] = True
            salida.at[
                indice,
                "Concepto propuesto",
            ] = str(
                salida.at[indice, "Clasificación inicial"]
            )
            salida.at[
                indice,
                "Código de catálogo",
            ] = str(
                salida.at[indice, "Código inicial"]
            )
            salida.at[
                indice,
                "Rol propuesto en la unidad",
            ] = "Representante de la unidad"

        elif relacion == "componente_unidad":
            salida.at[indice, "Coincide catálogo"] = False
            salida.at[indice, "Concepto propuesto"] = ""
            salida.at[indice, "Código de catálogo"] = ""
            salida.at[
                indice,
                "Rol propuesto en la unidad",
            ] = "Componente de la misma unidad"

        elif relacion == "soporte":
            salida.at[indice, "Coincide catálogo"] = False
            salida.at[indice, "Concepto propuesto"] = ""
            salida.at[indice, "Código de catálogo"] = ""
            salida.at[
                indice,
                "Rol propuesto en la unidad",
            ] = "Soporte / fuera de catálogo"

        else:
            salida.at[indice, "Coincide catálogo"] = False
            salida.at[indice, "Concepto propuesto"] = ""
            salida.at[indice, "Código de catálogo"] = ""
            salida.at[
                indice,
                "Rol propuesto en la unidad",
            ] = "Revisión necesaria"

        filas_comparacion.append(
            {
                "Alias": alias,
                "Archivo": alias_a_archivo[alias],
                "Título detectado": str(
                    salida.at[indice, "Título detectado"]
                ),
                "Clasificación inicial": str(
                    salida.at[indice, "Clasificación inicial"]
                ),
                "Relación comparativa": relacion,
                "Confianza comparativa (%)": confianza,
                "Evidencia comparativa": evidencia,
            }
        )

    representante_alias = str(comparacion["Representante"])
    representante_archivo = (
        ""
        if representante_alias == "ninguno"
        else alias_a_archivo.get(representante_alias, "")
    )

    meta = {
        "Estado": "Comparación ejecutada",
        "Candidatos comparados": len(candidatos),
        "Representante": representante_archivo,
        "Evidencia unidad": str(
            comparacion["Evidencia unidad"]
        ),
        "Costo comparación (USD)": float(
            comparacion["Costo comparación (USD)"]
        ),
        "Tiempo comparación (s)": float(
            comparacion["Tiempo comparación (s)"]
        ),
        "Modelo comparación": str(
            comparacion["Modelo comparación"]
        ),
    }

    return salida, pd.DataFrame(filas_comparacion), meta



def _resolver_conflictos_codigo_por_confianza(
    resultados: pd.DataFrame,
    codigo_unidad: str,
) -> pd.DataFrame:
    """
    Resuelve de forma determinística conflictos entre varios documentos que
    terminaron proponiendo el mismo código propio de representación única.

    No hace nuevas llamadas de IA. Usa la confianza final consolidada ya
    calculada por el flujo. Solo selecciona un ganador cuando:
    - el mejor candidato alcanza una confianza mínima; y
    - supera al segundo candidato por una diferencia mínima.

    Si el conflicto no es concluyente, ningún documento recibe el código hasta
    revisión manual. La clasificación inicial se conserva para trazabilidad.
    """
    salida = resultados.copy()

    columnas = {
        "Resolución conflicto código": "No aplica",
        "Código en conflicto": "",
        "Confianza conflicto (%)": 0.0,
        "Diferencia confianza conflicto (%)": 0.0,
        "Ganador conflicto": "",
        "Motivo conflicto": "",
    }
    for columna, valor in columnas.items():
        if columna not in salida.columns:
            salida[columna] = valor

    if salida.empty:
        return salida

    codigo_unidad_normalizado = _codigo_sin_extension(
        codigo_unidad
    ).upper()

    candidatos = salida[
        salida["Error"].astype(str).str.strip().eq("")
        & salida["Coincide catálogo"].astype(bool)
        & salida["Código de catálogo"].astype(str).str.strip().ne("")
    ].copy()

    if candidatos.empty:
        return salida

    candidatos["_codigo_conflicto"] = (
        candidatos["Código de catálogo"]
        .astype(str)
        .apply(_codigo_sin_extension)
        .str.upper()
    )

    candidatos = candidatos[
        candidatos["_codigo_conflicto"].isin(
            CODIGOS_REPRESENTACION_UNICA_CONFIANZA
        )
        & candidatos["_codigo_conflicto"].ne(
            codigo_unidad_normalizado
        )
    ]

    if candidatos.empty:
        return salida

    for codigo, bloque in candidatos.groupby(
        "_codigo_conflicto",
        sort=False,
    ):
        if len(bloque) <= 1:
            continue

        puntajes = pd.to_numeric(
            bloque["Confianza (%)"],
            errors="coerce",
        ).fillna(0.0)

        orden = puntajes.sort_values(
            ascending=False,
            kind="stable",
        )
        indice_ganador = orden.index[0]
        confianza_ganador = float(orden.iloc[0])
        confianza_segundo = float(orden.iloc[1])
        diferencia = confianza_ganador - confianza_segundo
        archivo_ganador = str(
            salida.at[indice_ganador, "Archivo"]
        )

        conflicto_concluyente = (
            confianza_ganador
            >= CONFIANZA_MIN_GANADOR_CONFLICTO
            and diferencia
            >= DIFERENCIA_MIN_CONFLICTO
        )

        for indice in bloque.index:
            confianza_actual = float(
                pd.to_numeric(
                    pd.Series(
                        [salida.at[indice, "Confianza (%)"]]
                    ),
                    errors="coerce",
                ).fillna(0.0).iloc[0]
            )
            salida.at[
                indice,
                "Código en conflicto",
            ] = codigo
            salida.at[
                indice,
                "Confianza conflicto (%)",
            ] = confianza_actual
            salida.at[
                indice,
                "Diferencia confianza conflicto (%)",
            ] = round(diferencia, 2)
            salida.at[
                indice,
                "Ganador conflicto",
            ] = archivo_ganador

            if conflicto_concluyente:
                salida.at[
                    indice,
                    "Motivo conflicto",
                ] = (
                    f"Confianza máxima {confianza_ganador:.1f}% "
                    f"vs. segundo candidato {confianza_segundo:.1f}% "
                    f"(diferencia {diferencia:.1f} pp)."
                )

                if indice == indice_ganador:
                    salida.at[
                        indice,
                        "Resolución conflicto código",
                    ] = "Ganador por confianza"
                else:
                    salida.at[
                        indice,
                        "Resolución conflicto código",
                    ] = "Desplazado por mayor confianza"
                    salida.at[
                        indice,
                        "Coincide catálogo",
                    ] = False
                    salida.at[
                        indice,
                        "Concepto propuesto",
                    ] = ""
                    salida.at[
                        indice,
                        "Código de catálogo",
                    ] = ""
                    salida.at[
                        indice,
                        "Relación con la unidad",
                    ] = "soporte"
                    salida.at[
                        indice,
                        "Rol propuesto en la unidad",
                    ] = (
                        "Soporte / candidato desplazado "
                        "por conflicto de código"
                    )
            else:
                salida.at[
                    indice,
                    "Resolución conflicto código",
                ] = "Revisión manual - conflicto no concluyente"
                salida.at[
                    indice,
                    "Motivo conflicto",
                ] = (
                    f"Mejor confianza {confianza_ganador:.1f}%, "
                    f"segunda {confianza_segundo:.1f}% "
                    f"(diferencia {diferencia:.1f} pp). "
                    f"Se exige >= {CONFIANZA_MIN_GANADOR_CONFLICTO:.0f}% "
                    "para el ganador y >= "
                    f"{DIFERENCIA_MIN_CONFLICTO:.0f} pp de diferencia."
                )
                salida.at[
                    indice,
                    "Coincide catálogo",
                ] = False
                salida.at[
                    indice,
                    "Concepto propuesto",
                ] = ""
                salida.at[
                    indice,
                    "Código de catálogo",
                ] = ""
                salida.at[
                    indice,
                    "Relación con la unidad",
                ] = "indeterminado"
                salida.at[
                    indice,
                    "Rol propuesto en la unidad",
                ] = "Revisión necesaria por conflicto de código"

    return salida


def agrupar_resultados_unidad(
    resultados: pd.DataFrame,
    procedimiento: str,
    consecutivo: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Consolida resultados finales.

    El orden de prioridad es:
    1. código propio completo;
    2. representante de la unidad;
    3. componente de la unidad;
    4. soporte / extracto;
    5. revisión.
    """
    if resultados.empty:
        return resultados.copy(), pd.DataFrame()

    salida = resultados.copy()
    codigo_unidad = _codigo_unidad(
        procedimiento,
        consecutivo,
    )

    salida = _resolver_conflictos_codigo_por_confianza(
        resultados=salida,
        codigo_unidad=codigo_unidad,
    )

    grupos = []
    relaciones = []
    acciones = []

    for _, fila in salida.iterrows():
        if str(fila.get("Error", "")).strip():
            grupos.append("Error de análisis")
            relaciones.append("No determinada")
            acciones.append("Revisar error")
            continue

        coincide = bool(fila.get("Coincide catálogo", False))
        codigo_final = str(
            fila.get("Código de catálogo", "")
        ).strip()
        relacion = str(
            fila.get("Relación con la unidad", "indeterminado")
        )
        alcance = str(
            fila.get("Alcance documental", "no_evaluado")
        )

        if (
            coincide
            and codigo_final
            and _codigo_sin_extension(codigo_final).upper()
            != codigo_unidad.upper()
        ):
            grupos.append(_codigo_sin_extension(codigo_final))
            relaciones.append("Documento con identidad propia")
            acciones.append(
                f"Codificación propuesta: {codigo_final}"
            )
            continue

        if (
            coincide
            and codigo_final
            and _codigo_sin_extension(codigo_final).upper()
            == codigo_unidad.upper()
            and relacion == "representante_unidad"
        ):
            grupos.append(codigo_unidad)
            relaciones.append("Posible representante de la unidad")
            acciones.append(
                "Candidato a recibir código de la unidad"
            )
            continue

        if relacion == "componente_unidad":
            grupos.append(codigo_unidad)
            relaciones.append(
                "Componente de la misma unidad lógica"
            )
            acciones.append("Conservar nombre original")
            continue

        if alcance == "parcial_extracto":
            grupos.append("Soporte / fuera de catálogo")
            relaciones.append("Documento parcial / extracto")
            acciones.append("Conservar nombre original")
            continue

        if relacion == "soporte":
            grupos.append("Soporte / fuera de catálogo")
            if (
                str(
                    fila.get(
                        "Resolución conflicto código",
                        "",
                    )
                )
                == "Desplazado por mayor confianza"
            ):
                relaciones.append(
                    "Candidato desplazado por conflicto de código"
                )
            else:
                relaciones.append("Componente de soporte")
            acciones.append("Conservar nombre original")
            continue

        grupos.append("Revisión necesaria")
        relaciones.append("Relación indeterminada")
        acciones.append("Revisión manual")

    salida["Grupo lógico"] = grupos
    salida["Relación consolidada"] = relaciones
    salida["Acción provisional"] = acciones

    resumen_registros = []

    for grupo, bloque in salida.groupby(
        "Grupo lógico",
        dropna=False,
        sort=False,
    ):
        archivos = list(bloque["Archivo"].astype(str))

        if grupo == codigo_unidad:
            representantes = int(
                (
                    bloque["Relación con la unidad"]
                    == "representante_unidad"
                ).sum()
            )
            componentes = int(
                (
                    bloque["Relación con la unidad"]
                    == "componente_unidad"
                ).sum()
            )

            if representantes == 1:
                estado = (
                    "Un representante candidato y "
                    f"{componentes} componente(s) de la unidad"
                )
            elif representantes > 1:
                estado = (
                    "Varios representantes candidatos; "
                    "requiere comparación"
                )
            else:
                estado = (
                    "Componentes asociados a la unidad; "
                    "representante pendiente"
                )

            codigo_asociado = f"{codigo_unidad}.pdf"

        elif grupo == "Soporte / fuera de catálogo":
            parciales = int(
                (
                    bloque["Alcance documental"]
                    == "parcial_extracto"
                ).sum()
            )
            estado = (
                f"Conservar como soporte; "
                f"{parciales} parcial(es)/extracto(s)"
            )
            codigo_asociado = ""

        elif grupo in (
            "Error de análisis",
            "Revisión necesaria",
        ):
            estado = "Revisión necesaria"
            codigo_asociado = ""

        elif len(bloque) > 1:
            estado = (
                "Varios archivos proponen el mismo código; "
                "revisar duplicado, variante o documento compuesto"
            )
            codigo_asociado = str(
                bloque.iloc[0]["Código de catálogo"]
            )

        else:
            estado = "Documento con código propio candidato"
            codigo_asociado = str(
                bloque.iloc[0]["Código de catálogo"]
            )

        resumen_registros.append(
            {
                "Grupo lógico": grupo,
                "Archivos": len(bloque),
                "Componentes": " | ".join(archivos),
                "Código asociado": codigo_asociado,
                "Estado del grupo": estado,
            }
        )

    return salida, pd.DataFrame(resumen_registros)
