from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import fitz
import pandas as pd

from services.catalog_family_service import (
    construir_catalogo_operativo_estimacion,
)
from services.evaluation_service import extraer_paginas_pdf_del_zip
from services.openai_multimodal_service import (
    clasificar_pdf_multimodal,
    comparar_candidatos_unidad_multimodal,
    identificar_documento_multimodal,
    resolver_catalogo_desde_identidad_multimodal,
    validar_equivalencia_documental_multimodal,
    validar_integridad_documental_multimodal,
)


PAGINAS_INICIALES = 2
MAX_INTENTOS_ANALISIS = 2
MAX_PAGINAS_INTEGRIDAD_COMPLETA = 10
USAR_IDENTIDAD_PURA = True


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


def _analizar_componente_unidad_legacy(
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

        etapa_equivalencia = validar_equivalencia_documental_multimodal(
            documento_alias="componente_unidad",
            pdf_bytes=pdf_contexto,
            modelo=modelo_multimodal,
            concepto_objetivo=concepto_inicial,
            identidad_detectada=titulo,
        )

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
            confianza_inicial,
            confianza_equivalencia,
        )
        costo += float(
            etapa_equivalencia["Costo equivalencia (USD)"]
        )
        tiempo += float(
            etapa_equivalencia["Tiempo equivalencia (s)"]
        )

        if equivalencia == "equivalente":
            etapa_integridad = validar_integridad_documental_multimodal(
                documento_alias="componente_unidad",
                pdf_bytes=pdf_contexto,
                modelo=modelo_multimodal,
                concepto_objetivo=concepto_inicial,
                identidad_detectada=titulo,
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
            "Multimodal: identidad + equivalencia + integridad selectiva"
        ),
        "Motivo de ruta": (
            "La clasificación inicial propone una identidad y un concepto; "
            "los códigos propios solo se aceptan después de validar de forma "
            "independiente equivalencia documental-funcional e integridad."
        ),
        "Título detectado": titulo,
        "Clasificación inicial": concepto_inicial,
        "Código inicial": codigo_inicial,
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



def _analizar_componente_unidad_identidad_pura(
    contenido_zip: bytes,
    ruta_pdf: str,
    catalogo: pd.DataFrame,
    procedimiento: str,
    modelo_multimodal: str,
    tipo_unidad: str = "Estimación",
    consecutivo: int = 1,
) -> dict:
    """
    Flujo experimental controlado:
    1) identidad pura sin catálogo;
    2) resolución catálogo desde identidad fija;
    3) integridad solo para códigos propios equivalentes;
    4) comparación conjunta para candidatos a la unidad.
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

    codigo_unidad = _codigo_unidad(
        procedimiento,
        consecutivo,
    )

    paginas_identidad = list(
        range(
            1,
            min(
                int(diagnostico["Páginas totales"]),
                PAGINAS_INICIALES,
            )
            + 1,
        )
    )
    pdf_identidad = _extraer_pdf_paginas(
        contenido_zip,
        ruta_pdf,
        paginas_identidad,
    )

    identidad = identificar_documento_multimodal(
        documento_alias="documento_sin_catalogo",
        pdf_bytes=pdf_identidad,
        modelo=modelo_multimodal,
    )

    titulo = str(identidad["Título detectado"])
    funcion_formal = str(identidad["Función formal"])
    acto_documentado = str(identidad["Acto documentado"])
    confianza_identidad = float(
        identidad["Confianza identidad (%)"]
    )
    evidencia_identidad = str(
        identidad["Evidencia identidad"]
    )

    costo = float(identidad["Costo identidad (USD)"])
    tiempo = float(identidad["Tiempo identidad (s)"])

    paginas_resolucion = _paginas_contexto(
        diagnostico["Páginas totales"]
    )
    pdf_resolucion = _extraer_pdf_paginas(
        contenido_zip,
        ruta_pdf,
        paginas_resolucion,
    )

    resolucion = resolver_catalogo_desde_identidad_multimodal(
        documento_alias="documento_identificado",
        pdf_bytes=pdf_resolucion,
        catalogo=catalogo_operativo,
        modelo=modelo_multimodal,
        identidad_detectada=titulo,
        funcion_formal=funcion_formal,
        acto_documentado=acto_documentado,
        codigo_unidad=codigo_unidad,
    )

    concepto_inicial = str(
        resolucion["Resultado catálogo"]
    )
    codigo_inicial = str(
        resolucion["Código candidato"]
    )
    equivalencia = str(
        resolucion["Relación catálogo"]
    )
    confianza_resolucion = float(
        resolucion["Confianza resolución (%)"]
    )
    evidencia_resolucion = str(
        resolucion["Evidencia resolución"]
    )

    costo += float(resolucion["Costo resolución (USD)"])
    tiempo += float(resolucion["Tiempo resolución (s)"])

    confianza_final = min(
        confianza_identidad,
        confianza_resolucion,
    )

    alcance = "no_evaluado"
    relacion = "soporte"
    validacion_secundaria = (
        "Identidad pura + resolución de catálogo"
    )
    evidencia_integridad = ""
    coincide_final = False
    concepto_final = ""
    codigo_final = ""

    concepto_relacionado = (
        concepto_inicial if codigo_inicial else ""
    )
    codigo_relacionado = codigo_inicial

    codigo_inicial_normalizado = _codigo_sin_extension(
        codigo_inicial
    ).upper()

    if (
        codigo_inicial_normalizado == codigo_unidad.upper()
        and equivalencia in ("candidato_unidad", "equivalente")
    ):
        relacion = "candidato_unidad"
        equivalencia = "candidato_unidad"
        validacion_secundaria = (
            "Pendiente comparación conjunta"
        )
        rol = "Candidato a comparación de unidad"

    elif codigo_inicial and equivalencia == "equivalente":
        paginas_integridad = _paginas_integridad(
            diagnostico["Páginas totales"]
        )
        pdf_integridad = _extraer_pdf_paginas(
            contenido_zip,
            ruta_pdf,
            paginas_integridad,
        )

        etapa_integridad = validar_integridad_documental_multimodal(
            documento_alias="documento_identificado",
            pdf_bytes=pdf_integridad,
            modelo=modelo_multimodal,
            concepto_objetivo=concepto_inicial,
            identidad_detectada=titulo,
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
        evidencia_integridad = str(
            etapa_integridad["Evidencia alcance"]
        )
        costo += float(
            etapa_integridad["Costo alcance (USD)"]
        )
        tiempo += float(
            etapa_integridad["Tiempo alcance (s)"]
        )

        validacion_secundaria = (
            "Resolución de catálogo + integridad"
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

    elif equivalencia == "indeterminado":
        relacion = "indeterminado"
        rol = "Revisión necesaria"

    else:
        relacion = "soporte"
        rol = "Soporte / fuera de catálogo"

    evidencias = [
        "Identidad pura: " + evidencia_identidad,
        "Resolución catálogo: " + evidencia_resolucion,
    ]
    if evidencia_integridad:
        evidencias.append(
            "Integridad: " + evidencia_integridad
        )

    paginas_reporte = sorted(
        set(
            paginas_identidad
            + paginas_resolucion
            + (
                _paginas_integridad(
                    diagnostico["Páginas totales"]
                )
                if evidencia_integridad
                else []
            )
        )
    )

    return {
        "Ruta utilizada": (
            "Experimental: identidad pura -> catálogo -> integridad selectiva"
        ),
        "Motivo de ruta": (
            "La primera llamada no recibe catálogo. La identidad queda fija "
            "antes de comparar contra conceptos institucionales."
        ),
        "Título detectado": titulo,
        "Función formal": funcion_formal,
        "Acto documentado": acto_documentado,
        "Confianza identidad (%)": confianza_identidad,
        "Clasificación inicial": concepto_inicial,
        "Código inicial": codigo_inicial,
        "Equivalencia funcional": equivalencia,
        "Confianza equivalencia (%)": confianza_resolucion,
        "Evidencia equivalencia": evidencia_resolucion,
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
        "Evidencia": " ".join(evidencias),
        "Modelo": str(identidad["Modelo identidad"]),
        "Costo (USD)": costo,
        "Tiempo (s)": tiempo,
        "Páginas usadas": ", ".join(
            str(p) for p in paginas_reporte
        ),
    }


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
    Punto único de entrada. El flujo anterior queda intacto como respaldo.
    Para regresar de inmediato basta cambiar USAR_IDENTIDAD_PURA a False.
    """
    if USAR_IDENTIDAD_PURA:
        return _analizar_componente_unidad_identidad_pura(
            contenido_zip=contenido_zip,
            ruta_pdf=ruta_pdf,
            catalogo=catalogo,
            procedimiento=procedimiento,
            modelo_multimodal=modelo_multimodal,
            tipo_unidad=tipo_unidad,
            consecutivo=consecutivo,
        )

    return _analizar_componente_unidad_legacy(
        contenido_zip=contenido_zip,
        ruta_pdf=ruta_pdf,
        catalogo=catalogo,
        procedimiento=procedimiento,
        modelo_multimodal=modelo_multimodal,
        tipo_unidad=tipo_unidad,
        consecutivo=consecutivo,
    )


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
                "Función formal": "",
                "Acto documentado": "",
                "Confianza identidad (%)": 0.0,
                "Clasificación inicial": "",
                "Código inicial": "",
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

    mascara_codigo = (
        salida["Código inicial"]
        .astype(str)
        .apply(_codigo_sin_extension)
        .str.upper()
        == codigo_unidad.upper()
    )
    mascara_sin_error = (
        salida["Error"].astype(str).str.strip() == ""
    )

    if "Equivalencia funcional" in salida.columns:
        equivalencias = (
            salida["Equivalencia funcional"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        mascara_relacion = equivalencias.isin(
            ["candidato_unidad", "no_evaluada"]
        )
    else:
        mascara_relacion = pd.Series(
            True,
            index=salida.index,
        )

    mascara = (
        mascara_codigo
        & mascara_sin_error
        & mascara_relacion
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
