from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import fitz
import pandas as pd

from services.evaluation_service import extraer_paginas_pdf_del_zip
from services.openai_multimodal_service import (
    clasificar_pdf_multimodal,
    comparar_candidatos_unidad_multimodal,
    validar_integridad_documental_multimodal,
)


PAGINAS_INICIALES = 2


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
    Flujo en dos etapas.

    ETAPA 1
    Identifica el documento y su posible equivalencia con el catálogo,
    sin contexto de carpeta ni nombre real.

    ETAPA 2
    Solo si hace falta:
    - si apunta al código de la propia unidad, decide si es representante,
      componente o soporte;
    - si apunta a un código propio distinto, valida únicamente si el archivo
      es completo o un parcial/extracto.

    La segunda etapa nunca puede cambiar un código propio válido por el código
    de la estimación.
    """
    diagnostico = diagnosticar_componente_pdf(
        contenido_zip,
        ruta_pdf,
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
        catalogo=catalogo,
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
        # No se decide el papel de este archivo de forma aislada.
        # Todos los candidatos al código de la unidad se comparan juntos
        # después de terminar la clasificación individual.
        relacion = "candidato_unidad"
        validacion_secundaria = "Pendiente comparación conjunta"
        rol = "Candidato a comparación de unidad"

    else:
        paginas_contexto = _paginas_contexto(
            diagnostico["Páginas totales"]
        )
        pdf_contexto = _extraer_pdf_paginas(
            contenido_zip,
            ruta_pdf,
            paginas_contexto,
        )

        etapa2 = validar_integridad_documental_multimodal(
            documento_alias="componente_unidad",
            pdf_bytes=pdf_contexto,
            modelo=modelo_multimodal,
            concepto_objetivo=concepto_inicial,
            identidad_detectada=titulo,
        )

        alcance = str(etapa2["Alcance documental"])
        confianza_alcance = float(
            etapa2["Confianza alcance (%)"]
        )
        confianza_final = min(
            confianza_inicial,
            confianza_alcance,
        )
        evidencia_secundaria = str(
            etapa2["Evidencia alcance"]
        )
        costo += float(etapa2["Costo alcance (USD)"])
        tiempo += float(etapa2["Tiempo alcance (s)"])
        validacion_secundaria = "Integridad documental"
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

    if evidencia_secundaria:
        evidencia = (
            f"Etapa 1: {evidencia_inicial} "
            f"Validación secundaria: {evidencia_secundaria}"
        )
    else:
        evidencia = evidencia_inicial

    paginas_usadas = ", ".join(
        str(p)
        for p in _paginas_contexto(
            diagnostico["Páginas totales"]
        )
    )

    return {
        "Ruta utilizada": (
            "Multimodal: clasificación inicial + validación selectiva"
        ),
        "Motivo de ruta": (
            "La clasificación inicial conserva la identidad documental; "
            "la segunda etapa solo valida relación o integridad cuando aplica."
        ),
        "Título detectado": titulo,
        "Clasificación inicial": concepto_inicial,
        "Código inicial": codigo_inicial,
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
    """
    resultados = []
    total = len(archivos_pdf)

    for posicion, (_, fila) in enumerate(
        archivos_pdf.reset_index(drop=True).iterrows(),
        start=1,
    ):
        archivo = str(fila["Archivo"])
        ruta_pdf = str(fila["Ruta original"])

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

            registro = {
                "Archivo": archivo,
                "Ruta original": ruta_pdf,
                **resultado,
                "Error": "",
            }

        except Exception as error:
            registro = {
                "Archivo": archivo,
                "Ruta original": ruta_pdf,
                "Ruta utilizada": "Error",
                "Motivo de ruta": "",
                "Título detectado": "",
                "Clasificación inicial": "",
                "Código inicial": "",
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
                "Error": str(error),
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
