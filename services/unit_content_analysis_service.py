from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import fitz
import pandas as pd

from services.evaluation_service import extraer_paginas_pdf_del_zip
from services.jev_classifier_service import JevError, clasificar_texto_con_jev
from services.openai_multimodal_service import (
    OpenAIMultimodalError,
    analizar_componente_unidad_multimodal,
    clasificar_pdf_multimodal,
)


PAGINAS_INICIALES = 2
MIN_CARACTERES_TEXTO = 200
UMBRAL_JEV = 70.0


def diagnosticar_componente_pdf(
    contenido_zip: bytes,
    ruta_pdf: str,
) -> dict:
    """
    Revisa únicamente las primeras páginas de un componente de la unidad.

    No usa IA. Determina si existe texto nativo suficiente para intentar Jev
    o si, en el prototipo web actual, conviene usar multimodal.
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

    ruta_sugerida = "Multimodal contextual"
    if caracteres >= MIN_CARACTERES_TEXTO:
        motivo = (
            "Existe texto nativo, pero esta fase necesita evaluar también "
            "alcance y función documental dentro de la unidad."
        )
    else:
        motivo = (
            "Texto nativo insuficiente. Esta fase usa análisis multimodal "
            "contextual para evaluar identidad, alcance y función."
        )

    return {
        "Páginas totales": total_paginas,
        "Páginas evaluadas": paginas,
        "Caracteres útiles": caracteres,
        "Ruta sugerida": ruta_sugerida,
        "Motivo": motivo,
        "Texto nativo": texto,
    }


def _interpretar_rol(
    coincide_catalogo: bool,
    codigo: str,
    procedimiento: str,
) -> str:
    if not coincide_catalogo:
        return "Posible soporte / fuera de catálogo"

    codigo_normalizado = str(codigo).upper()
    prefijo_estimacion = f"EJE_{procedimiento.upper()}_EST_"

    if prefijo_estimacion in codigo_normalizado:
        return "Posible representación de la unidad"

    return "Posible documento con código propio"


def _paginas_contexto_unidad(total_paginas: int) -> list[int]:
    """
    Selecciona hasta cuatro páginas para evaluar identidad y alcance.

    En documentos cortos usa todas. En documentos largos toma inicio y final,
    porque el alcance documental no siempre puede inferirse solo de la portada.
    """
    total = int(total_paginas)
    if total <= 0:
        return []
    if total <= 4:
        return list(range(1, total + 1))

    return [1, 2, total - 1, total]


def analizar_componente_unidad(
    contenido_zip: bytes,
    ruta_pdf: str,
    catalogo: pd.DataFrame,
    procedimiento: str,
    modelo_multimodal: str,
    umbral_jev: float = UMBRAL_JEV,
    tipo_unidad: str = "Estimación",
    consecutivo: int = 1,
) -> dict:
    """
    Analiza un componente dentro de una unidad documental.

    En esta fase se prioriza precisión sobre costo: se usa análisis multimodal
    contextual para separar identidad, alcance, relación con la unidad y
    equivalencia real con el catálogo. El nombre real y la ruta no se envían
    al modelo.
    """
    diagnostico = diagnosticar_componente_pdf(
        contenido_zip,
        ruta_pdf,
    )

    paginas = _paginas_contexto_unidad(
        diagnostico["Páginas totales"]
    )

    pdf_reducido = extraer_paginas_pdf_del_zip(
        contenido_zip,
        ruta_pdf,
        paginas,
    )

    mm = analizar_componente_unidad_multimodal(
        documento_alias="componente_unidad",
        pdf_bytes=pdf_reducido,
        catalogo=catalogo,
        modelo=modelo_multimodal,
        tipo_unidad=tipo_unidad,
        consecutivo=int(consecutivo),
    )

    alcance = str(mm["Alcance documental"])
    relacion = str(mm["Relación con la unidad"])
    coincide = bool(mm["Coincide catálogo"])

    # Salvaguarda determinista: un parcial/extracto no recibe el código de un
    # documento completo, aunque el modelo haya relacionado correctamente el
    # tipo documental.
    if alcance == "parcial_extracto":
        coincide = False

    mapa_roles = {
        "representante_unidad": "Posible representante de la unidad",
        "componente_unidad": "Componente de la misma unidad",
        "documento_independiente": (
            "Posible documento con código propio"
            if coincide
            else "Documento independiente sin equivalencia suficiente"
        ),
        "soporte": "Soporte / fuera de catálogo",
        "indeterminado": "Relación indeterminada",
    }

    return {
        "Ruta utilizada": "Multimodal contextual",
        "Motivo de ruta": (
            "Se evalúan identidad, alcance y función documental dentro "
            "de la unidad antes de permitir una codificación."
        ),
        "Título detectado": str(mm["Título detectado"]),
        "Alcance documental": alcance,
        "Relación con la unidad": relacion,
        "Concepto relacionado": str(mm["Concepto relacionado"]),
        "Código relacionado": str(mm["Código relacionado"]),
        "Coincide catálogo": coincide,
        "Concepto propuesto": (
            str(mm["Concepto propuesto"]) if coincide else ""
        ),
        "Código de catálogo": (
            str(mm["Código de catálogo"]) if coincide else ""
        ),
        "Confianza (%)": float(mm["Confianza (%)"]),
        "Rol propuesto en la unidad": mapa_roles.get(
            relacion,
            "Relación indeterminada",
        ),
        "Evidencia": str(mm["Evidencia"]),
        "Modelo": str(mm["Modelo"]),
        "Costo (USD)": float(mm["Costo (USD)"]),
        "Tiempo (s)": float(mm["Tiempo (s)"]),
        "Páginas usadas": ", ".join(str(p) for p in paginas),
    }


def _codigo_sin_extension(codigo: str) -> str:
    valor = str(codigo).strip()
    if valor.lower().endswith(".pdf"):
        return valor[:-4]
    return valor


def analizar_unidad_completa(
    contenido_zip: bytes,
    archivos_pdf: pd.DataFrame,
    catalogo: pd.DataFrame,
    procedimiento: str,
    modelo_multimodal: str,
    tipo_unidad: str,
    consecutivo: int,
    umbral_jev: float = UMBRAL_JEV,
    on_progress=None,
) -> pd.DataFrame:
    """
    Analiza todos los PDF directos de una unidad documental.

    Cada archivo se analiza de forma independiente y con alias neutro.
    El nombre real y la ruta nunca se envían al modelo.
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
                umbral_jev=umbral_jev,
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
                "Coincide catálogo": False,
                "Concepto propuesto": "",
                "Código de catálogo": "",
                "Confianza (%)": 0.0,
                "Alcance documental": "indeterminado",
                "Relación con la unidad": "indeterminado",
                "Concepto relacionado": "",
                "Código relacionado": "",
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


def agrupar_resultados_unidad(
    resultados: pd.DataFrame,
    procedimiento: str,
    consecutivo: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Consolida los componentes de una unidad usando relación y alcance.

    Un parcial/extracto nunca recibe automáticamente el código de un documento
    completo. Los componentes de la estimación se agrupan bajo el código de la
    unidad, pero solo un posible representante queda como candidato a recibirlo.
    """
    if resultados.empty:
        return resultados.copy(), pd.DataFrame()

    salida = resultados.copy()
    codigo_unidad = (
        f"EJE_{procedimiento.upper()}_EST_{int(consecutivo)}"
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

        alcance = str(
            fila.get("Alcance documental", "indeterminado")
        )
        relacion = str(
            fila.get("Relación con la unidad", "indeterminado")
        )
        coincide = bool(fila.get("Coincide catálogo", False))
        codigo = str(fila.get("Código de catálogo", "")).strip()

        if alcance == "parcial_extracto":
            grupos.append("Soporte / fuera de catálogo")
            relaciones.append("Documento parcial / extracto")
            acciones.append("Conservar nombre original")
            continue

        if relacion == "representante_unidad":
            grupos.append(codigo_unidad)
            relaciones.append("Posible representante de la unidad")
            acciones.append("Candidato a recibir código de la unidad")
            continue

        if relacion == "componente_unidad":
            grupos.append(codigo_unidad)
            relaciones.append("Componente de la misma unidad lógica")
            acciones.append("Conservar nombre original")
            continue

        if (
            relacion == "documento_independiente"
            and coincide
            and codigo
        ):
            grupos.append(_codigo_sin_extension(codigo))
            relaciones.append("Documento con identidad propia")
            acciones.append(f"Codificación propuesta: {codigo}")
            continue

        if relacion == "indeterminado":
            grupos.append("Revisión necesaria")
            relaciones.append("Relación indeterminada")
            acciones.append("Revisión manual")
            continue

        grupos.append("Soporte / fuera de catálogo")
        relaciones.append("Componente de soporte")
        acciones.append("Conservar nombre original")

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
        codigos = [
            str(valor)
            for valor in bloque["Código de catálogo"].astype(str)
            if str(valor).strip()
        ]

        if grupo == codigo_unidad:
            representantes = int(
                (
                    bloque["Relación con la unidad"]
                    == "representante_unidad"
                ).sum()
            )
            componentes = len(bloque) - representantes

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
        elif grupo == "Soporte / fuera de catálogo":
            parciales = int(
                (
                    bloque["Alcance documental"]
                    == "parcial_extracto"
                ).sum()
            )
            estado = (
                f"Conservar como soporte; {parciales} parcial(es)/extracto(s)"
            )
        elif grupo in ("Error de análisis", "Revisión necesaria"):
            estado = "Revisión necesaria"
        elif len(bloque) > 1:
            estado = (
                "Varios archivos proponen el mismo código; "
                "revisar duplicado, variante o documento compuesto"
            )
        else:
            estado = "Documento con código propio candidato"

        resumen_registros.append(
            {
                "Grupo lógico": grupo,
                "Archivos": len(bloque),
                "Componentes": " | ".join(archivos),
                "Código asociado": (
                    f"{codigo_unidad}.pdf"
                    if grupo == codigo_unidad
                    else (codigos[0] if codigos else "")
                ),
                "Estado del grupo": estado,
            }
        )

    resumen = pd.DataFrame(resumen_registros)
    return salida, resumen
