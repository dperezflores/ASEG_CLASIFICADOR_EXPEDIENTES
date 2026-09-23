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

    if caracteres >= MIN_CARACTERES_TEXTO:
        ruta_sugerida = "Texto nativo + Jev"
        motivo = "Texto nativo suficiente en las primeras páginas."
    else:
        ruta_sugerida = "Multimodal"
        motivo = (
            "Texto nativo insuficiente. En el prototipo web se evita OCR "
            "local y se usa análisis multimodal."
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


def analizar_componente_unidad(
    contenido_zip: bytes,
    ruta_pdf: str,
    catalogo: pd.DataFrame,
    procedimiento: str,
    modelo_multimodal: str,
    umbral_jev: float = UMBRAL_JEV,
) -> dict:
    """
    Analiza un solo componente de una unidad documental.

    El modelo nunca recibe el nombre real ni la ruta del archivo.
    """
    diagnostico = diagnosticar_componente_pdf(
        contenido_zip,
        ruta_pdf,
    )

    alias = "componente_unidad"
    ruta_utilizada = diagnostico["Ruta sugerida"]
    motivo_ruta = diagnostico["Motivo"]

    if ruta_utilizada == "Texto nativo + Jev":
        try:
            jev = clasificar_texto_con_jev(
                documento_alias=alias,
                texto_ocr=diagnostico["Texto nativo"],
                catalogo=catalogo,
            )

            fuera = (
                jev["Resultado Ruta A"]
                == "Fuera de catálogo / no identificado"
            )
            confianza = float(jev["Confianza A"])

            if not fuera and confianza >= float(umbral_jev):
                coincide = True
                concepto = str(jev["Resultado Ruta A"])
                codigo = str(jev["Código Ruta A"])
                titulo = concepto
                evidencia = (
                    "Clasificación Jev sobre texto nativo de las primeras "
                    "páginas."
                )
                modelo = str(jev["Modelo A"])
                costo = float(jev["Costo A (USD)"])
                tiempo = float(jev["Tiempo A (s)"])
            else:
                motivo_ruta = (
                    "Jev indicó fuera de catálogo"
                    if fuera
                    else (
                        f"Confianza Jev {confianza:.1f}% menor al umbral "
                        f"{float(umbral_jev):.1f}%"
                    )
                )
                ruta_utilizada = "Multimodal (escalado)"
                jev_costo = float(jev["Costo A (USD)"])
                jev_tiempo = float(jev["Tiempo A (s)"])

                paginas = diagnostico["Páginas evaluadas"]
                pdf_reducido = extraer_paginas_pdf_del_zip(
                    contenido_zip,
                    ruta_pdf,
                    paginas,
                )
                mm = clasificar_pdf_multimodal(
                    documento_alias=alias,
                    pdf_bytes=pdf_reducido,
                    catalogo=catalogo,
                    modelo=modelo_multimodal,
                )

                coincide = bool(mm["Coincide catálogo"])
                concepto = (
                    str(mm["Resultado Ruta B"])
                    if coincide
                    else ""
                )
                codigo = str(mm["Código Ruta B"]) if coincide else ""
                titulo = str(mm["Título detectado"])
                confianza = float(mm["Confianza B"])
                evidencia = str(mm["Evidencia B"])
                modelo = str(mm["Modelo B"])
                costo = jev_costo + float(mm["Costo B (USD)"])
                tiempo = jev_tiempo + float(mm["Tiempo B (s)"])

        except JevError:
            ruta_utilizada = "Multimodal (fallo Jev)"
            motivo_ruta = "Jev no pudo completar la clasificación."
            paginas = diagnostico["Páginas evaluadas"]
            pdf_reducido = extraer_paginas_pdf_del_zip(
                contenido_zip,
                ruta_pdf,
                paginas,
            )
            mm = clasificar_pdf_multimodal(
                documento_alias=alias,
                pdf_bytes=pdf_reducido,
                catalogo=catalogo,
                modelo=modelo_multimodal,
            )
            coincide = bool(mm["Coincide catálogo"])
            concepto = str(mm["Resultado Ruta B"]) if coincide else ""
            codigo = str(mm["Código Ruta B"]) if coincide else ""
            titulo = str(mm["Título detectado"])
            confianza = float(mm["Confianza B"])
            evidencia = str(mm["Evidencia B"])
            modelo = str(mm["Modelo B"])
            costo = float(mm["Costo B (USD)"])
            tiempo = float(mm["Tiempo B (s)"])

    else:
        paginas = diagnostico["Páginas evaluadas"]
        pdf_reducido = extraer_paginas_pdf_del_zip(
            contenido_zip,
            ruta_pdf,
            paginas,
        )
        mm = clasificar_pdf_multimodal(
            documento_alias=alias,
            pdf_bytes=pdf_reducido,
            catalogo=catalogo,
            modelo=modelo_multimodal,
        )
        coincide = bool(mm["Coincide catálogo"])
        concepto = str(mm["Resultado Ruta B"]) if coincide else ""
        codigo = str(mm["Código Ruta B"]) if coincide else ""
        titulo = str(mm["Título detectado"])
        confianza = float(mm["Confianza B"])
        evidencia = str(mm["Evidencia B"])
        modelo = str(mm["Modelo B"])
        costo = float(mm["Costo B (USD)"])
        tiempo = float(mm["Tiempo B (s)"])

    rol = _interpretar_rol(
        coincide_catalogo=coincide,
        codigo=codigo,
        procedimiento=procedimiento,
    )

    return {
        "Ruta utilizada": ruta_utilizada,
        "Motivo de ruta": motivo_ruta,
        "Título detectado": titulo,
        "Coincide catálogo": coincide,
        "Concepto propuesto": concepto,
        "Código de catálogo": codigo,
        "Confianza (%)": confianza,
        "Rol propuesto en la unidad": rol,
        "Evidencia": evidencia,
        "Modelo": modelo,
        "Costo (USD)": costo,
        "Tiempo (s)": tiempo,
        "Páginas usadas": ", ".join(
            str(p) for p in diagnostico["Páginas evaluadas"]
        ),
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
    Agrupa clasificaciones individuales para interpretar relaciones internas.

    No elige todavía un archivo representativo. Si varios componentes apuntan
    al código de la propia estimación, quedan agrupados como partes de la misma
    unidad lógica y se marca que la representación está pendiente.
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

        coincide = bool(fila["Coincide catálogo"])
        codigo = str(fila["Código de catálogo"]).strip()

        if not coincide or not codigo:
            grupos.append("Soporte / fuera de catálogo")
            relaciones.append("Componente de soporte")
            acciones.append("Conservar nombre original")
            continue

        if _codigo_sin_extension(codigo).upper() == codigo_unidad.upper():
            grupos.append(codigo_unidad)
            relaciones.append("Componente de la misma unidad lógica")
            acciones.append(
                "Pendiente elegir archivo representativo"
            )
        else:
            grupos.append(_codigo_sin_extension(codigo))
            relaciones.append("Documento con identidad propia")
            acciones.append(
                f"Codificación propuesta: {codigo}"
            )

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
            if len(bloque) > 1:
                estado = (
                    "Varios componentes asociados a la misma unidad; "
                    "representante pendiente"
                )
            else:
                estado = (
                    "Un componente asociado a la unidad; "
                    "representante pendiente de validar"
                )
        elif grupo == "Soporte / fuera de catálogo":
            estado = "Conservar como soporte"
        elif grupo == "Error de análisis":
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
                "Código asociado": codigos[0] if codigos else "",
                "Estado del grupo": estado,
            }
        )

    resumen = pd.DataFrame(resumen_registros)
    return salida, resumen
