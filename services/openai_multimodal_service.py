from __future__ import annotations

import base64
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

from services.evaluation_service import extraer_paginas_pdf_del_zip


BASE_URL = "https://api.openai.com/v1"
MODELOS = {
    "gpt-5.6-luna": {
        "input_per_million": 0.20,
        "output_per_million": 1.20,
        "label": "GPT-5.6 Luna",
    },
    "gpt-5.6-terra": {
        "input_per_million": 2.00,
        "output_per_million": 12.00,
        "label": "GPT-5.6 Terra",
    },
}


class OpenAIMultimodalError(RuntimeError):
    pass


def openai_configurado() -> bool:
    try:
        return bool(str(st.secrets["openai"]["api_key"]).strip())
    except Exception:
        return False


def _api_key() -> str:
    if not openai_configurado():
        raise OpenAIMultimodalError(
            "No se encontró la API key de OpenAI en Streamlit Secrets."
        )
    return str(st.secrets["openai"]["api_key"]).strip()


def _request_json(
    method: str,
    path: str,
    payload: dict | None = None,
    timeout: int = 180,
) -> dict:
    body = None
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Accept": "application/json",
    }

    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(
        f"{BASE_URL}{path}",
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detalle = error.read().decode("utf-8", errors="replace")
        raise OpenAIMultimodalError(
            f"OpenAI respondió HTTP {error.code}: {detalle}"
        ) from error
    except URLError as error:
        raise OpenAIMultimodalError(
            f"No fue posible conectar con OpenAI: {error.reason}"
        ) from error


def probar_conexion_openai() -> None:
    _request_json("GET", "/models", timeout=60)


def _crear_opciones(catalogo: pd.DataFrame) -> tuple[list[str], dict]:
    lineas = []
    mapa = {}

    for indice, fila in catalogo.reset_index(drop=True).iterrows():
        opcion = f"opcion_{indice + 1:03d}"
        concepto = str(fila["Concepto"]).strip()
        codigo = str(fila["Código"]).strip()

        lineas.append(f"{opcion}: {concepto}")
        mapa[opcion] = {
            "concepto": concepto,
            "codigo": codigo,
        }

    lineas.append(
        "fuera_catalogo: El documento no corresponde de forma razonable "
        "a ninguno de los conceptos anteriores."
    )
    mapa["fuera_catalogo"] = {
        "concepto": "Fuera de catálogo / no identificado",
        "codigo": "",
    }

    return lineas, mapa


def _extraer_output_text(respuesta: dict) -> str:
    textos = []

    for item in respuesta.get("output", []):
        if item.get("type") != "message":
            continue

        for parte in item.get("content", []):
            if parte.get("type") == "output_text":
                textos.append(str(parte.get("text", "")))

    texto = "\n".join(textos).strip()

    if not texto:
        raise OpenAIMultimodalError(
            "La respuesta de OpenAI no contiene texto estructurado utilizable."
        )

    return texto


def clasificar_pdf_multimodal(
    documento_alias: str,
    pdf_bytes: bytes,
    catalogo: pd.DataFrame,
    modelo: str,
) -> dict:
    if modelo not in MODELOS:
        raise OpenAIMultimodalError(f"Modelo no admitido: {modelo}")

    if len(pdf_bytes) >= 50 * 1024 * 1024:
        raise OpenAIMultimodalError(
            f"{documento_alias} supera el límite de 50 MB por archivo."
        )

    opciones, mapa = _crear_opciones(catalogo)
    enum_opciones = list(mapa.keys())

    prompt = (
        "Analiza este documento de auditoría de obra pública usando "
        "exclusivamente su contenido visual y textual. No uses ni infieras "
        "información de nombres de archivo o rutas. Primero identifica qué "
        "tipo de documento es y proporciona un título documental breve. "
        "Después determina si corresponde realmente a una de las opciones "
        "del catálogo. Una coincidencia exige equivalencia documental y "
        "funcional; compartir palabras, tema, etapa de obra o contexto "
        "administrativo no es suficiente. No elijas la opción más cercana "
        "solo por similitud. Si el documento cumple otra función o solamente "
        "sirve como soporte de un documento listado, usa fuera_catalogo.\n\n"
        "Opciones permitidas:\n"
        + "\n".join(opciones)
    )

    schema = {
        "type": "object",
        "properties": {
            "detected_title": {
                "type": "string",
            },
            "option": {
                "type": "string",
                "enum": enum_opciones,
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
            },
            "evidence": {
                "type": "string",
            },
        },
        "required": [
            "detected_title",
            "option",
            "confidence",
            "evidence",
        ],
        "additionalProperties": False,
    }

    encoded = base64.b64encode(pdf_bytes).decode("ascii")

    payload = {
        "model": modelo,
        "reasoning": {
            "effort": "low",
        },
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": f"{documento_alias}.pdf",
                        "file_data": (
                            "data:application/pdf;base64,"
                            + encoded
                        ),
                        "detail": "high",
                    },
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "document_classification",
                "strict": True,
                "schema": schema,
            }
        },
        "max_output_tokens": 500,
    }

    inicio = time.perf_counter()
    respuesta = _request_json(
        "POST",
        "/responses",
        payload,
        timeout=240,
    )
    duracion = time.perf_counter() - inicio

    try:
        resultado = json.loads(_extraer_output_text(respuesta))
    except json.JSONDecodeError as error:
        raise OpenAIMultimodalError(
            "OpenAI devolvió una respuesta que no pudo convertirse a JSON."
        ) from error

    opcion = resultado.get("option")
    if opcion not in mapa:
        raise OpenAIMultimodalError(
            "OpenAI devolvió una opción fuera del catálogo permitido."
        )

    usage = respuesta.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)

    precios = MODELOS[modelo]
    costo = (
        input_tokens / 1_000_000 * precios["input_per_million"]
        + output_tokens / 1_000_000 * precios["output_per_million"]
    )

    confianza_raw = float(resultado["confidence"])
    confianza_pct = (
        confianza_raw * 100
        if 0 <= confianza_raw <= 1
        else confianza_raw
    )

    coincide_catalogo = opcion != "fuera_catalogo"

    return {
        "Documento": documento_alias,
        "Título detectado": str(resultado["detected_title"]).strip(),
        "Coincide catálogo": coincide_catalogo,
        "Resultado Ruta B": mapa[opcion]["concepto"],
        "Código Ruta B": mapa[opcion]["codigo"],
        "Confianza B": round(confianza_pct, 2),
        "Evidencia B": str(resultado["evidence"]).strip(),
        "Modelo B": str(respuesta.get("model") or modelo),
        "Tokens entrada B": input_tokens,
        "Tokens salida B": output_tokens,
        "Costo B (USD)": costo,
        "Tiempo B (s)": duracion,
    }


def clasificar_muestra_multimodal(
    muestra: pd.DataFrame,
    catalogo: pd.DataFrame,
    contenido_zip: bytes,
    modelo: str,
) -> pd.DataFrame:
    resultados = []

    for _, fila in muestra.iterrows():
        paginas = list(fila["Páginas evaluadas"])
        pdf_reducido = extraer_paginas_pdf_del_zip(
            contenido_zip,
            str(fila["Ruta original"]),
            paginas,
        )

        resultado = clasificar_pdf_multimodal(
            documento_alias=str(fila["Documento"]),
            pdf_bytes=pdf_reducido,
            catalogo=catalogo,
            modelo=modelo,
        )

        concepto_real = str(fila["Concepto real"])
        resultado["Concepto real"] = concepto_real
        resultado["Acierto B"] = (
            resultado["Resultado Ruta B"].strip().casefold()
            == concepto_real.strip().casefold()
        )
        resultado["Páginas B"] = ",".join(
            str(p) for p in paginas
        )

        resultados.append(resultado)

    return pd.DataFrame(resultados)



def analizar_componente_unidad_multimodal(
    documento_alias: str,
    pdf_bytes: bytes,
    catalogo: pd.DataFrame,
    modelo: str,
    tipo_unidad: str,
    consecutivo: int,
) -> dict:
    """
    Analiza un componente dentro de una unidad documental.

    A diferencia de clasificar_pdf_multimodal, esta función separa:
    - identidad documental;
    - alcance (completo, parcial/extracto o indeterminado);
    - relación con la unidad;
    - equivalencia real con un concepto del catálogo.

    El modelo no recibe el nombre real ni la ruta del archivo.
    """
    if modelo not in MODELOS:
        raise OpenAIMultimodalError(f"Modelo no admitido: {modelo}")

    if len(pdf_bytes) >= 50 * 1024 * 1024:
        raise OpenAIMultimodalError(
            f"{documento_alias} supera el límite de 50 MB por archivo."
        )

    opciones, mapa = _crear_opciones(catalogo)
    enum_opciones = list(mapa.keys())

    prompt = (
        "Analiza este componente documental de un expediente de obra pública. "
        "No uses ni infieras nombres de archivo o rutas. El componente se "
        f"encuentra dentro de una unidad candidata de tipo {tipo_unidad}, "
        f"consecutivo {int(consecutivo)}.\n\n"
        "Debes separar cinco decisiones:\n"
        "1) IDENTIDAD: describe qué documento es por su propia función.\n"
        "2) ALCANCE: indica si el documento está completo, si es solo una "
        "parte/extracto de un documento mayor, o si no puede determinarse.\n"
        "3) RELACIÓN CON LA UNIDAD: determina si funciona como posible "
        "representante de la unidad, como componente de la misma unidad, "
        "como documento independiente con identidad propia, como soporte, "
        "o si es indeterminado.\n"
        "4) CATÁLOGO: identifica la opción del catálogo más relacionada, si "
        "existe. Esto NO significa todavía que el código deba asignarse.\n"
        "5) EQUIVALENCIA: solo marca catalog_equivalent=true cuando ESTE "
        "archivo, por sí mismo y con el alcance observado, satisface "
        "documental y funcionalmente el concepto del catálogo.\n\n"
        "Reglas estrictas:\n"
        "- Que un archivo mencione una estimación no lo convierte en la "
        "estimación. Croquis, plantillas, listas de verificación, facturas, "
        "generadores, fotografías y otros soportes siguen siendo soporte "
        "salvo que el catálogo contemple expresamente ese documento.\n"
        "- Hojas, notas o páginas parciales de una bitácora NO equivalen a "
        "la Bitácora de obra completa. Si solo se observa una parte de la "
        "bitácora, usa scope=parcial_extracto y catalog_equivalent=false.\n"
        "- Usa scope=parcial_extracto para fragmentos de otro documento "
        "mayor (por ejemplo, algunas hojas de la bitácora completa, páginas "
        "aisladas de un contrato o una parte de un presupuesto). No uses "
        "parcial_extracto solo porque el archivo sea un componente legítimo "
        "de la unidad: una carátula completa o una hoja de estimación completa "
        "pueden tener scope=completo y, al mismo tiempo, relación de "
        "representante_unidad o componente_unidad.\n"
        "- Un documento parcial o extracto no recibe el código del documento "
        "completo, salvo que el propio concepto del catálogo describa "
        "expresamente un parcial/extracto.\n"
        "- Un archivo puede formar parte de la unidad documental sin ser un "
        "documento independiente del catálogo.\n"
        "- No elijas una opción solo por similitud temática o palabras "
        "compartidas.\n\n"
        "Opciones permitidas del catálogo:\n"
        + "\n".join(opciones)
    )

    schema = {
        "type": "object",
        "properties": {
            "detected_title": {"type": "string"},
            "scope": {
                "type": "string",
                "enum": [
                    "completo",
                    "parcial_extracto",
                    "indeterminado",
                ],
            },
            "unit_relation": {
                "type": "string",
                "enum": [
                    "representante_unidad",
                    "componente_unidad",
                    "documento_independiente",
                    "soporte",
                    "indeterminado",
                ],
            },
            "catalog_option": {
                "type": "string",
                "enum": enum_opciones,
            },
            "catalog_equivalent": {"type": "boolean"},
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
            },
            "evidence": {"type": "string"},
        },
        "required": [
            "detected_title",
            "scope",
            "unit_relation",
            "catalog_option",
            "catalog_equivalent",
            "confidence",
            "evidence",
        ],
        "additionalProperties": False,
    }

    encoded = base64.b64encode(pdf_bytes).decode("ascii")

    payload = {
        "model": modelo,
        "reasoning": {"effort": "low"},
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": f"{documento_alias}.pdf",
                        "file_data": (
                            "data:application/pdf;base64," + encoded
                        ),
                        "detail": "high",
                    },
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "unit_component_analysis",
                "strict": True,
                "schema": schema,
            }
        },
        "max_output_tokens": 700,
    }

    inicio = time.perf_counter()
    respuesta = _request_json(
        "POST",
        "/responses",
        payload,
        timeout=240,
    )
    duracion = time.perf_counter() - inicio

    try:
        resultado = json.loads(_extraer_output_text(respuesta))
    except json.JSONDecodeError as error:
        raise OpenAIMultimodalError(
            "OpenAI devolvió una respuesta que no pudo convertirse a JSON."
        ) from error

    opcion = resultado.get("catalog_option")
    if opcion not in mapa:
        raise OpenAIMultimodalError(
            "OpenAI devolvió una opción fuera del catálogo permitido."
        )

    equivalente = bool(resultado["catalog_equivalent"])
    if opcion == "fuera_catalogo":
        equivalente = False

    usage = respuesta.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)

    precios = MODELOS[modelo]
    costo = (
        input_tokens / 1_000_000 * precios["input_per_million"]
        + output_tokens / 1_000_000 * precios["output_per_million"]
    )

    confianza_raw = float(resultado["confidence"])
    confianza_pct = (
        confianza_raw * 100
        if 0 <= confianza_raw <= 1
        else confianza_raw
    )

    return {
        "Documento": documento_alias,
        "Título detectado": str(resultado["detected_title"]).strip(),
        "Alcance documental": str(resultado["scope"]),
        "Relación con la unidad": str(resultado["unit_relation"]),
        "Concepto relacionado": mapa[opcion]["concepto"],
        "Código relacionado": mapa[opcion]["codigo"],
        "Coincide catálogo": equivalente,
        "Concepto propuesto": (
            mapa[opcion]["concepto"] if equivalente else ""
        ),
        "Código de catálogo": (
            mapa[opcion]["codigo"] if equivalente else ""
        ),
        "Confianza (%)": round(confianza_pct, 2),
        "Evidencia": str(resultado["evidence"]).strip(),
        "Modelo": str(respuesta.get("model") or modelo),
        "Tokens entrada": input_tokens,
        "Tokens salida": output_tokens,
        "Costo (USD)": costo,
        "Tiempo (s)": duracion,
    }



def analizar_relacion_unidad_multimodal(
    documento_alias: str,
    pdf_bytes: bytes,
    modelo: str,
    tipo_unidad: str,
    consecutivo: int,
    identidad_detectada: str,
) -> dict:
    """
    Segunda etapa para archivos inicialmente asociados al código de la unidad.

    No vuelve a clasificar contra el catálogo. Solo determina qué función tiene
    el archivo respecto de la unidad documental ya detectada.
    """
    if modelo not in MODELOS:
        raise OpenAIMultimodalError(f"Modelo no admitido: {modelo}")

    prompt = (
        "Analiza la función de este archivo dentro de una unidad documental "
        f"de tipo {tipo_unidad}, consecutivo {int(consecutivo)}. "
        f"La primera etapa identificó el archivo como: {identidad_detectada}. "
        "No vuelvas a clasificarlo contra un catálogo y no uses el nombre del "
        "archivo ni su ruta.\n\n"
        "Elige exactamente una relación:\n"
        "- representante_unidad: el archivo funciona como portada, carátula, "
        "resumen formal o pieza de identificación principal de la unidad y es "
        "el mejor candidato para portar el código de la unidad cuando esta se "
        "encuentra fragmentada en varios archivos.\n"
        "- componente_unidad: el archivo contiene el cuerpo sustantivo propio "
        "de la unidad, pero no es la pieza de identificación principal.\n"
        "- soporte: el archivo está relacionado con la unidad, pero cumple una "
        "función documental distinta y auxiliar. Ejemplos típicos son listas "
        "de verificación, croquis, plantillas, facturas, números generadores, "
        "reportes fotográficos, solicitudes de pago u otros anexos de soporte.\n"
        "- indeterminado: no existe evidencia suficiente para decidir.\n\n"
        "Regla crítica: mencionar la estimación, incluir conceptos, cantidades "
        "o datos del contrato no convierte por sí solo al archivo en parte "
        "sustantiva de la estimación."
    )

    schema = {
        "type": "object",
        "properties": {
            "unit_relation": {
                "type": "string",
                "enum": [
                    "representante_unidad",
                    "componente_unidad",
                    "soporte",
                    "indeterminado",
                ],
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
            },
            "evidence": {"type": "string"},
        },
        "required": [
            "unit_relation",
            "confidence",
            "evidence",
        ],
        "additionalProperties": False,
    }

    encoded = base64.b64encode(pdf_bytes).decode("ascii")
    payload = {
        "model": modelo,
        "reasoning": {"effort": "low"},
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": f"{documento_alias}.pdf",
                        "file_data": (
                            "data:application/pdf;base64," + encoded
                        ),
                        "detail": "high",
                    },
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "unit_relation_validation",
                "strict": True,
                "schema": schema,
            }
        },
        "max_output_tokens": 400,
    }

    inicio = time.perf_counter()
    respuesta = _request_json(
        "POST",
        "/responses",
        payload,
        timeout=240,
    )
    duracion = time.perf_counter() - inicio

    try:
        resultado = json.loads(_extraer_output_text(respuesta))
    except json.JSONDecodeError as error:
        raise OpenAIMultimodalError(
            "OpenAI devolvió una respuesta inválida al validar la unidad."
        ) from error

    usage = respuesta.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    precios = MODELOS[modelo]
    costo = (
        input_tokens / 1_000_000 * precios["input_per_million"]
        + output_tokens / 1_000_000 * precios["output_per_million"]
    )

    confianza = float(resultado["confidence"])
    if 0 <= confianza <= 1:
        confianza *= 100

    return {
        "Relación con la unidad": str(resultado["unit_relation"]),
        "Confianza relación (%)": round(confianza, 2),
        "Evidencia relación": str(resultado["evidence"]).strip(),
        "Costo relación (USD)": costo,
        "Tiempo relación (s)": duracion,
    }


def validar_integridad_documental_multimodal(
    documento_alias: str,
    pdf_bytes: bytes,
    modelo: str,
    concepto_objetivo: str,
    identidad_detectada: str,
) -> dict:
    """
    Segunda etapa para un documento con código propio candidato.

    No cambia su identidad. Solo determina si el archivo es una instancia
    completa del concepto o un parcial/extracto de un documento mayor.
    """
    if modelo not in MODELOS:
        raise OpenAIMultimodalError(f"Modelo no admitido: {modelo}")

    prompt = (
        "Evalúa exclusivamente la INTEGRIDAD DOCUMENTAL de este archivo. "
        f"La primera etapa lo identificó como: {identidad_detectada}. "
        f"El concepto objetivo del catálogo es: {concepto_objetivo}. "
        "No cambies la identidad documental y no selecciones otro concepto. "
        "No uses el nombre del archivo ni su ruta.\n\n"
        "Clasifica el alcance como:\n"
        "- completo: el archivo constituye por sí mismo una instancia completa "
        "del concepto objetivo para efectos de identificación documental.\n"
        "- parcial_extracto: contiene solamente algunas hojas, notas, páginas "
        "o una sección de un documento mayor. Por ejemplo, algunas notas de "
        "bitácora incluidas como soporte de una estimación no constituyen la "
        "Bitácora de obra completa.\n"
        "- indeterminado: con las páginas disponibles no puede establecerse "
        "razonablemente si está completo.\n\n"
        "No confundas un documento corto pero completo con un extracto. La "
        "decisión depende de su función y alcance, no del número de páginas."
    )

    schema = {
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": [
                    "completo",
                    "parcial_extracto",
                    "indeterminado",
                ],
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
            },
            "evidence": {"type": "string"},
        },
        "required": ["scope", "confidence", "evidence"],
        "additionalProperties": False,
    }

    encoded = base64.b64encode(pdf_bytes).decode("ascii")
    payload = {
        "model": modelo,
        "reasoning": {"effort": "low"},
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": f"{documento_alias}.pdf",
                        "file_data": (
                            "data:application/pdf;base64," + encoded
                        ),
                        "detail": "high",
                    },
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "document_integrity_validation",
                "strict": True,
                "schema": schema,
            }
        },
        "max_output_tokens": 400,
    }

    inicio = time.perf_counter()
    respuesta = _request_json(
        "POST",
        "/responses",
        payload,
        timeout=240,
    )
    duracion = time.perf_counter() - inicio

    try:
        resultado = json.loads(_extraer_output_text(respuesta))
    except json.JSONDecodeError as error:
        raise OpenAIMultimodalError(
            "OpenAI devolvió una respuesta inválida al validar integridad."
        ) from error

    usage = respuesta.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    precios = MODELOS[modelo]
    costo = (
        input_tokens / 1_000_000 * precios["input_per_million"]
        + output_tokens / 1_000_000 * precios["output_per_million"]
    )

    confianza = float(resultado["confidence"])
    if 0 <= confianza <= 1:
        confianza *= 100

    return {
        "Alcance documental": str(resultado["scope"]),
        "Confianza alcance (%)": round(confianza, 2),
        "Evidencia alcance": str(resultado["evidence"]).strip(),
        "Costo alcance (USD)": costo,
        "Tiempo alcance (s)": duracion,
    }
