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

# Versiones explícitas para persistir/reutilizar la Ficha V2 sin mezclar
# resultados generados por prompts o esquemas distintos.
FICHA_DOCUMENTAL_SCHEMA_VERSION = 2
FICHA_DOCUMENTAL_PROMPT_VERSION = "v2_20260924_internal_records_relations_limits"

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


def identificar_documento_multimodal(
    documento_alias: str,
    pdf_bytes: bytes,
    modelo: str,
) -> dict:
    """
    Identidad documental pura para verificación selectiva.

    No recibe catálogo, códigos, nombre real ni ruta. No clasifica ni asigna
    códigos: únicamente describe qué documento es, su función formal y el acto
    que documenta.
    """
    if modelo not in MODELOS:
        raise OpenAIMultimodalError(f"Modelo no admitido: {modelo}")

    if len(pdf_bytes) >= 50 * 1024 * 1024:
        raise OpenAIMultimodalError(
            f"{documento_alias} supera el límite de 50 MB por archivo."
        )

    prompt = (
        "Analiza este archivo de un expediente de obra pública usando "
        "exclusivamente su contenido visual y textual. NO tienes acceso al "
        "catálogo institucional, NO conoces códigos y NO debes intentar "
        "adivinarlos. No uses ni infieras información del nombre real del "
        "archivo ni de su ruta.\n\n"
        "Tu única tarea es establecer la IDENTIDAD DOCUMENTAL PROPIA del "
        "archivo: qué clase de documento es, cuál es su función formal y qué "
        "acto, hecho u operación documenta.\n\n"
        "Reglas estrictas:\n"
        "1. Identifica el documento por lo que ES, no por la etapa a la que "
        "pertenece ni por otros documentos que mencione.\n"
        "2. Un soporte no se convierte en el documento principal al que hace "
        "referencia. Un recibo que menciona un finiquito sigue siendo un "
        "recibo; una constancia de entrega física sigue siendo una constancia "
        "de entrega física aunque se relacione con una entrega-recepción.\n"
        "3. Distingue finalidades documentales cercanas: presupuesto de "
        "referencia, propuesta, contratado o finiquitado; y distingue también "
        "los diferentes objetos de las fianzas o garantías.\n"
        "4. No evalúes si el documento pertenece al catálogo, no selecciones "
        "conceptos del catálogo y no decidas ningún código.\n"
        "5. Si la evidencia visible no permite una identificación precisa, "
        "describe la identidad más específica que sí esté sustentada y baja "
        "la confianza."
    )

    schema = {
        "type": "object",
        "properties": {
            "detected_title": {"type": "string"},
            "formal_function": {"type": "string"},
            "documented_act": {"type": "string"},
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
            },
            "evidence": {"type": "string"},
        },
        "required": [
            "detected_title",
            "formal_function",
            "documented_act",
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
                "name": "selective_pure_document_identity",
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
            "OpenAI devolvió una identidad documental inválida."
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
        "Título identidad pura": str(
            resultado["detected_title"]
        ).strip(),
        "Función identidad pura": str(
            resultado["formal_function"]
        ).strip(),
        "Acto identidad pura": str(
            resultado["documented_act"]
        ).strip(),
        "Confianza identidad pura (%)": round(confianza, 2),
        "Evidencia identidad pura": str(
            resultado["evidence"]
        ).strip(),
        "Costo identidad pura (USD)": costo,
        "Tiempo identidad pura (s)": duracion,
    }


def generar_ficha_documental_multimodal(
    documento_alias: str,
    pdf_bytes: bytes,
    modelo: str,
) -> dict:
    """
    Ficha documental canónica V2.

    Una sola lectura multimodal del PDF produce una representación reutilizable
    que distingue:
    - documentos lógicos autónomos;
    - registros internos que NO deben separarse como documentos;
    - relaciones entre documentos lógicos;
    - alcance y límites de identidad;
    - texto representativo y datos clave.

    No recibe catálogo, código, nombre real ni ruta y no decide codificación.
    """
    if modelo not in MODELOS:
        raise OpenAIMultimodalError(f"Modelo no admitido: {modelo}")

    if len(pdf_bytes) >= 50 * 1024 * 1024:
        raise OpenAIMultimodalError(
            f"{documento_alias} supera el límite de 50 MB por archivo."
        )

    prompt = (
        "Analiza este PDF de un expediente de obra pública como un lector "
        "documental experto. NO tienes acceso al catálogo institucional, NO "
        "conoces códigos y NO debes asignar ninguno. No uses ni infieras el "
        "nombre real del archivo ni su ruta.\n\n"
        "OBJETIVO: producir una FICHA DOCUMENTAL CANÓNICA V2 reutilizable por "
        "otros motores sin volver a mirar el PDF.\n\n"
        "DISTINCIÓN CRÍTICA:\n"
        "A) DOCUMENTO LÓGICO: pieza documental autónoma con identidad, función "
        "y efecto propios. Puede abarcar una o varias páginas.\n"
        "B) REGISTRO INTERNO: asiento, nota, concepto, cláusula, partida, fila, "
        "evento o sección que vive dentro de un documento lógico y NO debe "
        "separarse como documento autónomo solo porque tenga número, fecha o "
        "acto propio.\n\n"
        "Ejemplos obligatorios de criterio:\n"
        "- Una hoja o extracto de BITÁCORA que contiene notas 37, 38 y 39 es "
        "UN documento lógico (extracto/hoja de bitácora). Las notas 37, 38 y "
        "39 son REGISTROS INTERNOS, no tres documentos lógicos.\n"
        "- Una ESTIMACIÓN con muchos conceptos sigue siendo un solo documento "
        "lógico; los conceptos son registros internos.\n"
        "- Una póliza con cláusulas, XML o condiciones inseparables sigue "
        "siendo una sola póliza cuando esas piezas forman parte de la misma "
        "garantía.\n"
        "- Una carta de autenticidad con identidad, fecha y función propia SÍ "
        "puede ser un documento lógico separado, pero debe relacionarse como "
        "soporte_de o autentica_a respecto de la póliza correspondiente.\n\n"
        "INSTRUCCIONES:\n"
        "1. Determina si el PDF contiene uno o varios documentos lógicos. "
        "Separa únicamente cuando exista autonomía documental real.\n"
        "2. Para cada documento lógico, indica página inicial y final usando "
        "numeración 1..N del PDF. No inventes fronteras. Si una misma hoja está "
        "fotografiada o escaneada dos veces, no conviertas la repetición en "
        "otro documento lógico; descríbelo en segmentation_uncertainty.\n"
        "3. Identifica cada documento por lo que ES, no por la etapa general "
        "del expediente ni por documentos que mencione.\n"
        "4. Describe formal_function y documented_act.\n"
        "5. Añade identity_scope: qué formaliza/acredita específicamente ESA "
        "pieza. Añade identity_limits: qué NO establece por sí misma cuando "
        "ese límite sea documentalmente relevante. No inventes limitaciones "
        "que no puedan sostenerse en el propio documento.\n"
        "6. Evalúa document_scope como completo, parcial_extracto o "
        "indeterminado. Una selección de páginas/notas tomada de una bitácora "
        "mayor debe ser parcial_extracto aunque las notas visibles estén "
        "completas. La integridad se refiere al tipo documental identificado, "
        "no a toda la etapa del expediente.\n"
        "7. representative_text debe ser una transcripción SELECTIVA y fiel "
        "para comparación posterior. Conserva nombres, números, fechas, "
        "importes, folios, contratos, números de estimación/póliza y frases "
        "distintivas. No inventes texto ilegible y no transcribas todo sin "
        "necesidad.\n"
        "8. key_facts contiene datos breves verificables visibles.\n"
        "9. internal_records contiene registros internos relevantes. No "
        "dupliques como registro lo que ya es un documento lógico. Para una "
        "bitácora, registra cada nota visible; para una estimación puedes "
        "registrar solo hitos relevantes, no cada concepto si no aporta a la "
        "identidad.\n"
        "10. page_markers contiene solo textos breves que ayuden a justificar "
        "inicio, cierre, identidad o fronteras.\n"
        "11. logical_relationships describe relaciones entre documentos "
        "lógicos del MISMO PDF, por ejemplo soporte_de, autentica_a, anexo_de, "
        "complementa_a o relacionado_con. Usa los logical_id exactos.\n"
        "12. reading_quality indica legibilidad. needs_additional_ocr solo es "
        "true si la lectura multimodal no recupera contenido suficiente.\n"
        "13. NO compares con otros archivos, NO decidas duplicados globales y "
        "NO selecciones conceptos del catálogo.\n\n"
        "La prioridad es modelar correctamente la estructura documental, no "
        "maximizar el número de segmentos."
    )

    internal_record_schema = {
        "type": "object",
        "properties": {
            "record_id": {"type": "string"},
            "record_type": {"type": "string"},
            "page_start": {"type": "integer", "minimum": 1},
            "page_end": {"type": "integer", "minimum": 1},
            "label": {"type": "string"},
            "documented_act": {"type": "string"},
            "representative_text": {"type": "string"},
            "key_facts": {
                "type": "array",
                "items": {"type": "string"},
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
            },
        },
        "required": [
            "record_id",
            "record_type",
            "page_start",
            "page_end",
            "label",
            "documented_act",
            "representative_text",
            "key_facts",
            "confidence",
        ],
        "additionalProperties": False,
    }

    logical_document_schema = {
        "type": "object",
        "properties": {
            "logical_id": {"type": "string"},
            "page_start": {"type": "integer", "minimum": 1},
            "page_end": {"type": "integer", "minimum": 1},
            "detected_title": {"type": "string"},
            "formal_function": {"type": "string"},
            "documented_act": {"type": "string"},
            "identity_scope": {"type": "string"},
            "identity_limits": {"type": "string"},
            "document_scope": {
                "type": "string",
                "enum": [
                    "completo",
                    "parcial_extracto",
                    "indeterminado",
                ],
            },
            "representative_text": {"type": "string"},
            "key_facts": {
                "type": "array",
                "items": {"type": "string"},
            },
            "internal_records": {
                "type": "array",
                "items": internal_record_schema,
            },
            "page_markers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer", "minimum": 1},
                        "text": {"type": "string"},
                    },
                    "required": ["page", "text"],
                    "additionalProperties": False,
                },
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
            },
            "evidence": {"type": "string"},
        },
        "required": [
            "logical_id",
            "page_start",
            "page_end",
            "detected_title",
            "formal_function",
            "documented_act",
            "identity_scope",
            "identity_limits",
            "document_scope",
            "representative_text",
            "key_facts",
            "internal_records",
            "page_markers",
            "confidence",
            "evidence",
        ],
        "additionalProperties": False,
    }

    schema = {
        "type": "object",
        "properties": {
            "physical_document": {
                "type": "object",
                "properties": {
                    "page_count_observed": {
                        "type": "integer",
                        "minimum": 1,
                    },
                    "is_compound": {"type": "boolean"},
                    "reading_quality": {
                        "type": "string",
                        "enum": ["alta", "media", "baja"],
                    },
                    "needs_additional_ocr": {"type": "boolean"},
                    "general_summary": {"type": "string"},
                    "segmentation_uncertainty": {"type": "string"},
                },
                "required": [
                    "page_count_observed",
                    "is_compound",
                    "reading_quality",
                    "needs_additional_ocr",
                    "general_summary",
                    "segmentation_uncertainty",
                ],
                "additionalProperties": False,
            },
            "logical_documents": {
                "type": "array",
                "items": logical_document_schema,
            },
            "logical_relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_logical_id": {"type": "string"},
                        "target_logical_id": {"type": "string"},
                        "relation_type": {
                            "type": "string",
                            "enum": [
                                "soporte_de",
                                "autentica_a",
                                "anexo_de",
                                "complementa_a",
                                "relacionado_con",
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
                        "source_logical_id",
                        "target_logical_id",
                        "relation_type",
                        "confidence",
                        "evidence",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "physical_document",
            "logical_documents",
            "logical_relationships",
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
                "name": "canonical_document_profile_v2",
                "strict": True,
                "schema": schema,
            }
        },
        "max_output_tokens": 6000,
    }

    inicio = time.perf_counter()
    respuesta = _request_json(
        "POST",
        "/responses",
        payload,
        timeout=300,
    )
    duracion = time.perf_counter() - inicio

    try:
        resultado = json.loads(_extraer_output_text(respuesta))
    except json.JSONDecodeError as error:
        raise OpenAIMultimodalError(
            "OpenAI devolvió una ficha documental V2 inválida."
        ) from error

    usage = respuesta.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    precios = MODELOS[modelo]
    costo = (
        input_tokens / 1_000_000 * precios["input_per_million"]
        + output_tokens / 1_000_000 * precios["output_per_million"]
    )

    return {
        "physical_document": resultado["physical_document"],
        "logical_documents": resultado["logical_documents"],
        "logical_relationships": resultado["logical_relationships"],
        "model": str(respuesta.get("model") or modelo),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": costo,
        "duration_s": duracion,
    }


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


def validar_equivalencia_documental_multimodal(
    documento_alias: str,
    pdf_bytes: bytes,
    modelo: str,
    concepto_objetivo: str,
    identidad_detectada: str,
) -> dict:
    """
    Valida de forma independiente si la identidad/función documental del
    archivo corresponde al concepto candidato del catálogo.

    Esta etapa NO decide completitud y NO puede seleccionar otro concepto.
    Su propósito es impedir que una similitud temática o de etapa se convierta
    automáticamente en una equivalencia documental.
    """
    if modelo not in MODELOS:
        raise OpenAIMultimodalError(f"Modelo no admitido: {modelo}")

    prompt = (
        "Evalúa exclusivamente la EQUIVALENCIA DOCUMENTAL Y FUNCIONAL entre "
        "este archivo y un concepto candidato del catálogo de obra pública. "
        f"La identidad detectada previamente es: {identidad_detectada}. "
        f"El concepto candidato es: {concepto_objetivo}. "
        "No cambies la identidad detectada, no selecciones otro concepto y no "
        "uses el nombre del archivo ni su ruta.\n\n"
        "IMPORTANTE: en esta etapa NO evalúes si el documento está completo. "
        "Un extracto o algunas páginas pueden seguir perteneciendo al mismo "
        "tipo documental; la integridad se revisará después.\n\n"
        "La equivalencia debe demostrarse por la FUNCIÓN FORMAL del documento, "
        "no por semejanza semántica. Antes de responder, contrasta mentalmente "
        "estos cuatro elementos del archivo contra el concepto candidato:\n"
        "A) identidad documental: qué clase de instrumento o constancia es;\n"
        "B) finalidad formal: para qué acto administrativo existe;\n"
        "C) hecho o acto que acredita: qué suceso formaliza o deja asentado;\n"
        "D) efecto documental: qué acredita, acepta, autoriza, entrega, recibe, "
        "cierra, garantiza o modifica frente al expediente.\n\n"
        "Responde:\n"
        "- equivalente: los elementos A-D son compatibles con el MISMO tipo "
        "documental y la misma función formal del concepto candidato. No basta "
        "con que pertenezcan a la misma etapa.\n"
        "- no_equivalente: existe una diferencia sustantiva en identidad, "
        "finalidad, acto acreditado o efecto documental, aunque ambos archivos "
        "se relacionen con la misma obra o trámite.\n"
        "- indeterminado: la evidencia visible no permite verificar esos "
        "elementos con seguridad razonable.\n\n"
        "Reglas de decisión estrictas:\n"
        "1. La carga de prueba está en la equivalencia: solo usa equivalente "
        "cuando puedas señalar evidencia positiva del mismo acto y función.\n"
        "2. No infieras formalidades que no sean visibles en el archivo. Si el "
        "documento solo acredita una fase material, inspección, constatación, "
        "aviso o preparación de un acto posterior, no lo conviertas en el "
        "documento que formaliza ese acto posterior.\n"
        "3. Compartir palabras, participantes, contrato, fecha, obra, etapa, "
        "firmas o montos NO demuestra por sí mismo equivalencia.\n"
        "4. Para un concepto de ACTA DE ENTREGA-RECEPCIÓN, una simple entrega "
        "física, constatación de terminación o recepción material no basta. "
        "Debe haber evidencia de que el propio documento formaliza el acto de "
        "entrega y recepción/aceptación entre las partes con esa finalidad "
        "administrativa. Si solo documenta la entrega física, usa "
        "no_equivalente.\n"
        "5. Para conceptos presupuestales, distingue la finalidad: referencia, "
        "propuesta, contratado, definitivo o finiquitado no son equivalentes "
        "solo porque contengan conceptos, cantidades y precios.\n"
        "6. Para garantías, distingue el riesgo u obligación garantizada; una "
        "póliza relacionada con la obra no equivale a otra garantía solo por "
        "ser una fianza o póliza.\n"
        "7. Si el archivo es claramente una parte del MISMO tipo documental, "
        "puede ser equivalente aquí; la etapa de integridad decidirá después "
        "si es parcial_extracto.\n"
        "8. Si detectas una diferencia funcional real, usa no_equivalente; "
        "reserva indeterminado para falta de evidencia, no para diferencias "
        "documentales observables.\n\n"
        "En la evidencia explica brevemente QUÉ elemento formal coincide o "
        "difiere. Evita justificar por palabras similares o por pertenecer a "
        "la misma etapa."
    )

    schema = {
        "type": "object",
        "properties": {
            "equivalence": {
                "type": "string",
                "enum": [
                    "equivalente",
                    "no_equivalente",
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
        "required": ["equivalence", "confidence", "evidence"],
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
                "name": "document_functional_equivalence",
                "strict": True,
                "schema": schema,
            }
        },
        "max_output_tokens": 450,
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
            "OpenAI devolvió una respuesta inválida al validar equivalencia."
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
        "Equivalencia funcional": str(resultado["equivalence"]),
        "Confianza equivalencia (%)": round(confianza, 2),
        "Evidencia equivalencia": str(resultado["evidence"]).strip(),
        "Costo equivalencia (USD)": costo,
        "Tiempo equivalencia (s)": duracion,
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
        "La pregunta NO es si aquí está todo el expediente, toda la etapa "
        "administrativa ni todos los documentos relacionados. La pregunta es "
        "si ESTE ARCHIVO constituye por sí mismo una instancia documental "
        "completa del concepto objetivo.\n\n"
        "Clasifica el alcance como:\n"
        "- completo: el archivo tiene inicio y cierre documental suficientes "
        "para constituir una instancia completa del concepto objetivo. Pueden "
        "ser señales de completitud: encabezado o identificación del documento, "
        "desarrollo coherente, numeración o continuidad de páginas, totales o "
        "conclusiones cuando correspondan y firmas/cierres cuando sean propios "
        "del tipo documental. La existencia de otros documentos relacionados "
        "en el expediente NO vuelve parcial a este archivo.\n"
        "- parcial_extracto: el archivo contiene solamente algunas hojas, "
        "notas, páginas o una sección desprendida de un documento mayor del "
        "MISMO concepto. Debe existir evidencia positiva de que faltan partes "
        "del propio documento, no solo de que existan otros documentos de la "
        "misma etapa. Por ejemplo, algunas notas de bitácora incluidas como "
        "soporte de una estimación no constituyen la Bitácora de obra completa.\n"
        "- indeterminado: con el material visible no puede establecerse "
        "razonablemente si el propio documento está completo.\n\n"
        "Reglas críticas:\n"
        "1. No confundas 'documento relacionado con otros' con 'documento "
        "incompleto'.\n"
        "2. Un cuadro, hoja-resumen o finiquito de varias páginas puede ser una "
        "instancia completa si integra su contenido, llega a totales/cierre y "
        "presenta las formalidades propias del documento, aunque existan actas, "
        "garantías u otros documentos de cierre por separado.\n"
        "3. Usa parcial_extracto solo cuando haya evidencia de fragmentación "
        "del propio documento objetivo.\n"
        "4. No decidas por el número de páginas; decide por estructura, "
        "continuidad y cierre documental."
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



def comparar_candidatos_unidad_multimodal(
    candidatos: list[dict],
    modelo: str,
    tipo_unidad: str,
    consecutivo: int,
) -> dict:
    """
    Compara simultáneamente los archivos que la primera etapa relacionó con
    el código de la propia unidad.

    Esta etapa no clasifica contra el catálogo ni puede crear códigos. Solo
    distribuye funciones dentro de la unidad: representante, componente,
    soporte o indeterminado.
    """
    if modelo not in MODELOS:
        raise OpenAIMultimodalError(f"Modelo no admitido: {modelo}")

    if not candidatos:
        raise OpenAIMultimodalError(
            "No hay candidatos para comparar dentro de la unidad."
        )

    aliases = [str(item["alias"]) for item in candidatos]
    if len(set(aliases)) != len(aliases):
        raise OpenAIMultimodalError(
            "Los alias de comparación deben ser únicos."
        )

    resumenes = []
    contenido = []

    for item in candidatos:
        alias = str(item["alias"])
        pdf_bytes = item["pdf_bytes"]

        if len(pdf_bytes) >= 50 * 1024 * 1024:
            raise OpenAIMultimodalError(
                f"{alias} supera el límite de 50 MB por archivo."
            )

        resumenes.append(
            (
                f"{alias}: título detectado={item.get('titulo', '')}; "
                f"clasificación inicial={item.get('clasificacion', '')}; "
                f"evidencia inicial={item.get('evidencia', '')}"
            )
        )

        encoded = base64.b64encode(pdf_bytes).decode("ascii")
        contenido.append(
            {
                "type": "input_file",
                "filename": f"{alias}.pdf",
                "file_data": (
                    "data:application/pdf;base64," + encoded
                ),
                "detail": "high",
            }
        )

    prompt = (
        "Compara CONJUNTAMENTE estos archivos, todos candidatos a estar "
        f"relacionados con una unidad documental de tipo {tipo_unidad}, "
        f"consecutivo {int(consecutivo)}. Los nombres candidate_XX son alias "
        "neutros y no contienen información documental. No vuelvas a "
        "clasificar contra ningún catálogo y no inventes códigos.\n\n"
        "Tu tarea es distribuir funciones DOCUMENTALES relativas entre los "
        "candidatos. Evalúalos entre sí, no de forma aislada.\n\n"
        "Relaciones permitidas:\n"
        "- representante_unidad: pieza que mejor identifica formalmente la "
        "unidad completa cuando esta está fragmentada. Suele ser portada, "
        "carátula o resumen formal que concentra número de estimación, periodo, "
        "importes, avances, contrato y firmas. Debe haber COMO MÁXIMO uno.\n"
        "- componente_unidad: contiene el cuerpo sustantivo propio de la "
        "estimación, por ejemplo relación de conceptos, cantidades, precios, "
        "importes o desglose de la estimación, pero no es la pieza que mejor "
        "identifica formalmente la unidad.\n"
        "- soporte: documento auxiliar relacionado con la estimación pero con "
        "función distinta. Croquis, planos de apoyo, plantillas de importación, "
        "listas de verificación, facturas, generadores, fotografías y "
        "solicitudes de pago normalmente son soporte salvo evidencia clara de "
        "que constituyen el cuerpo sustantivo de la estimación.\n"
        "- indeterminado: evidencia insuficiente.\n\n"
        "Reglas críticas:\n"
        "1. No confundas 'habla de la estimación' con 'es la estimación'.\n"
        "2. Si hay una carátula/resumen formal y también una hoja con el cuerpo "
        "de conceptos/importes, normalmente la primera es representante y la "
        "segunda componente.\n"
        "3. Un croquis con cantidades o referencias a conceptos sigue siendo "
        "croquis de soporte si su función principal es gráfica/ubicacional.\n"
        "4. Si ningún archivo merece claramente ser representante, devuelve "
        "representative_alias='ninguno'.\n"
        "5. No modifiques la identidad inicial; solo asigna el papel relativo "
        "dentro de esta unidad.\n\n"
        "Información previa de cada candidato:\n"
        + "\n".join(resumenes)
    )
    contenido.append({"type": "input_text", "text": prompt})

    schema = {
        "type": "object",
        "properties": {
            "representative_alias": {
                "type": "string",
                "enum": aliases + ["ninguno"],
            },
            "decisions": {
                "type": "array",
                "minItems": len(aliases),
                "maxItems": len(aliases),
                "items": {
                    "type": "object",
                    "properties": {
                        "alias": {
                            "type": "string",
                            "enum": aliases,
                        },
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
                        "alias",
                        "unit_relation",
                        "confidence",
                        "evidence",
                    ],
                    "additionalProperties": False,
                },
            },
            "unit_evidence": {"type": "string"},
        },
        "required": [
            "representative_alias",
            "decisions",
            "unit_evidence",
        ],
        "additionalProperties": False,
    }

    payload = {
        "model": modelo,
        "reasoning": {"effort": "low"},
        "input": [
            {
                "role": "user",
                "content": contenido,
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "comparative_unit_consolidation",
                "strict": True,
                "schema": schema,
            }
        },
        "max_output_tokens": 900,
    }

    inicio = time.perf_counter()
    respuesta = _request_json(
        "POST",
        "/responses",
        payload,
        timeout=300,
    )
    duracion = time.perf_counter() - inicio

    try:
        resultado = json.loads(_extraer_output_text(respuesta))
    except json.JSONDecodeError as error:
        raise OpenAIMultimodalError(
            "OpenAI devolvió una comparación de unidad inválida."
        ) from error

    decisiones = resultado.get("decisions") or []
    aliases_devuelto = [str(item.get("alias", "")) for item in decisiones]

    if (
        len(decisiones) != len(aliases)
        or set(aliases_devuelto) != set(aliases)
        or len(set(aliases_devuelto)) != len(aliases_devuelto)
    ):
        raise OpenAIMultimodalError(
            "La comparación no devolvió exactamente una decisión por candidato."
        )

    representante = str(resultado["representative_alias"])
    representantes_decision = [
        str(item["alias"])
        for item in decisiones
        if str(item["unit_relation"]) == "representante_unidad"
    ]

    if representante == "ninguno":
        if representantes_decision:
            raise OpenAIMultimodalError(
                "La comparación marcó representante aunque indicó 'ninguno'."
            )
    else:
        if representantes_decision != [representante]:
            raise OpenAIMultimodalError(
                "La comparación no produjo un único representante coherente."
            )

    usage = respuesta.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)

    precios = MODELOS[modelo]
    costo = (
        input_tokens / 1_000_000 * precios["input_per_million"]
        + output_tokens / 1_000_000 * precios["output_per_million"]
    )

    decisiones_normalizadas = []
    for item in decisiones:
        confianza = float(item["confidence"])
        if 0 <= confianza <= 1:
            confianza *= 100

        decisiones_normalizadas.append(
            {
                "Alias": str(item["alias"]),
                "Relación comparativa": str(item["unit_relation"]),
                "Confianza comparativa (%)": round(confianza, 2),
                "Evidencia comparativa": str(item["evidence"]).strip(),
            }
        )

    return {
        "Representante": representante,
        "Decisiones": decisiones_normalizadas,
        "Evidencia unidad": str(resultado["unit_evidence"]).strip(),
        "Costo comparación (USD)": costo,
        "Tiempo comparación (s)": duracion,
        "Modelo comparación": str(respuesta.get("model") or modelo),
    }
