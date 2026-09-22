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
        "Clasifica este documento de auditoría de obra pública usando "
        "exclusivamente su contenido visual y textual. No uses ni infieras "
        "información de nombres de archivo o rutas. Selecciona exactamente "
        "una opción del catálogo. Usa fuera_catalogo solo si ninguna opción "
        "corresponde razonablemente.\n\n"
        "Opciones permitidas:\n"
        + "\n".join(opciones)
    )

    schema = {
        "type": "object",
        "properties": {
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

    return {
        "Documento": documento_alias,
        "Resultado Ruta B": mapa[opcion]["concepto"],
        "Código Ruta B": mapa[opcion]["codigo"],
        "Confianza B": round(float(resultado["confidence"]), 2),
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
