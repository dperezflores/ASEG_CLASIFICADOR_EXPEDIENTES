from __future__ import annotations

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st


BASE_URL = "https://api.typesafe.ai"
MODEL = "jev-latest"
PRECIO_USD_POR_MILLON_TOKENS = 0.042


class JevError(RuntimeError):
    pass


def jev_configurado() -> bool:
    try:
        return bool(str(st.secrets["typesafe"]["api_key"]).strip())
    except Exception:
        return False


def _api_key() -> str:
    if not jev_configurado():
        raise JevError(
            "No se encontró la API key de TypeSafe en Streamlit Secrets."
        )
    return str(st.secrets["typesafe"]["api_key"]).strip()


def _request_json(method: str, path: str, payload: dict | None = None) -> dict:
    body = None
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = Request(
        f"{BASE_URL}{path}",
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detalle = error.read().decode("utf-8", errors="replace")
        raise JevError(
            f"TypeSafe respondió HTTP {error.code}: {detalle}"
        ) from error
    except URLError as error:
        raise JevError(
            f"No fue posible conectar con TypeSafe: {error.reason}"
        ) from error


def probar_conexion_jev() -> list[str]:
    respuesta = _request_json("GET", "/v1/models")
    modelos = respuesta.get("models", respuesta)

    if isinstance(modelos, list):
        salida = []
        for modelo in modelos:
            if isinstance(modelo, dict):
                salida.append(
                    str(
                        modelo.get("id")
                        or modelo.get("name")
                        or modelo.get("model")
                        or modelo
                    )
                )
            else:
                salida.append(str(modelo))
        return salida

    return [str(modelos)]


def _crear_criterios(catalogo: pd.DataFrame) -> tuple[dict, dict]:
    """
    Convierte los conceptos oficiales en opciones opacas.

    Jev ve las descripciones de los conceptos, pero no los códigos oficiales
    ni el ground truth del documento.
    """
    criterios = {}
    mapa = {}

    for indice, fila in catalogo.reset_index(drop=True).iterrows():
        etiqueta = f"opcion_{indice + 1:03d}"
        concepto = str(fila["Concepto"]).strip()
        codigo = str(fila["Código"]).strip()

        criterios[etiqueta] = concepto
        mapa[etiqueta] = {
            "concepto": concepto,
            "codigo": codigo,
        }

    # Evita forzar una clasificación cuando el documento no encaja.
    criterios["fuera_catalogo"] = (
        "El documento no corresponde de forma razonable a ninguno de los "
        "conceptos anteriores."
    )
    mapa["fuera_catalogo"] = {
        "concepto": "Fuera de catálogo / no identificado",
        "codigo": "",
    }

    return criterios, mapa


def clasificar_texto_con_jev(
    documento_alias: str,
    texto_ocr: str,
    catalogo: pd.DataFrame,
) -> dict:
    texto_ocr = str(texto_ocr).strip()

    if not texto_ocr:
        raise JevError(
            f"{documento_alias} no contiene texto OCR utilizable."
        )

    criterios, mapa = _crear_criterios(catalogo)

    payload = {
        "model": MODEL,
        "state": {
            "document_id": documento_alias,
            "document_text": texto_ocr,
        },
        "questions": {
            "document_type": {
                "type": "choice",
                "instructions": (
                    "Classify this public-works audit document into exactly "
                    "one of the supplied document types. Use only the document "
                    "content. Do not infer from filenames or folder paths. "
                    "Choose fuera_catalogo only when none of the listed "
                    "document types reasonably matches."
                ),
                "criteria": criterios,
            }
        },
    }

    inicio = time.perf_counter()
    respuesta = _request_json("POST", "/v1/systemone", payload)
    duracion = time.perf_counter() - inicio

    answer = respuesta.get("answers", {}).get("document_type", {})
    eleccion = answer.get("choice")

    if not eleccion or eleccion not in mapa:
        raise JevError(
            "La respuesta de Jev no contiene una opción de clasificación válida."
        )

    probabilidades = answer.get("probabilities") or {}
    confianza = float(answer.get("confidence") or 0.0)

    ranking = sorted(
        (
            {
                "opcion": opcion,
                "concepto": mapa.get(opcion, {}).get("concepto", opcion),
                "probabilidad": float(probabilidad),
            }
            for opcion, probabilidad in probabilidades.items()
        ),
        key=lambda item: item["probabilidad"],
        reverse=True,
    )

    usage = respuesta.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    costo = input_tokens / 1_000_000 * PRECIO_USD_POR_MILLON_TOKENS

    return {
        "Documento": documento_alias,
        "Resultado Ruta A": mapa[eleccion]["concepto"],
        "Código Ruta A": mapa[eleccion]["codigo"],
        "Confianza A": round(confianza * 100, 2),
        "Probabilidad elegida (%)": round(
            float(probabilidades.get(eleccion, 0.0)) * 100,
            2,
        ),
        "Top 3": ranking[:3],
        "Modelo A": str(respuesta.get("model") or MODEL),
        "Tokens entrada A": input_tokens,
        "Tokens salida A": output_tokens,
        "Costo A (USD)": costo,
        "Tiempo A (s)": duracion,
    }


def clasificar_muestra_con_jev(
    muestra: pd.DataFrame,
    catalogo: pd.DataFrame,
) -> pd.DataFrame:
    resultados = []

    for _, fila in muestra.iterrows():
        resultado = clasificar_texto_con_jev(
            documento_alias=str(fila["Documento"]),
            texto_ocr=str(fila["Texto OCR"]),
            catalogo=catalogo,
        )

        concepto_real = str(fila["Concepto real"])
        resultado["Concepto real"] = concepto_real
        resultado["Acierto A"] = (
            resultado["Resultado Ruta A"].strip().casefold()
            == concepto_real.strip().casefold()
        )
        resultados.append(resultado)

    return pd.DataFrame(resultados)
