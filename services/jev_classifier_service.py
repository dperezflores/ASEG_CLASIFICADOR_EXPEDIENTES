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
UMBRAL_EQUIVALENCIA_JEV = 0.85
MARGEN_EQUIVALENCIA_JEV = 0.15


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
                    "A match requires documentary and functional equivalence, "
                    "not merely similar words, subject matter, project phase "
                    "or administrative context. Do not choose the nearest "
                    "option just because it is similar. Choose fuera_catalogo "
                    "when the document has a different function or is only "
                    "supporting documentation for a listed document type."
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



def validar_equivalencia_identidad_con_jev(
    documento_alias: str,
    identidad_documental: str,
    funcion_formal: str,
    acto_documentado: str,
    concepto_objetivo: str,
) -> dict:
    """
    Compara texto contra texto: identidad pura vs. concepto de catálogo.

    No recibe PDF, nombre real ni ruta. Si Jev no alcanza suficiente seguridad,
    fuerza "indeterminado" para que el flujo pueda escalar al multimodal.
    """
    identidad_documental = str(identidad_documental).strip()
    funcion_formal = str(funcion_formal).strip()
    acto_documentado = str(acto_documentado).strip()
    concepto_objetivo = str(concepto_objetivo).strip()

    if not identidad_documental:
        raise JevError(
            f"{documento_alias} no contiene identidad documental utilizable."
        )
    if not concepto_objetivo:
        raise JevError(
            f"{documento_alias} no contiene concepto objetivo utilizable."
        )

    texto_comparacion = (
        f"IDENTIDAD DOCUMENTAL FIJA:\n{identidad_documental}\n\n"
        f"FUNCIÓN FORMAL:\n{funcion_formal}\n\n"
        f"ACTO DOCUMENTADO:\n{acto_documentado}\n\n"
        f"CONCEPTO CANDIDATO DEL CATÁLOGO:\n{concepto_objetivo}"
    )

    criterios = {
        "equivalente": (
            "La identidad documental fija corresponde al mismo tipo de "
            "documento y a la misma función formal que el concepto candidato. "
            "Coinciden sustancialmente la finalidad, el acto acreditado y el "
            "efecto documental."
        ),
        "no_equivalente": (
            "La identidad documental fija y el concepto candidato son "
            "documentos distintos o cumplen funciones formales distintas, "
            "aunque puedan pertenecer a la misma obra, etapa o trámite."
        ),
        "indeterminado": (
            "La información textual disponible no permite decidir con "
            "seguridad razonable si existe equivalencia documental-funcional."
        ),
    }

    payload = {
        "model": MODEL,
        "state": {
            "document_id": documento_alias,
            "document_text": texto_comparacion,
        },
        "questions": {
            "document_equivalence": {
                "type": "choice",
                "instructions": (
                    "Compare ONLY the fixed documentary identity against the "
                    "catalog candidate. Do not reinterpret the document and do "
                    "not infer from filenames, paths, project phase, dates, "
                    "participants or shared words. Equivalence requires the "
                    "same documentary type and formal function. A receipt that "
                    "mentions a final estimate is not a finiquito merely for "
                    "that reason. A physical handover/constatation is not an "
                    "administrative entrega-recepcion act unless the fixed "
                    "identity itself says it formalizes that act. Different "
                    "budget purposes and different guarantee objects are not "
                    "equivalent. Choose indeterminado when evidence is "
                    "insufficient rather than forcing the nearest option."
                ),
                "criteria": criterios,
            }
        },
    }

    inicio = time.perf_counter()
    respuesta = _request_json("POST", "/v1/systemone", payload)
    duracion = time.perf_counter() - inicio

    answer = respuesta.get("answers", {}).get(
        "document_equivalence",
        {},
    )
    eleccion = str(answer.get("choice") or "").strip()

    if eleccion not in criterios:
        raise JevError(
            "La respuesta de Jev no contiene una decisión de equivalencia válida."
        )

    probabilidades = answer.get("probabilities") or {}
    confianza = float(answer.get("confidence") or 0.0)
    prob_elegida = float(probabilidades.get(eleccion, 0.0) or 0.0)

    ranking = sorted(
        (
            (str(opcion), float(probabilidad))
            for opcion, probabilidad in probabilidades.items()
            if str(opcion) in criterios
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    margen = None
    if len(ranking) >= 2:
        margen = ranking[0][1] - ranking[1][1]

    seguridad = max(confianza, prob_elegida)
    decision_original = eleccion
    motivo_control = ""

    if eleccion != "indeterminado":
        if seguridad < UMBRAL_EQUIVALENCIA_JEV:
            eleccion = "indeterminado"
            motivo_control = (
                "Jev no alcanzó el umbral mínimo de seguridad "
                f"({seguridad:.3f} < {UMBRAL_EQUIVALENCIA_JEV:.2f})."
            )
        elif (
            margen is not None
            and margen < MARGEN_EQUIVALENCIA_JEV
        ):
            eleccion = "indeterminado"
            motivo_control = (
                "La diferencia entre las dos decisiones más probables fue "
                f"insuficiente ({margen:.3f} < "
                f"{MARGEN_EQUIVALENCIA_JEV:.2f})."
            )

    usage = respuesta.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    costo = input_tokens / 1_000_000 * PRECIO_USD_POR_MILLON_TOKENS

    return {
        "Equivalencia JEV": eleccion,
        "Decisión original JEV": decision_original,
        "Confianza JEV (%)": round(confianza * 100, 2),
        "Probabilidad elegida JEV (%)": round(
            prob_elegida * 100,
            2,
        ),
        "Margen JEV (%)": (
            round(margen * 100, 2)
            if margen is not None
            else None
        ),
        "Control JEV": motivo_control,
        "Modelo JEV": str(respuesta.get("model") or MODEL),
        "Tokens entrada JEV": input_tokens,
        "Tokens salida JEV": output_tokens,
        "Costo JEV (USD)": costo,
        "Tiempo JEV (s)": duracion,
    }



def validar_equivalencia_ficha_con_jev(
    documento_alias: str,
    identidad_documental: str,
    funcion_formal: str,
    acto_documentado: str,
    alcance_identidad: str,
    limites_identidad: str,
    alcance_documental: str,
    concepto_objetivo: str,
) -> dict:
    """
    Validación textual estricta de una Ficha V2 contra UN concepto candidato.

    No recibe PDF, nombre real, ruta ni catálogo completo. JEV no puede elegir
    otro concepto: únicamente decide si la identidad ya fijada por la ficha es
    documental y funcionalmente equivalente al candidato propuesto.
    """
    identidad_documental = str(identidad_documental).strip()
    concepto_objetivo = str(concepto_objetivo).strip()

    if not identidad_documental:
        raise JevError(
            f"{documento_alias} no contiene identidad documental utilizable."
        )
    if not concepto_objetivo:
        raise JevError(
            f"{documento_alias} no contiene concepto objetivo utilizable."
        )

    texto_comparacion = (
        f"IDENTIDAD DOCUMENTAL FIJA:\n{identidad_documental}\n\n"
        f"FUNCIÓN FORMAL FIJA:\n{str(funcion_formal).strip()}\n\n"
        f"ACTO DOCUMENTADO FIJO:\n{str(acto_documentado).strip()}\n\n"
        f"ALCANCE DE IDENTIDAD FIJO:\n{str(alcance_identidad).strip()}\n\n"
        f"LÍMITES DE IDENTIDAD FIJOS:\n{str(limites_identidad).strip()}\n\n"
        f"ALCANCE DOCUMENTAL FIJO:\n{str(alcance_documental).strip()}\n\n"
        f"ÚNICO CONCEPTO CANDIDATO A VALIDAR:\n{concepto_objetivo}"
    )

    criterios = {
        "equivalente": (
            "La identidad fija corresponde al MISMO tipo documental y a la "
            "MISMA función formal que el concepto candidato. El alcance y los "
            "efectos documentales son compatibles y los límites de identidad "
            "no contradicen el concepto candidato."
        ),
        "no_equivalente": (
            "La identidad fija corresponde a otro tipo documental, a una pieza "
            "de soporte o a una función formal distinta. También aplica cuando "
            "el concepto candidato exige un alcance o efecto documental más "
            "amplio que la identidad fija y sus límites permiten."
        ),
        "indeterminado": (
            "La ficha textual no aporta evidencia suficiente para confirmar o "
            "rechazar equivalencia sin reinterpretar el documento."
        ),
    }

    payload = {
        "model": MODEL,
        "state": {
            "document_id": documento_alias,
            "document_text": texto_comparacion,
        },
        "questions": {
            "strict_document_equivalence": {
                "type": "choice",
                "instructions": (
                    "Validate ONLY whether the FIXED documentary identity is "
                    "truly equivalent to the ONE candidate concept. Do not "
                    "reinterpret the identity, do not search for a nearest "
                    "catalog option, and do not infer from filenames, paths, "
                    "shared parties, dates, amounts or project phase. The burden "
                    "of proof is positive: equivalente requires the same "
                    "documentary type, formal purpose and documentary effect. "
                    "Respect identity_limits as hard evidence against broader "
                    "interpretations. A supporting authenticity letter is not "
                    "the guarantee/policy it authenticates. A payment receipt "
                    "is not a finiquito. A physical handover/receipt act is not "
                    "automatically the formal administrative entrega-recepcion "
                    "act: if the fixed identity is specifically 'acta de "
                    "entrega fisica' or limits itself to material/physical "
                    "handover, choose no_equivalente for a broader 'acta de "
                    "entrega-recepcion' unless the fixed identity itself "
                    "explicitly establishes that formal documentary act. "
                    "Choose indeterminado when the fixed ficha is genuinely "
                    "insufficient rather than stretching the candidate label."
                ),
                "criteria": criterios,
            }
        },
    }

    inicio = time.perf_counter()
    respuesta = _request_json("POST", "/v1/systemone", payload)
    duracion = time.perf_counter() - inicio

    answer = respuesta.get("answers", {}).get(
        "strict_document_equivalence",
        {},
    )
    eleccion = str(answer.get("choice") or "").strip()

    if eleccion not in criterios:
        raise JevError(
            "La respuesta de Jev no contiene una validación estricta válida."
        )

    probabilidades = answer.get("probabilities") or {}
    confianza = float(answer.get("confidence") or 0.0)
    prob_elegida = float(
        probabilidades.get(eleccion, 0.0) or 0.0
    )

    ranking = sorted(
        (
            (str(opcion), float(probabilidad))
            for opcion, probabilidad in probabilidades.items()
            if str(opcion) in criterios
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    margen = None
    if len(ranking) >= 2:
        margen = ranking[0][1] - ranking[1][1]

    seguridad = max(confianza, prob_elegida)
    decision_original = eleccion
    motivo_control = ""

    if eleccion != "indeterminado":
        if seguridad < UMBRAL_EQUIVALENCIA_JEV:
            eleccion = "indeterminado"
            motivo_control = (
                "Jev no alcanzó el umbral mínimo de seguridad "
                f"({seguridad:.3f} < {UMBRAL_EQUIVALENCIA_JEV:.2f})."
            )
        elif (
            margen is not None
            and margen < MARGEN_EQUIVALENCIA_JEV
        ):
            eleccion = "indeterminado"
            motivo_control = (
                "La diferencia entre las dos decisiones más probables fue "
                f"insuficiente ({margen:.3f} < "
                f"{MARGEN_EQUIVALENCIA_JEV:.2f})."
            )

    usage = respuesta.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    costo = (
        input_tokens
        / 1_000_000
        * PRECIO_USD_POR_MILLON_TOKENS
    )

    return {
        "Equivalencia estricta JEV": eleccion,
        "Decisión original estricta JEV": decision_original,
        "Confianza estricta JEV (%)": round(
            confianza * 100,
            2,
        ),
        "Probabilidad elegida estricta JEV (%)": round(
            prob_elegida * 100,
            2,
        ),
        "Margen estricta JEV (%)": (
            round(margen * 100, 2)
            if margen is not None
            else None
        ),
        "Control estricta JEV": motivo_control,
        "Modelo estricta JEV": str(
            respuesta.get("model") or MODEL
        ),
        "Tokens entrada estricta JEV": input_tokens,
        "Tokens salida estricta JEV": output_tokens,
        "Costo estricta JEV (USD)": costo,
        "Tiempo estricta JEV (s)": duracion,
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
