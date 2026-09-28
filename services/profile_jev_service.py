from __future__ import annotations

from pathlib import PurePosixPath
import re
import unicodedata

import pandas as pd

from services.catalog_family_service import (
    construir_catalogo_operativo_estimacion,
)
from services.jev_classifier_service import (
    clasificar_texto_con_jev,
    validar_equivalencia_ficha_con_jev,
)
from services.structural_analysis_service import (
    construir_mapa_estructural,
)


MAX_CARACTERES_FICHA_JEV = 18_000
PIPELINE_VERSION_FICHA_JEV = 6

UMBRAL_CANDIDATO_REFUERZO_ESTRUCTURAL = 95.0
UMBRAL_RELACION_REFUERZO_ESTRUCTURAL = 90.0
UMBRAL_SOPORTE_REFUERZO_ESTRUCTURAL = 80.0
RELACIONES_REFUERZO_ESTRUCTURAL = {
    "soporte_de",
    "autentica_a",
    "anexo_de",
    "complementa_a",
}


CODIGOS_COMPETENCIA_INDETERMINADA = {
    "ETR_LSI_ETR",
    "ETR_LSI_FIN",
    "ETR_LSI_GVO",
    "CNT_LSI_PTC",
}
CONFIANZA_MIN_GANADOR_COMPETENCIA = 90.0
DIFERENCIA_MIN_COMPETENCIA = 10.0

EXCLUSIONES_IDENTIDAD_POR_CODIGO = {
    "ETR_LSI_ETR": {
        "marcadores_exclusion": (
            "acta de entrega fisica",
            "entrega fisica de obra",
            "recepcion de planos",
            "control de fechas para entrega de planos",
        ),
        "marcadores_formales_titulo": (
            "acta de entrega recepcion",
            "entrega recepcion final",
        ),
    },
}


# Conceptos cuyo propósito no puede inferirse solo por similitud de tipo
# documental. La primera regla controlada corresponde a Factura(s) de anticipo.
MARCADORES_IDENTIDAD_CUERPO_ESTIMACION = (
    "caratula de estimacion",
    "hoja de estimacion",
    "estimacion constructor de obra",
)


PROPOSITOS_SENSIBLES = {
    "factura_anticipo": {
        "concepto_requiere": ("factura", "anticipo"),
        "evidencia_positiva": (
            "factura de anticipo",
            "factura por anticipo",
            "pago de anticipo",
            "por concepto de anticipo",
            "anticipo contractual",
            "porcentaje de anticipo",
            "amortizacion de anticipo",
            "amortizacion del anticipo",
        ),
        "evidencia_contextual_estimacion": (
            "recibo por pago de estimacion",
            "pago de estimacion",
        ),
    },
}


def _normalizar_texto_regla(valor: str) -> str:
    texto = unicodedata.normalize(
        "NFKD",
        str(valor),
    )
    texto = "".join(
        caracter
        for caracter in texto
        if not unicodedata.combining(caracter)
    )
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def _regla_proposito_sensible(
    detalle: pd.DataFrame,
    fila: pd.Series,
) -> dict:
    """
    Valida conceptos cuyo propósito debe estar expresamente acreditado.

    Resultado:
    - aplica=False: no es un concepto sensible;
    - acreditado=True: puede continuar a equivalencia estricta;
    - acreditado=False + soporte_contextual=True: no asignar código y tratar
      como soporte contextual de la estimación;
    - acreditado=False sin soporte contextual: revisión.
    """
    concepto = _normalizar_texto_regla(
        fila.get("Concepto JEV", "")
    )

    regla_id = None
    regla = None
    for identificador, candidata in (
        PROPOSITOS_SENSIBLES.items()
    ):
        if all(
            termino in concepto
            for termino in candidata[
                "concepto_requiere"
            ]
        ):
            regla_id = identificador
            regla = candidata
            break

    if regla is None:
        return {
            "aplica": False,
            "acreditado": True,
            "soporte_contextual": False,
            "evidencia": "",
            "regla": "",
        }

    texto_propio = _normalizar_texto_regla(
        " ".join(
            [
                str(
                    fila.get(
                        "Título detectado",
                        "",
                    )
                ),
                str(
                    fila.get(
                        "Función formal",
                        "",
                    )
                ),
                str(
                    fila.get(
                        "Acto documentado",
                        "",
                    )
                ),
                str(
                    fila.get(
                        "Alcance de identidad",
                        "",
                    )
                ),
                str(
                    fila.get(
                        "Texto representativo ficha",
                        "",
                    )
                ),
                str(
                    fila.get(
                        "Datos clave ficha",
                        "",
                    )
                ),
            ]
        )
    )

    coincidencias_positivas = [
        frase
        for frase in regla["evidencia_positiva"]
        if _normalizar_texto_regla(
            frase
        ) in texto_propio
    ]

    if coincidencias_positivas:
        return {
            "aplica": True,
            "acreditado": True,
            "soporte_contextual": False,
            "evidencia": (
                "Propósito acreditado en la Ficha V2: "
                + ", ".join(
                    coincidencias_positivas
                )
            ),
            "regla": regla_id,
        }

    ruta = str(
        fila.get("Ruta original", "")
    )
    mismo_archivo = detalle[
        detalle["Ruta original"]
        .astype(str)
        .eq(ruta)
    ]

    texto_contextual = _normalizar_texto_regla(
        " ".join(
            mismo_archivo[
                "Título detectado"
            ].fillna("").astype(str).tolist()
            + mismo_archivo[
                "Función formal"
            ].fillna("").astype(str).tolist()
            + mismo_archivo[
                "Acto documentado"
            ].fillna("").astype(str).tolist()
            + mismo_archivo[
                "Texto representativo ficha"
            ].fillna("").astype(str).tolist()
        )
    )

    coincidencias_contextuales = [
        frase
        for frase
        in regla["evidencia_contextual_estimacion"]
        if _normalizar_texto_regla(
            frase
        ) in texto_contextual
    ]

    soporte_contextual = bool(
        coincidencias_contextuales
        and str(
            fila.get("Unidad estructural", "")
        )
        == "Estimación"
    )

    evidencia = (
        "No se encontró evidencia positiva del propósito "
        f"'{regla_id}' en la Ficha V2."
    )
    if coincidencias_contextuales:
        evidencia += (
            " El mismo PDF contiene evidencia de pago de "
            "estimación: "
            + ", ".join(
                coincidencias_contextuales
            )
            + "."
        )

    return {
        "aplica": True,
        "acreditado": False,
        "soporte_contextual": soporte_contextual,
        "evidencia": evidencia,
        "regla": regla_id,
    }


def _codigo_sin_extension(codigo: str) -> str:
    valor = str(codigo).strip()
    if valor.lower().endswith(".pdf"):
        return valor[:-4]
    return valor


def _evaluar_exclusion_identidad_candidato(
    fila: pd.Series,
    codigo_candidato: str,
) -> dict:
    """
    Excluye candidatos cuya propia identidad documental contradice de forma
    explícita el concepto formal propuesto.

    Primera regla controlada:
    ETR_LSI_ETR no debe absorber una entrega física ni una recepción de planos.
    """
    codigo_base = _codigo_sin_extension(
        codigo_candidato
    ).upper()
    regla = EXCLUSIONES_IDENTIDAD_POR_CODIGO.get(
        codigo_base
    )
    if not regla:
        return {
            "aplica": False,
            "regla": "",
            "evidencia": "",
        }

    titulo = _normalizar_texto_regla(
        fila.get("Título detectado", "")
    )
    acto = _normalizar_texto_regla(
        fila.get("Acto documentado", "")
    )
    funcion = _normalizar_texto_regla(
        fila.get("Función formal", "")
    )
    alcance = _normalizar_texto_regla(
        fila.get("Alcance de identidad", "")
    )

    identidad = " ".join(
        item
        for item in (
            titulo,
            acto,
            funcion,
            alcance,
        )
        if item
    )

    formal_en_titulo = any(
        marcador in titulo
        for marcador
        in regla["marcadores_formales_titulo"]
    )
    if formal_en_titulo:
        return {
            "aplica": False,
            "regla": "",
            "evidencia": "",
        }

    marcador = next(
        (
            item
            for item
            in regla["marcadores_exclusion"]
            if item in identidad
        ),
        None,
    )
    if not marcador:
        return {
            "aplica": False,
            "regla": "",
            "evidencia": "",
        }

    return {
        "aplica": True,
        "regla": "identidad_excluye_concepto_formal",
        "evidencia": (
            f"La identidad documental contiene el marcador '{marcador}', "
            f"que describe un acto distinto del concepto formal "
            f"{codigo_candidato}. La exclusión se basa en la Ficha V2, "
            "no en el nombre físico del archivo."
        ),
    }


def _resolver_competencia_indeterminada(
    detalle: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resuelve competencia entre varios candidatos INDETERMINADOS al mismo
    código de representación única.

    Solo actúa si:
    - no existe ya un candidato validado para ese código;
    - hay al menos dos candidatos indeterminados;
    - el mejor alcanza >= 90%;
    - supera al segundo por >= 10 pp.

    Usa la confianza inicial de JEV y no realiza nuevas llamadas.
    """
    salida = detalle.copy()

    for columna, valor in (
        ("Competencia indeterminada aplicada", False),
        ("Resolución competencia indeterminada", "No aplica"),
        ("Confianza competencia (%)", 0.0),
        ("Diferencia competencia (%)", 0.0),
        ("Ganador competencia", ""),
        ("Evidencia competencia", ""),
    ):
        if columna not in salida.columns:
            salida[columna] = valor

    if salida.empty:
        return salida

    codigos_base = (
        salida["Código JEV"]
        .fillna("")
        .astype(str)
        .apply(_codigo_sin_extension)
        .str.upper()
    )

    for codigo in CODIGOS_COMPETENCIA_INDETERMINADA:
        bloque_codigo = salida[
            codigos_base.eq(codigo)
        ].copy()

        if bloque_codigo.empty:
            continue

        ya_validado = bloque_codigo[
            bloque_codigo["Decisión provisional"]
            .astype(str)
            .eq("Código propio candidato validado")
        ]
        if not ya_validado.empty:
            continue

        candidatos = bloque_codigo[
            bloque_codigo["Decisión provisional"]
            .astype(str)
            .eq("Revisión por equivalencia indeterminada")
        ].copy()

        if len(candidatos) < 2:
            continue

        puntajes = pd.to_numeric(
            candidatos["Confianza JEV (%)"],
            errors="coerce",
        ).fillna(0.0)

        orden = puntajes.sort_values(
            ascending=False,
            kind="stable",
        )
        indice_ganador = orden.index[0]
        confianza_ganador = float(orden.iloc[0])
        confianza_segundo = float(orden.iloc[1])
        diferencia = (
            confianza_ganador
            - confianza_segundo
        )

        if not (
            confianza_ganador
            >= CONFIANZA_MIN_GANADOR_COMPETENCIA
            and diferencia
            >= DIFERENCIA_MIN_COMPETENCIA
        ):
            for indice in candidatos.index:
                salida.at[
                    indice,
                    "Resolución competencia indeterminada",
                ] = "No concluyente"
                salida.at[
                    indice,
                    "Confianza competencia (%)",
                ] = float(
                    pd.to_numeric(
                        pd.Series(
                            [
                                salida.at[
                                    indice,
                                    "Confianza JEV (%)",
                                ]
                            ]
                        ),
                        errors="coerce",
                    ).fillna(0.0).iloc[0]
                )
                salida.at[
                    indice,
                    "Diferencia competencia (%)",
                ] = round(diferencia, 2)
            continue

        ganador = str(
            salida.at[
                indice_ganador,
                "Archivo",
            ]
        )

        for indice in candidatos.index:
            confianza_actual = float(
                pd.to_numeric(
                    pd.Series(
                        [
                            salida.at[
                                indice,
                                "Confianza JEV (%)",
                            ]
                        ]
                    ),
                    errors="coerce",
                ).fillna(0.0).iloc[0]
            )

            salida.at[
                indice,
                "Competencia indeterminada aplicada",
            ] = True
            salida.at[
                indice,
                "Confianza competencia (%)",
            ] = confianza_actual
            salida.at[
                indice,
                "Diferencia competencia (%)",
            ] = round(diferencia, 2)
            salida.at[
                indice,
                "Ganador competencia",
            ] = ganador

            evidencia = (
                f"Competencia por {codigo}: mejor candidato "
                f"{confianza_ganador:.1f}% vs. segundo "
                f"{confianza_segundo:.1f}% "
                f"(diferencia {diferencia:.1f} pp)."
            )
            salida.at[
                indice,
                "Evidencia competencia",
            ] = evidencia

            if indice == indice_ganador:
                salida.at[
                    indice,
                    "Resolución competencia indeterminada",
                ] = "Ganador por confianza"
                salida.at[
                    indice,
                    "Decisión provisional",
                ] = "Código propio candidato validado"
                salida.at[
                    indice,
                    "Código provisional",
                ] = str(
                    salida.at[
                        indice,
                        "Código JEV",
                    ]
                ).strip()
                salida.at[
                    indice,
                    "Motivo",
                ] = (
                    "La equivalencia estricta quedó indeterminada, "
                    "pero existe competencia concluyente entre varios "
                    "candidatos al mismo código de representación única. "
                    + evidencia
                )
            else:
                salida.at[
                    indice,
                    "Resolución competencia indeterminada",
                ] = "Desplazado por mayor confianza"
                salida.at[
                    indice,
                    "Decisión provisional",
                ] = "Soporte por competencia de confianza"
                salida.at[
                    indice,
                    "Código provisional",
                ] = ""
                salida.at[
                    indice,
                    "Motivo",
                ] = (
                    "El candidato fue desplazado por otro documento "
                    "con el mismo código propuesto y una confianza "
                    "claramente superior. "
                    + evidencia
                )

    return salida


def _resolver_jerarquia_estimacion_finiquito(
    fila_ficha: pd.Series,
    contexto: dict,
    procedimiento: str,
    catalogo_operativo: pd.DataFrame,
    concepto_candidato: str,
    codigo_candidato: str,
) -> dict:
    """
    Evita que la palabra 'finiquito' cambie la identidad principal de una
    pieza que la Ficha V2 reconoce como cuerpo formal de una EST_n.

    Solo actúa cuando:
    - la carpeta estructural es una unidad Estimación con consecutivo;
    - JEV propuso un código/concepto de Finiquito;
    - la identidad detectada contiene un marcador fuerte de cuerpo de
      estimación (carátula, hoja de estimación o estimación constructor).

    No usa el nombre físico del archivo. El candidato se redirige al EST_n
    operativo y la comparación conjunta decide después representante,
    componente o soporte.
    """
    if (
        str(contexto.get("unidad", ""))
        != "Estimación"
        or contexto.get("consecutivo") is None
    ):
        return {
            "aplica": False,
            "concepto": concepto_candidato,
            "codigo": codigo_candidato,
            "regla": "",
            "evidencia": "",
        }

    concepto_norm = _normalizar_texto_regla(
        concepto_candidato
    )
    codigo_norm = _codigo_sin_extension(
        codigo_candidato
    ).upper()

    candidato_finiquito = (
        "finiquito" in concepto_norm
        or codigo_norm.endswith("_FIN")
    )
    if not candidato_finiquito:
        return {
            "aplica": False,
            "concepto": concepto_candidato,
            "codigo": codigo_candidato,
            "regla": "",
            "evidencia": "",
        }

    identidad = _normalizar_texto_regla(
        " ".join(
            [
                str(
                    fila_ficha.get(
                        "Título detectado",
                        "",
                    )
                ),
                str(
                    fila_ficha.get(
                        "Función formal",
                        "",
                    )
                ),
                str(
                    fila_ficha.get(
                        "Acto documentado",
                        "",
                    )
                ),
                str(
                    fila_ficha.get(
                        "Alcance de identidad",
                        "",
                    )
                ),
            ]
        )
    )

    marcador = next(
        (
            item
            for item
            in MARCADORES_IDENTIDAD_CUERPO_ESTIMACION
            if item in identidad
        ),
        None,
    )
    if not marcador:
        return {
            "aplica": False,
            "concepto": concepto_candidato,
            "codigo": codigo_candidato,
            "regla": "",
            "evidencia": "",
        }

    consecutivo = int(
        contexto["consecutivo"]
    )
    codigo_objetivo_base = (
        f"EJE_{procedimiento.upper()}_EST_"
        f"{consecutivo}"
    )

    catalogo = catalogo_operativo.copy()
    codigos = (
        catalogo["Código"]
        .fillna("")
        .astype(str)
        .apply(_codigo_sin_extension)
        .str.upper()
    )
    coincidencias = catalogo[
        codigos.eq(
            codigo_objetivo_base.upper()
        )
    ]

    if coincidencias.empty:
        return {
            "aplica": False,
            "concepto": concepto_candidato,
            "codigo": codigo_candidato,
            "regla": "",
            "evidencia": (
                "La identidad parece pertenecer al cuerpo de la "
                "estimación, pero no se encontró el código EST_n "
                "correspondiente en el catálogo operativo."
            ),
        }

    fila_catalogo = coincidencias.iloc[0]
    codigo_objetivo = str(
        fila_catalogo["Código"]
    ).strip()
    concepto_objetivo = str(
        fila_catalogo["Concepto"]
    ).strip()

    return {
        "aplica": True,
        "concepto": concepto_objetivo,
        "codigo": codigo_objetivo,
        "regla": "estimacion_finiquito_no_cambia_identidad",
        "evidencia": (
            f"La Ficha V2 identifica la pieza mediante el marcador "
            f"'{marcador}' dentro de la unidad Estimación {consecutivo}. "
            f"'Finiquito' se interpreta como condición/fase de esa "
            f"estimación y no como identidad documental independiente. "
            f"El candidato {codigo_candidato or concepto_candidato} se "
            f"redirige a {codigo_objetivo} para comparación conjunta."
        ),
    }


def _contexto_unidad_por_ruta(
    inventario: pd.DataFrame,
) -> dict[str, dict]:
    mapa = construir_mapa_estructural(inventario)
    if mapa.empty:
        return {}

    salida = {}
    for _, fila in mapa.iterrows():
        ruta = str(fila["Ruta carpeta"])
        consecutivo = fila.get("Consecutivo", "")
        try:
            consecutivo_int = (
                int(consecutivo)
                if str(consecutivo).strip() != ""
                else None
            )
        except Exception:
            consecutivo_int = None

        salida[ruta] = {
            "unidad": str(
                fila.get("Unidad candidata", "—")
            ),
            "consecutivo": consecutivo_int,
        }

    return salida


def _relaciones_por_documento(
    relaciones: pd.DataFrame,
) -> dict[tuple[str, str], list[str]]:
    salida: dict[tuple[str, str], list[str]] = {}

    if relaciones.empty:
        return salida

    for _, fila in relaciones.iterrows():
        ruta = str(fila.get("Ruta original", ""))
        origen = str(fila.get("Origen", ""))
        destino = str(fila.get("Destino", ""))
        tipo = str(fila.get("Relación", ""))

        if origen:
            salida.setdefault((ruta, origen), []).append(
                f"{origen} {tipo} {destino}"
            )

        if destino:
            salida.setdefault((ruta, destino), []).append(
                f"{origen} {tipo} {destino}"
            )

    return salida


def _registros_por_documento(
    registros: pd.DataFrame,
) -> dict[tuple[str, str], list[str]]:
    salida: dict[tuple[str, str], list[str]] = {}

    if registros.empty:
        return salida

    for _, fila in registros.iterrows():
        clave = (
            str(fila.get("Ruta original", "")),
            str(fila.get("ID lógico", "")),
        )
        etiqueta = str(fila.get("Etiqueta", "")).strip()
        acto = str(fila.get("Acto documentado", "")).strip()

        texto = etiqueta
        if acto:
            texto = f"{etiqueta}: {acto}" if etiqueta else acto

        if texto:
            salida.setdefault(clave, []).append(texto)

    return salida


def _texto_ficha_para_jev(
    fila: pd.Series,
    relaciones: list[str],
    registros: list[str],
) -> str:
    partes = [
        "IDENTIDAD DOCUMENTAL FIJA:",
        str(fila.get("Título detectado", "")).strip(),
        "",
        "FUNCIÓN FORMAL:",
        str(fila.get("Función formal", "")).strip(),
        "",
        "ACTO DOCUMENTADO:",
        str(fila.get("Acto documentado", "")).strip(),
        "",
        "ALCANCE DE IDENTIDAD:",
        str(fila.get("Alcance de identidad", "")).strip(),
        "",
        "LÍMITES DE IDENTIDAD:",
        str(fila.get("Límites de identidad", "")).strip(),
        "",
        "ALCANCE DOCUMENTAL:",
        str(fila.get("Alcance documental", "")).strip(),
        "",
        "TEXTO REPRESENTATIVO EXTRAÍDO:",
        str(fila.get("Texto representativo", "")).strip(),
        "",
        "DATOS CLAVE:",
        str(fila.get("Datos clave", "")).strip(),
    ]

    if registros:
        partes.extend(
            [
                "",
                "REGISTROS INTERNOS RELEVANTES:",
                "\n".join(
                    f"- {registro}"
                    for registro in registros[:12]
                ),
            ]
        )

    if relaciones:
        partes.extend(
            [
                "",
                "RELACIONES INTERNAS DEL MISMO PDF:",
                "\n".join(
                    f"- {relacion}"
                    for relacion in relaciones[:8]
                ),
            ]
        )

    texto = "\n".join(partes).strip()
    if len(texto) > MAX_CARACTERES_FICHA_JEV:
        texto = texto[:MAX_CARACTERES_FICHA_JEV]

    return texto


def _relaciones_subordinadas(
    relaciones: pd.DataFrame,
) -> dict[tuple[str, str], list[dict]]:
    """
    Relaciones que pueden demostrar que una pieza es apoyo de otra.

    Se usan de forma conservadora: solo desplazan el código cuando la pieza
    subordinada y su destino fueron clasificados inicialmente con el MISMO
    código de catálogo.
    """
    salida: dict[tuple[str, str], list[dict]] = {}

    if relaciones.empty:
        return salida

    for _, fila in relaciones.iterrows():
        tipo = str(fila.get("Relación", "")).strip()
        if tipo not in {"autentica_a", "soporte_de"}:
            continue

        ruta = str(fila.get("Ruta original", ""))
        origen = str(fila.get("Origen", ""))
        destino = str(fila.get("Destino", ""))

        if not origen or not destino:
            continue

        salida.setdefault(
            (ruta, origen),
            [],
        ).append(
            {
                "tipo": tipo,
                "destino": destino,
                "confianza": float(
                    fila.get("Confianza (%)", 0.0)
                    or 0.0
                ),
            }
        )

    return salida


def _evaluar_refuerzo_estructural(
    relaciones: pd.DataFrame,
    detalle: pd.DataFrame,
    ruta: str,
    logical_id: str,
    codigo_candidato: str,
    confianza_candidato: float,
) -> tuple[bool, str]:
    """
    Refuerzo determinístico para una equivalencia JEV indeterminada.

    Solo aplica cuando:
    - el documento principal tiene candidato inicial de alta confianza;
    - existe una relación interna entrante fuerte hacia ese documento;
    - la pieza subordinada recibió el MISMO código candidato;
    - la pieza subordinada también tiene confianza inicial suficiente.

    Nunca convierte una decisión no_equivalente en válida y no se activa sin
    una relación explícita de la Ficha V2.
    """
    if (
        not codigo_candidato
        or confianza_candidato
        < UMBRAL_CANDIDATO_REFUERZO_ESTRUCTURAL
        or relaciones.empty
        or detalle.empty
    ):
        return False, ""

    codigo_base = _codigo_sin_extension(
        codigo_candidato
    ).upper()

    relacionadas = relaciones[
        relaciones["Ruta original"]
        .astype(str)
        .eq(str(ruta))
    ].copy()

    if relacionadas.empty:
        return False, ""

    relacionadas = relacionadas[
        relacionadas["Destino"]
        .astype(str)
        .eq(str(logical_id))
        & relacionadas["Relación"]
        .astype(str)
        .isin(RELACIONES_REFUERZO_ESTRUCTURAL)
    ].copy()

    if relacionadas.empty:
        return False, ""

    evidencias = []

    for _, relacion in relacionadas.iterrows():
        confianza_relacion = float(
            relacion.get("Confianza (%)", 0.0)
            or 0.0
        )
        ids_validos = bool(
            relacion.get("IDs válidos", True)
        )

        if (
            not ids_validos
            or confianza_relacion
            < UMBRAL_RELACION_REFUERZO_ESTRUCTURAL
        ):
            continue

        origen = str(
            relacion.get("Origen", "")
        ).strip()
        tipo = str(
            relacion.get("Relación", "")
        ).strip()

        soporte = detalle[
            detalle["Ruta original"]
            .astype(str)
            .eq(str(ruta))
            & detalle["ID lógico"]
            .astype(str)
            .eq(origen)
        ]

        if soporte.empty:
            continue

        soporte_fila = soporte.iloc[0]
        codigo_soporte = _codigo_sin_extension(
            soporte_fila.get("Código JEV", "")
        ).upper()
        confianza_soporte = float(
            soporte_fila.get(
                "Confianza JEV (%)",
                0.0,
            )
            or 0.0
        )

        if (
            codigo_soporte != codigo_base
            or confianza_soporte
            < UMBRAL_SOPORTE_REFUERZO_ESTRUCTURAL
        ):
            continue

        evidencias.append(
            (
                f"{origen} {tipo} {logical_id}; "
                f"relación {confianza_relacion:.1f}%, "
                f"mismo candidato {codigo_candidato} con "
                f"{confianza_soporte:.1f}% en la pieza subordinada"
            )
        )

    if not evidencias:
        return False, ""

    return True, " | ".join(evidencias)


def clasificar_fichas_v2_con_jev(
    documentos: pd.DataFrame,
    registros: pd.DataFrame,
    relaciones: pd.DataFrame,
    inventario: pd.DataFrame,
    catalogo_base: pd.DataFrame,
    procedimiento: str,
    on_progress=None,
) -> tuple[pd.DataFrame, dict]:
    """
    Ficha V2 -> candidato JEV -> resolución determinística/validación estricta.

    Fase A:
    - una llamada JEV por documento lógico contra el catálogo operativo.

    Fase B, sin volver al PDF:
    - extractos quedan bloqueados por Python;
    - candidatos EST_n se reservan para la comparación conjunta ya validada;
    - piezas autentica_a/soporte_de no heredan el código de su documento
      principal cuando ambos recibieron el mismo candidato;
    - los demás códigos propios se validan con una segunda llamada JEV
      estrictamente binaria contra UN solo concepto candidato.

    Nunca realiza llamadas multimodales.
    """
    if documentos.empty:
        raise ValueError(
            "No hay documentos lógicos de la Ficha V2 para clasificar."
        )

    procedimiento = str(procedimiento).strip().upper()
    contexto_unidades = _contexto_unidad_por_ruta(
        inventario
    )
    mapa_relaciones = _relaciones_por_documento(
        relaciones
    )
    mapa_registros = _registros_por_documento(
        registros
    )
    relaciones_subordinadas = _relaciones_subordinadas(
        relaciones
    )

    resultados_iniciales = []

    costo_clasificacion = 0.0
    tiempo_clasificacion = 0.0
    tokens_entrada_clasificacion = 0
    tokens_salida_clasificacion = 0

    total = len(documentos)

    # ------------------------------------------------------------------
    # FASE A · Candidato inicial contra catálogo
    # ------------------------------------------------------------------
    for posicion, (_, fila) in enumerate(
        documentos.iterrows(),
        start=1,
    ):
        archivo = str(fila.get("Archivo", ""))
        ruta = str(fila.get("Ruta original", ""))
        logical_id = str(fila.get("ID lógico", ""))
        carpeta = str(PurePosixPath(ruta).parent)
        contexto = contexto_unidades.get(
            carpeta,
            {
                "unidad": "—",
                "consecutivo": None,
            },
        )

        catalogo = catalogo_base
        cambios_familia = pd.DataFrame()

        if (
            contexto["unidad"] == "Estimación"
            and contexto["consecutivo"] is not None
        ):
            catalogo, cambios_familia = (
                construir_catalogo_operativo_estimacion(
                    catalogo=catalogo_base,
                    procedimiento=procedimiento,
                    consecutivo=contexto["consecutivo"],
                )
            )

        clave = (ruta, logical_id)
        texto_ficha = _texto_ficha_para_jev(
            fila=fila,
            relaciones=mapa_relaciones.get(
                clave,
                [],
            ),
            registros=mapa_registros.get(
                clave,
                [],
            ),
        )

        if on_progress is not None:
            on_progress(
                posicion,
                total,
                archivo,
                logical_id,
            )

        resultado = clasificar_texto_con_jev(
            documento_alias=(
                f"ficha_{posicion:02d}_{logical_id}"
            ),
            texto_ocr=texto_ficha,
            catalogo=catalogo,
        )

        concepto_jev_original = str(
            resultado["Resultado Ruta A"]
        ).strip()
        codigo_jev_original = str(
            resultado["Código Ruta A"]
        ).strip()

        jerarquia_unidad = (
            _resolver_jerarquia_estimacion_finiquito(
                fila_ficha=fila,
                contexto=contexto,
                procedimiento=procedimiento,
                catalogo_operativo=catalogo,
                concepto_candidato=concepto_jev_original,
                codigo_candidato=codigo_jev_original,
            )
        )

        concepto_jev_resuelto = str(
            jerarquia_unidad["concepto"]
        ).strip()
        codigo_jev_resuelto = str(
            jerarquia_unidad["codigo"]
        ).strip()

        top3 = resultado.get("Top 3", [])
        top3_texto = " | ".join(
            (
                f"{item.get('concepto', '')}: "
                f"{float(item.get('probabilidad', 0.0)) * 100:.1f}%"
            )
            for item in top3
        )

        costo = float(
            resultado.get("Costo A (USD)", 0.0)
        )
        tiempo = float(
            resultado.get("Tiempo A (s)", 0.0)
        )
        entrada = int(
            resultado.get("Tokens entrada A", 0)
        )
        salida = int(
            resultado.get("Tokens salida A", 0)
        )

        costo_clasificacion += costo
        tiempo_clasificacion += tiempo
        tokens_entrada_clasificacion += entrada
        tokens_salida_clasificacion += salida

        resultados_iniciales.append(
            {
                "Archivo": archivo,
                "Ruta original": ruta,
                "ID lógico": logical_id,
                "Título detectado": str(
                    fila.get("Título detectado", "")
                ),
                "Función formal": str(
                    fila.get("Función formal", "")
                ),
                "Acto documentado": str(
                    fila.get("Acto documentado", "")
                ),
                "Alcance de identidad": str(
                    fila.get("Alcance de identidad", "")
                ),
                "Límites de identidad": str(
                    fila.get("Límites de identidad", "")
                ),
                "Texto representativo ficha": str(
                    fila.get("Texto representativo", "")
                ),
                "Datos clave ficha": str(
                    fila.get("Datos clave", "")
                ),
                "Alcance documental": str(
                    fila.get("Alcance documental", "")
                ),
                "Unidad estructural": contexto["unidad"],
                "Consecutivo unidad": (
                    contexto["consecutivo"]
                    if contexto["consecutivo"] is not None
                    else ""
                ),
                "Catálogo operativo aplicado": (
                    not cambios_familia.empty
                ),
                "Concepto JEV original": (
                    concepto_jev_original
                ),
                "Código JEV original": (
                    codigo_jev_original
                ),
                "Jerarquía contexto unidad aplicada": bool(
                    jerarquia_unidad["aplica"]
                ),
                "Regla jerarquía unidad": str(
                    jerarquia_unidad["regla"]
                ),
                "Evidencia jerarquía unidad": str(
                    jerarquia_unidad["evidencia"]
                ),
                "Concepto JEV": concepto_jev_resuelto,
                "Código JEV": codigo_jev_resuelto,
                "Confianza JEV (%)": float(
                    resultado.get("Confianza A", 0.0)
                ),
                "Probabilidad elegida JEV (%)": float(
                    resultado.get(
                        "Probabilidad elegida (%)",
                        0.0,
                    )
                ),
                "Top 3 JEV": top3_texto,
                "Texto enviado a JEV": texto_ficha,
                "Tokens entrada JEV clasificación": entrada,
                "Tokens salida JEV clasificación": salida,
                "Costo JEV clasificación (USD)": costo,
                "Tiempo JEV clasificación (s)": tiempo,
            }
        )

    detalle = pd.DataFrame(resultados_iniciales)

    candidatos_por_clave = {
        (
            str(fila["Ruta original"]),
            str(fila["ID lógico"]),
        ): str(fila["Código JEV"]).strip()
        for _, fila in detalle.iterrows()
    }

    # Columnas de Fase B.
    detalle["Regla relación interna"] = ""
    detalle["Validación estricta JEV"] = "No requerida"
    detalle["Decisión original estricta JEV"] = ""
    detalle["Confianza estricta JEV (%)"] = 0.0
    detalle["Probabilidad estricta JEV (%)"] = 0.0
    detalle["Margen estricta JEV (%)"] = None
    detalle["Control estricta JEV"] = ""
    detalle["Exclusión identidad aplicada"] = False
    detalle["Regla exclusión identidad"] = ""
    detalle["Evidencia exclusión identidad"] = ""
    detalle["Refuerzo estructural aplicado"] = False
    detalle["Evidencia refuerzo estructural"] = ""
    detalle["Validación propósito sensible"] = "No aplica"
    detalle["Regla propósito sensible"] = ""
    detalle["Evidencia propósito sensible"] = ""
    detalle["Error validación estricta"] = ""
    detalle["Tokens entrada JEV validación"] = 0
    detalle["Tokens salida JEV validación"] = 0
    detalle["Costo JEV validación (USD)"] = 0.0
    detalle["Tiempo JEV validación (s)"] = 0.0
    detalle["Decisión provisional"] = ""
    detalle["Código provisional"] = ""
    detalle["Motivo"] = ""

    llamadas_validacion = 0
    costo_validacion = 0.0
    tiempo_validacion = 0.0
    tokens_entrada_validacion = 0
    tokens_salida_validacion = 0

    # ------------------------------------------------------------------
    # FASE B · Reglas + equivalencia estricta contra un solo candidato
    # ------------------------------------------------------------------
    for indice, fila in detalle.iterrows():
        ruta = str(fila["Ruta original"])
        logical_id = str(fila["ID lógico"])
        codigo_jev = str(fila["Código JEV"]).strip()
        concepto_jev = str(
            fila["Concepto JEV"]
        ).strip()
        alcance = str(
            fila["Alcance documental"]
        ).strip()

        codigo_unidad = ""
        consecutivo = fila["Consecutivo unidad"]
        if (
            str(fila["Unidad estructural"])
            == "Estimación"
            and str(consecutivo).strip() != ""
        ):
            codigo_unidad = (
                f"EJE_{procedimiento}_EST_"
                f"{int(consecutivo)}"
            )

        codigo_jev_base = _codigo_sin_extension(
            codigo_jev
        ).upper()
        codigo_unidad_base = _codigo_sin_extension(
            codigo_unidad
        ).upper()

        if not codigo_jev:
            detalle.at[
                indice,
                "Decisión provisional",
            ] = "Fuera de catálogo"
            detalle.at[
                indice,
                "Motivo",
            ] = (
                "JEV no encontró equivalencia documental-funcional "
                "con un concepto del catálogo."
            )
            continue

        exclusion_identidad = (
            _evaluar_exclusion_identidad_candidato(
                fila=fila,
                codigo_candidato=codigo_jev,
            )
        )
        if exclusion_identidad["aplica"]:
            detalle.at[
                indice,
                "Exclusión identidad aplicada",
            ] = True
            detalle.at[
                indice,
                "Regla exclusión identidad",
            ] = exclusion_identidad["regla"]
            detalle.at[
                indice,
                "Evidencia exclusión identidad",
            ] = exclusion_identidad["evidencia"]
            detalle.at[
                indice,
                "Decisión provisional",
            ] = (
                "Soporte por identidad no equivalente "
                "al concepto formal"
            )
            detalle.at[
                indice,
                "Motivo",
            ] = exclusion_identidad["evidencia"]
            continue

        if alcance == "parcial_extracto":
            detalle.at[
                indice,
                "Decisión provisional",
            ] = (
                "Coincidencia conceptual; extracto sin código completo"
            )
            detalle.at[
                indice,
                "Motivo",
            ] = (
                "La Ficha V2 identifica el tipo documental, pero su "
                "alcance es parcial_extracto. Python bloquea el código "
                "destinado al documento completo."
            )
            continue

        if alcance == "indeterminado":
            detalle.at[
                indice,
                "Decisión provisional",
            ] = "Revisión por alcance indeterminado"
            detalle.at[
                indice,
                "Motivo",
            ] = (
                "Existe candidato de catálogo, pero la Ficha V2 no "
                "confirma que la pieza sea una instancia completa."
            )
            continue

        if (
            codigo_unidad_base
            and codigo_jev_base == codigo_unidad_base
        ):
            detalle.at[
                indice,
                "Decisión provisional",
            ] = "Candidato al código de la unidad"
            detalle.at[
                indice,
                "Motivo",
            ] = (
                "Coincide con EST_n. La selección entre representante "
                "y componente sigue reservada a la comparación conjunta "
                "ya validada."
            )
            continue

        # Una pieza de autenticación/soporte no hereda el mismo código
        # del documento principal al que sirve.
        subordinada = None
        for relacion in relaciones_subordinadas.get(
            (ruta, logical_id),
            [],
        ):
            codigo_destino = candidatos_por_clave.get(
                (ruta, relacion["destino"]),
                "",
            )
            if (
                codigo_destino
                and _codigo_sin_extension(
                    codigo_destino
                ).upper()
                == codigo_jev_base
            ):
                subordinada = relacion
                break

        if subordinada is not None:
            regla = (
                f"{logical_id} "
                f"{subordinada['tipo']} "
                f"{subordinada['destino']}"
            )
            detalle.at[
                indice,
                "Regla relación interna",
            ] = regla
            detalle.at[
                indice,
                "Decisión provisional",
            ] = "Soporte por relación interna"
            detalle.at[
                indice,
                "Motivo",
            ] = (
                "La Ficha V2 identifica esta pieza como "
                f"{subordinada['tipo']} de otro documento lógico. "
                "Ambas piezas recibieron el mismo código candidato; "
                "por regla determinística la pieza subordinada no "
                "hereda el código de su documento principal."
            )
            continue

        # Algunos conceptos requieren que su finalidad esté expresamente
        # acreditada en la Ficha V2 antes de permitir la equivalencia estricta.
        proposito = _regla_proposito_sensible(
            detalle=detalle,
            fila=fila,
        )

        if proposito["aplica"]:
            detalle.at[
                indice,
                "Regla propósito sensible",
            ] = proposito["regla"]
            detalle.at[
                indice,
                "Evidencia propósito sensible",
            ] = proposito["evidencia"]

            if proposito["acreditado"]:
                detalle.at[
                    indice,
                    "Validación propósito sensible",
                ] = "Acreditado"
            elif proposito["soporte_contextual"]:
                detalle.at[
                    indice,
                    "Validación propósito sensible",
                ] = (
                    "No acreditado; soporte contextual"
                )
                detalle.at[
                    indice,
                    "Decisión provisional",
                ] = (
                    "Soporte por propósito sensible "
                    "no acreditado"
                )
                detalle.at[
                    indice,
                    "Motivo",
                ] = (
                    "El concepto candidato exige una finalidad "
                    "documental explícita que la Ficha V2 no acredita. "
                    "Además, el mismo PDF contiene evidencia de que la "
                    "pieza forma parte del pago de una estimación. "
                    + proposito["evidencia"]
                )
                continue
            else:
                detalle.at[
                    indice,
                    "Validación propósito sensible",
                ] = "No acreditado"
                detalle.at[
                    indice,
                    "Decisión provisional",
                ] = (
                    "Revisión por propósito sensible "
                    "no acreditado"
                )
                detalle.at[
                    indice,
                    "Motivo",
                ] = (
                    "El concepto candidato exige una finalidad "
                    "documental explícita, pero la Ficha V2 no aporta "
                    "evidencia positiva suficiente. "
                    + proposito["evidencia"]
                )
                continue

        # El resto de códigos propios debe superar una comparación
        # estricta contra UN solo concepto candidato.
        llamadas_validacion += 1
        if on_progress is not None:
            on_progress(
                llamadas_validacion,
                total,
                str(fila["Archivo"]),
                f"{logical_id} · validación estricta",
            )

        try:
            validacion = validar_equivalencia_ficha_con_jev(
                documento_alias=(
                    f"validacion_{logical_id}"
                ),
                identidad_documental=str(
                    fila["Título detectado"]
                ),
                funcion_formal=str(
                    fila["Función formal"]
                ),
                acto_documentado=str(
                    fila["Acto documentado"]
                ),
                alcance_identidad=str(
                    fila["Alcance de identidad"]
                ),
                limites_identidad=str(
                    fila["Límites de identidad"]
                ),
                alcance_documental=alcance,
                concepto_objetivo=concepto_jev,
            )
        except Exception as error:
            detalle.at[
                indice,
                "Validación estricta JEV",
            ] = "Error"
            detalle.at[
                indice,
                "Error validación estricta",
            ] = str(error)
            detalle.at[
                indice,
                "Decisión provisional",
            ] = "Revisión por error de validación"
            detalle.at[
                indice,
                "Motivo",
            ] = (
                "La clasificación candidata se conserva para "
                "trazabilidad, pero no se asigna código porque falló "
                "la validación textual estricta."
            )
            continue

        equivalencia = str(
            validacion["Equivalencia estricta JEV"]
        )
        decision_original = str(
            validacion[
                "Decisión original estricta JEV"
            ]
        )
        confianza = float(
            validacion[
                "Confianza estricta JEV (%)"
            ]
        )
        probabilidad = float(
            validacion[
                "Probabilidad elegida estricta JEV (%)"
            ]
        )
        margen = validacion[
            "Margen estricta JEV (%)"
        ]
        control = str(
            validacion["Control estricta JEV"]
        )
        entrada = int(
            validacion[
                "Tokens entrada estricta JEV"
            ]
        )
        salida = int(
            validacion[
                "Tokens salida estricta JEV"
            ]
        )
        costo = float(
            validacion[
                "Costo estricta JEV (USD)"
            ]
        )
        tiempo = float(
            validacion[
                "Tiempo estricta JEV (s)"
            ]
        )

        detalle.at[
            indice,
            "Validación estricta JEV",
        ] = equivalencia
        detalle.at[
            indice,
            "Decisión original estricta JEV",
        ] = decision_original
        detalle.at[
            indice,
            "Confianza estricta JEV (%)",
        ] = confianza
        detalle.at[
            indice,
            "Probabilidad estricta JEV (%)",
        ] = probabilidad
        detalle.at[
            indice,
            "Margen estricta JEV (%)",
        ] = margen
        detalle.at[
            indice,
            "Control estricta JEV",
        ] = control
        detalle.at[
            indice,
            "Tokens entrada JEV validación",
        ] = entrada
        detalle.at[
            indice,
            "Tokens salida JEV validación",
        ] = salida
        detalle.at[
            indice,
            "Costo JEV validación (USD)",
        ] = costo
        detalle.at[
            indice,
            "Tiempo JEV validación (s)",
        ] = tiempo

        costo_validacion += costo
        tiempo_validacion += tiempo
        tokens_entrada_validacion += entrada
        tokens_salida_validacion += salida

        if equivalencia == "equivalente":
            detalle.at[
                indice,
                "Decisión provisional",
            ] = "Código propio candidato validado"
            detalle.at[
                indice,
                "Código provisional",
            ] = codigo_jev
            detalle.at[
                indice,
                "Motivo",
            ] = (
                "JEV propuso el candidato y una segunda llamada JEV, "
                "sin PDF, confirmó equivalencia documental-funcional "
                "contra ese único concepto."
            )
        elif equivalencia == "no_equivalente":
            detalle.at[
                indice,
                "Decisión provisional",
            ] = "Candidato rechazado por equivalencia estricta"
            detalle.at[
                indice,
                "Motivo",
            ] = (
                "La identidad fija de la Ficha V2 no es documental "
                "y funcionalmente equivalente al concepto candidato."
            )
        else:
            refuerzo, evidencia_refuerzo = (
                _evaluar_refuerzo_estructural(
                    relaciones=relaciones,
                    detalle=detalle,
                    ruta=ruta,
                    logical_id=logical_id,
                    codigo_candidato=codigo_jev,
                    confianza_candidato=float(
                        fila.get(
                            "Confianza JEV (%)",
                            0.0,
                        )
                        or 0.0
                    ),
                )
            )

            detalle.at[
                indice,
                "Refuerzo estructural aplicado",
            ] = refuerzo
            detalle.at[
                indice,
                "Evidencia refuerzo estructural",
            ] = evidencia_refuerzo

            if refuerzo:
                detalle.at[
                    indice,
                    "Decisión provisional",
                ] = "Código propio candidato validado"
                detalle.at[
                    indice,
                    "Código provisional",
                ] = codigo_jev
                detalle.at[
                    indice,
                    "Motivo",
                ] = (
                    "La equivalencia estricta JEV quedó indeterminada, "
                    "pero se acepta de forma controlada por refuerzo "
                    "estructural fuerte de la Ficha V2: candidato inicial "
                    "de alta confianza, relación interna explícita hacia "
                    "el documento principal y pieza subordinada con el "
                    "mismo código candidato. "
                    + evidencia_refuerzo
                )
            else:
                detalle.at[
                    indice,
                    "Decisión provisional",
                ] = "Revisión por equivalencia indeterminada"
                detalle.at[
                    indice,
                    "Motivo",
                ] = (
                    "JEV no alcanzó evidencia suficiente para confirmar "
                    "o rechazar el candidato y la Ficha V2 no aportó un "
                    "refuerzo estructural fuerte compatible."
                )

    detalle = _resolver_competencia_indeterminada(
        detalle
    )

    llamadas_clasificacion = len(detalle)
    llamadas_total = (
        llamadas_clasificacion
        + llamadas_validacion
    )

    costo_total = (
        costo_clasificacion
        + costo_validacion
    )
    tiempo_total = (
        tiempo_clasificacion
        + tiempo_validacion
    )
    tokens_entrada_total = (
        tokens_entrada_clasificacion
        + tokens_entrada_validacion
    )
    tokens_salida_total = (
        tokens_salida_clasificacion
        + tokens_salida_validacion
    )

    resumen = {
        "pipeline_version": PIPELINE_VERSION_FICHA_JEV,
        "documentos_logicos_evaluados": len(detalle),
        "jerarquias_unidad_aplicadas": int(
            detalle[
                "Jerarquía contexto unidad aplicada"
            ].fillna(False).astype(bool).sum()
        ),
        "llamadas_jev_clasificacion": llamadas_clasificacion,
        "llamadas_jev_validacion": llamadas_validacion,
        "llamadas_jev": llamadas_total,
        "multimodales_adicionales": 0,
        "fuera_catalogo": int(
            (
                detalle["Decisión provisional"]
                == "Fuera de catálogo"
            ).sum()
        ),
        "extractos_sin_codigo": int(
            (
                detalle["Decisión provisional"]
                == (
                    "Coincidencia conceptual; "
                    "extracto sin código completo"
                )
            ).sum()
        ),
        "candidatos_unidad": int(
            (
                detalle["Decisión provisional"]
                == "Candidato al código de la unidad"
            ).sum()
        ),
        "soportes_relacion_interna": int(
            (
                detalle["Decisión provisional"]
                == "Soporte por relación interna"
            ).sum()
        ),
        "codigos_propios_validados": int(
            (
                detalle["Decisión provisional"]
                == "Código propio candidato validado"
            ).sum()
        ),
        "codigos_validados_por_refuerzo": int(
            detalle[
                "Refuerzo estructural aplicado"
            ].fillna(False).astype(bool).sum()
        ),
        "propositos_sensibles_bloqueados": int(
            detalle[
                "Decisión provisional"
            ].astype(str).isin(
                [
                    "Soporte por propósito sensible no acreditado",
                    "Revisión por propósito sensible no acreditado",
                ]
            ).sum()
        ),
        "exclusiones_identidad_aplicadas": int(
            detalle[
                "Exclusión identidad aplicada"
            ].fillna(False).astype(bool).sum()
        ),
        "competencias_indeterminadas_resueltas": int(
            detalle[
                "Resolución competencia indeterminada"
            ].astype(str).eq(
                "Ganador por confianza"
            ).sum()
        ),
        "candidatos_rechazados": int(
            (
                detalle["Decisión provisional"]
                == (
                    "Candidato rechazado por "
                    "equivalencia estricta"
                )
            ).sum()
        ),
        "revision": int(
            detalle[
                "Decisión provisional"
            ].astype(str).str.startswith(
                "Revisión"
            ).sum()
        ),
        "tokens_entrada_jev_clasificacion": (
            tokens_entrada_clasificacion
        ),
        "tokens_salida_jev_clasificacion": (
            tokens_salida_clasificacion
        ),
        "tokens_entrada_jev_validacion": (
            tokens_entrada_validacion
        ),
        "tokens_salida_jev_validacion": (
            tokens_salida_validacion
        ),
        "tokens_entrada_jev": tokens_entrada_total,
        "tokens_salida_jev": tokens_salida_total,
        "costo_jev_clasificacion_usd": (
            costo_clasificacion
        ),
        "costo_jev_validacion_usd": costo_validacion,
        "costo_total_jev_usd": costo_total,
        "tiempo_jev_clasificacion_s": (
            tiempo_clasificacion
        ),
        "tiempo_jev_validacion_s": tiempo_validacion,
        "tiempo_total_jev_s": tiempo_total,
    }

    return detalle, resumen
