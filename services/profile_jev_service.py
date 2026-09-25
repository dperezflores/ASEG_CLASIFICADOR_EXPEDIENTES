from __future__ import annotations

from pathlib import PurePosixPath

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


def _codigo_sin_extension(codigo: str) -> str:
    valor = str(codigo).strip()
    if valor.lower().endswith(".pdf"):
        return valor[:-4]
    return valor


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
                "Concepto JEV": str(
                    resultado["Resultado Ruta A"]
                ).strip(),
                "Código JEV": str(
                    resultado["Código Ruta A"]
                ).strip(),
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
            detalle.at[
                indice,
                "Decisión provisional",
            ] = "Revisión por equivalencia indeterminada"
            detalle.at[
                indice,
                "Motivo",
            ] = (
                "JEV no alcanzó evidencia suficiente para confirmar "
                "o rechazar el candidato sin reinterpretar la ficha."
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
        "pipeline_version": 2,
        "documentos_logicos_evaluados": len(detalle),
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
