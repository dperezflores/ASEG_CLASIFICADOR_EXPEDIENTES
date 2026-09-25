from __future__ import annotations

from pathlib import PurePosixPath

import pandas as pd

from services.catalog_family_service import (
    construir_catalogo_operativo_estimacion,
)
from services.jev_classifier_service import (
    clasificar_texto_con_jev,
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
    Clasifica documentos lógicos de la Ficha V2 exclusivamente con su texto.

    No recibe PDF ni hace llamadas multimodales. Cada documento lógico genera
    una llamada JEV. Para documentos ubicados dentro de una unidad EST_n, usa
    la vista operativa del catálogo ya validada para parametrizar EST_n/AVE_n.
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

    resultados = []
    costo_total = 0.0
    tiempo_total = 0.0
    tokens_entrada = 0
    tokens_salida = 0
    total = len(documentos)

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

        codigo_jev = str(
            resultado["Código Ruta A"]
        ).strip()
        concepto_jev = str(
            resultado["Resultado Ruta A"]
        ).strip()
        alcance = str(
            fila.get("Alcance documental", "")
        ).strip()

        codigo_unidad = ""
        if (
            contexto["unidad"] == "Estimación"
            and contexto["consecutivo"] is not None
        ):
            codigo_unidad = (
                f"EJE_{procedimiento}_EST_"
                f"{int(contexto['consecutivo'])}"
            )

        codigo_jev_base = _codigo_sin_extension(
            codigo_jev
        ).upper()
        codigo_unidad_base = _codigo_sin_extension(
            codigo_unidad
        ).upper()

        if not codigo_jev:
            decision = "Fuera de catálogo"
            codigo_provisional = ""
            motivo = (
                "JEV no encontró equivalencia documental-funcional "
                "con un concepto del catálogo."
            )
        elif alcance == "parcial_extracto":
            decision = (
                "Coincidencia conceptual; extracto sin código completo"
            )
            codigo_provisional = ""
            motivo = (
                "La ficha identifica el tipo documental, pero su alcance "
                "es parcial_extracto; no se asigna el código destinado al "
                "documento completo."
            )
        elif alcance == "indeterminado":
            decision = "Revisión por alcance indeterminado"
            codigo_provisional = ""
            motivo = (
                "Existe un candidato de catálogo, pero la ficha no permite "
                "confirmar que la pieza sea una instancia completa."
            )
        elif (
            codigo_unidad_base
            and codigo_jev_base == codigo_unidad_base
        ):
            decision = "Candidato al código de la unidad"
            codigo_provisional = ""
            motivo = (
                "El concepto coincide con el código EST_n de la unidad. "
                "La selección de representante/componente se deja a la "
                "comparación conjunta ya validada."
            )
        else:
            decision = "Código propio candidato"
            codigo_provisional = codigo_jev
            motivo = (
                "JEV encontró equivalencia y la ficha marca la pieza "
                "como completa. La codificación sigue siendo provisional "
                "en este experimento."
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

        costo_total += costo
        tiempo_total += tiempo
        tokens_entrada += entrada
        tokens_salida += salida

        resultados.append(
            {
                "Archivo": archivo,
                "Ruta original": ruta,
                "ID lógico": logical_id,
                "Título detectado": str(
                    fila.get("Título detectado", "")
                ),
                "Alcance documental": alcance,
                "Unidad estructural": contexto["unidad"],
                "Consecutivo unidad": (
                    contexto["consecutivo"]
                    if contexto["consecutivo"] is not None
                    else ""
                ),
                "Catálogo operativo aplicado": (
                    not cambios_familia.empty
                ),
                "Concepto JEV": concepto_jev,
                "Código JEV": codigo_jev,
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
                "Decisión provisional": decision,
                "Código provisional": codigo_provisional,
                "Motivo": motivo,
                "Texto enviado a JEV": texto_ficha,
                "Tokens entrada JEV": entrada,
                "Tokens salida JEV": salida,
                "Costo JEV (USD)": costo,
                "Tiempo JEV (s)": tiempo,
            }
        )

    detalle = pd.DataFrame(resultados)

    resumen = {
        "documentos_logicos_evaluados": len(detalle),
        "llamadas_jev": len(detalle),
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
        "codigos_propios_candidatos": int(
            (
                detalle["Decisión provisional"]
                == "Código propio candidato"
            ).sum()
        ),
        "revision": int(
            (
                detalle["Decisión provisional"]
                == "Revisión por alcance indeterminado"
            ).sum()
        ),
        "tokens_entrada_jev": tokens_entrada,
        "tokens_salida_jev": tokens_salida,
        "costo_total_jev_usd": costo_total,
        "tiempo_total_jev_s": tiempo_total,
    }

    return detalle, resumen
