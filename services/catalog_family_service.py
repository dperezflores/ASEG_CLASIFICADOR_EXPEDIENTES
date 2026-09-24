from __future__ import annotations

import re

import pandas as pd


# Primera versión controlada.
# Solo estas familias toman el consecutivo de la unidad de estimación.
FAMILIAS_CONSECUTIVAS_ESTIMACION = {
    "EST": {
        "actualizar_concepto": True,
    },
    "AVE": {
        "actualizar_concepto": False,
    },
}


def _separar_extension(codigo: str) -> tuple[str, str]:
    valor = str(codigo).strip()
    if valor.lower().endswith(".pdf"):
        return valor[:-4], valor[-4:]
    return valor, ""


def _actualizar_numero_concepto_estimacion(
    concepto: str,
    consecutivo: int,
) -> str:
    """
    Ajusta únicamente conceptos explícitos del tipo 'Estimación 1'.

    No modifica conceptos genéricos ni otros números que puedan formar parte
    legítima del nombre documental.
    """
    texto = str(concepto).strip()

    patron = re.compile(
        r"(\bESTIMACI[ÓO]N\s*(?:NO\.?\s*)?)(\d+)",
        re.IGNORECASE,
    )

    if not patron.search(texto):
        return texto

    return patron.sub(
        lambda coincidencia: (
            f"{coincidencia.group(1)}{int(consecutivo)}"
        ),
        texto,
        count=1,
    )


def construir_catalogo_operativo_estimacion(
    catalogo: pd.DataFrame,
    procedimiento: str,
    consecutivo: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Genera una vista operativa del catálogo para una estimación concreta.

    La fuente institucional permanece intacta. Solo se parametrizan familias
    explícitamente autorizadas. En esta V1:
    - EST_n toma el consecutivo de la estimación.
    - AVE_n toma el consecutivo de la estimación.

    Si el catálogo llegara a contener varias filas de una misma familia
    (EST_1, EST_2, ...), se colapsan en una sola opción operativa para evitar
    duplicidades. No se hace ninguna sustitución general de '_1'.
    """
    if catalogo.empty:
        return catalogo.copy(), pd.DataFrame()

    procedimiento = str(procedimiento).strip().upper()
    consecutivo = int(consecutivo)

    filas_operativas = []
    cambios = []
    familias_emitidas: set[str] = set()
    coincidencias_por_familia: dict[str, int] = {
        familia: 0
        for familia in FAMILIAS_CONSECUTIVAS_ESTIMACION
    }

    for _, fila in catalogo.iterrows():
        codigo_original = str(fila["Código"]).strip()
        concepto_original = str(fila["Concepto"]).strip()
        base, extension = _separar_extension(codigo_original)

        familia_detectada = None
        numero_catalogo = None

        for familia in FAMILIAS_CONSECUTIVAS_ESTIMACION:
            patron_codigo = re.compile(
                rf"^EJE_{re.escape(procedimiento)}_"
                rf"{re.escape(familia)}_(\d+)$",
                re.IGNORECASE,
            )
            coincidencia = patron_codigo.match(base)

            if coincidencia:
                familia_detectada = familia
                numero_catalogo = int(coincidencia.group(1))
                coincidencias_por_familia[familia] += 1
                break

        if familia_detectada is None:
            filas_operativas.append(
                {
                    "Código": codigo_original,
                    "Concepto": concepto_original,
                }
            )
            continue

        if familia_detectada in familias_emitidas:
            continue

        reglas = FAMILIAS_CONSECUTIVAS_ESTIMACION[
            familia_detectada
        ]
        nuevo_base = re.sub(
            r"_\d+$",
            f"_{consecutivo}",
            base,
        )
        nuevo_codigo = nuevo_base + extension

        nuevo_concepto = concepto_original
        if reglas["actualizar_concepto"]:
            nuevo_concepto = _actualizar_numero_concepto_estimacion(
                concepto_original,
                consecutivo,
            )

        filas_operativas.append(
            {
                "Código": nuevo_codigo,
                "Concepto": nuevo_concepto,
            }
        )
        cambios.append(
            {
                "Familia": familia_detectada,
                "Número catálogo base": numero_catalogo,
                "Consecutivo operativo": consecutivo,
                "Código base": codigo_original,
                "Código operativo": nuevo_codigo,
                "Concepto base": concepto_original,
                "Concepto operativo": nuevo_concepto,
            }
        )
        familias_emitidas.add(familia_detectada)

    operativo = pd.DataFrame(
        filas_operativas,
        columns=["Código", "Concepto"],
    )

    cambios_df = pd.DataFrame(cambios)
    if not cambios_df.empty:
        cambios_df["Filas de familia en catálogo"] = (
            cambios_df["Familia"].map(coincidencias_por_familia)
        )

    return operativo.reset_index(drop=True), cambios_df
