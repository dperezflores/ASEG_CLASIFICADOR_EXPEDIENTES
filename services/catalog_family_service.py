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
        r"^(.*?\bESTIMACI[ÓO]N\s*(?:NO\.?\s*)?)(\d+)(\s*)$",
        re.IGNORECASE,
    )
    coincidencia = patron.match(texto)

    if not coincidencia:
        return texto

    return (
        f"{coincidencia.group(1)}"
        f"{int(consecutivo)}"
        f"{coincidencia.group(3)}"
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

    No se hace ninguna sustitución general de '_1' por otro número.
    """
    if catalogo.empty:
        return catalogo.copy(), pd.DataFrame()

    procedimiento = str(procedimiento).strip().upper()
    consecutivo = int(consecutivo)

    operativo = catalogo.copy()
    cambios = []

    for indice, fila in operativo.iterrows():
        codigo_original = str(fila["Código"]).strip()
        concepto_original = str(fila["Concepto"]).strip()

        base, extension = _separar_extension(codigo_original)

        for familia, reglas in FAMILIAS_CONSECUTIVAS_ESTIMACION.items():
            patron_codigo = re.compile(
                rf"^EJE_{re.escape(procedimiento)}_"
                rf"{re.escape(familia)}_(\d+)$",
                re.IGNORECASE,
            )
            coincidencia = patron_codigo.match(base)

            if not coincidencia:
                continue

            numero_catalogo = int(coincidencia.group(1))
            nuevo_base = re.sub(
                r"_\d+$",
                f"_{consecutivo}",
                base,
            )
            nuevo_codigo = nuevo_base + extension

            nuevo_concepto = concepto_original
            if reglas["actualizar_concepto"]:
                nuevo_concepto = (
                    _actualizar_numero_concepto_estimacion(
                        concepto_original,
                        consecutivo,
                    )
                )

            operativo.at[indice, "Código"] = nuevo_codigo
            operativo.at[indice, "Concepto"] = nuevo_concepto

            cambios.append(
                {
                    "Familia": familia,
                    "Número catálogo base": numero_catalogo,
                    "Consecutivo operativo": consecutivo,
                    "Código base": codigo_original,
                    "Código operativo": nuevo_codigo,
                    "Concepto base": concepto_original,
                    "Concepto operativo": nuevo_concepto,
                }
            )
            break

    return operativo.reset_index(drop=True), pd.DataFrame(cambios)
