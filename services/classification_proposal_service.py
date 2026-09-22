from __future__ import annotations

import re
import unicodedata
from pathlib import PurePosixPath

import pandas as pd


UMBRAL_REVISION = 80.0


def _normalizar_titulo_archivo(titulo: str) -> str:
    texto = unicodedata.normalize("NFKD", str(titulo))
    texto = "".join(
        caracter for caracter in texto
        if not unicodedata.combining(caracter)
    )
    texto = texto.upper().strip()
    texto = re.sub(r"[^A-Z0-9]+", "_", texto)
    texto = re.sub(r"_+", "_", texto).strip("_")
    return texto or "DOCUMENTO_SIN_IDENTIFICAR"


def construir_propuesta_clasificacion(
    resultados: pd.DataFrame,
    muestra: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convierte los resultados técnicos de clasificación en una propuesta
    orientada al usuario: archivo original, nombre final y estado de revisión.

    - Si existe coincidencia con catálogo, conserva literalmente el código
      oficial devuelto por el catálogo.
    - Si no existe coincidencia, propone un nombre informativo derivado del
      título detectado, sin convertirlo en código institucional.
    """
    if resultados.empty:
        return pd.DataFrame()

    referencia = muestra[
        [
            "Documento",
            "Archivo real",
            "Ruta original",
        ]
    ].copy()

    propuesta = resultados.merge(
        referencia,
        on="Documento",
        how="left",
    )

    nombres = []
    estados = []
    observaciones = []

    for _, fila in propuesta.iterrows():
        coincide = bool(fila["Coincide catálogo"])
        confianza = float(fila["Confianza (%)"])
        archivo_real = str(fila.get("Archivo real", ""))
        extension = PurePosixPath(archivo_real).suffix or ".pdf"

        if coincide:
            nombre = str(fila["Código propuesto"]).strip()
            if not nombre:
                nombre = str(fila["Concepto propuesto"]).strip()

            if confianza >= UMBRAL_REVISION:
                estado = "CLASIFICADO"
                observacion = "Coincidencia con catálogo con confianza suficiente."
            else:
                estado = "REVISIÓN RECOMENDADA"
                observacion = (
                    "La IA encontró coincidencia con catálogo, pero la confianza "
                    f"({confianza:.1f} %) es menor al umbral de revisión "
                    f"({UMBRAL_REVISION:.0f} %)."
                )
        else:
            titulo = str(fila["Título detectado"]).strip()
            nombre = _normalizar_titulo_archivo(titulo) + extension.lower()
            estado = "FUERA DE CATÁLOGO"
            observacion = (
                "No se encontró una coincidencia documental suficiente con el "
                "catálogo. El nombre propuesto es únicamente informativo."
            )

        nombres.append(nombre)
        estados.append(estado)
        observaciones.append(observacion)

    propuesta["Nombre propuesto"] = nombres
    propuesta["Estado"] = estados
    propuesta["Observación"] = observaciones

    columnas = [
        "Documento",
        "Archivo real",
        "Ruta original",
        "Título detectado",
        "Coincide catálogo",
        "Concepto propuesto",
        "Código propuesto",
        "Nombre propuesto",
        "Confianza (%)",
        "Estado",
        "Ruta utilizada",
        "Observación",
        "Evidencia",
        "Costo (USD)",
        "Tiempo (s)",
    ]

    return propuesta[columnas]
