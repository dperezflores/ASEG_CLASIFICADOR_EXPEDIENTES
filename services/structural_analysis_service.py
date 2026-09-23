from __future__ import annotations

import re
import unicodedata
from pathlib import PurePosixPath

import pandas as pd


PATRON_ESTIMACION = re.compile(
    r"^EST(?:IMACION)?[ _-]*(\d+)(?:[ _-]*FIN)?$",
    re.IGNORECASE,
)


def _normalizar(texto: str) -> str:
    valor = unicodedata.normalize("NFKD", str(texto))
    valor = "".join(
        caracter for caracter in valor
        if not unicodedata.combining(caracter)
    )
    return re.sub(r"\s+", " ", valor).strip().upper()


def _carpetas_desde_inventario(inventario: pd.DataFrame) -> list[str]:
    carpetas: set[str] = set()

    if inventario.empty:
        return []

    for ruta_texto in inventario["Ruta original"].astype(str):
        ruta = PurePosixPath(ruta_texto)
        for padre in ruta.parents:
            padre_texto = str(padre)
            if padre_texto not in ("", "."):
                carpetas.add(padre_texto)

    return sorted(
        carpetas,
        key=lambda ruta: (
            len(PurePosixPath(ruta).parts),
            ruta.lower(),
        ),
    )


def _detectar_unidad_documental(nombre_carpeta: str) -> tuple[str, str, int | None]:
    """
    Primera regla estructural del prototipo.

    Deliberadamente solo detecta estimaciones cuando el nombre de la carpeta
    contiene una estructura suficientemente clara (EST 1, ESTIMACION 1,
    EST 7 FIN, etc.). No intenta todavía inferir otras unidades.
    """
    normalizado = _normalizar(nombre_carpeta)
    coincidencia = PATRON_ESTIMACION.fullmatch(normalizado)

    if coincidencia:
        consecutivo = int(coincidencia.group(1))
        return "Estimación", "ALTA", consecutivo

    return "—", "—", None


def construir_mapa_estructural(inventario: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un mapa de carpetas a partir del inventario ya existente.

    No usa IA, no abre documentos y no modifica el expediente.
    """
    carpetas = _carpetas_desde_inventario(inventario)
    registros = []

    for carpeta in carpetas:
        ruta = PurePosixPath(carpeta)
        prefijo = carpeta.rstrip("/") + "/"

        archivos_directos = inventario[
            inventario["Carpeta"].astype(str) == carpeta
        ]

        archivos_descendientes = inventario[
            inventario["Ruta original"].astype(str).str.startswith(
                prefijo,
                na=False,
            )
        ]

        subcarpetas_directas = 0
        for otra in carpetas:
            otra_ruta = PurePosixPath(otra)
            if str(otra_ruta.parent) == carpeta:
                subcarpetas_directas += 1

        tipo_unidad, confianza, consecutivo = _detectar_unidad_documental(
            ruta.name
        )

        registros.append(
            {
                "Nivel": len(ruta.parts),
                "Ruta carpeta": carpeta,
                "Carpeta": ruta.name,
                "Subcarpetas directas": subcarpetas_directas,
                "Archivos directos": len(archivos_directos),
                "Archivos totales": len(archivos_descendientes),
                "Unidad candidata": tipo_unidad,
                "Consecutivo": consecutivo if consecutivo is not None else "",
                "Confianza estructural": confianza,
            }
        )

    return pd.DataFrame(registros)


def obtener_estimaciones_detectadas(
    mapa: pd.DataFrame,
) -> pd.DataFrame:
    if mapa.empty:
        return pd.DataFrame()

    return mapa[
        mapa["Unidad candidata"] == "Estimación"
    ].copy()
