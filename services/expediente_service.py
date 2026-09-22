from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

import pandas as pd

from config.constants import CARPETAS_IGNORADAS


TIPOS_ARCHIVO = {
    ".pdf": "PDF",
    ".xlsx": "Excel",
    ".xls": "Excel",
    ".xlsm": "Excel",
    ".docx": "Word",
    ".doc": "Word",
    ".dwg": "CAD",
    ".dxf": "CAD",
    ".jpg": "Imagen",
    ".jpeg": "Imagen",
    ".png": "Imagen",
    ".tif": "Imagen",
    ".tiff": "Imagen",
}


def _ruta_ignorada(ruta: PurePosixPath) -> bool:
    """
    Indica si una ruta pertenece a una carpeta que no debe analizarse.
    La comparación se hace contra cada parte de la ruta.
    """
    return any(parte in CARPETAS_IGNORADAS for parte in ruta.parts)


def _clasificar_extension(extension: str) -> str:
    """
    Asigna una categoría general a partir de la extensión del archivo.
    """
    if not extension:
        return "Sin extensión"

    return TIPOS_ARCHIVO.get(extension.lower(), "Otro")


def inventariar_expediente_zip(contenido_zip: bytes) -> tuple[pd.DataFrame, dict]:
    """
    Lee la estructura interna de un ZIP sin extraer ni modificar sus archivos.

    Devuelve:
    1. Un DataFrame con un registro por archivo.
    2. Un resumen con totales del expediente.
    """
    registros = []
    carpetas = set()

    try:
        with ZipFile(BytesIO(contenido_zip)) as archivo_zip:
            for info in archivo_zip.infolist():
                ruta = PurePosixPath(info.filename)

                # Ignoramos entradas auxiliares creadas por algunos sistemas.
                if not ruta.parts or ruta.parts[0] == "__MACOSX":
                    continue

                if _ruta_ignorada(ruta):
                    continue

                if info.is_dir():
                    carpetas.add(str(ruta).rstrip("/"))
                    continue

                # Guardamos también las carpetas implícitas de cada archivo.
                padres = ruta.parents
                for padre in padres:
                    if str(padre) not in ("", "."):
                        carpetas.add(str(padre))

                extension = ruta.suffix.lower()
                carpeta = str(ruta.parent)
                if carpeta == ".":
                    carpeta = "/"

                registros.append(
                    {
                        "Ruta original": str(ruta),
                        "Carpeta": carpeta,
                        "Archivo": ruta.name,
                        "Extensión": extension if extension else "—",
                        "Tipo": _clasificar_extension(extension),
                        "Tamaño (bytes)": info.file_size,
                    }
                )

    except BadZipFile as error:
        raise ValueError(
            "El archivo cargado no es un ZIP válido o está dañado."
        ) from error

    inventario = pd.DataFrame(registros)

    if inventario.empty:
        resumen = {
            "archivos": 0,
            "carpetas": len(carpetas),
            "tamano_bytes": 0,
            "tipos": {},
        }
        return inventario, resumen

    tipos = Counter(inventario["Tipo"])

    resumen = {
        "archivos": len(inventario),
        "carpetas": len(carpetas),
        "tamano_bytes": int(inventario["Tamaño (bytes)"].sum()),
        "tipos": dict(sorted(tipos.items())),
    }

    return inventario, resumen


def formatear_tamano(numero_bytes: int) -> str:
    """
    Convierte bytes a una unidad legible.
    """
    unidades = ("B", "KB", "MB", "GB", "TB")
    valor = float(numero_bytes)

    for unidad in unidades:
        if valor < 1024 or unidad == unidades[-1]:
            return f"{valor:.2f} {unidad}"
        valor /= 1024

    return f"{numero_bytes} B"
