from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile

import pandas as pd


def analizar_duplicados_exactos(
    contenido_zip: bytes,
    inventario: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Detecta archivos físicamente idénticos en todo el expediente.

    Esta fase es completamente determinística:
    - no usa nombres para decidir duplicidad;
    - no usa IA;
    - no modifica el ZIP;
    - dos archivos solo se consideran duplicados exactos cuando su SHA-256
      calculado sobre los bytes completos es idéntico.

    Devuelve:
    1. detalle: una fila por archivo que pertenece a un grupo duplicado;
    2. grupos: una fila por grupo de duplicidad exacta;
    3. resumen: métricas generales del análisis.
    """
    if inventario.empty:
        resumen = {
            "archivos_evaluados": 0,
            "grupos_duplicados_exactos": 0,
            "archivos_en_duplicados": 0,
            "copias_adicionales": 0,
            "bytes_repetidos_adicionales": 0,
        }
        return pd.DataFrame(), pd.DataFrame(), resumen

    rutas_validas = set(
        inventario["Ruta original"].astype(str).tolist()
    )
    metadatos = (
        inventario.drop_duplicates(
            subset=["Ruta original"],
            keep="first",
        )
        .set_index("Ruta original")
        .to_dict("index")
    )

    huellas = []
    with ZipFile(BytesIO(contenido_zip)) as archivo_zip:
        for info in archivo_zip.infolist():
            if info.is_dir():
                continue

            ruta = str(info.filename)
            if ruta not in rutas_validas:
                continue

            with archivo_zip.open(info, "r") as origen:
                digest = sha256()
                while True:
                    bloque = origen.read(1024 * 1024)
                    if not bloque:
                        break
                    digest.update(bloque)

            meta = metadatos.get(ruta, {})
            huellas.append(
                {
                    "Ruta original": ruta,
                    "Archivo": str(meta.get("Archivo", "")),
                    "Carpeta": str(meta.get("Carpeta", "")),
                    "Extensión": str(meta.get("Extensión", "")),
                    "Tipo": str(meta.get("Tipo", "")),
                    "Tamaño (bytes)": int(
                        meta.get("Tamaño (bytes)", info.file_size) or 0
                    ),
                    "SHA-256": digest.hexdigest(),
                }
            )

    huellas_df = pd.DataFrame(huellas)
    if huellas_df.empty:
        resumen = {
            "archivos_evaluados": 0,
            "grupos_duplicados_exactos": 0,
            "archivos_en_duplicados": 0,
            "copias_adicionales": 0,
            "bytes_repetidos_adicionales": 0,
        }
        return pd.DataFrame(), pd.DataFrame(), resumen

    grupos_hash: dict[str, list[int]] = defaultdict(list)
    for indice, fila in huellas_df.iterrows():
        grupos_hash[str(fila["SHA-256"])].append(indice)

    grupos_repetidos = [
        (hash_archivo, indices)
        for hash_archivo, indices in grupos_hash.items()
        if len(indices) > 1
    ]

    grupos_repetidos.sort(
        key=lambda item: min(
            str(huellas_df.at[i, "Ruta original"]).casefold()
            for i in item[1]
        )
    )

    detalle_registros = []
    grupo_registros = []
    bytes_repetidos_adicionales = 0

    for numero, (hash_archivo, indices) in enumerate(
        grupos_repetidos,
        start=1,
    ):
        grupo_id = f"DUP_EXACTO_{numero:03d}"
        bloque = huellas_df.loc[indices].copy().sort_values(
            by="Ruta original",
            key=lambda serie: serie.astype(str).str.casefold(),
        )

        tamano = int(
            pd.to_numeric(
                bloque["Tamaño (bytes)"],
                errors="coerce",
            ).fillna(0).max()
        )
        cantidad = len(bloque)
        bytes_repetidos_adicionales += tamano * (cantidad - 1)

        rutas = bloque["Ruta original"].astype(str).tolist()
        archivos = bloque["Archivo"].astype(str).tolist()
        carpetas = bloque["Carpeta"].astype(str).tolist()

        grupo_registros.append(
            {
                "Grupo duplicado": grupo_id,
                "Archivos": cantidad,
                "Tamaño por archivo (bytes)": tamano,
                "Copias adicionales": cantidad - 1,
                "Rutas": " | ".join(rutas),
                "Nombres": " | ".join(archivos),
                "Carpetas": " | ".join(carpetas),
                "SHA-256": hash_archivo,
                "Relación detectada": "Duplicado exacto por contenido",
                "Acción provisional": (
                    "No decidir eliminación ni codificación todavía; "
                    "conservar relación para resolución global posterior."
                ),
            }
        )

        for _, fila in bloque.iterrows():
            detalle_registros.append(
                {
                    "Grupo duplicado": grupo_id,
                    "Ruta original": fila["Ruta original"],
                    "Archivo": fila["Archivo"],
                    "Carpeta": fila["Carpeta"],
                    "Extensión": fila["Extensión"],
                    "Tipo": fila["Tipo"],
                    "Tamaño (bytes)": int(
                        fila["Tamaño (bytes)"]
                    ),
                    "SHA-256": hash_archivo,
                    "Relación detectada": (
                        "Duplicado exacto por contenido"
                    ),
                }
            )

    detalle = pd.DataFrame(detalle_registros)
    grupos = pd.DataFrame(grupo_registros)

    archivos_en_duplicados = int(
        grupos["Archivos"].sum()
    ) if not grupos.empty else 0
    copias_adicionales = int(
        grupos["Copias adicionales"].sum()
    ) if not grupos.empty else 0

    resumen = {
        "archivos_evaluados": len(huellas_df),
        "grupos_duplicados_exactos": len(grupos),
        "archivos_en_duplicados": archivos_en_duplicados,
        "copias_adicionales": copias_adicionales,
        "bytes_repetidos_adicionales": bytes_repetidos_adicionales,
    }

    return detalle, grupos, resumen
