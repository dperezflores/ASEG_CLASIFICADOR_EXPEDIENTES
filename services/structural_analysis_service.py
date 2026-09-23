from __future__ import annotations

import re
import unicodedata
from pathlib import PurePosixPath

import pandas as pd

from services.expediente_service import es_archivo_ignorado


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


def _inventario_analitico(inventario: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la limpieza analítica incluso a inventarios persistidos antes de
    que existiera el filtro de archivos técnicos.
    """
    if inventario.empty:
        return inventario.copy()

    mascara = ~inventario["Archivo"].astype(str).apply(es_archivo_ignorado)
    return inventario[mascara].copy()


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
    inventario = _inventario_analitico(inventario)
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



def obtener_archivos_unidad(
    inventario: pd.DataFrame,
    ruta_carpeta: str,
) -> pd.DataFrame:
    """
    Devuelve únicamente los archivos directamente contenidos en una unidad
    documental. No incorpora archivos de subcarpetas y no usa IA.
    """
    inventario = _inventario_analitico(inventario)

    resultado = inventario[
        inventario["Carpeta"].astype(str) == str(ruta_carpeta)
    ].copy()

    if resultado.empty:
        return resultado

    resultado = resultado.sort_values(
        by="Archivo",
        key=lambda serie: serie.astype(str).str.lower(),
    ).reset_index(drop=True)

    return resultado


def construir_vista_unidad_estimacion(
    inventario: pd.DataFrame,
    ruta_carpeta: str,
    procedimiento: str,
    consecutivo: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Primera interpretación interna de una estimación.

    Regla deliberadamente limitada:
    - CARATULA / CARÁTULA se marca como candidato a documento principal.
    - El resto se conserva como componente de la unidad pendiente de análisis.

    Todavía no se decide qué otros componentes tienen código propio.
    """
    archivos = obtener_archivos_unidad(inventario, ruta_carpeta)

    if archivos.empty:
        return archivos, {}

    filas = []
    for _, fila in archivos.iterrows():
        nombre = str(fila["Archivo"])
        raiz = PurePosixPath(nombre).stem
        normalizado = _normalizar(raiz)

        if normalizado == "CARATULA":
            rol = "Candidato a documento principal"
            motivo = (
                "El nombre del archivo coincide con la carátula de la "
                "estimación. Debe validarse su contenido en el siguiente paso."
            )
        else:
            rol = "Componente de la unidad"
            motivo = (
                "Pertenece a la carpeta de la estimación, pero todavía no se "
                "ha determinado si tiene código propio en el catálogo."
            )

        filas.append(
            {
                "Archivo": nombre,
                "Tipo": fila["Tipo"],
                "Tamaño (bytes)": fila["Tamaño (bytes)"],
                "Rol preliminar": rol,
                "Motivo": motivo,
            }
        )

    codigo_unidad = f"EJE_{procedimiento}_EST_{int(consecutivo)}"

    resumen = {
        "tipo_unidad": "Estimación",
        "consecutivo": int(consecutivo),
        "ruta_origen": str(ruta_carpeta),
        "codigo_unidad_candidato": codigo_unidad,
        "archivos": len(filas),
    }

    return pd.DataFrame(filas), resumen
