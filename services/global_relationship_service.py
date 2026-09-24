from __future__ import annotations

from collections import defaultdict
from hashlib import blake2b, sha256
from io import BytesIO
from itertools import combinations
import re
import unicodedata
from zipfile import ZipFile

import cv2
import fitz
import numpy as np
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


MIN_CARACTERES_TEXTO_SIMILITUD = 120
MAX_CARACTERES_TEXTO_SIMILITUD = 250_000
MAX_SHINGLES_FIRMA = 5_000
N_SHINGLE = 5
UMBRAL_SIMILITUD_MEDIA = 0.68
UMBRAL_SIMILITUD_ALTA = 0.80
UMBRAL_SIMILITUD_MUY_ALTA = 0.92


def _normalizar_texto_similitud(texto: str) -> str:
    valor = unicodedata.normalize("NFKD", str(texto))
    valor = "".join(
        caracter
        for caracter in valor
        if not unicodedata.combining(caracter)
    )
    valor = valor.lower()
    valor = re.sub(r"[^a-z0-9]+", " ", valor)
    return re.sub(r"\s+", " ", valor).strip()


def _recortar_texto_similitud(texto: str) -> str:
    texto = str(texto)
    if len(texto) <= MAX_CARACTERES_TEXTO_SIMILITUD:
        return texto

    inicio = 200_000
    final = MAX_CARACTERES_TEXTO_SIMILITUD - inicio
    return texto[:inicio] + " " + texto[-final:]


def _firma_shingles(texto_normalizado: str) -> frozenset[int]:
    tokens = str(texto_normalizado).split()
    if not tokens:
        return frozenset()

    if len(tokens) < N_SHINGLE:
        fragmentos = tokens
    else:
        fragmentos = (
            " ".join(tokens[i : i + N_SHINGLE])
            for i in range(len(tokens) - N_SHINGLE + 1)
        )

    valores = set()
    for fragmento in fragmentos:
        digest = blake2b(
            str(fragmento).encode("utf-8"),
            digest_size=8,
        ).digest()
        valores.add(int.from_bytes(digest, "big"))

    if len(valores) > MAX_SHINGLES_FIRMA:
        valores = set(sorted(valores)[:MAX_SHINGLES_FIRMA])

    return frozenset(valores)


def _paginas_muestra_visual(total_paginas: int) -> list[int]:
    total = int(total_paginas)
    if total <= 0:
        return []
    if total <= 5:
        return list(range(total))

    posiciones = (0.0, 0.25, 0.5, 0.75, 1.0)
    return sorted(
        {
            int(round((total - 1) * posicion))
            for posicion in posiciones
        }
    )


def _hash_visual_pagina(pagina: fitz.Page) -> int | None:
    pixmap = pagina.get_pixmap(
        matrix=fitz.Matrix(0.18, 0.18),
        colorspace=fitz.csGRAY,
        alpha=False,
    )

    if pixmap.width <= 0 or pixmap.height <= 0:
        return None

    matriz = np.frombuffer(
        pixmap.samples,
        dtype=np.uint8,
    ).reshape(pixmap.height, pixmap.width)

    reducida = cv2.resize(
        matriz,
        (16, 16),
        interpolation=cv2.INTER_AREA,
    )

    # Las páginas prácticamente vacías no aportan una huella visual fiable.
    if float(reducida.std()) < 4.0:
        return None

    media = float(reducida.mean())
    bits = (reducida >= media).reshape(-1)

    valor = 0
    for bit in bits:
        valor = (valor << 1) | int(bool(bit))

    return valor


def _similitud_hash_visual(hash_a: int, hash_b: int) -> float:
    distancia = int(hash_a ^ hash_b).bit_count()
    return 1.0 - (distancia / 256.0)


def _similitud_visual_documentos(
    hashes_a: tuple[int, ...],
    hashes_b: tuple[int, ...],
) -> float:
    if not hashes_a or not hashes_b:
        return 0.0

    menor, mayor = (
        (hashes_a, hashes_b)
        if len(hashes_a) <= len(hashes_b)
        else (hashes_b, hashes_a)
    )

    mejores = []
    for hash_menor in menor:
        mejores.append(
            max(
                _similitud_hash_visual(
                    hash_menor,
                    hash_mayor,
                )
                for hash_mayor in mayor
            )
        )

    return float(sum(mejores) / len(mejores))


def _similitud_textual_firmas(
    firma_a: frozenset[int],
    firma_b: frozenset[int],
) -> tuple[float, float, float]:
    if not firma_a or not firma_b:
        return 0.0, 0.0, 0.0

    interseccion = len(firma_a.intersection(firma_b))
    union = len(firma_a.union(firma_b))
    menor = min(len(firma_a), len(firma_b))

    jaccard = interseccion / union if union else 0.0
    contencion = interseccion / menor if menor else 0.0

    # La contención permite detectar un extracto dentro de un documento mayor;
    # Jaccard evita que una pequeña coincidencia domine por sí sola.
    puntuacion = 0.65 * contencion + 0.35 * jaccard
    return jaccard, contencion, puntuacion


def _nivel_similitud(puntuacion: float) -> str:
    if puntuacion >= UMBRAL_SIMILITUD_MUY_ALTA:
        return "MUY ALTA"
    if puntuacion >= UMBRAL_SIMILITUD_ALTA:
        return "ALTA"
    if puntuacion >= UMBRAL_SIMILITUD_MEDIA:
        return "MEDIA"
    return "BAJA"


def _relacion_provisional_similitud(
    puntuacion: float,
    jaccard: float,
    contencion: float,
    similitud_visual: float,
    paginas_a: int,
    paginas_b: int,
) -> str:
    paginas_mayor = max(int(paginas_a), int(paginas_b), 1)
    paginas_menor = min(int(paginas_a), int(paginas_b))
    proporcion_paginas = paginas_menor / paginas_mayor

    if (
        puntuacion >= UMBRAL_SIMILITUD_MUY_ALTA
        and proporcion_paginas >= 0.90
        and (
            jaccard >= 0.85
            or similitud_visual >= 0.96
        )
    ):
        return "Probable copia documental"

    if (
        contencion >= 0.90
        and proporcion_paginas < 0.85
    ):
        return "Posible parcial / extracto"

    if puntuacion >= UMBRAL_SIMILITUD_ALTA:
        return "Probable variante / versión"

    return "Candidato a revisión"


def _perfil_pdf_similitud(
    pdf_bytes: bytes,
    ruta: str,
    archivo: str,
    carpeta: str,
    tamano_bytes: int,
) -> dict:
    documento = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    try:
        total_paginas = int(documento.page_count)
        textos = []

        for indice in range(total_paginas):
            texto = documento.load_page(indice).get_text(
                "text"
            )
            if texto:
                textos.append(texto)

        texto_completo = _recortar_texto_similitud(
            "\n".join(textos)
        )
        texto_normalizado = _normalizar_texto_similitud(
            texto_completo
        )
        caracteres_utiles = len(
            "".join(texto_normalizado.split())
        )
        firma = _firma_shingles(texto_normalizado)

        hashes_visuales = []
        paginas_visuales = []
        for indice in _paginas_muestra_visual(
            total_paginas
        ):
            hash_pagina = _hash_visual_pagina(
                documento.load_page(indice)
            )
            if hash_pagina is not None:
                hashes_visuales.append(hash_pagina)
                paginas_visuales.append(indice + 1)

        texto_utilizable = (
            caracteres_utiles
            >= MIN_CARACTERES_TEXTO_SIMILITUD
            and len(firma) >= 10
        )

        return {
            "Ruta original": ruta,
            "Archivo": archivo,
            "Carpeta": carpeta,
            "Tamaño (bytes)": int(tamano_bytes),
            "Páginas": total_paginas,
            "Caracteres texto útiles": caracteres_utiles,
            "Texto utilizable": bool(texto_utilizable),
            "Páginas visuales usadas": ", ".join(
                str(p) for p in paginas_visuales
            ),
            "Firma texto": firma,
            "Hashes visuales": tuple(hashes_visuales),
            "Estado perfil": "OK",
            "Error perfil": "",
        }
    finally:
        documento.close()


def analizar_similitud_global_pdfs(
    contenido_zip: bytes,
    inventario: pd.DataFrame,
    on_progress=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Segunda fase global: genera pares sospechosos de PDFs similares.

    No usa IA, OCR ni nombres/rutas como señal de similitud. Combina:
    - similitud de shingles del texto nativo;
    - contención textual para detectar posibles extractos;
    - huellas visuales perceptuales sobre páginas muestreadas;
    - número de páginas únicamente como señal descriptiva/relacional.

    El resultado NO declara duplicados definitivos: produce candidatos para
    validación posterior.
    """
    if inventario.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "pdf_evaluados": 0,
            "pdf_con_texto": 0,
            "pdf_solo_visual": 0,
            "pdf_con_error": 0,
            "pares_comparados": 0,
            "pares_candidatos": 0,
            "muy_alta": 0,
            "alta": 0,
            "media": 0,
        }

    pdfs = inventario[
        inventario["Extensión"]
        .astype(str)
        .str.lower()
        .eq(".pdf")
    ].copy().reset_index(drop=True)

    if pdfs.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "pdf_evaluados": 0,
            "pdf_con_texto": 0,
            "pdf_solo_visual": 0,
            "pdf_con_error": 0,
            "pares_comparados": 0,
            "pares_candidatos": 0,
            "muy_alta": 0,
            "alta": 0,
            "media": 0,
        }

    perfiles = []
    total = len(pdfs)

    with ZipFile(BytesIO(contenido_zip)) as archivo_zip:
        for posicion, (_, fila) in enumerate(
            pdfs.iterrows(),
            start=1,
        ):
            ruta = str(fila["Ruta original"])

            try:
                pdf_bytes = archivo_zip.read(ruta)
                perfil = _perfil_pdf_similitud(
                    pdf_bytes=pdf_bytes,
                    ruta=ruta,
                    archivo=str(fila["Archivo"]),
                    carpeta=str(fila["Carpeta"]),
                    tamano_bytes=int(
                        fila["Tamaño (bytes)"]
                    ),
                )
            except Exception as error:
                perfil = {
                    "Ruta original": ruta,
                    "Archivo": str(fila["Archivo"]),
                    "Carpeta": str(fila["Carpeta"]),
                    "Tamaño (bytes)": int(
                        fila["Tamaño (bytes)"]
                    ),
                    "Páginas": 0,
                    "Caracteres texto útiles": 0,
                    "Texto utilizable": False,
                    "Páginas visuales usadas": "",
                    "Firma texto": frozenset(),
                    "Hashes visuales": tuple(),
                    "Estado perfil": "ERROR",
                    "Error perfil": str(error),
                }

            perfiles.append(perfil)

            if on_progress is not None:
                on_progress(
                    "perfil",
                    posicion,
                    total,
                    str(fila["Archivo"]),
                )

    pares_registros = []
    pares_totales = total * (total - 1) // 2

    for posicion, (indice_a, indice_b) in enumerate(
        combinations(range(len(perfiles)), 2),
        start=1,
    ):
        a = perfiles[indice_a]
        b = perfiles[indice_b]

        if (
            a["Estado perfil"] != "OK"
            or b["Estado perfil"] != "OK"
        ):
            continue

        ambos_texto = bool(
            a["Texto utilizable"]
            and b["Texto utilizable"]
        )

        if ambos_texto:
            jaccard, contencion, texto_score = (
                _similitud_textual_firmas(
                    a["Firma texto"],
                    b["Firma texto"],
                )
            )
        else:
            jaccard = 0.0
            contencion = 0.0
            texto_score = 0.0

        visual_score = _similitud_visual_documentos(
            a["Hashes visuales"],
            b["Hashes visuales"],
        )

        if ambos_texto:
            # La señal visual puede reforzar, pero nunca rebaja una señal
            # textual fuerte causada por reexportación o recomposición del PDF.
            puntuacion = max(
                texto_score,
                0.80 * texto_score
                + 0.20 * visual_score,
            )
            fuente = (
                "Texto nativo + visual"
                if visual_score > 0
                else "Texto nativo"
            )
        else:
            puntuacion = visual_score
            fuente = "Visual"

        nivel = _nivel_similitud(puntuacion)

        if nivel != "BAJA":
            relacion = _relacion_provisional_similitud(
                puntuacion=puntuacion,
                jaccard=jaccard,
                contencion=contencion,
                similitud_visual=visual_score,
                paginas_a=int(a["Páginas"]),
                paginas_b=int(b["Páginas"]),
            )

            pares_registros.append(
                {
                    "Nivel": nivel,
                    "Puntuación global (%)": round(
                        puntuacion * 100,
                        2,
                    ),
                    "Similitud texto Jaccard (%)": round(
                        jaccard * 100,
                        2,
                    ),
                    "Contención textual (%)": round(
                        contencion * 100,
                        2,
                    ),
                    "Similitud visual (%)": round(
                        visual_score * 100,
                        2,
                    ),
                    "Fuente principal": fuente,
                    "Relación provisional": relacion,
                    "Archivo A": a["Archivo"],
                    "Ruta A": a["Ruta original"],
                    "Páginas A": int(a["Páginas"]),
                    "Archivo B": b["Archivo"],
                    "Ruta B": b["Ruta original"],
                    "Páginas B": int(b["Páginas"]),
                    "Acción provisional": (
                        "No codificar ni eliminar por esta señal. "
                        "Validar relación documental en la siguiente etapa."
                    ),
                }
            )

        if (
            on_progress is not None
            and (
                posicion == pares_totales
                or posicion % 250 == 0
            )
        ):
            on_progress(
                "comparacion",
                posicion,
                pares_totales,
                f"{len(pares_registros)} candidato(s)",
            )

    pares = pd.DataFrame(pares_registros)
    if not pares.empty:
        orden_nivel = {
            "MUY ALTA": 0,
            "ALTA": 1,
            "MEDIA": 2,
        }
        pares["_orden_nivel"] = (
            pares["Nivel"].map(orden_nivel)
        )
        pares = (
            pares.sort_values(
                by=[
                    "_orden_nivel",
                    "Puntuación global (%)",
                ],
                ascending=[True, False],
            )
            .drop(columns=["_orden_nivel"])
            .reset_index(drop=True)
        )

    perfiles_publicos = pd.DataFrame(perfiles)
    if not perfiles_publicos.empty:
        perfiles_publicos = perfiles_publicos.drop(
            columns=["Firma texto", "Hashes visuales"],
            errors="ignore",
        )

    pdf_con_texto = sum(
        1
        for perfil in perfiles
        if perfil["Estado perfil"] == "OK"
        and perfil["Texto utilizable"]
    )
    pdf_con_error = sum(
        1
        for perfil in perfiles
        if perfil["Estado perfil"] != "OK"
    )
    pdf_solo_visual = sum(
        1
        for perfil in perfiles
        if perfil["Estado perfil"] == "OK"
        and not perfil["Texto utilizable"]
        and bool(perfil["Hashes visuales"])
    )

    resumen = {
        "pdf_evaluados": len(perfiles),
        "pdf_con_texto": pdf_con_texto,
        "pdf_solo_visual": pdf_solo_visual,
        "pdf_con_error": pdf_con_error,
        "pares_comparados": pares_totales,
        "pares_candidatos": len(pares),
        "muy_alta": (
            int((pares["Nivel"] == "MUY ALTA").sum())
            if not pares.empty
            else 0
        ),
        "alta": (
            int((pares["Nivel"] == "ALTA").sum())
            if not pares.empty
            else 0
        ),
        "media": (
            int((pares["Nivel"] == "MEDIA").sum())
            if not pares.empty
            else 0
        ),
    }

    return perfiles_publicos, pares, resumen
