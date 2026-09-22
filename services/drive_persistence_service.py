from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload


DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file"
TOKEN_URI = "https://oauth2.googleapis.com/token"
FOLDER_MIME = "application/vnd.google-apps.folder"


class DrivePersistenceError(RuntimeError):
    pass


def drive_configurado() -> bool:
    try:
        secretos = st.secrets["google_drive"]
        requeridos = ("client_id", "client_secret", "refresh_token")
        return all(str(secretos.get(k, "")).strip() for k in requeridos)
    except Exception:
        return False


def _servicio_drive():
    if not drive_configurado():
        raise DrivePersistenceError(
            "Google Drive no está configurado en Streamlit Secrets."
        )

    secretos = st.secrets["google_drive"]

    credenciales = Credentials(
        token=None,
        refresh_token=str(secretos["refresh_token"]),
        token_uri=TOKEN_URI,
        client_id=str(secretos["client_id"]),
        client_secret=str(secretos["client_secret"]),
        scopes=[DRIVE_SCOPE],
    )

    return build(
        "drive",
        "v3",
        credentials=credenciales,
        cache_discovery=False,
    )


def _root_folder_name() -> str:
    try:
        valor = str(
            st.secrets["google_drive"].get(
                "root_folder_name",
                "ASEG_CLASIFICADOR_EXPEDIENTES",
            )
        ).strip()
        return valor or "ASEG_CLASIFICADOR_EXPEDIENTES"
    except Exception:
        return "ASEG_CLASIFICADOR_EXPEDIENTES"


def _escapar_q(valor: str) -> str:
    return valor.replace("\\", "\\\\").replace("'", "\\'")


def probar_conexion() -> None:
    try:
        servicio = _servicio_drive()
        servicio.files().list(
            pageSize=1,
            spaces="drive",
            fields="files(id)",
        ).execute()
    except Exception as error:
        raise DrivePersistenceError(
            f"No fue posible autenticar con Google Drive: {error}"
        ) from error


def _buscar_hijo(
    servicio,
    parent_id: str,
    nombre: str,
    mime_type: str | None = None,
):
    condiciones = [
        f"'{_escapar_q(parent_id)}' in parents",
        f"name = '{_escapar_q(nombre)}'",
        "trashed = false",
    ]
    if mime_type:
        condiciones.append(f"mimeType = '{_escapar_q(mime_type)}'")

    respuesta = servicio.files().list(
        q=" and ".join(condiciones),
        spaces="drive",
        fields="files(id,name,mimeType,modifiedTime,appProperties)",
        pageSize=10,
    ).execute()

    archivos = respuesta.get("files", [])
    return archivos[0] if archivos else None


def _obtener_o_crear_root(servicio) -> str:
    nombre = _root_folder_name()

    respuesta = servicio.files().list(
        q=(
            f"name = '{_escapar_q(nombre)}' and "
            f"mimeType = '{FOLDER_MIME}' and trashed = false"
        ),
        spaces="drive",
        fields="files(id,name)",
        pageSize=10,
    ).execute()

    carpetas = respuesta.get("files", [])
    if carpetas:
        return carpetas[0]["id"]

    metadata = {
        "name": nombre,
        "mimeType": FOLDER_MIME,
        "appProperties": {"aseg_type": "root"},
    }
    creada = servicio.files().create(
        body=metadata,
        fields="id",
    ).execute()
    return creada["id"]


def _nombre_carpeta_expediente(nombre_zip: str, tamano: int) -> tuple[str, str]:
    stem = Path(nombre_zip).stem
    stem_seguro = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    stem_seguro = stem_seguro or "expediente"

    clave = hashlib.sha1(
        f"{nombre_zip}:{tamano}".encode("utf-8")
    ).hexdigest()[:10]

    return f"{stem_seguro}__{clave}", clave


def _obtener_o_crear_carpeta_expediente(
    servicio,
    root_id: str,
    nombre_zip: str,
    procedimiento: str,
    tamano: int,
) -> tuple[str, str]:
    nombre_carpeta, clave = _nombre_carpeta_expediente(nombre_zip, tamano)

    existente = _buscar_hijo(
        servicio,
        root_id,
        nombre_carpeta,
        FOLDER_MIME,
    )
    if existente:
        return existente["id"], clave

    metadata = {
        "name": nombre_carpeta,
        "mimeType": FOLDER_MIME,
        "parents": [root_id],
        "appProperties": {
            "aseg_type": "expediente",
            "expediente_key": clave,
            "nombre_zip": nombre_zip[:120],
            "procedimiento": procedimiento,
        },
    }

    creada = servicio.files().create(
        body=metadata,
        fields="id",
    ).execute()

    return creada["id"], clave


def _subir_o_actualizar(
    servicio,
    folder_id: str,
    nombre: str,
    contenido: bytes,
    mime_type: str,
) -> str:
    existente = _buscar_hijo(servicio, folder_id, nombre)

    stream = io.BytesIO(contenido)
    es_grande = len(contenido) > 8 * 1024 * 1024

    media = MediaIoBaseUpload(
        stream,
        mimetype=mime_type,
        resumable=es_grande,
        chunksize=8 * 1024 * 1024 if es_grande else -1,
    )

    if existente:
        request = servicio.files().update(
            fileId=existente["id"],
            media_body=media,
            fields="id",
        )
    else:
        request = servicio.files().create(
            body={
                "name": nombre,
                "parents": [folder_id],
            },
            media_body=media,
            fields="id",
        )

    if not es_grande:
        return request.execute()["id"]

    respuesta = None
    while respuesta is None:
        _, respuesta = request.next_chunk()

    return respuesta["id"]


def _descargar_archivo(servicio, file_id: str) -> bytes:
    request = servicio.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)

    terminado = False
    while not terminado:
        _, terminado = downloader.next_chunk()

    return buffer.getvalue()


def _leer_archivo_por_nombre(
    servicio,
    folder_id: str,
    nombre: str,
) -> bytes | None:
    archivo = _buscar_hijo(servicio, folder_id, nombre)
    if not archivo:
        return None
    return _descargar_archivo(servicio, archivo["id"])


def _json_bytes(datos) -> bytes:
    return json.dumps(
        datos,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")


def _df_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def guardar_expediente(
    nombre_zip: str,
    contenido_zip: bytes,
    procedimiento: str,
    inventario: pd.DataFrame,
    resumen: dict,
) -> dict:
    try:
        servicio = _servicio_drive()
        root_id = _obtener_o_crear_root(servicio)
        folder_id, clave = _obtener_o_crear_carpeta_expediente(
            servicio,
            root_id,
            nombre_zip,
            procedimiento,
            len(contenido_zip),
        )

        _subir_o_actualizar(
            servicio,
            folder_id,
            "expediente.zip",
            contenido_zip,
            "application/zip",
        )

        _subir_o_actualizar(
            servicio,
            folder_id,
            "inventario.csv",
            _df_csv_bytes(inventario),
            "text/csv",
        )

        estado = {
            "version": 1,
            "expediente_id": f"{nombre_zip}:{len(contenido_zip)}",
            "expediente_key": clave,
            "expediente_nombre": nombre_zip,
            "procedimiento": procedimiento,
            "resumen_expediente": resumen,
        }

        _subir_o_actualizar(
            servicio,
            folder_id,
            "estado.json",
            _json_bytes(estado),
            "application/json",
        )

        return {
            "folder_id": folder_id,
            "estado": estado,
        }

    except DrivePersistenceError:
        raise
    except Exception as error:
        raise DrivePersistenceError(
            f"No fue posible guardar el expediente en Drive: {error}"
        ) from error


def guardar_diagnostico(
    folder_id: str,
    analisis_pdf: pd.DataFrame,
    resumen_pdf: dict,
) -> None:
    try:
        servicio = _servicio_drive()
        _subir_o_actualizar(
            servicio,
            folder_id,
            "diagnostico_pdf.csv",
            _df_csv_bytes(analisis_pdf),
            "text/csv",
        )
        _subir_o_actualizar(
            servicio,
            folder_id,
            "resumen_pdf.json",
            _json_bytes(resumen_pdf),
            "application/json",
        )
    except Exception as error:
        raise DrivePersistenceError(
            f"No fue posible guardar el diagnóstico en Drive: {error}"
        ) from error


def guardar_ocr(
    folder_id: str,
    resultado_ocr: pd.DataFrame,
) -> None:
    try:
        servicio = _servicio_drive()
        _subir_o_actualizar(
            servicio,
            folder_id,
            "ocr.csv",
            _df_csv_bytes(resultado_ocr),
            "text/csv",
        )
    except Exception as error:
        raise DrivePersistenceError(
            f"No fue posible guardar el OCR en Drive: {error}"
        ) from error


def guardar_ground_truth(
    folder_id: str,
    ground_truth: dict,
) -> None:
    try:
        servicio = _servicio_drive()
        _subir_o_actualizar(
            servicio,
            folder_id,
            "ground_truth.json",
            _json_bytes(ground_truth),
            "application/json",
        )
    except Exception as error:
        raise DrivePersistenceError(
            f"No fue posible guardar las referencias en Drive: {error}"
        ) from error


def listar_expedientes() -> list[dict]:
    try:
        servicio = _servicio_drive()
        root_id = _obtener_o_crear_root(servicio)

        respuesta = servicio.files().list(
            q=(
                f"'{_escapar_q(root_id)}' in parents and "
                f"mimeType = '{FOLDER_MIME}' and trashed = false"
            ),
            spaces="drive",
            fields=(
                "files(id,name,modifiedTime,appProperties)"
            ),
            orderBy="modifiedTime desc",
            pageSize=100,
        ).execute()

        resultado = []
        for carpeta in respuesta.get("files", []):
            props = carpeta.get("appProperties") or {}
            if props.get("aseg_type") != "expediente":
                continue

            resultado.append(
                {
                    "folder_id": carpeta["id"],
                    "folder_name": carpeta["name"],
                    "modified_time": carpeta.get("modifiedTime", ""),
                    "nombre_zip": props.get(
                        "nombre_zip",
                        carpeta["name"],
                    ),
                    "procedimiento": props.get("procedimiento", ""),
                }
            )

        return resultado

    except Exception as error:
        raise DrivePersistenceError(
            f"No fue posible consultar expedientes guardados: {error}"
        ) from error


def cargar_expediente(folder_id: str) -> dict:
    try:
        servicio = _servicio_drive()

        estado_raw = _leer_archivo_por_nombre(
            servicio,
            folder_id,
            "estado.json",
        )
        zip_raw = _leer_archivo_por_nombre(
            servicio,
            folder_id,
            "expediente.zip",
        )
        inventario_raw = _leer_archivo_por_nombre(
            servicio,
            folder_id,
            "inventario.csv",
        )

        if not estado_raw or not zip_raw or not inventario_raw:
            raise DrivePersistenceError(
                "El expediente guardado está incompleto en Google Drive."
            )

        estado = json.loads(estado_raw.decode("utf-8"))
        inventario = pd.read_csv(io.BytesIO(inventario_raw))

        resultado = {
            "drive_folder_id": folder_id,
            "contenido_zip": zip_raw,
            "inventario": inventario,
            **estado,
        }

        diagnostico_raw = _leer_archivo_por_nombre(
            servicio,
            folder_id,
            "diagnostico_pdf.csv",
        )
        resumen_pdf_raw = _leer_archivo_por_nombre(
            servicio,
            folder_id,
            "resumen_pdf.json",
        )

        if diagnostico_raw and resumen_pdf_raw:
            resultado["analisis_pdf"] = pd.read_csv(
                io.BytesIO(diagnostico_raw)
            )
            resultado["resumen_pdf"] = json.loads(
                resumen_pdf_raw.decode("utf-8")
            )

        ocr_raw = _leer_archivo_por_nombre(
            servicio,
            folder_id,
            "ocr.csv",
        )
        if ocr_raw:
            resultado["resultado_ocr"] = pd.read_csv(
                io.BytesIO(ocr_raw)
            )

        ground_truth_raw = _leer_archivo_por_nombre(
            servicio,
            folder_id,
            "ground_truth.json",
        )
        if ground_truth_raw:
            resultado["ground_truth_manual"] = json.loads(
                ground_truth_raw.decode("utf-8")
            )

        return resultado

    except DrivePersistenceError:
        raise
    except Exception as error:
        raise DrivePersistenceError(
            f"No fue posible restaurar el expediente: {error}"
        ) from error



def guardar_clasificacion_jev(
    folder_id: str,
    resultados: pd.DataFrame,
) -> None:
    try:
        servicio = _servicio_drive()
        serializable = resultados.copy()
        if "Top 3" in serializable.columns:
            serializable["Top 3"] = serializable["Top 3"].apply(
                lambda valor: json.dumps(valor, ensure_ascii=False)
            )

        _subir_o_actualizar(
            servicio,
            folder_id,
            "clasificacion_jev.csv",
            _df_csv_bytes(serializable),
            "text/csv",
        )
    except Exception as error:
        raise DrivePersistenceError(
            f"No fue posible guardar la clasificación de Jev en Drive: {error}"
        ) from error
