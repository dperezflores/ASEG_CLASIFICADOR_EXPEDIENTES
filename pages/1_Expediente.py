import streamlit as st

from config.constants import CARPETAS_IGNORADAS
from services.catalog_service import cargar_catalogo, obtener_hojas_catalogo
from services.drive_persistence_service import (
    DrivePersistenceError,
    cargar_expediente,
    drive_configurado,
    guardar_expediente,
    listar_expedientes,
    probar_conexion,
)
from services.expediente_service import formatear_tamano, inventariar_expediente_zip
from ui.common import (
    PROCEDIMIENTOS,
    expediente_activo,
    mostrar_encabezado,
)


CLAVES_EXPEDIENTE = (
    "expediente_id",
    "expediente_nombre",
    "contenido_zip",
    "inventario",
    "resumen_expediente",
    "procedimiento",
    "drive_folder_id",
    "analisis_pdf",
    "resumen_pdf",
    "resultado_ocr",
    "ground_truth_manual",
)


def _limpiar_expediente():
    for clave in CLAVES_EXPEDIENTE:
        st.session_state.pop(clave, None)


def _restaurar_en_sesion(datos: dict):
    _limpiar_expediente()

    claves = (
        "expediente_id",
        "expediente_nombre",
        "contenido_zip",
        "inventario",
        "resumen_expediente",
        "procedimiento",
        "drive_folder_id",
        "analisis_pdf",
        "resumen_pdf",
        "resultado_ocr",
        "ground_truth_manual",
    )

    for clave in claves:
        if clave in datos:
            st.session_state[clave] = datos[clave]


mostrar_encabezado(
    "Expediente",
    "Carga, persistencia e inventario del expediente de obra pública",
)

if not drive_configurado():
    st.error(
        "Google Drive no está configurado en Streamlit Secrets. "
        "La persistencia está deshabilitada."
    )
    st.stop()

with st.expander("Estado de Google Drive"):
    if st.button("Probar conexión con Google Drive"):
        try:
            probar_conexion()
            st.success("Conexión con Google Drive correcta.")
        except DrivePersistenceError as error:
            st.error(str(error))

procedimiento_actual = st.session_state.get("procedimiento", "DIR")

procedimiento = st.selectbox(
    "Tipo de procedimiento",
    options=list(PROCEDIMIENTOS.keys()),
    index=list(PROCEDIMIENTOS.keys()).index(procedimiento_actual),
    format_func=lambda x: PROCEDIMIENTOS[x],
)

try:
    hojas = obtener_hojas_catalogo()
    if procedimiento not in hojas:
        st.error(f"La hoja {procedimiento} no fue encontrada en el catálogo.")
        st.stop()

    catalogo = cargar_catalogo(procedimiento)
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

st.caption(
    f"Catálogo activo: {procedimiento} · "
    f"{len(catalogo)} conceptos de codificación."
)

if expediente_activo():
    st.success(
        f"Expediente activo: {st.session_state['expediente_nombre']}"
    )

    if "drive_folder_id" in st.session_state:
        st.caption("Persistencia en Google Drive: activa ✓")

    if procedimiento != st.session_state["procedimiento"]:
        st.warning(
            "El procedimiento seleccionado es distinto al procedimiento "
            "del expediente activo. Usa 'Cambiar expediente' para evitar "
            "mezclar resultados."
        )

    if st.button("Cambiar expediente"):
        _limpiar_expediente()
        st.rerun()

else:
    tab_drive, tab_nuevo = st.tabs(
        ["Abrir desde Google Drive", "Cargar nuevo expediente"]
    )

    with tab_drive:
        try:
            expedientes = listar_expedientes()
        except DrivePersistenceError as error:
            st.error(str(error))
            expedientes = []

        if not expedientes:
            st.info(
                "Todavía no hay expedientes guardados por esta aplicación "
                "en Google Drive."
            )
        else:
            mapa = {
                (
                    f"{item['nombre_zip']} · "
                    f"{item['procedimiento'] or 'Sin procedimiento'}"
                ): item
                for item in expedientes
            }

            seleccion_drive = st.selectbox(
                "Expediente guardado",
                options=list(mapa.keys()),
            )

            if st.button(
                "Abrir expediente guardado",
                type="primary",
            ):
                item = mapa[seleccion_drive]
                with st.spinner(
                    "Restaurando expediente y resultados desde Google Drive..."
                ):
                    try:
                        datos = cargar_expediente(item["folder_id"])
                        _restaurar_en_sesion(datos)
                        st.success("Expediente restaurado correctamente.")
                        st.rerun()
                    except DrivePersistenceError as error:
                        st.error(str(error))

    with tab_nuevo:
        archivo_zip = st.file_uploader(
            "Selecciona el expediente comprimido",
            type=["zip"],
            help=(
                "El ZIP se guardará en Google Drive junto con sus resultados "
                "para poder recuperar el proceso después."
            ),
        )

        st.caption(
            "Carpetas excluidas del análisis: "
            + ", ".join(CARPETAS_IGNORADAS)
        )

        if archivo_zip is not None:
            contenido_zip = archivo_zip.getvalue()

            try:
                inventario, resumen = inventariar_expediente_zip(
                    contenido_zip
                )
            except ValueError as error:
                st.error(str(error))
                st.stop()

            st.write(
                f"**{resumen['archivos']} archivos** · "
                f"{formatear_tamano(resumen['tamano_bytes'])} sin comprimir"
            )

            if st.button(
                "Guardar en Drive y abrir expediente",
                type="primary",
            ):
                with st.spinner(
                    "Guardando expediente en Google Drive. "
                    "Un ZIP grande puede tardar varios minutos..."
                ):
                    try:
                        guardado = guardar_expediente(
                            archivo_zip.name,
                            contenido_zip,
                            procedimiento,
                            inventario,
                            resumen,
                        )

                        _restaurar_en_sesion(
                            {
                                "expediente_id": (
                                    f"{archivo_zip.name}:{archivo_zip.size}"
                                ),
                                "expediente_nombre": archivo_zip.name,
                                "contenido_zip": contenido_zip,
                                "inventario": inventario,
                                "resumen_expediente": resumen,
                                "procedimiento": procedimiento,
                                "drive_folder_id": guardado["folder_id"],
                            }
                        )

                        st.success(
                            "Expediente guardado en Drive y abierto."
                        )
                        st.rerun()

                    except DrivePersistenceError as error:
                        st.error(str(error))

if expediente_activo():
    resumen = st.session_state["resumen_expediente"]
    inventario = st.session_state["inventario"]

    st.subheader("Resumen")

    col1, col2, col3 = st.columns(3)
    col1.metric("Archivos encontrados", resumen["archivos"])
    col2.metric("Carpetas encontradas", resumen["carpetas"])
    col3.metric(
        "Tamaño sin comprimir",
        formatear_tamano(resumen["tamano_bytes"]),
    )

    if resumen["tipos"]:
        st.subheader("Tipos de archivo")
        columnas = st.columns(len(resumen["tipos"]))
        for columna, (tipo, cantidad) in zip(
            columnas,
            resumen["tipos"].items(),
        ):
            columna.metric(tipo, cantidad)

    st.subheader("Inventario")

    inventario_visual = inventario.copy()
    inventario_visual["Tamaño"] = inventario_visual[
        "Tamaño (bytes)"
    ].apply(formatear_tamano)

    st.dataframe(
        inventario_visual[
            [
                "Ruta original",
                "Archivo",
                "Extensión",
                "Tipo",
                "Tamaño",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Ver catálogo de codificación activo"):
    st.dataframe(
        catalogo,
        use_container_width=True,
        hide_index=True,
    )
