import streamlit as st

from config.constants import CARPETAS_IGNORADAS
from services.catalog_service import cargar_catalogo, obtener_hojas_catalogo
from services.expediente_service import formatear_tamano, inventariar_expediente_zip
from ui.common import (
    PROCEDIMIENTOS,
    expediente_activo,
    mostrar_encabezado,
)


mostrar_encabezado(
    "Expediente",
    "Carga, configuración e inventario del expediente de obra pública",
)

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

    if procedimiento != st.session_state["procedimiento"]:
        st.session_state["procedimiento"] = procedimiento
        st.session_state.pop("analisis_pdf", None)
        st.session_state.pop("resumen_pdf", None)
        st.session_state.pop("resultado_ocr", None)
        st.info(
            "Se actualizó el procedimiento. Los resultados posteriores se "
            "limpiaron para evitar mezclar análisis de otro catálogo."
        )

    if st.button("Cambiar expediente"):
        for clave in (
            "expediente_id",
            "expediente_nombre",
            "contenido_zip",
            "inventario",
            "resumen_expediente",
            "procedimiento",
            "analisis_pdf",
            "resumen_pdf",
            "resultado_ocr",
        ):
            st.session_state.pop(clave, None)
        st.rerun()

else:
    archivo_zip = st.file_uploader(
        "Selecciona el expediente comprimido",
        type=["zip"],
        help=(
            "El ZIP puede contener carpetas, subcarpetas, PDF, Excel, DWG "
            "y otros archivos. Los originales no se modifican."
        ),
    )

    st.caption(
        "Carpetas excluidas del análisis: "
        + ", ".join(CARPETAS_IGNORADAS)
    )

    if archivo_zip is not None:
        contenido_zip = archivo_zip.getvalue()

        try:
            inventario, resumen = inventariar_expediente_zip(contenido_zip)
        except ValueError as error:
            st.error(str(error))
            st.stop()

        st.session_state["expediente_id"] = (
            f"{archivo_zip.name}:{archivo_zip.size}"
        )
        st.session_state["expediente_nombre"] = archivo_zip.name
        st.session_state["contenido_zip"] = contenido_zip
        st.session_state["inventario"] = inventario
        st.session_state["resumen_expediente"] = resumen
        st.session_state["procedimiento"] = procedimiento

        st.success("Expediente cargado y registrado como expediente activo.")
        st.rerun()

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
