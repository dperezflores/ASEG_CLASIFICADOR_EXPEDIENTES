from pathlib import Path

import streamlit as st

from config.constants import APP_NAME, APP_SUBTITLE, CARPETAS_IGNORADAS
from services.catalog_service import cargar_catalogo, obtener_hojas_catalogo
from services.expediente_service import formatear_tamano, inventariar_expediente_zip


st.set_page_config(
    page_title=APP_NAME,
    page_icon="📁",
    layout="wide",
)

css_path = Path("assets/styles.css")
if css_path.exists():
    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="app-header">
        <div class="app-kicker">ASEG</div>
        <h1>{APP_NAME}</h1>
        <p>{APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Etapa actual: recibir un expediente completo en ZIP y construir "
    "su inventario sin modificar los archivos originales."
)

st.subheader("1. Configuración del expediente")

procedimiento = st.selectbox(
    "Tipo de procedimiento",
    options=["DIR", "LPU", "LSI"],
    format_func=lambda x: {
        "DIR": "Adjudicación directa",
        "LPU": "Licitación pública",
        "LSI": "Licitación simplificada",
    }[x],
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

st.subheader("2. Cargar expediente")

archivo_zip = st.file_uploader(
    "Selecciona el expediente comprimido",
    type=["zip"],
    help=(
        "El ZIP puede contener carpetas, subcarpetas y distintos tipos de "
        "archivo. En esta etapa solo se construye el inventario."
    ),
)

st.caption(
    "Carpetas excluidas del análisis: "
    + ", ".join(CARPETAS_IGNORADAS)
)

if archivo_zip is not None:
    try:
        inventario, resumen = inventariar_expediente_zip(
            archivo_zip.getvalue()
        )
    except ValueError as error:
        st.error(str(error))
        st.stop()

    st.success("Expediente leído correctamente.")

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

    st.subheader("Inventario del expediente")

    if inventario.empty:
        st.warning(
            "No se encontraron archivos utilizables dentro del expediente."
        )
    else:
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
            column_config={
                "Ruta original": st.column_config.TextColumn(
                    "Ruta original",
                    width="large",
                ),
                "Archivo": st.column_config.TextColumn(
                    "Archivo",
                    width="large",
                ),
                "Extensión": st.column_config.TextColumn(
                    "Extensión",
                    width="small",
                ),
                "Tipo": st.column_config.TextColumn(
                    "Tipo",
                    width="small",
                ),
                "Tamaño": st.column_config.TextColumn(
                    "Tamaño",
                    width="small",
                ),
            },
        )

        st.caption(
            "En esta etapa el ZIP se lee en memoria. Los documentos no se "
            "extraen, renombran, clasifican ni modifican."
        )

with st.expander("Ver catálogo de codificación activo"):
    st.dataframe(
        catalogo,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Código": st.column_config.TextColumn(
                "Código",
                width="medium",
            ),
            "Concepto": st.column_config.TextColumn(
                "Concepto",
                width="large",
            ),
        },
    )
