from pathlib import Path

import streamlit as st

from config.constants import APP_NAME, APP_SUBTITLE
from services.catalog_service import cargar_catalogo, obtener_hojas_catalogo


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
    "Etapa actual: validar que la aplicación interprete correctamente "
    "el catálogo institucional de codificación."
)

st.subheader("Configuración del expediente")

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
        st.error(
            f"La hoja {procedimiento} no fue encontrada en el catálogo."
        )
        st.stop()

    catalogo = cargar_catalogo(procedimiento)

except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

st.success(
    f"Catálogo cargado correctamente: {len(catalogo)} documentos "
    f"disponibles para {procedimiento}."
)

st.subheader("Catálogo activo")

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

st.caption(
    "En esta etapa la aplicación únicamente lee y presenta el catálogo. "
    "Todavía no se realiza clasificación automática de documentos."
)
