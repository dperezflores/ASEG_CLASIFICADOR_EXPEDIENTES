from pathlib import Path

import streamlit as st

from config.constants import APP_NAME, APP_SUBTITLE
from services.catalog_service import obtener_hojas_catalogo


st.set_page_config(
    page_title=APP_NAME,
    page_icon="📁",
    layout="wide",
)

css_path = Path("assets/styles.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

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
    "Primera etapa del proyecto: validar la estructura de la aplicación y la lectura del catálogo de codificación."
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

st.write(f"Procedimiento seleccionado: **{procedimiento}**")

hojas = obtener_hojas_catalogo()
if hojas:
    st.caption("Hojas detectadas en el catálogo: " + ", ".join(hojas))
else:
    st.warning(
        "El catálogo todavía no se encuentra cargado en la carpeta /catalogo. "
        "Lo incorporaremos en el siguiente paso."
    )
