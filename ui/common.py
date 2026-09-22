from pathlib import Path

import streamlit as st

from config.constants import APP_NAME, APP_SUBTITLE
from services.expediente_service import formatear_tamano


PROCEDIMIENTOS = {
    "DIR": "Adjudicación directa",
    "LPU": "Licitación pública",
    "LSI": "Licitación simplificada",
}


def configurar_pagina(titulo: str | None = None) -> None:
    st.set_page_config(
        page_title=titulo or APP_NAME,
        page_icon="📁",
        layout="wide",
    )

    css_path = Path("assets/styles.css")
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def mostrar_encabezado(titulo: str, subtitulo: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-kicker">ASEG</div>
            <h1>{titulo}</h1>
            <p>{subtitulo or APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def expediente_activo() -> bool:
    return all(
        clave in st.session_state
        for clave in (
            "expediente_id",
            "expediente_nombre",
            "contenido_zip",
            "inventario",
            "resumen_expediente",
            "procedimiento",
        )
    )


def mostrar_sidebar_expediente() -> None:
    st.sidebar.markdown("### Expediente activo")

    if not expediente_activo():
        st.sidebar.caption("No hay un expediente cargado.")
        return

    resumen = st.session_state["resumen_expediente"]
    procedimiento = st.session_state["procedimiento"]

    st.sidebar.markdown(
        f"**{st.session_state['expediente_nombre']}**"
    )
    st.sidebar.caption(
        f"{PROCEDIMIENTOS.get(procedimiento, procedimiento)} · "
        f"{resumen['archivos']} archivos · "
        f"{formatear_tamano(resumen['tamano_bytes'])}"
    )

    diagnostico = "✓" if "analisis_pdf" in st.session_state else "—"
    ocr = "✓" if "resultado_ocr" in st.session_state else "—"

    st.sidebar.markdown(
        f"""
        **Estado**
        
        Inventario: ✓  
        Diagnóstico PDF: {diagnostico}  
        OCR controlado: {ocr}  
        Clasificación IA: —
        """
    )


def requerir_expediente() -> None:
    if expediente_activo():
        return

    st.warning(
        "Primero carga un expediente desde la página **Expediente**."
    )
    st.stop()
