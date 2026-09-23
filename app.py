import streamlit as st

from config.constants import APP_NAME
from ui.common import configurar_pagina, mostrar_sidebar_expediente


configurar_pagina(APP_NAME)

paginas = {
    "": [
        st.Page(
            "pages/0_Inicio.py",
            title="Inicio",
            icon="🏠",
            default=True,
        ),
    ],
    "Expediente": [
        st.Page(
            "pages/1_Expediente.py",
            title="Expediente",
            icon="📁",
        ),
        st.Page(
            "pages/2_Mapa_Estructural.py",
            title="Mapa estructural",
            icon="🗂️",
        ),
        st.Page(
            "pages/2_Unidad_Documental.py",
            title="Unidad documental",
            icon="📂",
        ),
        st.Page(
            "pages/2_Validacion_Contenido.py",
            title="Validación por contenido",
            icon="🔬",
        ),
        st.Page(
            "pages/2_Diagnostico_PDF.py",
            title="Diagnóstico PDF",
            icon="📄",
        ),
        st.Page(
            "pages/3_OCR_Controlado.py",
            title="OCR controlado",
            icon="🔎",
        ),
        st.Page(
            "pages/4_Clasificacion_IA.py",
            title="Clasificación IA",
            icon="🧠",
        ),
        st.Page(
            "pages/5_Clasificacion_Hibrida.py",
            title="Clasificación híbrida",
            icon="⚡",
        ),
        st.Page(
            "pages/6_Resultado_Clasificacion.py",
            title="Resultado clasificación",
            icon="✅",
        ),
    ],
}

pagina = st.navigation(paginas, position="sidebar")
mostrar_sidebar_expediente()
pagina.run()
