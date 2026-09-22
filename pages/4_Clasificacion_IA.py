import streamlit as st

from ui.common import (
    mostrar_encabezado,
    requerir_expediente,
)


mostrar_encabezado(
    "Clasificación IA",
    "Comparativa experimental: OCR + clasificador frente a IA multimodal",
)
requerir_expediente()

st.info(
    "Esta página ya está reservada para el siguiente experimento. "
    "Aquí compararemos los dos caminos usando los mismos documentos."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Ruta A")
    st.markdown(
        """
        **OCR / texto → clasificador**
        
        - Texto nativo cuando exista.
        - RapidOCR cuando sea necesario.
        - Clasificador textual, inicialmente Jev.
        """
    )

with col2:
    st.subheader("Ruta B")
    st.markdown(
        """
        **PDF original → modelo multimodal**
        
        - El modelo recibe el documento visual.
        - Conserva estructura, tablas y distribución.
        - Probaremos un proveedor multimodal.
        """
    )

st.caption(
    "Todavía no se ejecuta ninguna llamada de IA desde esta página."
)
