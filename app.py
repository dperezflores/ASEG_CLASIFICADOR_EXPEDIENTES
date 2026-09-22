import streamlit as st

from config.constants import APP_NAME, APP_SUBTITLE
from ui.common import (
    configurar_pagina,
    expediente_activo,
    mostrar_encabezado,
    mostrar_sidebar_expediente,
)


configurar_pagina()
mostrar_sidebar_expediente()
mostrar_encabezado(APP_NAME, APP_SUBTITLE)

st.info(
    "La aplicación ahora está organizada por páginas. Cada etapa conserva "
    "sus resultados mientras la sesión de Streamlit permanezca activa."
)

if expediente_activo():
    resumen = st.session_state["resumen_expediente"]

    st.subheader("Expediente activo")

    col1, col2, col3 = st.columns(3)
    col1.metric("Archivos", resumen["archivos"])
    col2.metric("Carpetas", resumen["carpetas"])
    col3.metric(
        "Procedimiento",
        st.session_state["procedimiento"],
    )

    st.write(
        "Usa el menú lateral para continuar con **Diagnóstico PDF**, "
        "**OCR controlado** o las etapas posteriores."
    )
else:
    st.subheader("Comenzar")
    st.write(
        "Abre la página **Expediente** desde el menú lateral para cargar "
        "el ZIP de trabajo. Después podrás navegar entre módulos sin repetir "
        "los pasos ya ejecutados durante esta sesión."
    )

st.divider()

st.subheader("Flujo del prototipo")

st.markdown(
    """
    1. **Expediente** — carga, procedimiento e inventario.
    2. **Diagnóstico PDF** — identifica texto nativo, PDF mixtos y PDF que requieren OCR.
    3. **OCR controlado** — procesa una muestra seleccionada.
    4. **Clasificación IA** — siguiente etapa: comparación OCR + clasificador vs multimodal.
    """
)

st.caption(
    "La persistencia entre reinicios o nuevos despliegues de Streamlit se "
    "implementará en una etapa separada. La navegación entre páginas ya no "
    "obliga a repetir el trabajo dentro de la misma sesión."
)
