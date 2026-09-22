import streamlit as st

from config.constants import APP_NAME, APP_SUBTITLE
from ui.common import expediente_activo, mostrar_encabezado


mostrar_encabezado(APP_NAME, APP_SUBTITLE)

st.info(
    "La aplicación está organizada por módulos. El expediente y los "
    "resultados permanecen disponibles al navegar entre páginas durante "
    "la misma sesión."
)

if expediente_activo():
    resumen = st.session_state["resumen_expediente"]

    st.subheader("Expediente activo")

    col1, col2, col3 = st.columns(3)
    col1.metric("Archivos", resumen["archivos"])
    col2.metric("Carpetas", resumen["carpetas"])
    col3.metric("Procedimiento", st.session_state["procedimiento"])

    st.write(
        "Continúa desde el menú lateral con **Diagnóstico PDF**, "
        "**OCR controlado** o **Clasificación IA**."
    )
else:
    st.subheader("Comenzar")
    st.write(
        "Selecciona **Expediente** en el menú lateral para cargar el ZIP "
        "y crear el expediente activo."
    )

st.divider()

st.subheader("Flujo del prototipo")
st.markdown(
    """
    1. **Expediente** — carga, procedimiento e inventario.
    2. **Diagnóstico PDF** — identifica texto nativo, PDF mixtos y PDF que requieren OCR.
    3. **OCR controlado** — procesa una muestra seleccionada.
    4. **Clasificación IA** — comparación OCR + clasificador frente a multimodal.
    """
)

st.caption(
    "La persistencia después de reinicios o nuevos despliegues se "
    "implementará en una etapa posterior."
)
