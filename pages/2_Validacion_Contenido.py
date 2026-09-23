import streamlit as st

from services.catalog_service import cargar_catalogo
from services.openai_multimodal_service import (
    MODELOS,
    OpenAIMultimodalError,
    openai_configurado,
)
from services.jev_classifier_service import JevError, jev_configurado
from services.structural_analysis_service import (
    construir_mapa_estructural,
    obtener_archivos_unidad,
    obtener_estimaciones_detectadas,
)
from services.unit_content_analysis_service import (
    analizar_componente_unidad,
    diagnosticar_componente_pdf,
)
from ui.common import mostrar_encabezado, requerir_expediente


mostrar_encabezado(
    "Validación por contenido",
    "Análisis controlado de un solo componente dentro de una unidad documental",
)
requerir_expediente()

if not openai_configurado():
    st.error(
        "La API key de OpenAI no está configurada en Streamlit Secrets."
    )
    st.stop()

if not jev_configurado():
    st.error(
        "La API key de TypeSafe/Jev no está configurada en Streamlit Secrets."
    )
    st.stop()

inventario = st.session_state["inventario"]
contenido_zip = st.session_state["contenido_zip"]
procedimiento = st.session_state["procedimiento"]
catalogo = cargar_catalogo(procedimiento)

mapa = construir_mapa_estructural(inventario)
estimaciones = obtener_estimaciones_detectadas(mapa)

st.info(
    "En esta prueba se analiza un solo archivo a la vez. "
    "El nombre real y la ruta se muestran únicamente en la interfaz; "
    "la IA recibe un alias neutro y el contenido del documento."
)

if estimaciones.empty:
    st.warning("No hay estimaciones detectadas en el mapa estructural.")
    st.stop()

mapa_unidades = {
    f"{fila['Carpeta']} · {fila['Ruta carpeta']}": fila
    for _, fila in estimaciones.iterrows()
}

seleccion_unidad = st.selectbox(
    "Unidad documental",
    options=list(mapa_unidades.keys()),
    index=0,
)
unidad = mapa_unidades[seleccion_unidad]

archivos = obtener_archivos_unidad(
    inventario,
    unidad["Ruta carpeta"],
)
archivos_pdf = archivos[
    archivos["Extensión"].astype(str).str.lower() == ".pdf"
].copy()

if archivos_pdf.empty:
    st.warning("La unidad seleccionada no contiene archivos PDF directos.")
    st.stop()

mapa_archivos = {
    str(fila["Archivo"]): str(fila["Ruta original"])
    for _, fila in archivos_pdf.iterrows()
}

archivo = st.selectbox(
    "Componente a analizar",
    options=list(mapa_archivos.keys()),
)
ruta_pdf = mapa_archivos[archivo]

st.subheader("1. Diagnóstico previo")

try:
    diagnostico = diagnosticar_componente_pdf(
        contenido_zip,
        ruta_pdf,
    )
except Exception as error:
    st.error(f"No fue posible revisar el PDF: {error}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Páginas totales", diagnostico["Páginas totales"])
c2.metric("Caracteres útiles", diagnostico["Caracteres útiles"])
c3.metric("Ruta sugerida", diagnostico["Ruta sugerida"])

st.write(f"**Motivo:** {diagnostico['Motivo']}")
st.caption(
    "En este prototipo web, un PDF con poco texto nativo se envía "
    "directamente al multimodal. En una futura versión portable, "
    "esa ruta podrá sustituirse por OCR local + Jev."
)

st.subheader("2. Ejecutar análisis de contenido")

modelo = st.selectbox(
    "Modelo multimodal de respaldo",
    options=list(MODELOS.keys()),
    index=0,
    format_func=lambda m: MODELOS[m]["label"],
)

if st.button(
    "Analizar este componente",
    type="primary",
):
    with st.spinner(
        "Analizando únicamente el componente seleccionado..."
    ):
        try:
            resultado = analizar_componente_unidad(
                contenido_zip=contenido_zip,
                ruta_pdf=ruta_pdf,
                catalogo=catalogo,
                procedimiento=procedimiento,
                modelo_multimodal=modelo,
            )
            st.session_state["resultado_componente_unidad"] = {
                "archivo": archivo,
                "ruta": ruta_pdf,
                "unidad": str(unidad["Carpeta"]),
                "resultado": resultado,
            }
            st.rerun()

        except (
            JevError,
            OpenAIMultimodalError,
            ValueError,
            KeyError,
        ) as error:
            st.error(str(error))

guardado = st.session_state.get("resultado_componente_unidad")

if (
    guardado
    and guardado.get("ruta") == ruta_pdf
):
    resultado = guardado["resultado"]

    st.subheader("3. Resultado")

    st.write(f"**Archivo analizado:** {archivo}")
    st.write(f"**Ruta utilizada:** {resultado['Ruta utilizada']}")
    st.write(f"**Título detectado:** {resultado['Título detectado']}")
    st.write(
        f"**Coincide con catálogo:** "
        f"{'Sí' if resultado['Coincide catálogo'] else 'No'}"
    )

    if resultado["Coincide catálogo"]:
        st.write(
            f"**Concepto propuesto:** "
            f"{resultado['Concepto propuesto']}"
        )
        st.write(
            f"**Código de catálogo:** "
            f"{resultado['Código de catálogo']}"
        )

    st.write(
        f"**Confianza:** {float(resultado['Confianza (%)']):.1f} %"
    )
    st.write(
        f"**Rol propuesto dentro de la unidad:** "
        f"{resultado['Rol propuesto en la unidad']}"
    )
    st.write(f"**Evidencia:** {resultado['Evidencia']}")

    with st.expander("Ver información técnica"):
        st.write(
            f"**Motivo de ruta:** {resultado['Motivo de ruta']}"
        )
        st.write(f"**Modelo:** {resultado['Modelo']}")
        st.write(f"**Páginas usadas:** {resultado['Páginas usadas']}")
        st.write(
            f"**Costo:** USD {float(resultado['Costo (USD)']):.8f}"
        )
        st.write(
            f"**Tiempo IA:** {float(resultado['Tiempo (s)']):.2f} s"
        )

st.divider()

st.caption(
    "Esta pantalla todavía no renombra, mueve ni codifica físicamente archivos. "
    "Solo prueba si el contenido permite distinguir: representación de la "
    "unidad, documento con código propio o soporte/fuera de catálogo."
)
