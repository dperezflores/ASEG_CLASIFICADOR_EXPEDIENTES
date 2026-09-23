import streamlit as st

from services.structural_analysis_service import (
    construir_mapa_estructural,
    obtener_estimaciones_detectadas,
)
from ui.common import mostrar_encabezado, requerir_expediente


mostrar_encabezado(
    "Mapa estructural",
    "Primera lectura del expediente antes de utilizar inteligencia artificial",
)
requerir_expediente()

inventario = st.session_state["inventario"]
mapa = construir_mapa_estructural(inventario)
estimaciones = obtener_estimaciones_detectadas(mapa)

st.info(
    "Este paso no usa IA y no modifica ningún archivo. "
    "Solo reconstruye la estructura de carpetas y aplica una primera regla "
    "conservadora para detectar carpetas de estimaciones."
)

st.subheader("1. Resumen estructural")

c1, c2, c3 = st.columns(3)
c1.metric("Carpetas detectadas", len(mapa))
c2.metric("Estimaciones candidatas", len(estimaciones))
c3.metric(
    "Archivos del expediente",
    len(inventario),
)

st.subheader("2. Carpetas detectadas")

if mapa.empty:
    st.warning("No se encontraron carpetas en el expediente.")
else:
    st.dataframe(
        mapa,
        use_container_width=True,
        hide_index=True,
    )

st.subheader("3. Primera regla: estimaciones")

if estimaciones.empty:
    st.info(
        "No se detectaron carpetas con un patrón claro de estimación "
        "(por ejemplo: EST 1, ESTIMACION 2 o EST 7 FIN)."
    )
else:
    st.success(
        "Estas carpetas se identificaron únicamente por su estructura y "
        "nombre. Todavía no se ha revisado su contenido."
    )

    st.dataframe(
        estimaciones[
            [
                "Ruta carpeta",
                "Carpeta",
                "Consecutivo",
                "Archivos directos",
                "Archivos totales",
                "Confianza estructural",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.divider()

st.caption(
    "Al validar este mapa, el siguiente paso será analizar el contenido "
    "interno de una sola unidad documental (por ejemplo, EST 1) para "
    "distinguir documento principal, documentos codificables y soporte."
)
