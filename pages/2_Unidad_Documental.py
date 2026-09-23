import streamlit as st

from services.expediente_service import formatear_tamano
from services.structural_analysis_service import (
    construir_mapa_estructural,
    construir_vista_unidad_estimacion,
    obtener_estimaciones_detectadas,
)
from ui.common import mostrar_encabezado, requerir_expediente


mostrar_encabezado(
    "Unidad documental",
    "Lectura controlada del interior de una estimación",
)
requerir_expediente()

inventario = st.session_state["inventario"]
procedimiento = st.session_state["procedimiento"]

mapa = construir_mapa_estructural(inventario)
estimaciones = obtener_estimaciones_detectadas(mapa)

st.info(
    "Este paso todavía no usa IA. Su objetivo es comprobar que el sistema "
    "entiende una estimación como una unidad documental formada por varios "
    "archivos, no como un único PDF."
)

if estimaciones.empty:
    st.warning(
        "No hay estimaciones detectadas en el mapa estructural. "
        "Regresa a Mapa estructural para revisar el expediente."
    )
    st.stop()

opciones = estimaciones.to_dict("records")
mapa_opciones = {
    f"{fila['Carpeta']} · {fila['Ruta carpeta']}": fila
    for fila in opciones
}

seleccion = st.selectbox(
    "Unidad a revisar",
    options=list(mapa_opciones.keys()),
    index=0,
)

unidad = mapa_opciones[seleccion]

detalle, resumen = construir_vista_unidad_estimacion(
    inventario,
    unidad["Ruta carpeta"],
    procedimiento,
    int(unidad["Consecutivo"]),
)

st.subheader("1. Identificación de la unidad")

c1, c2, c3 = st.columns(3)
c1.metric("Tipo", resumen["tipo_unidad"])
c2.metric("Consecutivo", resumen["consecutivo"])
c3.metric("Archivos directos", resumen["archivos"])

st.write("**Ruta original:**")
st.code(resumen["ruta_origen"], language=None)

st.write("**Código de unidad candidato:**")
st.code(resumen["codigo_unidad_candidato"], language=None)

st.caption(
    "El código anterior se deriva de la estructura detectada y del tipo de "
    "procedimiento. Todavía no implica que un archivo haya sido validado o "
    "renombrado."
)

st.subheader("2. Archivos que forman la unidad")

vista = detalle.copy()
vista["Tamaño"] = vista["Tamaño (bytes)"].apply(formatear_tamano)

st.dataframe(
    vista[
        [
            "Archivo",
            "Tipo",
            "Tamaño",
            "Rol preliminar",
            "Motivo",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

principales = detalle[
    detalle["Rol preliminar"] == "Candidato a documento principal"
]

st.subheader("3. Resultado preliminar")

if len(principales) == 1:
    principal = principales.iloc[0]["Archivo"]
    st.success(
        f"Se encontró un candidato a documento principal: {principal}."
    )
elif len(principales) > 1:
    st.warning(
        "Se encontró más de un candidato a documento principal. "
        "Será necesaria validación adicional."
    )
else:
    st.warning(
        "No se encontró un archivo con nombre de carátula. "
        "En una etapa posterior deberá identificarse por contenido."
    )

st.write(
    "Los demás archivos se conservan como componentes de la unidad. "
    "Todavía no se intenta decidir cuáles tienen código propio."
)

st.divider()

st.caption(
    "Siguiente paso, después de validar esta vista: revisar el contenido de "
    "los componentes para separar tres grupos: documento principal, "
    "documentos con código propio y documentos de soporte sin código."
)
