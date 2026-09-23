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
    "Este paso todavía no usa IA. El sistema identifica una unidad documental "
    "y describe cómo está representada físicamente en este expediente. "
    "Los nombres de archivos solo generan señales; no determinan por sí solos "
    "qué archivo debe recibir un código."
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

st.write(f"**Representación física:** {resumen['representacion_fisica']}")
st.write(
    "**Archivo representativo:** pendiente de validar por contenido"
)

st.write("**Ruta original:**")
st.code(resumen["ruta_origen"], language=None)

st.write("**Código de unidad candidato:**")
st.code(resumen["codigo_unidad_candidato"], language=None)

st.caption(
    "El código anterior corresponde a la unidad documental candidata, no a "
    "un archivo específico. Ningún archivo ha sido validado, codificado o "
    "renombrado todavía."
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
            "Señal estructural",
            "Fuente de la señal",
            "Estado",
            "Motivo",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

senales = detalle[
    detalle["Señal estructural"] != "Sin señal estructural específica"
]

st.subheader("3. Resultado preliminar")

if senales.empty:
    st.info(
        "La estructura no aporta una señal clara sobre qué archivo representa "
        "la unidad. Será necesario revisar contenido."
    )
else:
    st.warning(
        "Se encontraron señales estructurales útiles, pero ninguna se toma "
        "como conclusión. Deben validarse mediante contenido."
    )

    for _, fila in senales.iterrows():
        st.write(
            f"- **{fila['Archivo']}** → {fila['Señal estructural']} "
            f"({fila['Fuente de la señal']})"
        )

st.write(
    "Todos los archivos siguen siendo componentes de la unidad. En el "
    "siguiente paso se determinará, mediante contenido, cuál representa la "
    "estimación, cuáles tienen código propio y cuáles son soporte."
)

st.divider()

st.caption(
    "Siguiente paso: validar la unidad por contenido y determinar cómo está "
    "representada realmente en este expediente: un archivo representativo, "
    "documentos con código propio y documentación de soporte."
)
