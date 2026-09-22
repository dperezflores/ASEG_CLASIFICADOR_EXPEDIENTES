import pandas as pd
import streamlit as st

from services.catalog_service import cargar_catalogo
from services.classification_proposal_service import (
    construir_propuesta_clasificacion,
)
from services.drive_persistence_service import (
    DrivePersistenceError,
    guardar_propuesta_clasificacion,
)
from services.hybrid_classification_service import preparar_muestra_hibrida
from ui.common import mostrar_encabezado, requerir_expediente


mostrar_encabezado(
    "Resultado de clasificación",
    "Propuesta de codificación y nombres del expediente",
)
requerir_expediente()

if "resultados_hibridos" not in st.session_state:
    st.warning(
        "Primero ejecuta una muestra desde **Clasificación híbrida**. "
        "Aquí se transforma el resultado técnico en una propuesta de nombres."
    )
    st.stop()

procedimiento = st.session_state["procedimiento"]
catalogo = cargar_catalogo(procedimiento)
analisis_pdf = st.session_state["analisis_pdf"]

muestra = preparar_muestra_hibrida(
    analisis_pdf,
    catalogo,
)
resultados = st.session_state["resultados_hibridos"]

propuesta = construir_propuesta_clasificacion(
    resultados,
    muestra,
)

st.info(
    "Cuando existe coincidencia, el nombre propuesto corresponde al código "
    "oficial del catálogo. Cuando no existe coincidencia, el sistema propone "
    "un nombre descriptivo únicamente informativo; no crea un código nuevo."
)

st.subheader("1. Resumen")

clasificados = int((propuesta["Estado"] == "CLASIFICADO").sum())
revision = int(
    (propuesta["Estado"] == "REVISIÓN RECOMENDADA").sum()
)
fuera = int(
    (propuesta["Estado"] == "FUERA DE CATÁLOGO").sum()
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Documentos procesados", len(propuesta))
c2.metric("Clasificados", clasificados)
c3.metric("Revisión recomendada", revision)
c4.metric("Fuera de catálogo", fuera)

st.subheader("2. Propuesta de nombres")

st.dataframe(
    propuesta[
        [
            "Archivo real",
            "Título detectado",
            "Nombre propuesto",
            "Confianza (%)",
            "Estado",
        ]
    ],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Confianza (%)": st.column_config.NumberColumn(
            "Confianza (%)",
            format="%.1f %%",
        )
    },
)

st.caption(
    "En esta etapa todavía no se renombra ni mueve ningún archivo. "
    "Solo se presenta la propuesta para revisión."
)

st.subheader("3. Casos que requieren atención")

casos = propuesta[
    propuesta["Estado"].isin(
        ["REVISIÓN RECOMENDADA", "FUERA DE CATÁLOGO"]
    )
].copy()

if casos.empty:
    st.success(
        "No hay excepciones en esta muestra. Todos los documentos fueron "
        "clasificados con confianza suficiente."
    )
else:
    for _, fila in casos.iterrows():
        with st.expander(
            f"{fila['Archivo real']} · {fila['Estado']}"
        ):
            st.write(f"**Título detectado:** {fila['Título detectado']}")
            st.write(
                f"**Nombre propuesto:** {fila['Nombre propuesto']}"
            )
            st.write(
                f"**Confianza:** {float(fila['Confianza (%)']):.1f} %"
            )

            if bool(fila["Coincide catálogo"]):
                st.write(
                    f"**Concepto de catálogo:** "
                    f"{fila['Concepto propuesto']}"
                )
                st.write(
                    f"**Código:** {fila['Código propuesto']}"
                )
            else:
                st.write(
                    "**Catálogo:** No se encontró coincidencia suficiente."
                )

            st.write(f"**Motivo:** {fila['Observación']}")
            st.write(f"**Evidencia IA:** {fila['Evidencia']}")

if st.button(
    "Guardar propuesta en Google Drive",
    type="primary",
):
    try:
        guardar_propuesta_clasificacion(
            st.session_state["drive_folder_id"],
            propuesta,
        )
        st.session_state["propuesta_clasificacion"] = propuesta
        st.success(
            "Propuesta guardada. El siguiente paso será generar una copia "
            "clasificada del expediente sin modificar los originales."
        )
    except (DrivePersistenceError, KeyError) as error:
        st.error(str(error))

st.divider()

st.caption(
    "Próxima etapa: generar una copia del expediente aplicando los nombres "
    "propuestos y separando los casos de revisión/fuera de catálogo."
)
