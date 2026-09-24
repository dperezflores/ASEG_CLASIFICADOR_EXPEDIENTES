import streamlit as st

from services.expediente_service import formatear_tamano
from services.global_relationship_service import (
    analizar_duplicados_exactos,
)
from ui.common import mostrar_encabezado, requerir_expediente


mostrar_encabezado(
    "Relaciones globales del expediente",
    "Primera fase: duplicados exactos entre archivos de cualquier carpeta",
)
requerir_expediente()

inventario = st.session_state["inventario"]
contenido_zip = st.session_state["contenido_zip"]

st.info(
    "Esta primera fase global no usa IA y no modifica ningún archivo. "
    "Calcula una huella SHA-256 sobre el contenido completo de cada archivo. "
    "Dos archivos solo se agrupan como duplicados exactos cuando sus bytes "
    "son idénticos, aunque tengan nombres o rutas diferentes."
)

st.caption(
    "Todavía no se detectan variantes, versiones, copias escaneadas ni "
    "documentos semánticamente equivalentes con contenido distinto. Esa será "
    "la siguiente capa después de validar esta prueba."
)

if st.button(
    "Analizar duplicados exactos del expediente",
    type="primary",
):
    with st.spinner(
        "Calculando huellas de todos los archivos del expediente..."
    ):
        detalle, grupos, resumen = analizar_duplicados_exactos(
            contenido_zip=contenido_zip,
            inventario=inventario,
        )

        st.session_state["relaciones_globales_exactas"] = {
            "detalle": detalle,
            "grupos": grupos,
            "resumen": resumen,
            "expediente_id": st.session_state.get(
                "expediente_id",
                "",
            ),
        }

guardado = st.session_state.get("relaciones_globales_exactas")

if (
    guardado
    and guardado.get("expediente_id", "")
    == st.session_state.get("expediente_id", "")
):
    detalle = guardado["detalle"]
    grupos = guardado["grupos"]
    resumen = guardado["resumen"]

    st.subheader("Resultado global")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Archivos evaluados",
        resumen["archivos_evaluados"],
    )
    c2.metric(
        "Grupos duplicados exactos",
        resumen["grupos_duplicados_exactos"],
    )
    c3.metric(
        "Archivos dentro de duplicados",
        resumen["archivos_en_duplicados"],
    )
    c4.metric(
        "Copias adicionales",
        resumen["copias_adicionales"],
    )

    st.caption(
        "Espacio correspondiente a copias adicionales exactas: "
        + formatear_tamano(
            resumen["bytes_repetidos_adicionales"]
        )
        + ". Este dato es diagnóstico; el sistema no elimina archivos."
    )

    if grupos.empty:
        st.success(
            "No se encontraron archivos con contenido binario exactamente "
            "idéntico en el expediente."
        )
    else:
        st.subheader("Grupos de duplicados exactos")
        st.dataframe(
            grupos[
                [
                    "Grupo duplicado",
                    "Archivos",
                    "Tamaño por archivo (bytes)",
                    "Copias adicionales",
                    "Nombres",
                    "Carpetas",
                    "Relación detectada",
                    "Acción provisional",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Ver rutas y huellas SHA-256"):
            st.dataframe(
                detalle[
                    [
                        "Grupo duplicado",
                        "Ruta original",
                        "Archivo",
                        "Carpeta",
                        "Tipo",
                        "Tamaño (bytes)",
                        "SHA-256",
                        "Relación detectada",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.warning(
            "Duplicado exacto no significa todavía que una copia deba "
            "eliminarse. Algunas copias pueden tener una función válida dentro "
            "de distintas carpetas. En la siguiente etapa cruzaremos estas "
            "relaciones con identidad documental, código propuesto y ubicación "
            "para decidir principal, copia o soporte."
        )
