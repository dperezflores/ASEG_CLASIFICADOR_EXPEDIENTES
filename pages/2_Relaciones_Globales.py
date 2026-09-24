import streamlit as st

from services.expediente_service import formatear_tamano
from services.global_relationship_service import (
    analizar_duplicados_exactos,
    analizar_similitud_global_pdfs,
)
from ui.common import mostrar_encabezado, requerir_expediente


mostrar_encabezado(
    "Relaciones globales del expediente",
    "Duplicados exactos y similitud documental entre PDFs del expediente",
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
    "La fase 1 detecta únicamente igualdad binaria exacta. La fase 2 genera "
    "candidatos de similitud documental entre PDFs mediante texto nativo y "
    "huellas visuales locales. Ninguna de las dos fases elimina, mueve o "
    "renombra archivos."
)

st.subheader("Fase 1 · Duplicados exactos")

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


st.divider()
st.subheader("Fase 2 · Similitud global de PDFs")

st.info(
    "Esta fase tampoco usa IA ni OCR. Compara todos los PDFs mediante dos "
    "señales locales: shingles del texto nativo y huellas visuales perceptuales "
    "de páginas muestreadas. El nombre del archivo y la ruta se muestran para "
    "trazabilidad, pero NO participan en el cálculo de similitud."
)

st.caption(
    "Los niveles iniciales son deliberadamente provisionales: MUY ALTA >= 92%, "
    "ALTA >= 80% y MEDIA >= 68%. Un resultado es solamente un candidato a "
    "relación documental; todavía no significa duplicado definitivo."
)

if st.button(
    "Analizar similitud global de PDFs",
    type="primary",
):
    barra_similitud = st.progress(0.0)
    estado_similitud = st.empty()

    def actualizar_progreso_similitud(
        etapa,
        posicion,
        total,
        detalle,
    ):
        porcentaje = (
            posicion / total
            if total
            else 1.0
        )
        barra_similitud.progress(
            min(max(porcentaje, 0.0), 1.0)
        )

        if etapa == "perfil":
            estado_similitud.write(
                f"Perfilando PDF {posicion}/{total}: {detalle}"
            )
        else:
            estado_similitud.write(
                f"Comparando pares {posicion}/{total}: {detalle}"
            )

    with st.spinner(
        "Analizando texto y estructura visual de los PDFs..."
    ):
        perfiles_pdf, pares_pdf, resumen_pdf = (
            analizar_similitud_global_pdfs(
                contenido_zip=contenido_zip,
                inventario=inventario,
                on_progress=actualizar_progreso_similitud,
            )
        )

        st.session_state[
            "relaciones_globales_similitud_pdf"
        ] = {
            "perfiles": perfiles_pdf,
            "pares": pares_pdf,
            "resumen": resumen_pdf,
            "expediente_id": st.session_state.get(
                "expediente_id",
                "",
            ),
        }

    barra_similitud.empty()
    estado_similitud.empty()

guardado_similitud = st.session_state.get(
    "relaciones_globales_similitud_pdf"
)

if (
    guardado_similitud
    and guardado_similitud.get(
        "expediente_id",
        "",
    )
    == st.session_state.get("expediente_id", "")
):
    perfiles_pdf = guardado_similitud["perfiles"]
    pares_pdf = guardado_similitud["pares"]
    resumen_pdf = guardado_similitud["resumen"]

    st.subheader("Resultado de similitud")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric(
        "PDF evaluados",
        resumen_pdf["pdf_evaluados"],
    )
    s2.metric(
        "Con texto nativo útil",
        resumen_pdf["pdf_con_texto"],
    )
    s3.metric(
        "Solo señal visual",
        resumen_pdf["pdf_solo_visual"],
    )
    s4.metric(
        "Pares candidatos",
        resumen_pdf["pares_candidatos"],
    )

    st.caption(
        f"Pares comparados: {resumen_pdf['pares_comparados']} · "
        f"MUY ALTA: {resumen_pdf['muy_alta']} · "
        f"ALTA: {resumen_pdf['alta']} · "
        f"MEDIA: {resumen_pdf['media']} · "
        f"PDF con error: {resumen_pdf['pdf_con_error']}"
    )

    if pares_pdf.empty:
        st.success(
            "No se encontraron pares de PDFs que superen el umbral mínimo "
            "de similitud de esta primera versión."
        )
    else:
        niveles_disponibles = [
            nivel
            for nivel in ("MUY ALTA", "ALTA", "MEDIA")
            if nivel in set(
                pares_pdf["Nivel"].astype(str)
            )
        ]

        niveles = st.multiselect(
            "Nivel de similitud a mostrar",
            options=niveles_disponibles,
            default=niveles_disponibles,
        )

        if niveles:
            pares_visibles = pares_pdf[
                pares_pdf["Nivel"].isin(niveles)
            ].copy()
        else:
            pares_visibles = pares_pdf.iloc[0:0].copy()

        st.dataframe(
            pares_visibles[
                [
                    "Nivel",
                    "Puntuación global (%)",
                    "Relación provisional",
                    "Archivo A",
                    "Páginas A",
                    "Archivo B",
                    "Páginas B",
                    "Similitud texto Jaccard (%)",
                    "Contención textual (%)",
                    "Similitud visual (%)",
                    "Fuente principal",
                    "Acción provisional",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander(
            "Ver rutas completas de los pares candidatos"
        ):
            st.dataframe(
                pares_visibles[
                    [
                        "Nivel",
                        "Puntuación global (%)",
                        "Relación provisional",
                        "Ruta A",
                        "Ruta B",
                        "Similitud texto Jaccard (%)",
                        "Contención textual (%)",
                        "Similitud visual (%)",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with st.expander(
        "Ver diagnóstico de los PDFs evaluados"
    ):
        st.dataframe(
            perfiles_pdf[
                [
                    "Ruta original",
                    "Archivo",
                    "Carpeta",
                    "Páginas",
                    "Caracteres texto útiles",
                    "Texto utilizable",
                    "Páginas visuales usadas",
                    "Estado perfil",
                    "Error perfil",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.warning(
        "Todavía no se decide cuál documento es principal, copia, variante o "
        "extracto definitivo. Primero validaremos si los pares detectados son "
        "útiles y si los umbrales separan correctamente documentos realmente "
        "relacionados de coincidencias por formato o plantilla."
    )
