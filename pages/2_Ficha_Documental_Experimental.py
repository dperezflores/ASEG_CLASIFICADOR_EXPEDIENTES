import pandas as pd
import streamlit as st

from services.document_profile_service import (
    MAX_DOCUMENTOS_MUESTRA,
    analizar_muestra_fichas_documentales,
)
from services.openai_multimodal_service import (
    MODELOS,
    openai_configurado,
)
from ui.common import mostrar_encabezado, requerir_expediente


mostrar_encabezado(
    "Ficha documental experimental",
    "Prueba paralela de una sola lectura multimodal rica por PDF",
)
requerir_expediente()

if not openai_configurado():
    st.error(
        "La API key de OpenAI no está configurada en Streamlit Secrets."
    )
    st.stop()

inventario = st.session_state["inventario"]
contenido_zip = st.session_state["contenido_zip"]

st.info(
    "Esta prueba NO reemplaza ni modifica el análisis de estimaciones que ya "
    "funciona. Ejecuta una arquitectura paralela: una sola llamada multimodal "
    "por PDF para producir una ficha documental rica y reutilizable. No usa "
    "catálogo, no asigna códigos, no usa JEV y no altera el expediente."
)

st.caption(
    "Objetivo de la prueba: comprobar si una sola lectura puede identificar "
    "documentos lógicos internos, rangos de páginas, identidad, función, acto "
    "documentado y suficiente texto representativo para que después JEV y las "
    "reglas trabajen sin volver a mirar el PDF."
)

pdfs = inventario[
    inventario["Extensión"]
    .astype(str)
    .str.lower()
    .eq(".pdf")
].copy()

if pdfs.empty:
    st.warning("El expediente activo no contiene PDFs.")
    st.stop()

opciones = {
    f"{fila['Archivo']} · {fila['Ruta original']}": str(
        fila["Ruta original"]
    )
    for _, fila in pdfs.iterrows()
}

selecciones = st.multiselect(
    "Selecciona una muestra pequeña de PDFs",
    options=list(opciones.keys()),
    default=[],
    max_selections=MAX_DOCUMENTOS_MUESTRA,
    help=(
        f"Máximo {MAX_DOCUMENTOS_MUESTRA} PDFs. Cada PDF seleccionado "
        "genera exactamente una llamada multimodal en esta prueba."
    ),
)

rutas_seleccionadas = [
    opciones[seleccion]
    for seleccion in selecciones
]

modelo = st.selectbox(
    "Modelo multimodal",
    options=list(MODELOS.keys()),
    format_func=lambda clave: MODELOS[clave]["label"],
)

if rutas_seleccionadas:
    vista = pdfs[
        pdfs["Ruta original"].astype(str).isin(
            rutas_seleccionadas
        )
    ][
        [
            "Archivo",
            "Ruta original",
            "Tamaño (bytes)",
        ]
    ].copy()

    st.dataframe(
        vista,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Esta ejecución realizará {len(rutas_seleccionadas)} llamada(s) "
        "multimodal(es), una por PDF. El costo real se mostrará al terminar."
    )

puede_ejecutar = (
    1 <= len(rutas_seleccionadas) <= MAX_DOCUMENTOS_MUESTRA
)

if st.button(
    "Generar fichas documentales experimentales",
    type="primary",
    disabled=not puede_ejecutar,
):
    barra = st.progress(0.0)
    estado = st.empty()

    def actualizar_progreso(posicion, total, archivo):
        barra.progress(
            min(max(posicion / total, 0.0), 1.0)
            if total
            else 1.0
        )
        estado.write(
            f"Analizando PDF {posicion}/{total}: {archivo}"
        )

    with st.spinner(
        "Generando una ficha documental rica por cada PDF seleccionado..."
    ):
        try:
            (
                archivos,
                documentos,
                marcadores,
                resumen,
                perfiles_crudos,
            ) = analizar_muestra_fichas_documentales(
                contenido_zip=contenido_zip,
                inventario=inventario,
                rutas_pdf=rutas_seleccionadas,
                modelo=modelo,
                on_progress=actualizar_progreso,
            )
        except Exception as error:
            barra.empty()
            estado.empty()
            st.error(str(error))
            st.stop()

    barra.empty()
    estado.empty()

    st.session_state[
        "ficha_documental_experimental"
    ] = {
        "archivos": archivos,
        "documentos": documentos,
        "marcadores": marcadores,
        "resumen": resumen,
        "perfiles_crudos": perfiles_crudos,
        "expediente_id": st.session_state.get(
            "expediente_id",
            "",
        ),
        "modelo": modelo,
    }

guardado = st.session_state.get(
    "ficha_documental_experimental"
)

if (
    guardado
    and guardado.get("expediente_id", "")
    == st.session_state.get("expediente_id", "")
):
    archivos = guardado["archivos"]
    documentos = guardado["documentos"]
    marcadores = guardado["marcadores"]
    resumen = guardado["resumen"]

    st.divider()
    st.subheader("Resultado de la prueba")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "PDF analizados",
        resumen["pdf_analizados"],
    )
    c2.metric(
        "Llamadas multimodales",
        resumen["llamadas_multimodales"],
    )
    c3.metric(
        "Documentos lógicos",
        resumen["documentos_logicos"],
    )
    c4.metric(
        "PDF compuestos",
        resumen["documentos_compuestos"],
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        "Costo total (USD)",
        f"{resumen['costo_total_usd']:.6f}",
    )
    c6.metric(
        "Tiempo acumulado (s)",
        f"{resumen['tiempo_total_s']:.1f}",
    )
    c7.metric(
        "Tokens entrada",
        resumen["tokens_entrada"],
    )
    c8.metric(
        "Tokens salida",
        resumen["tokens_salida"],
    )

    if (
        resumen["rangos_invalidos"] > 0
        or resumen["conteos_pagina_incorrectos"] > 0
    ):
        st.warning(
            "La ficha contiene inconsistencias de páginas. No debe integrarse "
            "al flujo principal hasta revisar esos casos."
        )

    st.subheader("1. Ficha física por PDF")
    st.dataframe(
        archivos[
            [
                "Archivo",
                "Páginas reales",
                "Páginas observadas IA",
                "Conteo páginas coincide",
                "Documento compuesto",
                "Documentos lógicos detectados",
                "Calidad de lectura",
                "Requiere OCR adicional",
                "Resumen general",
                "Incertidumbre segmentación",
                "Costo (USD)",
                "Tiempo (s)",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("2. Documentos lógicos detectados")
    st.dataframe(
        documentos[
            [
                "Archivo",
                "ID lógico",
                "Página inicial",
                "Página final",
                "Rango válido",
                "Título detectado",
                "Función formal",
                "Acto documentado",
                "Alcance documental",
                "Confianza (%)",
                "Datos clave",
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

    st.subheader("3. Texto reutilizable para etapas posteriores")

    if documentos.empty:
        st.info("No se detectaron documentos lógicos.")
    else:
        for indice, fila in documentos.iterrows():
            titulo = (
                f"{fila['Archivo']} · {fila['ID lógico']} · "
                f"págs. {fila['Página inicial']}-{fila['Página final']}"
            )
            with st.expander(titulo):
                st.write(
                    f"**Identidad:** {fila['Título detectado']}"
                )
                st.write(
                    f"**Función:** {fila['Función formal']}"
                )
                st.write(
                    f"**Acto documentado:** {fila['Acto documentado']}"
                )
                st.write(
                    "**Texto representativo:**"
                )
                st.write(
                    fila["Texto representativo"]
                    or "Sin texto representativo."
                )
                st.write("**Datos clave:**")
                st.write(
                    fila["Datos clave"]
                    or "No se extrajeron datos clave."
                )
                st.write("**Evidencia:**")
                st.write(
                    fila["Evidencia"]
                    or "Sin evidencia adicional."
                )

    st.subheader("4. Marcadores de página")

    if marcadores.empty:
        st.info(
            "La IA no generó marcadores de página para esta muestra."
        )
    else:
        st.dataframe(
            marcadores[
                [
                    "Archivo",
                    "ID lógico",
                    "Página",
                    "Página válida",
                    "Texto marcador",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.warning(
        "Este resultado sigue siendo experimental. Todavía NO alimenta JEV, "
        "el catálogo, la comparación conjunta de estimaciones ni la "
        "codificación final. Primero compararemos estas fichas contra los "
        "resultados que ya conocemos para detectar posibles regresiones."
    )
