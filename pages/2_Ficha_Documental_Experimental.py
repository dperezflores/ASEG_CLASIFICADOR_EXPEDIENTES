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


FICHA_SCHEMA_VERSION = 2


mostrar_encabezado(
    "Ficha documental experimental",
    "V2 · documentos lógicos, registros internos y relaciones en una sola lectura",
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
    "V2 distingue documentos lógicos autónomos de registros internos. También "
    "extrae alcance y límites de identidad y relaciones entre piezas del mismo "
    "PDF. La prueba sigue aislada del catálogo, JEV y el motor de estimaciones."
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
                registros,
                relaciones,
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
        "schema_version": FICHA_SCHEMA_VERSION,
        "archivos": archivos,
        "documentos": documentos,
        "registros": registros,
        "relaciones": relaciones,
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

resultado_actual = (
    guardado
    and guardado.get("expediente_id", "")
    == st.session_state.get("expediente_id", "")
)

if (
    resultado_actual
    and guardado.get("schema_version")
    != FICHA_SCHEMA_VERSION
):
    st.info(
        "Existe un resultado de una versión experimental anterior. Se conserva "
        "sin reutilizarlo. Ejecuta nuevamente la misma muestra para generar la "
        "ficha V2; no se hace ningún reanálisis automático."
    )

if (
    resultado_actual
    and guardado.get("schema_version")
    == FICHA_SCHEMA_VERSION
):
    archivos = guardado["archivos"]
    documentos = guardado["documentos"]
    registros = guardado["registros"]
    relaciones = guardado["relaciones"]
    marcadores = guardado["marcadores"]
    resumen = guardado["resumen"]

    st.divider()
    st.subheader("Resultado de la prueba V2")

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
        "Registros internos",
        resumen["registros_internos"],
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        "Relaciones lógicas",
        resumen["relaciones_logicas"],
    )
    c6.metric(
        "PDF compuestos",
        resumen["documentos_compuestos"],
    )
    c7.metric(
        "Costo total (USD)",
        f"{resumen['costo_total_usd']:.6f}",
    )
    c8.metric(
        "Tiempo acumulado (s)",
        f"{resumen['tiempo_total_s']:.1f}",
    )

    c9, c10, c11, c12 = st.columns(4)
    c9.metric(
        "Tokens entrada",
        resumen["tokens_entrada"],
    )
    c10.metric(
        "Tokens salida",
        resumen["tokens_salida"],
    )
    c11.metric(
        "Requieren OCR adicional",
        resumen["requieren_ocr_adicional"],
    )
    c12.metric(
        "Relaciones inválidas",
        resumen["relaciones_invalidas"],
    )

    if (
        resumen["rangos_invalidos"] > 0
        or resumen["registros_rango_invalido"] > 0
        or resumen["conteos_pagina_incorrectos"] > 0
        or resumen["relaciones_invalidas"] > 0
    ):
        st.warning(
            "La ficha V2 contiene alguna inconsistencia estructural de páginas "
            "o relaciones. No debe integrarse al flujo principal hasta revisar "
            "esos casos."
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
                "Registros internos detectados",
                "Relaciones lógicas detectadas",
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
                "Alcance de identidad",
                "Límites de identidad",
                "Alcance documental",
                "Registros internos",
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
                    f"**Alcance de identidad:** {fila['Alcance de identidad']}"
                )
                st.write(
                    f"**Límites de identidad:** {fila['Límites de identidad']}"
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

    st.subheader("4. Registros internos")

    if registros.empty:
        st.info(
            "No se detectaron registros internos relevantes en esta muestra."
        )
    else:
        st.dataframe(
            registros[
                [
                    "Archivo",
                    "ID lógico",
                    "ID registro",
                    "Tipo registro",
                    "Página inicial",
                    "Página final",
                    "Rango válido",
                    "Etiqueta",
                    "Acto documentado",
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

    st.subheader("5. Relaciones entre documentos lógicos")

    if relaciones.empty:
        st.info(
            "No se detectaron relaciones entre documentos lógicos de un mismo PDF."
        )
    else:
        st.dataframe(
            relaciones[
                [
                    "Archivo",
                    "Origen",
                    "Relación",
                    "Destino",
                    "IDs válidos",
                    "Confianza (%)",
                    "Evidencia",
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

    st.subheader("6. Marcadores de página")

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
        "Este resultado V2 sigue siendo experimental. Todavía NO alimenta JEV, "
        "el catálogo, la comparación conjunta de estimaciones ni la "
        "codificación final. La siguiente validación es repetir la misma "
        "muestra de cinco PDFs y verificar que NOTAS DE BITÁCORA se modele como "
        "un documento lógico con notas internas, sin degradar los otros casos."
    )
