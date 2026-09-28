import pandas as pd
import streamlit as st

from services.expediente_v2_diagnostic_service import (
    DIAGNOSTICO_INTEGRAL_SCHEMA_VERSION,
    ejecutar_diagnostico_integral_v2,
)
from services.openai_multimodal_service import (
    MODELOS,
    openai_configurado,
)
from services.profile_jev_service import (
    PIPELINE_VERSION_FICHA_JEV,
)
from ui.common import (
    mostrar_encabezado,
    requerir_expediente,
)


RESULTADO_DIAGNOSTICO_INTEGRAL_SCHEMA = 1


mostrar_encabezado(
    "Diagnóstico integral · Ficha V2",
    (
        "Expediente completo · propuesta de codificación sin modificar "
        "archivos"
    ),
)
requerir_expediente()

if not openai_configurado():
    st.error(
        "La API key de OpenAI no está configurada en Streamlit Secrets."
    )
    st.stop()

inventario = st.session_state["inventario"]
contenido_zip = st.session_state["contenido_zip"]
procedimiento = st.session_state["procedimiento"]
drive_folder_id = st.session_state.get(
    "drive_folder_id",
)

pdf_totales = int(
    inventario["Extensión"]
    .fillna("")
    .astype(str)
    .str.lower()
    .eq(".pdf")
    .sum()
)
no_pdf_totales = len(inventario) - pdf_totales

st.info(
    "Esta fase es exclusivamente diagnóstica. No renombra, mueve, elimina ni "
    "divide archivos. Para EST_n reutiliza el flujo ya validado de Ficha V2 + "
    "JEV + comparación conjunta. El resto de familias se muestra como "
    "propuesta provisional para detectar qué reglas/unidades faltan."
)

c0, c1, c2 = st.columns(3)
c0.metric(
    "Archivos del expediente",
    len(inventario),
)
c1.metric(
    "PDF a evaluar",
    pdf_totales,
)
c2.metric(
    "Archivos no-PDF",
    no_pdf_totales,
)

modelo = st.selectbox(
    "Modelo multimodal para Fichas V2 nuevas",
    options=list(MODELOS.keys()),
    format_func=lambda clave: MODELOS[clave]["label"],
)

usar_cache = st.checkbox(
    "Reutilizar Fichas V2 compatibles guardadas en Drive",
    value=True,
    disabled=not bool(drive_folder_id),
)

forzar_reanalisis = st.checkbox(
    "Forzar nuevo análisis multimodal de todos los PDF",
    value=False,
    disabled=not bool(drive_folder_id),
    help=(
        "Déjalo desactivado. Activarlo vuelve a pagar la lectura multimodal "
        "de todos los PDF aunque ya exista una Ficha V2 compatible."
    ),
)

st.caption(
    f"Pipeline JEV requerido: v{PIPELINE_VERSION_FICHA_JEV}. "
    "Cada PDF sin Ficha V2 compatible genera una llamada multimodal. "
    "Las Fichas ya almacenadas se reutilizan sin volver a enviar el PDF."
)

confirmar = st.checkbox(
    "Entiendo que los PDF aún no cacheados generarán llamadas multimodales",
    value=False,
)

puede_ejecutar = (
    pdf_totales > 0
    and confirmar
)

if st.button(
    "Ejecutar diagnóstico integral del expediente",
    type="primary",
    disabled=not puede_ejecutar,
):
    barra = st.progress(0.0)
    estado = st.empty()

    def progreso(
        etapa,
        posicion,
        total,
        detalle,
    ):
        if etapa == "Ficha V2":
            base = 0.0
            amplitud = 0.58
        elif etapa == "JEV":
            base = 0.58
            amplitud = 0.27
        elif etapa == "Comparaciones EST":
            base = 0.85
            amplitud = 0.10
        else:
            base = 0.95
            amplitud = 0.05

        fraccion = (
            posicion / total
            if total
            else 1.0
        )
        barra.progress(
            min(
                max(
                    base
                    + amplitud * fraccion,
                    0.0,
                ),
                1.0,
            )
        )
        estado.write(
            f"{etapa}: {posicion}/{total} · {detalle}"
        )

    with st.spinner(
        "Analizando el expediente completo. Esta primera corrida puede tardar "
        "si aún faltan Fichas V2 por generar..."
    ):
        try:
            resultado = (
                ejecutar_diagnostico_integral_v2(
                    contenido_zip=contenido_zip,
                    inventario=inventario,
                    procedimiento=procedimiento,
                    modelo_multimodal=modelo,
                    drive_folder_id=drive_folder_id,
                    usar_cache=usar_cache,
                    forzar_reanalisis=forzar_reanalisis,
                    on_progress=progreso,
                )
            )
        except Exception as error:
            barra.empty()
            estado.empty()
            st.error(str(error))
            st.stop()

    barra.empty()
    estado.empty()

    st.session_state[
        "diagnostico_integral_v2"
    ] = {
        "ui_schema_version": (
            RESULTADO_DIAGNOSTICO_INTEGRAL_SCHEMA
        ),
        "service_schema_version": (
            DIAGNOSTICO_INTEGRAL_SCHEMA_VERSION
        ),
        "expediente_id": st.session_state.get(
            "expediente_id",
            "",
        ),
        "modelo": modelo,
        "resultado": resultado,
    }

guardado = st.session_state.get(
    "diagnostico_integral_v2"
)

if (
    guardado
    and guardado.get(
        "ui_schema_version"
    )
    == RESULTADO_DIAGNOSTICO_INTEGRAL_SCHEMA
    and guardado.get(
        "service_schema_version"
    )
    == DIAGNOSTICO_INTEGRAL_SCHEMA_VERSION
    and guardado.get(
        "expediente_id",
        "",
    )
    == st.session_state.get(
        "expediente_id",
        "",
    )
):
    resultado = guardado["resultado"]
    resumen = resultado["resumen"]
    decisiones = resultado[
        "decisiones_fisicas"
    ]
    logicos = resultado[
        "resultados_logicos"
    ]
    detalle_jev = resultado[
        "detalle_jev"
    ]
    comparaciones = resultado[
        "comparaciones_est"
    ]
    duplicados = resultado[
        "grupos_duplicados"
    ]
    no_pdf = resultado["no_pdf"]
    fichas = resultado["archivos_ficha"]

    st.divider()
    st.subheader(
        "Resultado del diagnóstico integral"
    )

    a1, a2, a3, a4 = st.columns(4)
    a1.metric(
        "PDF",
        resumen["pdf_totales"],
    )
    a2.metric(
        "Documentos lógicos",
        resumen["documentos_logicos"],
    )
    a3.metric(
        "Fichas desde cache",
        resumen[
            "fichas_cache_reutilizadas"
        ],
    )
    a4.metric(
        "Fichas multimodales nuevas",
        resumen[
            "fichas_multimodales_nuevas"
        ],
    )

    b1, b2, b3, b4 = st.columns(4)
    b1.metric(
        "Llamadas JEV",
        resumen["llamadas_jev"],
    )
    b2.metric(
        "Comparaciones EST multimodales",
        resumen[
            "comparaciones_est_multimodales"
        ],
    )
    b3.metric(
        "Archivos a dividir",
        resumen["archivos_a_dividir"],
    )
    b4.metric(
        "Revisión física",
        resumen["archivos_revision"],
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Pipeline JEV",
        resumen[
            "pipeline_jev_version"
        ],
    )
    c2.metric(
        "Estimaciones detectadas",
        resumen[
            "estimaciones_detectadas"
        ],
    )
    c3.metric(
        "Duplicados exactos",
        resumen[
            "grupos_duplicados_exactos"
        ],
    )
    c4.metric(
        "Costo API actual (USD)",
        f"{resumen['costo_api_total_usd']:.6f}",
    )

    st.caption(
        f"Costo Fichas nuevas: USD "
        f"{resumen['costo_fichas_usd']:.6f} · "
        f"Costo JEV: USD {resumen['costo_jev_usd']:.6f} · "
        f"Comparaciones EST: USD "
        f"{resumen['costo_comparaciones_est_usd']:.6f}"
    )

    st.warning(
        resumen["nota_alcance"]
    )

    st.subheader(
        "1. Propuesta por archivo físico"
    )

    columnas_fisicas = [
        "Archivo original",
        "Ruta original",
        "Documentos lógicos",
        "Códigos principales",
        "Decisión física",
        "Nombre de salida propuesto",
        "Páginas de salida",
        "Requiere división",
        "Requiere revisión",
        "Es duplicado exacto",
        "Grupo duplicado exacto",
        "Motivo",
    ]
    columnas_fisicas = [
        columna
        for columna in columnas_fisicas
        if columna in decisiones.columns
    ]

    st.dataframe(
        decisiones[columnas_fisicas],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Descargar propuesta física CSV",
        data=decisiones.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name=(
            "diagnostico_integral_propuesta_fisica.csv"
        ),
        mime="text/csv",
    )

    revision = decisiones[
        decisiones[
            "Requiere revisión"
        ].fillna(False).astype(bool)
    ].copy()

    st.subheader(
        "2. Casos que requieren revisión"
    )
    if revision.empty:
        st.success(
            "No hay archivos físicos marcados para revisión en esta corrida."
        )
    else:
        st.dataframe(
            revision[
                [
                    columna
                    for columna in (
                        "Archivo original",
                        "Ruta original",
                        "Códigos principales",
                        "Decisión física",
                        "Motivo",
                    )
                    if columna
                    in revision.columns
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader(
        "3. Comparaciones conjuntas de estimaciones"
    )
    if comparaciones.empty:
        st.info(
            "No se detectaron unidades EST_n para comparación conjunta."
        )
    else:
        st.dataframe(
            comparaciones,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader(
        "4. Duplicados exactos globales"
    )
    if duplicados.empty:
        st.success(
            "No se detectaron archivos binariamente idénticos."
        )
    else:
        st.dataframe(
            duplicados,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander(
        "5. Resultado por documento lógico"
    ):
        columnas_logicas = [
            "Archivo",
            "Ruta original",
            "ID lógico",
            "Título detectado",
            "Clasificación inicial",
            "Código inicial",
            "Alcance documental",
            "Relación con la unidad",
            "Relación comparativa",
            "Coincide catálogo",
            "Código de catálogo",
            "Grupo lógico",
            "Acción provisional",
        ]
        columnas_logicas = [
            columna
            for columna in columnas_logicas
            if columna in logicos.columns
        ]
        st.dataframe(
            logicos[columnas_logicas],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander(
        "6. Detalle JEV y reglas"
    ):
        columnas_jev = [
            "Archivo",
            "Ruta original",
            "ID lógico",
            "Título detectado",
            "Unidad estructural",
            "Consecutivo unidad",
            "Concepto JEV original",
            "Código JEV original",
            "Jerarquía contexto unidad aplicada",
            "Exclusión identidad aplicada",
            "Resolución competencia indeterminada",
            "Concepto JEV",
            "Código JEV",
            "Confianza JEV (%)",
            "Validación estricta JEV",
            "Confianza estricta JEV (%)",
            "Decisión provisional",
            "Código provisional",
            "Motivo",
        ]
        columnas_jev = [
            columna
            for columna in columnas_jev
            if columna in detalle_jev.columns
        ]
        st.dataframe(
            detalle_jev[columnas_jev],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander(
        "7. Uso de cache y costo original de Fichas V2"
    ):
        columnas_fichas = [
            "Archivo",
            "Ruta original",
            "Origen ficha",
            "SHA-256 PDF",
            "Modelo",
            "Costo (USD)",
            "Costo original ficha (USD)",
        ]
        columnas_fichas = [
            columna
            for columna in columnas_fichas
            if columna in fichas.columns
        ]
        st.dataframe(
            fichas[columnas_fichas],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander(
        "8. Archivos no-PDF"
    ):
        if no_pdf.empty:
            st.info(
                "No hay archivos no-PDF en el expediente."
            )
        else:
            st.dataframe(
                no_pdf,
                use_container_width=True,
                hide_index=True,
            )

    st.info(
        "Siguiente criterio de validación: comparar esta propuesta contra el "
        "expediente codificado manualmente. Todavía no se crea ni modifica "
        "ninguna carpeta de salida."
    )
