import pandas as pd
import streamlit as st

from services.estimation_v2_experimental_service import (
    ejecutar_est_completa_ficha_v2,
    listar_estimaciones_para_prueba,
)
from services.openai_multimodal_service import (
    MODELOS,
    openai_configurado,
)
from ui.common import mostrar_encabezado, requerir_expediente


RESULTADO_EST_V2_SCHEMA_VERSION = 2


mostrar_encabezado(
    "EST completa · Ficha V2 experimental",
    "Ficha V2 cacheable + JEV + comparación conjunta existente",
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

st.info(
    "Este modo NO reemplaza el análisis de unidad completa actual. Ejecuta en "
    "paralelo el nuevo flujo: estructura → Ficha V2 persistente → JEV → "
    "comparación conjunta ya validada → agrupación existente."
)

estimaciones = listar_estimaciones_para_prueba(
    inventario
)

if estimaciones.empty:
    st.warning(
        "No se detectaron carpetas de estimación en el expediente activo."
    )
    st.stop()

opciones = {}
for _, fila in estimaciones.iterrows():
    ruta = str(fila["Ruta carpeta"])
    consecutivo = int(fila["Consecutivo"])
    etiqueta = (
        f"EST {consecutivo} · {ruta} · "
        f"{int(fila['Archivos directos'])} archivo(s) directo(s)"
    )
    opciones[etiqueta] = {
        "ruta": ruta,
        "consecutivo": consecutivo,
    }

seleccion = st.selectbox(
    "Estimación a analizar",
    options=list(opciones.keys()),
)
unidad = opciones[seleccion]

modelo = st.selectbox(
    "Modelo multimodal",
    options=list(MODELOS.keys()),
    format_func=lambda clave: MODELOS[clave]["label"],
)

usar_cache = st.checkbox(
    "Reutilizar Fichas V2 compatibles guardadas en Drive",
    value=True,
    disabled=not bool(drive_folder_id),
)

forzar_reanalisis = st.checkbox(
    "Forzar nuevo análisis multimodal de todas las fichas",
    value=False,
    disabled=not bool(drive_folder_id),
    help=(
        "Déjalo desactivado durante pruebas normales. Úsalo solo para volver "
        "a medir deliberadamente la Ficha V2."
    ),
)

st.caption(
    "Las Fichas V2 ya existentes solo se reutilizan cuando coinciden SHA-256 "
    "del PDF, versión de esquema, versión de prompt y modelo. La comparación "
    "conjunta existente puede realizar una llamada multimodal adicional sobre "
    "los candidatos al código EST_n."
)

if st.button(
    "Ejecutar EST completa con Ficha V2",
    type="primary",
):
    barra = st.progress(0.0)
    estado = st.empty()

    def progreso(etapa, posicion, total, detalle):
        if etapa == "Ficha V2":
            base = 0.0
            amplitud = 0.65
        elif etapa == "JEV":
            base = 0.65
            amplitud = 0.25
        else:
            base = 0.90
            amplitud = 0.10

        fraccion = (
            posicion / total
            if total
            else 1.0
        )
        barra.progress(
            min(
                max(
                    base + amplitud * fraccion,
                    0.0,
                ),
                1.0,
            )
        )
        estado.write(
            f"{etapa}: {posicion}/{total} · {detalle}"
        )

    with st.spinner(
        "Ejecutando pipeline experimental de la estimación completa..."
    ):
        try:
            resultado = ejecutar_est_completa_ficha_v2(
                contenido_zip=contenido_zip,
                inventario=inventario,
                ruta_carpeta=unidad["ruta"],
                procedimiento=procedimiento,
                consecutivo=unidad["consecutivo"],
                modelo_multimodal=modelo,
                drive_folder_id=drive_folder_id,
                usar_cache=usar_cache,
                forzar_reanalisis=forzar_reanalisis,
                on_progress=progreso,
            )
        except Exception as error:
            barra.empty()
            estado.empty()
            st.error(str(error))
            st.stop()

    barra.empty()
    estado.empty()

    st.session_state[
        "est_completa_ficha_v2_experimental"
    ] = {
        "schema_version": RESULTADO_EST_V2_SCHEMA_VERSION,
        "expediente_id": st.session_state.get(
            "expediente_id",
            "",
        ),
        "ruta_carpeta": unidad["ruta"],
        "consecutivo": unidad["consecutivo"],
        "modelo": modelo,
        "resultado": resultado,
    }

guardado = st.session_state.get(
    "est_completa_ficha_v2_experimental"
)

if (
    guardado
    and guardado.get("schema_version")
    == RESULTADO_EST_V2_SCHEMA_VERSION
    and guardado.get("expediente_id", "")
    == st.session_state.get(
        "expediente_id",
        "",
    )
):
    resultado = guardado["resultado"]
    resumen = resultado["resumen_pipeline"]
    finales = resultado["resultados_finales"]
    comparacion = resultado["detalle_comparacion"]
    grupos = resultado["resumen_grupos"]
    decisiones_fisicas = resultado["decisiones_fisicas"]
    resumen_fisico = resultado["resumen_fisico"]
    archivos_ficha = resultado["archivos_ficha"]
    detalle_jev = resultado["detalle_jev"]
    meta_comparacion = resultado["meta_comparacion"]

    st.divider()
    st.subheader(
        f"Resultado experimental · EST {resumen['consecutivo']}"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "PDF directos",
        resumen["pdfs_directos"],
    )
    c2.metric(
        "Fichas desde cache",
        resumen[
            "fichas_cache_reutilizadas"
        ],
    )
    c3.metric(
        "Multimodales ficha nuevas",
        resumen[
            "multimodales_ficha_nuevas"
        ],
    )
    c4.metric(
        "Documentos lógicos",
        resumen["documentos_logicos"],
    )

    cf1, cf2, cf3, cf4 = st.columns(4)
    cf1.metric(
        "Archivos físicos consolidados",
        resumen_fisico["archivos_fisicos"],
    )
    cf2.metric(
        "Salidas físicas propuestas",
        resumen_fisico["salidas_fisicas"],
    )
    cf3.metric(
        "Archivos a dividir",
        resumen_fisico["archivos_a_dividir"],
    )
    cf4.metric(
        "Revisión física",
        resumen_fisico["archivos_revision"],
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        "Llamadas JEV",
        resumen["llamadas_jev"],
    )
    c6.metric(
        "Multimodal comparación conjunta",
        resumen[
            "multimodales_comparacion_conjunta"
        ],
    )
    c7.metric(
        "Multimodales nuevas totales",
        resumen[
            "multimodales_totales_nuevas"
        ],
    )
    c8.metric(
        "Costo API actual (USD)",
        f"{resumen['costo_api_total_usd']:.6f}",
    )

    st.caption(
        f"Representante propuesto: "
        f"{resumen['representante'] or 'pendiente'} · "
        f"Estado comparación: {resumen['estado_comparacion']} · "
        f"Costo fichas nuevas: USD "
        f"{resumen['costo_fichas_nuevas_usd']:.6f} · "
        f"Costo JEV: USD {resumen['costo_jev_usd']:.6f} · "
        f"Costo comparación conjunta: USD "
        f"{resumen['costo_comparacion_conjunta_usd']:.6f}"
    )

    st.subheader("1. Decisión final por archivo físico")

    st.caption(
        "Esta tabla vuelve a consolidar los documentos lógicos en el archivo "
        "físico real. Un PDF con varios documentos de soporte aparece una sola "
        "vez. Si existe un único documento principal con código, ese código "
        "domina el archivo completo. Solo se proponen varias salidas cuando "
        "existen códigos principales distintos y fronteras de páginas válidas."
    )

    columnas_fisicas = [
        "Archivo original",
        "Documentos lógicos",
        "Códigos principales",
        "Decisión física",
        "Nombre de salida propuesto",
        "Páginas de salida",
        "Requiere división",
        "Requiere revisión",
        "Motivo",
    ]
    columnas_fisicas = [
        columna
        for columna in columnas_fisicas
        if columna in decisiones_fisicas.columns
    ]

    st.dataframe(
        decisiones_fisicas[columnas_fisicas],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("2. Diagnóstico por documento lógico")

    columnas_finales = [
        "Archivo",
        "ID lógico",
        "Título detectado",
        "Clasificación inicial",
        "Código inicial",
        "Alcance documental",
        "Relación con la unidad",
        "Relación comparativa",
        "Confianza comparativa (%)",
        "Coincide catálogo",
        "Código de catálogo",
        "Rol propuesto en la unidad",
        "Grupo lógico",
        "Relación consolidada",
        "Acción provisional",
    ]
    columnas_finales = [
        columna
        for columna in columnas_finales
        if columna in finales.columns
    ]

    st.dataframe(
        finales[columnas_finales],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confianza comparativa (%)": (
                st.column_config.NumberColumn(
                    "Confianza comparativa (%)",
                    format="%.1f %%",
                )
            ),
        },
    )

    st.subheader("3. Comparación conjunta existente")

    if comparacion.empty:
        st.info(
            "No hubo candidatos EST_n que requirieran comparación conjunta."
        )
    else:
        st.dataframe(
            comparacion,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        str(
            meta_comparacion.get(
                "Evidencia unidad",
                "",
            )
        )
    )

    st.subheader("4. Grupos lógicos resultantes")
    st.dataframe(
        grupos,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander(
        "Ver origen de Fichas V2 y ahorro de API"
    ):
        columnas_cache = [
            "Archivo",
            "Ruta original",
            "Origen ficha",
            "SHA-256 PDF",
            "Modelo",
            "Costo (USD)",
            "Costo original ficha (USD)",
        ]
        columnas_cache = [
            columna
            for columna in columnas_cache
            if columna in archivos_ficha.columns
        ]
        st.dataframe(
            archivos_ficha[columnas_cache],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander(
        "Ver detalle JEV previo a comparación conjunta"
    ):
        columnas_jev = [
            "Archivo",
            "ID lógico",
            "Título detectado",
            "Concepto JEV",
            "Código JEV",
            "Confianza JEV (%)",
            "Regla relación interna",
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

    st.warning(
        "Este resultado es exclusivamente experimental. La consolidación "
        "físico-lógica NO modifica ni divide archivos todavía: solo propone "
        "la salida física. El módulo actual de Análisis unidad completa "
        "permanece intacto y sigue siendo el punto de referencia."
    )
