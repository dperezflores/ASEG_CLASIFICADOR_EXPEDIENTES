import pandas as pd
import streamlit as st

from services.catalog_service import cargar_catalogo
from services.document_profile_service import (
    MAX_DOCUMENTOS_MUESTRA,
    analizar_muestra_fichas_documentales,
)
from services.jev_classifier_service import jev_configurado
from services.openai_multimodal_service import (
    MODELOS,
    openai_configurado,
)
from services.profile_jev_service import (
    clasificar_fichas_v2_con_jev,
)
from ui.common import mostrar_encabezado, requerir_expediente


FICHA_SCHEMA_VERSION = 2
JEV_RESULT_SCHEMA_VERSION = 2


def _resultado_jev_pipeline_actual(detalle, resumen) -> bool:
    columnas_requeridas = {
        "Validación estricta JEV",
        "Regla relación interna",
        "Decisión provisional",
    }
    claves_requeridas = {
        "llamadas_jev_clasificacion",
        "llamadas_jev_validacion",
        "codigos_propios_validados",
        "soportes_relacion_interna",
        "candidatos_rechazados",
    }

    return (
        isinstance(detalle, pd.DataFrame)
        and columnas_requeridas.issubset(
            set(detalle.columns)
        )
        and isinstance(resumen, dict)
        and resumen.get("pipeline_version") == 2
        and claves_requeridas.issubset(
            set(resumen.keys())
        )
    )


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
procedimiento = st.session_state["procedimiento"]

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

    st.session_state.pop(
        "ficha_v2_jev_experimental",
        None,
    )

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
        "La Ficha V2 sigue siendo experimental y todavía no sustituye el flujo "
        "principal. La prueba siguiente clasifica su TEXTO con JEV sin volver "
        "a enviar los PDFs al modelo multimodal."
    )

    st.divider()
    st.subheader("7. JEV en dos etapas sobre la Ficha V2")

    st.info(
        "La Ficha V2 se reutiliza sin reenviar los PDFs a OpenAI. Primero JEV "
        "propone un concepto candidato entre el catálogo. Después Python aplica "
        "reglas determinísticas y solo los códigos propios que aún lo requieren "
        "pasan a una segunda llamada JEV de equivalencia estricta contra UN "
        "solo concepto candidato."
    )

    st.caption(
        f"Etapa 1: {len(documentos)} llamada(s) JEV, una por documento lógico. "
        "Etapa 2: llamadas JEV selectivas únicamente para códigos propios que "
        "no sean extractos, candidatos EST_n ni piezas subordinadas por una "
        "relación interna. Multimodales adicionales: 0."
    )

    if not jev_configurado():
        st.warning(
            "TypeSafe/JEV no está configurado en Streamlit Secrets; la Ficha V2 "
            "permanece disponible, pero no puede ejecutarse esta comparación."
        )
    else:
        if st.button(
            "Clasificar y validar Ficha V2 con JEV",
            type="primary",
        ):
            barra_jev = st.progress(0.0)
            estado_jev = st.empty()

            def actualizar_progreso_jev(
                posicion,
                total,
                archivo,
                logical_id,
            ):
                progreso = (
                    min(max(posicion / total, 0.0), 1.0)
                    if total
                    else 1.0
                )
                barra_jev.progress(progreso)
                estado_jev.write(
                    f"JEV {posicion}/{total}: "
                    f"{archivo} · {logical_id}"
                )

            with st.spinner(
                "Clasificando la ficha y validando candidatos sin reenviar PDFs..."
            ):
                try:
                    catalogo_base = cargar_catalogo(
                        procedimiento
                    )
                    (
                        detalle_jev,
                        resumen_jev,
                    ) = clasificar_fichas_v2_con_jev(
                        documentos=documentos,
                        registros=registros,
                        relaciones=relaciones,
                        inventario=inventario,
                        catalogo_base=catalogo_base,
                        procedimiento=procedimiento,
                        on_progress=actualizar_progreso_jev,
                    )
                except Exception as error:
                    barra_jev.empty()
                    estado_jev.empty()
                    st.error(str(error))
                    st.stop()

            barra_jev.empty()
            estado_jev.empty()

            resultado_pipeline_actual = (
                _resultado_jev_pipeline_actual(
                    detalle_jev,
                    resumen_jev,
                )
            )

            st.session_state[
                "ficha_v2_jev_experimental"
            ] = {
                "schema_version": FICHA_SCHEMA_VERSION,
                "jev_schema_version": (
                    JEV_RESULT_SCHEMA_VERSION
                    if resultado_pipeline_actual
                    else 1
                ),
                "expediente_id": st.session_state.get(
                    "expediente_id",
                    "",
                ),
                "detalle": detalle_jev,
                "resumen": resumen_jev,
            }

            if not resultado_pipeline_actual:
                st.warning(
                    "La ejecución terminó, pero el servidor todavía utilizó "
                    "una versión anterior del servicio JEV. No se perdió la "
                    "Ficha V2 ni es necesario repetir las llamadas multimodales. "
                    "Espera a que termine el despliegue y vuelve a pulsar "
                    "únicamente el botón JEV."
                )

        jev_guardado = st.session_state.get(
            "ficha_v2_jev_experimental"
        )

        jev_mismo_expediente = (
            jev_guardado
            and jev_guardado.get("schema_version")
            == FICHA_SCHEMA_VERSION
            and jev_guardado.get("expediente_id", "")
            == st.session_state.get(
                "expediente_id",
                "",
            )
        )

        if (
            jev_mismo_expediente
            and jev_guardado.get(
                "jev_schema_version"
            )
            != JEV_RESULT_SCHEMA_VERSION
        ):
            st.info(
                "El resultado JEV guardado no corresponde al pipeline actual "
                "de candidato + equivalencia estricta. Puede ser el resultado "
                "anterior o una ejecución realizada mientras Streamlit todavía "
                "estaba actualizando el servicio. La Ficha V2 se conserva: "
                "vuelve a pulsar únicamente el botón JEV. No repitas las "
                "llamadas multimodales."
            )

        if (
            jev_mismo_expediente
            and jev_guardado.get(
                "jev_schema_version"
            )
            == JEV_RESULT_SCHEMA_VERSION
        ):
            detalle_jev = jev_guardado["detalle"]
            resumen_jev = jev_guardado["resumen"]

            st.subheader(
                "Resultado JEV: candidato + equivalencia estricta"
            )

            j1, j2, j3, j4 = st.columns(4)
            j1.metric(
                "Documentos lógicos",
                resumen_jev[
                    "documentos_logicos_evaluados"
                ],
            )
            j2.metric(
                "JEV clasificación",
                resumen_jev[
                    "llamadas_jev_clasificacion"
                ],
            )
            j3.metric(
                "JEV validación",
                resumen_jev[
                    "llamadas_jev_validacion"
                ],
            )
            j4.metric(
                "Multimodales adicionales",
                resumen_jev[
                    "multimodales_adicionales"
                ],
            )

            j5, j6, j7, j8 = st.columns(4)
            j5.metric(
                "JEV totales",
                resumen_jev["llamadas_jev"],
            )
            j6.metric(
                "Costo JEV total (USD)",
                f"{resumen_jev['costo_total_jev_usd']:.6f}",
            )
            j7.metric(
                "Candidatos unidad",
                resumen_jev["candidatos_unidad"],
            )
            j8.metric(
                "Extractos sin código",
                resumen_jev[
                    "extractos_sin_codigo"
                ],
            )

            j9, j10, j11, j12 = st.columns(4)
            j9.metric(
                "Códigos propios validados",
                resumen_jev[
                    "codigos_propios_validados"
                ],
            )
            j10.metric(
                "Soportes por relación",
                resumen_jev[
                    "soportes_relacion_interna"
                ],
            )
            j11.metric(
                "Candidatos rechazados",
                resumen_jev[
                    "candidatos_rechazados"
                ],
            )
            j12.metric(
                "Revisión",
                resumen_jev["revision"],
            )

            st.caption(
                f"Tiempo JEV total: "
                f"{resumen_jev['tiempo_total_jev_s']:.2f} s · "
                f"Clasificación: "
                f"{resumen_jev['tiempo_jev_clasificacion_s']:.2f} s · "
                f"Validación: "
                f"{resumen_jev['tiempo_jev_validacion_s']:.2f} s · "
                f"Tokens entrada totales: "
                f"{resumen_jev['tokens_entrada_jev']} · "
                f"Tokens salida totales: "
                f"{resumen_jev['tokens_salida_jev']}"
            )

            st.dataframe(
                detalle_jev[
                    [
                        "Archivo",
                        "ID lógico",
                        "Título detectado",
                        "Alcance documental",
                        "Unidad estructural",
                        "Consecutivo unidad",
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
                ],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confianza JEV (%)": (
                        st.column_config.NumberColumn(
                            "Confianza JEV (%)",
                            format="%.1f %%",
                        )
                    ),
                    "Confianza estricta JEV (%)": (
                        st.column_config.NumberColumn(
                            "Confianza estricta JEV (%)",
                            format="%.1f %%",
                        )
                    ),
                },
            )

            with st.expander(
                "Ver detalle de clasificación, validación y consumo JEV"
            ):
                st.dataframe(
                    detalle_jev[
                        [
                            "Archivo",
                            "ID lógico",
                            "Top 3 JEV",
                            "Probabilidad elegida JEV (%)",
                            "Decisión original estricta JEV",
                            "Probabilidad estricta JEV (%)",
                            "Margen estricta JEV (%)",
                            "Control estricta JEV",
                            "Error validación estricta",
                            "Texto enviado a JEV",
                            "Tokens entrada JEV clasificación",
                            "Tokens salida JEV clasificación",
                            "Costo JEV clasificación (USD)",
                            "Tiempo JEV clasificación (s)",
                            "Tokens entrada JEV validación",
                            "Tokens salida JEV validación",
                            "Costo JEV validación (USD)",
                            "Tiempo JEV validación (s)",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            st.warning(
                "Los resultados siguen siendo experimentales. Para EST_n no se "
                "elige representante aquí: CARÁTULA/ESTIMACIÓN continúan hacia "
                "la comparación conjunta validada. Una pieza autentica_a o "
                "soporte_de no hereda el mismo código del documento principal. "
                "Los demás códigos propios solo se conservan cuando la segunda "
                "comparación JEV confirma equivalencia."
            )
