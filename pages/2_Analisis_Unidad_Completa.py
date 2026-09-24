import pandas as pd
import streamlit as st

from services.catalog_service import cargar_catalogo
from services.catalog_family_service import (
    construir_catalogo_operativo_estimacion,
)
from services.openai_multimodal_service import (
    MODELOS,
    openai_configurado,
)
from services.structural_analysis_service import (
    construir_mapa_estructural,
    obtener_archivos_unidad,
    obtener_estimaciones_detectadas,
)
from services.unit_content_analysis_service import (
    agrupar_resultados_unidad,
    analizar_unidad_completa,
    consolidar_candidatos_unidad,
)
from ui.common import mostrar_encabezado, requerir_expediente


RESULTADO_UNIDAD_SCHEMA_VERSION = 6


mostrar_encabezado(
    "Análisis de unidad completa",
    "Primera consolidación de todos los componentes de una estimación",
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
catalogo = cargar_catalogo(procedimiento)

mapa = construir_mapa_estructural(inventario)
estimaciones = obtener_estimaciones_detectadas(mapa)

st.info(
    "Se conserva el flujo validado. Para códigos propios sensibles, la "
    "identidad pura se compara por texto con JEV y existe fallback multimodal "
    "si hace falta. Además, si varios documentos terminan proponiendo el mismo "
    "código propio definido como de representación única, el sistema resuelve "
    "el conflicto sin nuevas llamadas de IA usando la confianza final: exige "
    "al menos 90% para el mejor candidato y una ventaja mínima de 10 puntos "
    "porcentuales. Si no se cumplen ambas condiciones, manda el conflicto a "
    "revisión manual."
)

if estimaciones.empty:
    st.warning("No hay estimaciones detectadas en el mapa estructural.")
    st.stop()

mapa_unidades = {
    f"{fila['Carpeta']} · {fila['Ruta carpeta']}": fila
    for _, fila in estimaciones.iterrows()
}

seleccion = st.selectbox(
    "Unidad documental a analizar",
    options=list(mapa_unidades.keys()),
    index=0,
    help=(
        "El sistema parametriza automáticamente las familias consecutivas "
        "con el número de la estimación seleccionada."
    ),
)

unidad = mapa_unidades[seleccion]

archivos = obtener_archivos_unidad(
    inventario,
    unidad["Ruta carpeta"],
)

archivos_pdf = archivos[
    archivos["Extensión"].astype(str).str.lower() == ".pdf"
].copy()

catalogo_operativo, cambios_familias = (
    construir_catalogo_operativo_estimacion(
        catalogo=catalogo,
        procedimiento=procedimiento,
        consecutivo=int(unidad["Consecutivo"]),
    )
)

st.subheader("1. Alcance de la prueba")

c1, c2, c3 = st.columns(3)
c1.metric("Unidad", str(unidad["Carpeta"]))
c2.metric("PDF directos", len(archivos_pdf))
c3.metric("Procedimiento", procedimiento)

st.caption(
    "La IA no recibe el nombre real ni la ruta. Un código propio solo se "
    "acepta si supera primero la validación independiente de equivalencia "
    "documental-funcional y después, cuando corresponde, la de integridad. "
    "Los candidatos al código de la estimación se resuelven mediante una "
    "comparación conjunta."
)

if not cambios_familias.empty:
    with st.expander("Ver familias consecutivas aplicadas"):
        st.dataframe(
            cambios_familias[
                [
                    "Familia",
                    "Código base",
                    "Código operativo",
                    "Concepto base",
                    "Concepto operativo",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "El catálogo institucional no se modifica. Esta es una vista "
            "operativa generada determinísticamente para la estimación "
            "seleccionada."
        )

with st.expander("Ver componentes que se analizarán"):
    st.dataframe(
        archivos_pdf[
            [
                "Archivo",
                "Tipo",
                "Tamaño (bytes)",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

modelo = st.selectbox(
    "Modelo multimodal",
    options=list(MODELOS.keys()),
    index=0,
    format_func=lambda m: MODELOS[m]["label"],
)

if st.button(
    "Analizar unidad completa",
    type="primary",
    disabled=archivos_pdf.empty,
):
    barra = st.progress(0.0)
    estado = st.empty()

    def actualizar_progreso(posicion, total, archivo):
        porcentaje = posicion / total if total else 1.0
        barra.progress(porcentaje)
        estado.write(
            f"Procesando {posicion}/{total}: {archivo}"
        )

    with st.spinner(
        "Analizando los componentes y consolidando la unidad..."
    ):
        resultados = analizar_unidad_completa(
            contenido_zip=contenido_zip,
            archivos_pdf=archivos_pdf,
            catalogo=catalogo,
            procedimiento=procedimiento,
            modelo_multimodal=modelo,
            tipo_unidad="Estimación",
            consecutivo=int(unidad["Consecutivo"]),
            on_progress=actualizar_progreso,
        )

        estado.write(
            "Comparando conjuntamente los candidatos a la unidad..."
        )

        comparado, comparacion, meta_comparacion = (
            consolidar_candidatos_unidad(
                contenido_zip=contenido_zip,
                resultados=resultados,
                procedimiento=procedimiento,
                consecutivo=int(unidad["Consecutivo"]),
                modelo_multimodal=modelo,
                tipo_unidad="Estimación",
            )
        )

        consolidado, grupos = agrupar_resultados_unidad(
            resultados=comparado,
            procedimiento=procedimiento,
            consecutivo=int(unidad["Consecutivo"]),
        )

        st.session_state["resultado_unidad_completa"] = {
            "schema_version": RESULTADO_UNIDAD_SCHEMA_VERSION,
            "ruta_unidad": str(unidad["Ruta carpeta"]),
            "procedimiento": procedimiento,
            "consecutivo": int(unidad["Consecutivo"]),
            "detalle": consolidado,
            "comparacion": comparacion,
            "meta_comparacion": meta_comparacion,
            "grupos": grupos,
        }

    barra.empty()
    estado.empty()

guardado = st.session_state.get("resultado_unidad_completa")

if (
    guardado
    and guardado.get("ruta_unidad") == str(unidad["Ruta carpeta"])
    and guardado.get("procedimiento") == procedimiento
):
    detalle = guardado["detalle"].copy()
    comparacion = guardado.get("comparacion", pd.DataFrame())
    meta_comparacion = guardado.get("meta_comparacion", {})
    grupos = guardado["grupos"]

    # Compatibilidad hacia atrás:
    # una actualización de la interfaz nunca debe obligar a repetir llamadas
    # de IA ya pagadas. Si un resultado anterior no trae columnas nuevas,
    # se completan localmente con valores neutros y se muestra lo disponible.
    columnas_compatibilidad = {
        "Clasificación inicial": "",
        "Código inicial": "",
        "Verificación identidad pura": "No aplicada",
        "Título identidad pura": "",
        "Función identidad pura": "",
        "Acto identidad pura": "",
        "Confianza identidad pura (%)": 0.0,
        "Evidencia identidad pura": "",
        "Equivalencia resuelta por": "No requerida",
        "Equivalencia JEV": "no_evaluada",
        "Decisión original JEV": "",
        "Confianza JEV (%)": 0.0,
        "Probabilidad elegida JEV (%)": 0.0,
        "Margen JEV (%)": None,
        "Control JEV": "",
        "Fallback multimodal JEV": "No",
        "Error JEV": "",
        "Llamadas multimodales documento": 0,
        "Llamadas JEV documento": 0,
        "Resolución conflicto código": "No aplica",
        "Código en conflicto": "",
        "Confianza conflicto (%)": 0.0,
        "Diferencia confianza conflicto (%)": 0.0,
        "Ganador conflicto": "",
        "Motivo conflicto": "",
        "Equivalencia funcional": "no_evaluada",
        "Confianza equivalencia (%)": 0.0,
        "Evidencia equivalencia": "",
        "Validación secundaria": "",
        "Alcance documental": "no_evaluado",
        "Relación comparativa": "",
        "Confianza comparativa (%)": 0.0,
        "Evidencia comparativa": "",
        "Relación con la unidad": "indeterminado",
        "Concepto relacionado": "",
        "Código relacionado": "",
        "Coincide catálogo": False,
        "Código de catálogo": "",
        "Confianza (%)": 0.0,
        "Relación consolidada": "",
        "Acción provisional": "",
        "Ruta utilizada": "",
        "Motivo de ruta": "",
        "Evidencia": "",
        "Modelo": "",
        "Páginas usadas": "",
        "Costo (USD)": 0.0,
        "Tiempo (s)": 0.0,
        "Intentos de análisis": 1,
        "Reintento aplicado": False,
        "Error": "",
    }

    columnas_agregadas = []
    for columna, valor in columnas_compatibilidad.items():
        if columna not in detalle.columns:
            detalle[columna] = valor
            columnas_agregadas.append(columna)

    if columnas_agregadas:
        st.info(
            "Este resultado fue generado con una versión anterior de la "
            "interfaz. Se muestra sin repetir el análisis de IA; las columnas "
            "nuevas que no existían se completaron localmente."
        )

    st.subheader("2. Resultado por componente")

    coinciden = int(detalle["Coincide catálogo"].astype(bool).sum())
    soportes = int(
        (detalle["Grupo lógico"] == "Soporte / fuera de catálogo").sum()
    )
    errores = int(
        detalle["Error"].astype(str).str.strip().ne("").sum()
    )
    costo_componentes = float(
        pd.to_numeric(
            detalle["Costo (USD)"],
            errors="coerce",
        ).fillna(0).sum()
    )
    costo_comparacion = float(
        meta_comparacion.get("Costo comparación (USD)", 0.0)
    )
    costo = costo_componentes + costo_comparacion

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PDF analizados", len(detalle))
    m2.metric("Códigos asignables", coinciden)
    m3.metric("Soporte / fuera catálogo", soportes)
    m4.metric("Errores", errores)

    st.dataframe(
        detalle[
            [
                "Archivo",
                "Título detectado",
                "Clasificación inicial",
                "Código inicial",
                "Verificación identidad pura",
                "Título identidad pura",
                "Equivalencia resuelta por",
                "Equivalencia JEV",
                "Fallback multimodal JEV",
                "Equivalencia funcional",
                "Confianza equivalencia (%)",
                "Llamadas multimodales documento",
                "Llamadas JEV documento",
                "Resolución conflicto código",
                "Código en conflicto",
                "Confianza conflicto (%)",
                "Diferencia confianza conflicto (%)",
                "Ganador conflicto",
                "Validación secundaria",
                "Alcance documental",
                "Relación comparativa",
                "Relación con la unidad",
                "Coincide catálogo",
                "Código de catálogo",
                "Confianza (%)",
                "Intentos de análisis",
                "Reintento aplicado",
                "Relación consolidada",
                "Acción provisional",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confianza (%)": st.column_config.NumberColumn(
                "Confianza (%)",
                format="%.1f %%",
            ),
            "Confianza identidad pura (%)": st.column_config.NumberColumn(
                "Confianza identidad pura (%)",
                format="%.1f %%",
            ),
            "Confianza JEV (%)": st.column_config.NumberColumn(
                "Confianza JEV (%)",
                format="%.1f %%",
            ),
            "Probabilidad elegida JEV (%)": st.column_config.NumberColumn(
                "Probabilidad elegida JEV (%)",
                format="%.1f %%",
            ),
            "Margen JEV (%)": st.column_config.NumberColumn(
                "Margen JEV (%)",
                format="%.1f %%",
            ),
            "Confianza equivalencia (%)": st.column_config.NumberColumn(
                "Confianza equivalencia (%)",
                format="%.1f %%",
            ),
            "Confianza conflicto (%)": st.column_config.NumberColumn(
                "Confianza conflicto (%)",
                format="%.1f %%",
            ),
            "Diferencia confianza conflicto (%)": st.column_config.NumberColumn(
                "Diferencia confianza conflicto (%)",
                format="%.1f pp",
            )
        },
    )

    parciales = int(
        (detalle["Alcance documental"] == "parcial_extracto").sum()
    )
    reintentos = int(
        detalle["Reintento aplicado"].astype(bool).sum()
    )
    llamadas_mm = int(
        pd.to_numeric(
            detalle["Llamadas multimodales documento"],
            errors="coerce",
        ).fillna(0).sum()
    )
    llamadas_jev = int(
        pd.to_numeric(
            detalle["Llamadas JEV documento"],
            errors="coerce",
        ).fillna(0).sum()
    )
    if meta_comparacion.get("Estado") == "Comparación ejecutada":
        llamadas_mm += 1

    st.caption(
        f"Costo IA acumulado de esta ejecución: USD {costo:.6f} · "
        f"Llamadas multimodales: {llamadas_mm} · "
        f"Llamadas JEV: {llamadas_jev} · "
        f"Parciales/extractos detectados: {parciales} · "
        f"Archivos con reintento: {reintentos}"
    )

    st.subheader("3. Comparación conjunta de candidatos")

    if comparacion.empty:
        st.info(
            "No fue necesario comparar candidatos para el código de la unidad."
        )
    else:
        representante = meta_comparacion.get("Representante") or "No definido"
        st.write(f"**Representante seleccionado:** {representante}")
        st.write(
            f"**Conclusión comparativa:** "
            f"{meta_comparacion.get('Evidencia unidad', '')}"
        )

        st.dataframe(
            comparacion,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Confianza comparativa (%)": st.column_config.NumberColumn(
                    "Confianza comparativa (%)",
                    format="%.1f %%",
                )
            },
        )

        st.caption(
            "Esta comparación solo puede decidir el papel relativo de los "
            "candidatos a la estimación. No puede alterar documentos que ya "
            "tengan otro código propio validado."
        )

    st.subheader("4. Agrupación lógica de la unidad")

    st.dataframe(
        grupos,
        use_container_width=True,
        hide_index=True,
    )

    codigo_unidad = (
        f"EJE_{procedimiento}_EST_{int(unidad['Consecutivo'])}"
    )
    grupo_unidad = grupos[
        grupos["Grupo lógico"] == codigo_unidad
    ]

    if not grupo_unidad.empty:
        cantidad = int(grupo_unidad.iloc[0]["Archivos"])
        representante = meta_comparacion.get("Representante") or ""
        if representante:
            st.success(
                f"La unidad {codigo_unidad} quedó consolidada con "
                f"{cantidad} archivo(s) asociados. Representante candidato: "
                f"{representante}."
            )
        elif cantidad > 0:
            st.warning(
                f"Hay {cantidad} archivo(s) asociados a {codigo_unidad}, "
                "pero la comparación no pudo definir un representante."
            )

    repetidos = grupos[
        (grupos["Archivos"] > 1)
        & (grupos["Grupo lógico"] != codigo_unidad)
        & (grupos["Grupo lógico"] != "Soporte / fuera de catálogo")
        & (grupos["Grupo lógico"] != "Error de análisis")
    ]

    if not repetidos.empty:
        st.warning(
            "Hay otros códigos propuestos por más de un archivo. "
            "En una etapa posterior habrá que determinar si se trata de "
            "duplicados, variantes, partes de un mismo documento o un "
            "documento compuesto."
        )

    with st.expander("Ver evidencia y ruta técnica"):
        st.dataframe(
            detalle[
                [
                    "Archivo",
                    "Ruta utilizada",
                    "Motivo de ruta",
                    "Clasificación inicial",
                    "Código inicial",
                    "Verificación identidad pura",
                    "Título identidad pura",
                    "Función identidad pura",
                    "Acto identidad pura",
                    "Confianza identidad pura (%)",
                    "Evidencia identidad pura",
                    "Equivalencia resuelta por",
                    "Equivalencia JEV",
                    "Decisión original JEV",
                    "Confianza JEV (%)",
                    "Probabilidad elegida JEV (%)",
                    "Margen JEV (%)",
                    "Control JEV",
                    "Fallback multimodal JEV",
                    "Error JEV",
                    "Llamadas multimodales documento",
                    "Llamadas JEV documento",
                    "Resolución conflicto código",
                    "Código en conflicto",
                    "Confianza conflicto (%)",
                    "Diferencia confianza conflicto (%)",
                    "Ganador conflicto",
                    "Motivo conflicto",
                    "Equivalencia funcional",
                    "Confianza equivalencia (%)",
                    "Evidencia equivalencia",
                    "Validación secundaria",
                    "Alcance documental",
                    "Relación comparativa",
                    "Confianza comparativa (%)",
                    "Evidencia comparativa",
                    "Relación con la unidad",
                    "Concepto relacionado",
                    "Código relacionado",
                    "Evidencia",
                    "Modelo",
                    "Páginas usadas",
                    "Costo (USD)",
                    "Tiempo (s)",
                    "Intentos de análisis",
                    "Reintento aplicado",
                    "Error",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

st.divider()

st.caption(
    "Esta versión ya puede proponer un representante de la unidad mediante "
    "comparación conjunta, pero todavía no renombra ni mueve archivos y no "
    "divide PDF que contengan varios documentos."
)
