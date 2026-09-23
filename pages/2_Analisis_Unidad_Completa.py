import pandas as pd
import streamlit as st

from services.catalog_service import cargar_catalogo
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
)
from ui.common import mostrar_encabezado, requerir_expediente


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
    "Esta etapa analiza todos los PDF directos de una sola estimación y "
    "separa identidad, alcance documental y función dentro de la unidad. "
    "Un parcial o extracto no recibe automáticamente el código del documento "
    "completo. Todavía no divide documentos compuestos."
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
        "Para esta primera validación se recomienda trabajar únicamente "
        "con EST 1."
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

st.subheader("1. Alcance de la prueba")

c1, c2, c3 = st.columns(3)
c1.metric("Unidad", str(unidad["Carpeta"]))
c2.metric("PDF directos", len(archivos_pdf))
c3.metric("Procedimiento", procedimiento)

st.caption(
    "Cada PDF se analiza con alias neutro; la IA no recibe el nombre real "
    "ni la ruta. Después Python consolida identidad, alcance, relación con "
    "la unidad y equivalencia real con el catálogo."
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

        consolidado, grupos = agrupar_resultados_unidad(
            resultados=resultados,
            procedimiento=procedimiento,
            consecutivo=int(unidad["Consecutivo"]),
        )

        st.session_state["resultado_unidad_completa"] = {
            "ruta_unidad": str(unidad["Ruta carpeta"]),
            "procedimiento": procedimiento,
            "consecutivo": int(unidad["Consecutivo"]),
            "detalle": consolidado,
            "grupos": grupos,
        }

    barra.empty()
    estado.empty()
    st.rerun()

guardado = st.session_state.get("resultado_unidad_completa")

if (
    guardado
    and guardado.get("ruta_unidad") == str(unidad["Ruta carpeta"])
    and guardado.get("procedimiento") == procedimiento
):
    detalle = guardado["detalle"]
    grupos = guardado["grupos"]

    st.subheader("2. Resultado por componente")

    coinciden = int(detalle["Coincide catálogo"].astype(bool).sum())
    soportes = int(
        (detalle["Grupo lógico"] == "Soporte / fuera de catálogo").sum()
    )
    errores = int(
        detalle["Error"].astype(str).str.strip().ne("").sum()
    )
    costo = float(
        pd.to_numeric(
            detalle["Costo (USD)"],
            errors="coerce",
        ).fillna(0).sum()
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PDF analizados", len(detalle))
    m2.metric("Coincidencias catálogo", coinciden)
    m3.metric("Soporte / fuera catálogo", soportes)
    m4.metric("Errores", errores)

    st.dataframe(
        detalle[
            [
                "Archivo",
                "Título detectado",
                "Alcance documental",
                "Relación con la unidad",
                "Concepto relacionado",
                "Coincide catálogo",
                "Código de catálogo",
                "Confianza (%)",
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
            )
        },
    )

    parciales = int(
        (detalle["Alcance documental"] == "parcial_extracto").sum()
    )
    st.caption(
        f"Costo IA acumulado de esta ejecución: USD {costo:.6f} · "
        f"Parciales/extractos detectados: {parciales}"
    )

    st.subheader("3. Agrupación lógica de la unidad")

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
        if cantidad > 1:
            st.warning(
                f"Se encontraron {cantidad} archivos asociados a "
                f"{codigo_unidad}. Esto no se trata todavía como duplicado: "
                "pueden ser componentes distintos de la misma unidad lógica. "
                "El archivo representativo sigue pendiente de decidir."
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
                    "Alcance documental",
                    "Relación con la unidad",
                    "Concepto relacionado",
                    "Código relacionado",
                    "Evidencia",
                    "Modelo",
                    "Páginas usadas",
                    "Costo (USD)",
                    "Tiempo (s)",
                    "Error",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

st.divider()

st.caption(
    "Esta versión solo consolida relaciones. No renombra archivos, no mueve "
    "documentos, no selecciona todavía el archivo representativo y no divide "
    "PDF que contengan varios documentos."
)
