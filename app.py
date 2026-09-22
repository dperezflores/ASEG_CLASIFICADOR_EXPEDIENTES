from pathlib import Path

import streamlit as st

from config.constants import APP_NAME, APP_SUBTITLE, CARPETAS_IGNORADAS
from services.catalog_service import cargar_catalogo, obtener_hojas_catalogo
from services.expediente_service import formatear_tamano, inventariar_expediente_zip
from services.ocr_service import ejecutar_ocr_controlado
from services.pdf_service import analizar_pdfs_zip


st.set_page_config(
    page_title=APP_NAME,
    page_icon="📁",
    layout="wide",
)

css_path = Path("assets/styles.css")
if css_path.exists():
    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="app-header">
        <div class="app-kicker">ASEG</div>
        <h1>{APP_NAME}</h1>
        <p>{APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Etapa actual: inventariar el expediente, diagnosticar PDF y probar "
    "OCR de forma controlada sobre una muestra pequeña."
)

st.subheader("1. Configuración del expediente")

procedimiento = st.selectbox(
    "Tipo de procedimiento",
    options=["DIR", "LPU", "LSI"],
    format_func=lambda x: {
        "DIR": "Adjudicación directa",
        "LPU": "Licitación pública",
        "LSI": "Licitación simplificada",
    }[x],
)

try:
    hojas = obtener_hojas_catalogo()

    if procedimiento not in hojas:
        st.error(f"La hoja {procedimiento} no fue encontrada en el catálogo.")
        st.stop()

    catalogo = cargar_catalogo(procedimiento)

except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

st.caption(
    f"Catálogo activo: {procedimiento} · "
    f"{len(catalogo)} conceptos de codificación."
)

st.subheader("2. Cargar expediente")

archivo_zip = st.file_uploader(
    "Selecciona el expediente comprimido",
    type=["zip"],
    help=(
        "El ZIP puede contener carpetas, subcarpetas y distintos tipos de "
        "archivo. Los documentos originales no se modifican."
    ),
)

st.caption(
    "Carpetas excluidas del análisis: "
    + ", ".join(CARPETAS_IGNORADAS)
)

if archivo_zip is not None:
    contenido_zip = archivo_zip.getvalue()
    expediente_id = f"{archivo_zip.name}:{archivo_zip.size}"

    if st.session_state.get("expediente_id") != expediente_id:
        st.session_state["expediente_id"] = expediente_id
        st.session_state.pop("analisis_pdf", None)
        st.session_state.pop("resumen_pdf", None)
        st.session_state.pop("resultado_ocr", None)

    try:
        inventario, resumen = inventariar_expediente_zip(contenido_zip)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    st.success("Expediente leído correctamente.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Archivos encontrados", resumen["archivos"])
    col2.metric("Carpetas encontradas", resumen["carpetas"])
    col3.metric(
        "Tamaño sin comprimir",
        formatear_tamano(resumen["tamano_bytes"]),
    )

    if resumen["tipos"]:
        st.subheader("Tipos de archivo")

        columnas = st.columns(len(resumen["tipos"]))
        for columna, (tipo, cantidad) in zip(
            columnas,
            resumen["tipos"].items(),
        ):
            columna.metric(tipo, cantidad)

    st.subheader("Inventario del expediente")

    if inventario.empty:
        st.warning(
            "No se encontraron archivos utilizables dentro del expediente."
        )
    else:
        inventario_visual = inventario.copy()
        inventario_visual["Tamaño"] = inventario_visual[
            "Tamaño (bytes)"
        ].apply(formatear_tamano)

        st.dataframe(
            inventario_visual[
                [
                    "Ruta original",
                    "Archivo",
                    "Extensión",
                    "Tipo",
                    "Tamaño",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("3. Diagnóstico de PDF")

    st.write(
        "Este diagnóstico todavía no hace OCR. Solo revisa si cada página "
        "contiene texto que Python puede extraer directamente."
    )

    if st.button(
        "Analizar PDF del expediente",
        type="primary",
        use_container_width=False,
    ):
        with st.spinner(
            "Revisando los PDF. El tiempo depende del número de páginas..."
        ):
            try:
                analisis_pdf, resumen_pdf = analizar_pdfs_zip(contenido_zip)
                st.session_state["analisis_pdf"] = analisis_pdf
                st.session_state["resumen_pdf"] = resumen_pdf
            except ValueError as error:
                st.error(str(error))

    if "analisis_pdf" in st.session_state:
        analisis_pdf = st.session_state["analisis_pdf"]
        resumen_pdf = st.session_state["resumen_pdf"]

        st.success(
            f"Diagnóstico terminado: {resumen_pdf['pdfs']} PDF y "
            f"{resumen_pdf['paginas']} páginas revisadas."
        )

        if resumen_pdf["estados"]:
            columnas_pdf = st.columns(len(resumen_pdf["estados"]))
            for columna, (estado, cantidad) in zip(
                columnas_pdf,
                resumen_pdf["estados"].items(),
            ):
                columna.metric(estado, cantidad)

        st.dataframe(
            analisis_pdf,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Texto (%)": st.column_config.NumberColumn(
                    "Texto (%)",
                    format="%.1f %%",
                )
            },
        )

        st.caption(
            "Texto extraíble: al menos 80 % de las páginas contienen texto "
            "suficiente. Mixto: solo parte de las páginas contiene texto. "
            "Requiere OCR: no se detectó texto extraíble."
        )

        st.subheader("4. OCR controlado")

        candidatos_ocr = analisis_pdf[
            analisis_pdf["Estado PDF"].isin(
                ["Requiere OCR", "Mixto"]
            )
        ].copy()

        if candidatos_ocr.empty:
            st.info("No hay PDF pendientes de OCR.")
        else:
            opciones_ocr = candidatos_ocr["Ruta original"].tolist()

            seleccion_ocr = st.multiselect(
                "Selecciona hasta 5 PDF para la prueba",
                options=opciones_ocr,
                max_selections=5,
                help=(
                    "Para esta etapa procesamos una muestra pequeña. "
                    "Elige documentos distintos para comparar la calidad."
                ),
            )

            max_paginas = st.slider(
                "Páginas a procesar por PDF",
                min_value=1,
                max_value=5,
                value=2,
                help=(
                    "Solo se procesan las primeras páginas de cada PDF "
                    "seleccionado."
                ),
            )

            st.caption(
                "Motor de prueba: RapidOCR. Las páginas se renderizan a "
                "180 DPI y el OCR se ejecuta localmente en Streamlit."
            )

            if st.button(
                "Ejecutar OCR controlado",
                disabled=not seleccion_ocr,
            ):
                with st.spinner(
                    "Ejecutando OCR sobre la muestra seleccionada..."
                ):
                    try:
                        resultado_ocr = ejecutar_ocr_controlado(
                            contenido_zip,
                            seleccion_ocr,
                            max_paginas,
                        )
                        st.session_state["resultado_ocr"] = resultado_ocr
                    except ValueError as error:
                        st.error(str(error))

            if "resultado_ocr" in st.session_state:
                resultado_ocr = st.session_state["resultado_ocr"]

                if resultado_ocr.empty:
                    st.warning("El OCR no generó resultados.")
                else:
                    st.success(
                        f"OCR terminado: {len(resultado_ocr)} páginas "
                        "procesadas."
                    )

                    st.dataframe(
                        resultado_ocr[
                            [
                                "Archivo",
                                "Página",
                                "Caracteres OCR",
                                "Confianza media (%)",
                                "Estado OCR",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Confianza media (%)": (
                                st.column_config.NumberColumn(
                                    "Confianza media (%)",
                                    format="%.1f %%",
                                )
                            )
                        },
                    )

                    st.subheader("Texto recuperado")

                    for indice, fila in resultado_ocr.iterrows():
                        titulo = (
                            f"{fila['Archivo']} · página {fila['Página']} · "
                            f"{fila['Confianza media (%)']:.1f}%"
                        )

                        with st.expander(titulo):
                            texto = fila["Texto OCR"].strip()
                            if texto:
                                st.text_area(
                                    "Texto OCR",
                                    value=texto,
                                    height=220,
                                    key=f"ocr_texto_{indice}",
                                    disabled=True,
                                )
                            else:
                                st.warning(
                                    "No se recuperó texto en esta página."
                                )

                    st.caption(
                        "La confianza es una señal técnica del motor OCR; "
                        "todavía no equivale a exactitud documental ni a una "
                        "clasificación correcta."
                    )

with st.expander("Ver catálogo de codificación activo"):
    st.dataframe(
        catalogo,
        use_container_width=True,
        hide_index=True,
    )
