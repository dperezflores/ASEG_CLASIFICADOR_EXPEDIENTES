import pandas as pd
import streamlit as st

from services.catalog_service import cargar_catalogo
from services.drive_persistence_service import (
    DrivePersistenceError,
    guardar_clasificacion_hibrida,
)
from services.hybrid_classification_service import (
    clasificar_muestra_hibrida,
    preparar_muestra_hibrida,
)
from services.openai_multimodal_service import (
    MODELOS,
    OpenAIMultimodalError,
    openai_configurado,
)
from services.jev_classifier_service import JevError, jev_configurado
from ui.common import mostrar_encabezado, requerir_expediente


mostrar_encabezado(
    "Clasificación híbrida",
    "Prueba ampliada orientada a precisión, tiempo y costo",
)
requerir_expediente()

if "analisis_pdf" not in st.session_state:
    st.warning(
        "Primero realiza el **Diagnóstico PDF**. "
        "Esta prueba no requiere ejecutar OCR."
    )
    st.stop()

if not openai_configurado():
    st.error(
        "La API key de OpenAI no está configurada en Streamlit Secrets."
    )
    st.stop()

if not jev_configurado():
    st.error(
        "La API key de TypeSafe/Jev no está configurada en Streamlit Secrets."
    )
    st.stop()

procedimiento = st.session_state["procedimiento"]
catalogo = cargar_catalogo(procedimiento)
analisis_pdf = st.session_state["analisis_pdf"]
contenido_zip = st.session_state["contenido_zip"]

muestra = preparar_muestra_hibrida(
    analisis_pdf,
    catalogo,
)

st.info(
    "Esta prueba evita OCR local. Los PDF con texto nativo se envían primero "
    "a Jev; los PDF escaneados o mixtos se envían directamente al modelo "
    "multimodal. Jev también escala al multimodal cuando su confianza es baja "
    "o detecta un posible documento fuera de catálogo."
)

st.subheader("1. Seleccionar muestra")

opciones = muestra["Documento"].tolist()

seleccion = st.multiselect(
    "Selecciona entre 1 y 30 documentos",
    options=opciones,
    max_selections=30,
    help=(
        "Los modelos solo reciben el alias documento_XXX. "
        "No reciben el nombre real ni la carpeta."
    ),
)

muestra_sel = muestra[
    muestra["Documento"].isin(seleccion)
].copy()

if not muestra_sel.empty:
    resumen_estados = (
        muestra_sel["Estado PDF"]
        .value_counts()
        .rename_axis("Estado PDF")
        .reset_index(name="Documentos")
    )

    st.dataframe(
        resumen_estados,
        use_container_width=True,
        hide_index=True,
    )

    conocidos = int(
        (
            muestra_sel["Concepto de referencia"]
            != "Sin referencia conocida"
        ).sum()
    )
    desconocidos = len(muestra_sel) - conocidos

    c1, c2, c3 = st.columns(3)
    c1.metric("Documentos seleccionados", len(muestra_sel))
    c2.metric("Con referencia conocida", conocidos)
    c3.metric("Sin referencia conocida", desconocidos)

with st.expander("Ver correspondencia interna de la muestra"):
    st.dataframe(
        muestra_sel[
            [
                "Documento",
                "Archivo real",
                "Ruta original",
                "Estado PDF",
                "Concepto de referencia",
            ]
        ] if not muestra_sel.empty else pd.DataFrame(),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("2. Configuración experimental")

col_modelo, col_umbral = st.columns(2)

with col_modelo:
    modelo = st.selectbox(
        "Modelo multimodal",
        options=list(MODELOS.keys()),
        index=0,
        format_func=lambda m: MODELOS[m]["label"],
    )

with col_umbral:
    umbral_jev = st.slider(
        "Umbral experimental de confianza Jev",
        min_value=50,
        max_value=95,
        value=70,
        step=5,
        help=(
            "Si Jev queda por debajo del umbral, el documento se escala "
            "automáticamente al multimodal. Este valor todavía no es definitivo."
        ),
    )

st.caption(
    "Se procesan inicialmente hasta 2 páginas por documento. "
    "El objetivo es medir si eso basta para identificar y clasificar "
    "correctamente sin elevar costos innecesariamente."
)

if st.button(
    "Ejecutar clasificación híbrida",
    type="primary",
    disabled=muestra_sel.empty,
):
    with st.spinner(
        "Clasificando la muestra. Los PDF escaneados se envían directamente "
        "al modelo multimodal; no se ejecuta RapidOCR..."
    ):
        try:
            resultados = clasificar_muestra_hibrida(
                muestra=muestra_sel,
                catalogo=catalogo,
                contenido_zip=contenido_zip,
                modelo_multimodal=modelo,
                umbral_jev=float(umbral_jev),
            )

            st.session_state["resultados_hibridos"] = resultados

            if "drive_folder_id" in st.session_state:
                guardar_clasificacion_hibrida(
                    st.session_state["drive_folder_id"],
                    resultados,
                )

            st.success(
                "Clasificación híbrida terminada y guardada en Google Drive."
            )
            st.rerun()

        except (
            JevError,
            OpenAIMultimodalError,
            DrivePersistenceError,
            ValueError,
        ) as error:
            st.error(str(error))

if "resultados_hibridos" in st.session_state:
    resultados = st.session_state["resultados_hibridos"]

    st.subheader("3. Resultados")

    st.dataframe(
        resultados[
            [
                "Documento",
                "Ruta utilizada",
                "Título detectado",
                "Coincide catálogo",
                "Concepto propuesto",
                "Código propuesto",
                "Confianza (%)",
                "Concepto de referencia",
                "Acierto conocido",
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

    total = len(resultados)
    multimodal = int(
        (resultados["Ruta utilizada"] == "Multimodal").sum()
    )
    jev = int(
        (resultados["Ruta utilizada"] == "Texto nativo + Jev").sum()
    )
    fuera = int((~resultados["Coincide catálogo"].astype(bool)).sum())
    costo = float(resultados["Costo (USD)"].sum())
    tiempo = float(resultados["Tiempo (s)"].sum())

    conocidos_df = resultados[
        resultados["Acierto conocido"].notna()
    ].copy()
    if not conocidos_df.empty:
        aciertos = int(
            conocidos_df["Acierto conocido"].astype(bool).sum()
        )
        precision_texto = f"{aciertos}/{len(conocidos_df)}"
    else:
        precision_texto = "Sin referencia"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precisión conocida", precision_texto)
    m2.metric("Jev sin escalamiento", jev)
    m3.metric("Multimodal / escalados", multimodal)
    m4.metric("Fuera de catálogo", fuera)

    m5, m6 = st.columns(2)
    m5.metric("Costo total", f"USD {costo:.6f}")
    m6.metric("Tiempo IA acumulado", f"{tiempo:.2f} s")

    with st.expander("Ver motivo de ruta y evidencia"):
        st.dataframe(
            resultados[
                [
                    "Documento",
                    "Motivo de ruta",
                    "Título detectado",
                    "Evidencia",
                    "Modelo",
                    "Páginas usadas",
                    "Costo (USD)",
                    "Tiempo (s)",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "En documentos sin referencia conocida, 'Acierto conocido' queda vacío. "
        "Su evaluación se hará revisando si el título detectado es correcto y "
        "si la decisión de fuera de catálogo es adecuada."
    )
