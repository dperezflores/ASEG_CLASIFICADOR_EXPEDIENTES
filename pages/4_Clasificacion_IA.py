import streamlit as st

from services.catalog_service import cargar_catalogo
from services.evaluation_service import preparar_muestra_ocr
from ui.common import mostrar_encabezado, requerir_expediente


mostrar_encabezado(
    "Clasificación IA",
    "Banco de pruebas: OCR + clasificador frente a PDF multimodal",
)
requerir_expediente()

if "resultado_ocr" not in st.session_state:
    st.warning(
        "Primero realiza una prueba en **OCR controlado**. "
        "Esta comparativa reutiliza esos resultados y no ejecuta OCR nuevamente."
    )
    st.stop()

procedimiento = st.session_state["procedimiento"]
catalogo = cargar_catalogo(procedimiento)
resultado_ocr = st.session_state["resultado_ocr"]

muestra = preparar_muestra_ocr(resultado_ocr, catalogo)

if muestra.empty:
    st.warning("No hay documentos OCR disponibles para construir la muestra.")
    st.stop()

st.info(
    "Los modelos no recibirán el nombre real del archivo ni su carpeta. "
    "Cada documento se identificará únicamente como documento_001, "
    "documento_002, etc."
)

st.subheader("1. Muestra de evaluación")

opciones = muestra["Documento"].tolist()

seleccion = st.multiselect(
    "Documentos a comparar",
    options=opciones,
    default=opciones,
    max_selections=5,
)

muestra_sel = muestra[muestra["Documento"].isin(seleccion)].copy()

tabla_visible = muestra_sel[
    [
        "Documento",
        "Páginas OCR",
        "Caracteres OCR",
        "Confianza OCR media (%)",
        "Concepto real",
    ]
]

st.dataframe(
    tabla_visible,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Confianza OCR media (%)": st.column_config.NumberColumn(
            "Confianza OCR media (%)",
            format="%.1f %%",
        )
    },
)

st.caption(
    "La columna 'Concepto real' se usa solo para evaluar después. "
    "No se incluirá en el contenido enviado a ningún modelo."
)

with st.expander("Ver correspondencia interna de la muestra"):
    st.dataframe(
        muestra_sel[
            [
                "Documento",
                "Archivo real",
                "Ruta original",
                "Código real",
                "Concepto real",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.subheader("2. Preparación de las dos rutas")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### Ruta A · OCR + clasificador")
    st.write(
        "Entrada preparada: texto OCR ya generado, anonimizado por documento."
    )

    if not muestra_sel.empty:
        ejemplo = muestra_sel.iloc[0]
        st.caption(
            f"Ejemplo de entrada: {ejemplo['Documento']} · "
            f"{ejemplo['Caracteres OCR']} caracteres"
        )

    st.success("Muestra OCR lista para conectar con Jev.")

with col_b:
    st.markdown("### Ruta B · PDF + multimodal")
    st.write(
        "Entrada preparada: se recuperará el PDF original desde el ZIP, "
        "pero el proveedor recibirá un alias neutro y no su nombre real."
    )

    st.success("Muestra PDF lista para conectar con un modelo multimodal.")

st.subheader("3. Resultado comparativo")

st.dataframe(
    muestra_sel.assign(
        **{
            "Resultado Ruta A": "Pendiente",
            "Confianza A": None,
            "Resultado Ruta B": "Pendiente",
            "Confianza B": None,
        }
    )[
        [
            "Documento",
            "Concepto real",
            "Resultado Ruta A",
            "Confianza A",
            "Resultado Ruta B",
            "Confianza B",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "En este paso todavía no se envía información a Jev, OpenAI ni Gemini. "
    "Solo dejamos lista y validable la muestra que ambos métodos usarán."
)
