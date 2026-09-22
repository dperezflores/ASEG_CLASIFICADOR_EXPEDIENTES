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

# Ground truth manual para archivos cuyo nombre no coincide con el catálogo.
if "ground_truth_manual" not in st.session_state:
    st.session_state["ground_truth_manual"] = {}

pendientes = muestra_sel[
    muestra_sel["Concepto real"] == "Pendiente de referencia"
].copy()

if not pendientes.empty:
    st.warning(
        "Hay documentos cuya clasificación real no pudo determinarse "
        "automáticamente por el nombre del archivo. Completa su referencia "
        "antes de ejecutar la comparación."
    )

    with st.expander("Completar clasificación real de documentos pendientes", expanded=True):
        opciones_catalogo = {
            f"{fila['Concepto']}  ·  {fila['Código']}": {
                "concepto": str(fila["Concepto"]).strip(),
                "codigo": str(fila["Código"]).strip(),
            }
            for _, fila in catalogo.iterrows()
        }

        etiquetas = ["— Seleccionar —"] + list(opciones_catalogo.keys())

        for _, fila in pendientes.iterrows():
            ruta = fila["Ruta original"]
            documento = fila["Documento"]
            archivo_real = fila["Archivo real"]

            valor_guardado = st.session_state["ground_truth_manual"].get(ruta)

            indice_actual = 0
            if valor_guardado:
                for i, etiqueta in enumerate(etiquetas[1:], start=1):
                    datos = opciones_catalogo[etiqueta]
                    if (
                        datos["concepto"] == valor_guardado["concepto"]
                        and datos["codigo"] == valor_guardado["codigo"]
                    ):
                        indice_actual = i
                        break

            seleccion_real = st.selectbox(
                f"{documento} · {archivo_real}",
                options=etiquetas,
                index=indice_actual,
                key=f"gt_{ruta}",
            )

            if seleccion_real != "— Seleccionar —":
                datos = opciones_catalogo[seleccion_real]
                st.session_state["ground_truth_manual"][ruta] = datos
            else:
                st.session_state["ground_truth_manual"].pop(ruta, None)

# Aplicar las referencias manuales únicamente para evaluación interna.
for indice, fila in muestra_sel.iterrows():
    ruta = fila["Ruta original"]
    manual = st.session_state["ground_truth_manual"].get(ruta)

    if (
        fila["Concepto real"] == "Pendiente de referencia"
        and manual is not None
    ):
        muestra_sel.at[indice, "Concepto real"] = manual["concepto"]
        muestra_sel.at[indice, "Código real"] = manual["codigo"]

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

referencias_completas = not (
    muestra_sel["Concepto real"] == "Pendiente de referencia"
).any()

if referencias_completas:
    st.success(
        "La muestra ya tiene clasificación real completa y está lista "
        "para medir precisión."
    )
else:
    st.info(
        "Completa las referencias pendientes antes de ejecutar los modelos."
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
