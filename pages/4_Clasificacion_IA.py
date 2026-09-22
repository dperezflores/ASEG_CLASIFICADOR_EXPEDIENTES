import streamlit as st

from services.catalog_service import cargar_catalogo
from services.drive_persistence_service import (\n    DrivePersistenceError,\n    guardar_clasificacion_jev,\n    guardar_ground_truth,\n)\nfrom services.evaluation_service import preparar_muestra_ocr\nfrom services.jev_classifier_service import (\n    JevError,\n    clasificar_muestra_con_jev,\n    jev_configurado,\n    probar_conexion_jev,\n)\nfrom ui.common import mostrar_encabezado, requerir_expediente


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

if st.session_state.get("ground_truth_manual"):
    if st.button("Guardar referencias en Google Drive"):
        try:
            guardar_ground_truth(
                st.session_state["drive_folder_id"],
                st.session_state["ground_truth_manual"],
            )
            st.success("Referencias guardadas en Google Drive.")
        except (DrivePersistenceError, KeyError) as error:
            st.error(
                "No fue posible guardar las referencias en Drive: "
                f"{error}"
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

st.subheader("3. Ejecutar Ruta A · Jev")

if not jev_configurado():
    st.error(
        "No se encontró la API key de TypeSafe en Streamlit Secrets."
    )
else:
    col_test, col_run = st.columns([1, 2])

    with col_test:
        if st.button("Probar conexión con Jev"):
            try:
                modelos = probar_conexion_jev()
                st.success("Conexión con TypeSafe correcta.")
                if modelos:
                    st.caption("Modelos disponibles: " + ", ".join(modelos))
            except JevError as error:
                st.error(str(error))

    with col_run:
        if st.button(
            "Clasificar muestra con Jev",
            type="primary",
            disabled=not referencias_completas or muestra_sel.empty,
        ):
            with st.spinner(
                "Jev está clasificando los documentos usando únicamente "
                "el texto OCR y el catálogo activo..."
            ):
                try:
                    resultados_jev = clasificar_muestra_con_jev(
                        muestra_sel,
                        catalogo,
                    )
                    st.session_state["resultados_jev"] = resultados_jev

                    if "drive_folder_id" in st.session_state:
                        guardar_clasificacion_jev(
                            st.session_state["drive_folder_id"],
                            resultados_jev,
                        )

                    st.success(
                        "Ruta A terminada y guardada en Google Drive."
                    )
                    st.rerun()
                except (JevError, DrivePersistenceError) as error:
                    st.error(str(error))

if "resultados_jev" in st.session_state:
    resultados_jev = st.session_state["resultados_jev"]

    st.dataframe(
        resultados_jev[
            [
                "Documento",
                "Concepto real",
                "Resultado Ruta A",
                "Confianza A",
                "Probabilidad elegida (%)",
                "Acierto A",
                "Tokens entrada A",
                "Costo A (USD)",
                "Tiempo A (s)",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confianza A": st.column_config.NumberColumn(
                "Confianza A",
                format="%.2f %%",
            ),
            "Probabilidad elegida (%)": st.column_config.NumberColumn(
                "Probabilidad elegida (%)",
                format="%.2f %%",
            ),
            "Costo A (USD)": st.column_config.NumberColumn(
                "Costo A (USD)",
                format="$%.8f",
            ),
            "Tiempo A (s)": st.column_config.NumberColumn(
                "Tiempo A (s)",
                format="%.3f",
            ),
        },
    )

    aciertos = int(resultados_jev["Acierto A"].sum())
    total = len(resultados_jev)
    costo_total = float(resultados_jev["Costo A (USD)"].sum())
    tiempo_total = float(resultados_jev["Tiempo A (s)"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Aciertos Ruta A", f"{aciertos}/{total}")
    c2.metric("Costo estimado", f"$ {costo_total:.8f}")
    c3.metric("Tiempo total", f"{tiempo_total:.3f} s")

    with st.expander("Ver probabilidades principales de Jev"):
        for _, fila in resultados_jev.iterrows():
            st.markdown(f"**{fila['Documento']}**")
            for item in fila["Top 3"]:
                st.write(
                    f"- {item['concepto']}: "
                    f"{item['probabilidad'] * 100:.2f}%"
                )

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
