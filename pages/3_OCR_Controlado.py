import streamlit as st

from services.drive_persistence_service import (
    DrivePersistenceError,
    guardar_ocr,
)
from services.ocr_service import ejecutar_ocr_controlado
from ui.common import (
    mostrar_encabezado,
    requerir_expediente,
)


mostrar_encabezado(
    "OCR controlado",
    "Prueba de recuperación de texto sobre una muestra de documentos",
)
requerir_expediente()

if "analisis_pdf" not in st.session_state:
    st.warning(
        "Antes de ejecutar OCR, realiza el **Diagnóstico PDF**."
    )
    st.stop()

contenido_zip = st.session_state["contenido_zip"]
analisis_pdf = st.session_state["analisis_pdf"]

candidatos_ocr = analisis_pdf[
    analisis_pdf["Estado PDF"].isin(["Requiere OCR", "Mixto"])
].copy()

if candidatos_ocr.empty:
    st.info("No hay PDF pendientes de OCR.")
    st.stop()

opciones_ocr = candidatos_ocr["Ruta original"].tolist()

if "resultado_ocr" in st.session_state:
    resultado_ocr = st.session_state["resultado_ocr"]

    st.success(
        f"Resultado OCR disponible: {len(resultado_ocr)} páginas procesadas."
    )
    st.caption("Persistencia en Google Drive: guardado ✓")

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
            "Confianza media (%)": st.column_config.NumberColumn(
                "Confianza media (%)",
                format="%.1f %%",
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
            texto = str(fila["Texto OCR"]).strip()
            if texto:
                st.text_area(
                    "Texto OCR",
                    value=texto,
                    height=220,
                    key=f"ocr_texto_{indice}",
                    disabled=True,
                )
            else:
                st.warning("No se recuperó texto en esta página.")

    if st.button("Realizar nueva prueba OCR"):
        st.session_state.pop("resultado_ocr", None)
        st.rerun()

else:
    st.write(
        "Selecciona hasta 5 PDF. Esta prueba no modifica los documentos "
        "originales."
    )

    seleccion_ocr = st.multiselect(
        "PDF para la muestra",
        options=opciones_ocr,
        max_selections=5,
    )

    max_paginas = st.slider(
        "Páginas a procesar por PDF",
        min_value=1,
        max_value=5,
        value=3,
    )

    st.caption(
        "Motor de prueba: RapidOCR · renderizado a 180 DPI · ejecución local."
    )

    if st.button(
        "Ejecutar OCR controlado",
        type="primary",
        disabled=not seleccion_ocr,
    ):
        with st.spinner("Ejecutando OCR sobre la muestra seleccionada..."):
            try:
                resultado_ocr = ejecutar_ocr_controlado(
                    contenido_zip,
                    seleccion_ocr,
                    max_paginas,
                )

                if "drive_folder_id" in st.session_state:
                    guardar_ocr(
                        st.session_state["drive_folder_id"],
                        resultado_ocr,
                    )

                st.session_state["resultado_ocr"] = resultado_ocr
                st.success("OCR terminado y guardado en Google Drive.")
                st.rerun()

            except (ValueError, DrivePersistenceError) as error:
                st.error(str(error))
