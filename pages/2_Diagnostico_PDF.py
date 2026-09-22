import streamlit as st

from services.pdf_service import analizar_pdfs_zip
from ui.common import (
    mostrar_encabezado,
    requerir_expediente,
)


mostrar_encabezado(
    "Diagnóstico PDF",
    "Identificación de texto nativo y necesidad preliminar de OCR",
)
requerir_expediente()

contenido_zip = st.session_state["contenido_zip"]

if "analisis_pdf" not in st.session_state:
    st.info(
        "Este análisis revisa todos los PDF del expediente sin aplicar OCR."
    )

    if st.button("Analizar PDF del expediente", type="primary"):
        with st.spinner(
            "Revisando los PDF. El tiempo depende del número de páginas..."
        ):
            try:
                analisis_pdf, resumen_pdf = analizar_pdfs_zip(contenido_zip)
                st.session_state["analisis_pdf"] = analisis_pdf
                st.session_state["resumen_pdf"] = resumen_pdf
                st.rerun()
            except ValueError as error:
                st.error(str(error))
else:
    analisis_pdf = st.session_state["analisis_pdf"]
    resumen_pdf = st.session_state["resumen_pdf"]

    st.success(
        f"Diagnóstico disponible: {resumen_pdf['pdfs']} PDF y "
        f"{resumen_pdf['paginas']} páginas revisadas."
    )

    if resumen_pdf["estados"]:
        columnas = st.columns(len(resumen_pdf["estados"]))
        for columna, (estado, cantidad) in zip(
            columnas,
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

    if st.button("Volver a ejecutar diagnóstico"):
        st.session_state.pop("analisis_pdf", None)
        st.session_state.pop("resumen_pdf", None)
        st.session_state.pop("resultado_ocr", None)
        st.rerun()
