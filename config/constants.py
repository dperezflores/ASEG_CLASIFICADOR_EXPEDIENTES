APP_NAME = "Clasificador Inteligente de Expedientes"
APP_SUBTITLE = "Identificación, codificación y organización documental de expedientes de obra pública"

CATALOGO_PATH = "catalogo/CODIFICACION_DOCUMENTOS.xlsx"
HOJAS_PROCEDIMIENTO = ("DIR", "LPU", "LSI")

# Carpetas que no forman parte del análisis del expediente.
CARPETAS_IGNORADAS = ("7_PT",)

# Archivos técnicos del sistema operativo que se conservan en el original,
# pero no deben entrar al inventario analítico ni a ningún flujo de IA.
ARCHIVOS_IGNORADOS = (
    "thumbs.db",
    ".ds_store",
    "desktop.ini",
)
