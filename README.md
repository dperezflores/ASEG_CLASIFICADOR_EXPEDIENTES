# ASEG Clasificador de Expedientes

Aplicación independiente para la identificación, codificación y organización automática de documentos que integran expedientes de obra pública.

## Objetivo

Recibir un expediente completo, identificar sus documentos, asignar la codificación institucional correspondiente y generar una copia organizada por carpetas, reservando para revisión humana los casos inciertos.

## Estado actual

**Etapa inicial de construcción.**

La primera versión valida:

- estructura base del proyecto;
- interfaz en Streamlit;
- selección del tipo de procedimiento;
- lectura posterior del catálogo oficial de codificación.

## Estructura inicial

```text
ASEG_CLASIFICADOR_EXPEDIENTES/
├── app.py
├── requirements.txt
├── assets/
│   └── styles.css
├── catalogo/
├── config/
│   └── constants.py
└── services/
    └── catalog_service.py
```

El proyecto se desarrollará por etapas, sin modificar nunca los expedientes originales.
