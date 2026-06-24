# 🤖 Agente Inteligente de Soporte: Triaje & RAG con LangGraph

Este repositorio contiene el desarrollo de un Agente de Soporte Técnico y RRHH de última generación. Combina el poder de LangGraph para la orquestación de flujos de trabajo basados en estados, LangChain para la estructuración y recuperación de información, y Gemini 2.5 Flash como motor principal de razonamiento.

El agente es capaz de analizar solicitudes de usuarios internos, clasificarlas de forma inteligente por nivel de urgencia e intención, y decidir dinámicamente si auto-resolver la duda mediante un motor RAG (Retrieval-Augmented Generation), solicitar más detalles o escalar el caso abriendo un ticket.

# 📐 Arquitectura y Flujo de Trabajo

El agente utiliza un flujo de trabajo de estados gestionado con LangGraph:

graph TD
    START([Inicio]) --> Triaje[Nodo: Triaje e Intención]
    
    Triaje -->|Decisión: AUTO_RESOLVER| RAG[Nodo: Auto Resolver RAG]
    Triaje -->|Decisión: PEDIR_INFO| Info[Nodo: Pedir Información]
    Triaje -->|Decisión: ABRIR_TICKET| Ticket[Nodo: Abrir Ticket]
    
    RAG -->|¿RAG con éxito?| Fin([Fin])
    RAG -->|¿RAG falló pero tiene palabras clave de ticket?| Ticket
    RAG -->|¿RAG falló sin contexto crítico?| Info
    
    Info --> Fin
    Ticket --> Fin

    style START fill:#4CAF50,stroke:#333,stroke-width:2px,color:#fff
    style Fin fill:#f44336,stroke:#333,stroke-width:2px,color:#fff
    style Triaje fill:#2196F3,stroke:#333,stroke-width:2px,color:#fff
    style RAG fill:#9C27B0,stroke:#333,stroke-width:2px,color:#fff


## Componentes Clave

Agente de Triaje Estructurado: Utiliza la capacidad de salida estructurada de Gemini con Pydantic (with_structured_output) para garantizar que la clasificación siempre retorne un JSON válido y estricto con los campos decision y urgencia.

Motor RAG Dinámico: Busca coincidencias semánticas en la base de datos vectorial local basada en FAISS indexada con documentos internos en PDF (políticas de home office, viajes, etc.).

Lógica de Transición (Aristas Condicionales): Si la consulta no es clara o el motor RAG no encuentra una respuesta de alta confianza, el agente redirige el estado de manera automática.

📂 Estructura del Proyecto

├── Docs/                          # Carpeta para colocar los PDFs de políticas internas\
├── RAG_optimizado_2026.py         # Script principal con la lógica del agente y LangGraph\
├── requirements.txt               # Archivo de dependencias del proyecto\
├── .env.example                   # Plantilla de variables de entorno para las credenciales\
└── README.md                      # Documentación del proyecto

----------------

# 🚀 Requisitos e Instalación Local

Si vas a ejecutar el proyecto de forma local en tu computadora, sigue estos pasos:

### 1. Clonar el repositorio y configurar entorno

git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
cd tu-repositorio

python -m venv venv
**Activar entorno:**
- En Windows: venv\Scripts\activate
- En macOS/Linux: source venv/bin/activate

pip install -r requirements.txt


### 2. Configurar las variables de entorno (.env)

Duplica el archivo de plantilla .env.example y renómbralo a .env:

cp .env.example .env


Abre tu archivo .env y añade tu clave:

GEMINI_API_KEY="tu-api-key-de-gemini-aqui"


### 3. Ejecutar el script

python RAG_optimizado_2026.py


## 📓 Alternativa: Ejecución en Google Colab

Si prefieres ejecutar y experimentar este agente en la nube utilizando Google Colab, no necesitas lidiar con archivos .env. Sigue estas sencillas instrucciones dentro de tu notebook:

### 1. Instalar las dependencias

Crea una celda de código en Colab y ejecuta el siguiente comando para instalar todos los paquetes requeridos directamente de tu repositorio:

!pip install -r requirements.txt


### 2. Configurar la API Key de forma segura

Google Colab proporciona un panel para almacenar claves privadas llamado Secrets (icono de llave 🔑 en el menú de la izquierda).

Abre el panel de Secrets (icono 🔑).

Añade un nuevo secreto con el nombre: Gemini-GC.

Pega tu API Key de Gemini en el valor.

Asegúrate de activar la casilla de "Notebook access" (Acceso del cuaderno) para ese secreto.

### 3. Preparar el entorno y ejecutar el Agente

Para ejecutar el archivo .py en Colab sin necesidad de editar el código que busca el entorno local, agrega esta celda inicial para mapear la clave del secreto de Colab directamente a la variable de entorno del sistema:

import os
from google.colab import userdata

- Mapeamos el secreto 'Api-key-gemini' de Colab a la variable que busca nuestro script
os.environ["GEMINI_API_KEY"] = userdata.get('Api-key-gemini')

- (Opcional) Si necesitas cargar documentos desde tu Google Drive, móntalo aquí:
- from google.colab import drive
- drive.mount('/content/drive')

- Ejecutamos el script principal
!python RAG_optimizado_2026.py


## 📈 Demostración de Resultados en Consola

=====================================

*PREGUNTA*: ¿Puedo solicitar mi reembolso por usar mi internet en home office?
Ejecutando nodo 'triaje'...
Ejecutando arista 'decision_triaje'...
Ejecutando nodo 'auto_resolver'...
Ejecutando arista 'decision_rag'...
RAG con éxito, finalizando flujo

*DECISION TRIAJE*: AUTO_RESOLVER | URGENCIA: BAJA\
*ACCIÓN FINAL DEL AGENTE*: AUTO_RESOLVER\
*RESPUESTA FINAL*: Sí, puedes solicitar un reembolso del 50% de tu factura de internet según la política de Home Office.
  - *CITACION 1*: politica_home_office.pdf -> Reembolso del 50% aplicable...

=====================================


# 🛠️ Tecnologías Utilizadas

- **LangChain**: Abstracción y conectores de LLM.

- **LangGraph**: Orquestador de estados y flujos conversacionales complejos.

- **FAISS**: Almacenamiento vectorial eficiente en memoria local.

- **PyMuPDF**: Extracción rápida de texto de archivos PDF para la ingesta del RAG.

- **Pydantic V2**: Validación y tipado fuerte para las respuestas estructuradas del LLM.

- **Python-dotenv**: Carga automatizada de configuraciones locales desde el archivo .env.
