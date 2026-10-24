# 📘 Glosario y Guía Didáctica del Agente RAG

Este documento está diseñado no solo para servir como **manual de mantenimiento**, sino también para facilitar el **onboarding de nuevos desarrolladores**. Explica el *por qué* de las decisiones arquitectónicas y el funcionamiento interno de nuestro `RAG_optimizado_2026.py`, asumiendo un nivel básico de Python para que cualquiera pueda entender, aprender y reutilizar el código.

## 🏗️ 1. Decisiones Arquitectónicas Clave

Antes de ensuciarnos las manos con el código, es vital entender qué herramientas elegimos y por qué:

| Tecnología / Patrón | ¿Por qué se usó? | Impacto en el sistema |
| :--- | :--- | :--- |
| **LangGraph (`StateGraph`)** | A diferencia de las cadenas lineales de LangChain, los agentes necesitan tomar decisiones y volver atrás. | Permite crear flujos cíclicos y condicionales (Triaje -> RAG -> Ticket). |
| **Pydantic (`BaseModel`)** | Los LLMs son propensos a alucinar formatos. | Obliga a Gemini a responder con un JSON estricto. Si falla, el código lanza un error controlado en lugar de romperse. |
| **FAISS** | Búsqueda vectorial rápida en local sin necesidad de pagar por una base de datos en la nube (Pinecone, etc.). | Ideal para despliegues ligeros y pruebas de concepto. |
| **Umbral de similitud (`score_threshold: 0.3`)** | Evita que el RAG responda con información irrelevante si no hay coincidencia real en los documentos. | Si la similitud es menor a 0.3, el agente asume que "no sabe la respuesta" y escala a un humano. |
| **`temperature=0.1`** | Los modelos creativos son malos para clasificar. | Garantiza que el triaje sea lógico, consistente y determinista. |

## 2. El Ciclo de Vida del Estado (`AgentState`)

El `AgentState` es la **memoria compartida** del grafo. Cualquier función (nodo) que definas en `RAG_optimizado_2026.py` puede leer lo que hicieron los nodos anteriores y escribir nuevas claves.

```python
class AgentState(TypedDict):
    pregunta: str                   # Texto original del usuario
    triaje: Dict                    # Salida estructurada de Pydantic (decisión y urgencia)
    respuesta_RAG: Optional[str]    # Respuesta generada por el motor de búsqueda
    citaciones: Optional[List]      # Documentos fuente usados para responder
    rag_exito: bool                 # Flag de control de calidad
    accion_final: str               # Trazabilidad de la decisión final
```

### ¿Cómo actualizar el estado?

Cuando un nodo (función) termina, no modifica el estado original, sino que devuelve un diccionario con las claves que quiere actualizar. LangGraph se encarga de fusionar ese diccionario con el estado global.

*Ejemplo:*

```python
def nodo_triaje(state):
    # El nodo lee state["pregunta"]
    resultado = ejecutar_triaje(state["pregunta"])
    # Y escribe en state["triaje"]
    return {"triaje": resultado}
```

> 💡 **Tip para desarrolladores**: Si necesitas añadir una nueva funcionalidad (ej. enviar un correo), debes agregar una nueva clave a este diccionario (ej. correo_enviado: bool).

## 🔄 3. Flujo Lógico y Enrutamiento Condicional

El sistema se divide en dos fases críticas:

### Fase 1: Triaje (Clasificación)

El LLM analiza la pregunta y decide el camino. Es como el recepcionista de un hospital que decide si te manda a urgencias, a consulta general o te pide más datos.

* `AUTO_RESOLVER`: Va al nodo RAG.
* `PEDIR_INFO`: Va al nodo de aclaración.
* `ABRIR_TICKET`: Va al nodo de escalamiento humano.

### Fase 2: Fallback Inteligente (Lógica de Aristas)

¿Qué pasa si el usuario pide un `AUTO_RESOLVER`, pero el RAG no encuentra la respuesta? Aquí es donde ocurre la magia en la función `arista_decision_rag`. No dejamos al usuario colgado; evaluamos palabras clave para decidir si escalamos a un humano o pedimos más contexto:

```python
def arista_decision_rag(state: AgentState) -> str:
    # Si el RAG encontró buena información, terminamos con éxito
    if state["rag_exito"]:
        return "ok"

    # Si el RAG falló, verificamos si hay palabras clave de escalamiento
    KEYWORDS_ABRIR_TICKET = ["aprobación", "aprobar", "excepción", "liberación", "autorización", "abrir ticket", "acceso especial"]
    if any(keyword in state["pregunta"].lower() for keyword in KEYWORDS_ABRIR_TICKET):
        return "ticket" # Escalar a humano

    return "info" # Pedir más datos al usuario
```

## 4. Referencia visual

Recuerda que al ejecutar `RAG_optimizado_2026.py`, el sistema genera automáticamente el archivo `flujo_agente.png`. Este diagrama es tu mejor aliado para explicar el flujo a otros colaboradores. Si el diagrama se vuelve complejo, considera dividir el flujo en sub-grafos.

> 💡 **Tip para colaboradores:** Mantén siempre el archivo `.env.example` actualizado si añades nuevas variables de entorno para que el resto del equipo sepa qué llaves configurar.

## 🛠️ 5. Manual de Extensiones (¿Cómo modificar el Agente?)

Este proyecto está diseñado para crecer. Aquí te explicamos cómo hacer cambios sin romper el flujo.

### A. ¿Cómo cambiar el modelo LLM (ej. de Gemini a GPT-4)?

Solo debes modificar la inicialización en la sección `6.2` del código principal. El resto de la lógica (Pydantic, Triaje, RAG) seguirá funcionando sin cambios.

```python
# 1. Reemplaza ChatGoogleGenerativeAI por ChatOpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.1)
```

### B. ¿Cómo añadir un nuevo nodo al grafo?

```python
# 1. Define la función del nodo:
def nodo_mi_nueva_funcion(state: AgentState) -> AgentState:
    print("Ejecutando nueva función...")
    return {"accion_final": "MI_NUEVA_ACCION"}

# 2. Regístralo en el grafo (en la sección de ensamblado):
workflow.add_node("mi_nueva_funcion", nodo_mi_nueva_funcion)

# 3. Ajusta las conexiones (aristas):
workflow.add_edge("auto_resolver", "mi_nueva_funcion")
workflow.add_edge("mi_nueva_funcion", END)
```

### C. ¿Cómo añadir nuevos documentos al RAG?

1. Copia tus archivos PDF en la carpeta `./Docs/`.
2. El script los leerá automáticamente al iniciar (usando `Path("./Docs/").glob("*.pdf")`).
3. Si añades archivos muy grandes, ajusta `chunk_size=300` y `chunk_overlap=30` en el `RecursiveCharacterTextSplitter` para mejorar la precisión.

## 🔍 6. Desglose del Código Paso a Paso

Si eres nuevo en Python o en LangChain, esta sección te explica qué hace cada línea del archivo `RAG_optimizado_2026.py` para que puedas entender un poco mejor cada línea.

### 6.1 Configuración e Importaciones

Entender qué importa nuestro código es el primer paso para dominar su lógica:

* `import os`: Es de la biblioteca estándar de Python; nos permite interactuar con el sistema operativo (lo usamos específicamente para leer variables de entorno como API Keys sin exponerlas en el código).
* `from pathlib import Path`: Nos permite manejar rutas de archivos y carpetas de manera orientada a objetos, resolviendo automáticamente las diferencias de formato entre Windows (con \) y macOS/Linux (con /).
* `from typing import TypedDict, Optional, Dict, Literal, List`: Proporciona herramientas de tipado estático para hacer nuestro código más legible, seguro y robusto.
   - TypedDict: Define un diccionario con claves fijas y tipos específicos, ideal para el "estado" de LangGraph.
   - Optional: Indica que una variable puede tener un tipo de dato específico (ej. str) o ser None.
   - Dict: Representa un tipo de diccionario estándar con tipos de clave/valor declarados.
   - Literal: Limita el valor de una variable a opciones exactas de texto (ej. "ALTA" | "BAJA").
   - List: Define una lista estructurada que contendrá elementos de un tipo específico (ej. List[str]).

* `from pydantic import BaseModel`: Es la biblioteca estándar en la industria de IA para la validación de datos. BaseModel nos permite crear clases estructuradas ("esquemas"). Cuando el LLM responde, Pydantic valida que la respuesta de la IA tenga exactamente la estructura definida.
*`from google.colab import userdata`:(Nota) Es un módulo exclusivo de Google Colab para leer claves en la nube. En entornos locales de Python se utiliza os.getenv() junto con .env.
* `from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings`: Contiene los conectores oficiales para interactuar con los modelos de Google. ChatGoogleGenerativeAI es el wrapper para llamar a los modelos de chat, y GoogleGenerativeAIEmbeddings transforma palabras en vectores matemáticos.
* `from langchain_community.document_loaders import PyMuPDFLoader`: Lector de archivos PDF de alto rendimiento. Extrae texto para que LangChain pueda procesar la documentación interna.
* `from langchain_community.vectorstores import FAISS`: Librería ultrarrápida para almacenar vectores de texto. Actúa como nuestra base de datos vectorial local (en memoria).
* `from langchain_text_splitters import RecursiveCharacterTextSplitter`: Segmentador de texto inteligente. Divide documentos largos en fragmentos más pequeños de forma recursiva.
* `from langchain_core.messages import SystemMessage, HumanMessage`: Mensajes estructurados para arquitecturas de chat basadas en roles (SystemMessage para instrucciones de la IA, HumanMessage para el usuario).
* `from langchain_core.prompts import ChatPromptTemplate`: Construye plantillas dinámicas de prompts para inyectar variables de forma ordenada.
* `from langchain.chains.combine_documents import create_stuff_documents_chain`: Asistente que junta (o "rellena") una lista de documentos recuperados directamente en el contexto del prompt del LLM.
* `from langgraph.graph import START, END, StateGraph`: Componentes para construir flujos de trabajo basados en estados (START inicio, END fin, StateGraph el motor principal).
* `from dotenv import load_dotenv`: Carga automáticamente las claves guardadas en tu archivo local .env al entorno de tu sistema.

### 6.2 Comprobación de API Key e Inicialización

Se obtiene la API Key del entorno y se inicializa el modelo. Si la clave no existe, el código lanza un error explícito para guiar al usuario (en lugar de fallar silenciosamente).


```python
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Si no existe la API Key, lanzamos un error explícito para guiar al usuario
if not GEMINI_API_KEY:
    raise ValueError(
        "⚠️ Error: No se encontró la variable GEMINI_API_KEY.\n"
        "Por favor, asegúrate de crear tu archivo '.env' con la clave correcta "
        "o configurarla en tu terminal."
    )

# Inicializamos el Modelo de Lenguaje Grande (LLM) de Google
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash', # Versión rápida, económica y altamente inteligente
    temperature=0.1,          # Temperatura baja para asegurar respuestas lógicas y consistentes
    google_api_key=GEMINI_API_KEY,
)
```
> **Tip de desarrollador**: Al momento de hacer este RAG se utilizó 'gemini-2.5-flash' pero es porbrable que si lees esto ene l futuro este modelo se haya reempalazado por uno mas nuevo. Por eso recomiendo revisar la documentación en [Google AI](https://ai.google.dev/gemini-api/docs/models?hl=es-419)

### 6.3 Lógica de Triaje (Clasificación e Intenciones)

Usamos `Pydantic` para crear un esquema estricto (`TriajeOut`). Luego, le decimos al LLM que use `with_structured_output`. Esto es permite que nos aseguremos de que la IA nunca devuelva texto libre, sino un JSON que nuestro código pueda leer sin errores.

```python
# Esquema de validación estricto para estructurar la respuesta de triaje
class TriajeOut(BaseModel):
    decision: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_TICKET"]
    urgencia: Literal["BAJA", "MEDIANA", "ALTA"]

# Prompt para instruir al Agente de Triaje
PROMPT_TRIAJE = """
Eres un Agente de Triaje del Service Desk. Analiza el mensaje del usuario y devuelve SOLO un objeto JSON con `decision` y `urgencia`.

Reglas de decisión (`decision`):
- "AUTO_RESOLVER": Consultas claras sobre políticas, FAQs o procedimientos estándar ya establecidos.
- "PEDIR_INFO": Mensajes ambiguos, incompletos o que requieren contexto extra antes de poder procesarse.
- "ABRIR_TICKET": Solicitudes de excepciones, aprobaciones, accesos especiales o incidentes técnicos complejos.

Reglas de urgencia (`urgencia`):
- "BAJA": Consultas generales sin afectación directa a la productividad diaria.
- "MEDIANA": Problemas que impactan parcialmente al usuario pero no detienen sus operaciones por completo.
- "ALTA": Bloqueo completo de actividades del usuario/equipo, incidentes graves de seguridad o plazos críticos.

Instrucciones:
1. Responde única y estrictamente con la estructura JSON solicitada.
2. Prioriza la máxima precisión en la clasificación semántica.
"""

# Vinculamos el LLM con el esquema estructurado de Pydantic
chain_de_triaje = llm.with_structured_output(TriajeOut)

def ejecutar_triaje(mensaje: str) -> Dict:
    """Envía la consulta del usuario a la IA y devuelve un diccionario con la clasificación."""
    salida: TriajeOut = chain_de_triaje.invoke([
        SystemMessage(content=PROMPT_TRIAJE),
        HumanMessage(content=mensaje)
    ])
    return salida.model_dump()
```

### 6.4 Lógica RAG (Búsqueda Vectorial)

* **Carga:** Se recorre `./Docs/` buscando PDFs con `PyMuPDFLoader`.
* **Procesamiento:** `RecursiveCharacterTextSplitter` divide el texto en fragmentos (chunks) para que no excedan el límite de tokens.
* **Vectorización:** `GoogleGenerativeAIEmbeddings` convierte cada fragmento en un vector matemático.
* **Búsqueda:** `FAISS` almacena los vectores y recupera los más relevantes según la consulta. Usamos un `score_threshold` de 0.3 para evitar respuestas inventadas.

```python
# 1. Definir la ruta local de los documentos
RUTA_DOCS = "./Docs/"
docs = []

# 2. Cargar documentos PDF de la carpeta local
if os.path.exists(RUTA_DOCS):
    for n in Path(RUTA_DOCS).glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(n))
            docs.extend(loader.load())
            print(f"📄 Archivo cargado exitosamente: {n.name}")
        except Exception as e:
            print(f"⚠️ Error cargando archivo: {n.name}: {e}")
else:
    print(f"⚠️ Advertencia: No se encontró la carpeta {RUTA_DOCS}. El RAG no tendrá contexto.")

# 3. Inicializar el motor de búsqueda vectorial
retriever = None
if docs:
    print(f"Total de documentos cargados en memoria: {len(docs)}")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs_splits = splitter.split_documents(docs)

    modelo_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    
    vectorstore = FAISS.from_documents(docs_splits, modelo_embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 3}
    )
```

### 6.5 Configuración del Estado y Nodos de LangGraph

Aquí definimos la estructura del `AgentState` y las funciones (nodos) que ejecutarán cada paso. Piensa en los nodos como "estaciones de trabajo" en una fábrica: cada una toma el estado actual, hace su trabajo y devuelve una actualización.

```python
# El 'AgentState' es la memoria compartida del grafo
class AgentState(TypedDict):
    pregunta: str                  
    triaje: Dict                   
    respuesta_RAG: Optional[str]   
    citaciones: Optional[List]     
    rag_exito: bool                
    accion_final: str              

# --- DEFINICIÓN DE NODOS ---

def nodo_triaje(state: AgentState) -> AgentState:
    print("\n[Nodo Triaje] Analizando la intención y prioridad de la consulta...")
    resultado = ejecutar_triaje(state["pregunta"])
    return {"triaje": resultado}

def nodo_auto_resolver(state: AgentState) -> AgentState:
    print("[Nodo Auto-Resolver] Consultando la base de datos de políticas internas...")
    respuesta_RAG = busqueda_respuesta_RAG(state["pregunta"])
    
    update: AgentState = {
        "respuesta_RAG": respuesta_RAG["respuesta"],
        "citaciones": respuesta_RAG.get("citaciones", []),
        "rag_exito": respuesta_RAG["documentos_encontrados"],
    }
    if respuesta_RAG["documentos_encontrados"]:
        update["accion_final"] = "AUTO_RESOLVER"
        
    return update

def nodo_pedir_info(state: AgentState) -> AgentState:
    print("[Nodo Pedir Info] La consulta requiere aclaraciones por parte del empleado.")
    return {
        "respuesta_RAG": "Por favor, proporciónanos más detalles para procesar tu solicitud.",
        "citaciones": [],
        "accion_final": "PEDIR_INFO"
    }

def nodo_abrir_ticket(state: AgentState) -> AgentState:
    print("[Nodo Abrir Ticket] Derivando caso al departamento humano de soporte...")
    return {
        "respuesta_RAG": f"Se ha abierto un ticket formal. Un agente revisará tu solicitud de: '{state['pregunta']}'",
        "citaciones": [],
        "accion_final": "ABRIR_TICKET"
    }

# --- DEFINICIÓN DE ARISTAS CONDICIONALES ---

def arista_decision_triaje(state: AgentState) -> str:
    decision = state["triaje"]["decision"]
    print(f"[Enrutador Triaje] Enrutando flujo hacia: {decision}")
    
    if decision == "AUTO_RESOLVER": return "rag"      
    elif decision == "PEDIR_INFO": return "info"     
    elif decision == "ABRIR_TICKET": return "ticket"   
    raise ValueError(f"Decisión inesperada en el triaje: {decision}")

def arista_decision_rag(state: AgentState) -> str:
    print("[Enrutador RAG] Evaluando resultados de la consulta vectorial...")
    if state["rag_exito"]:
        print(" -> Éxito total. Cerrando flujo.")
        return "ok"

    KEYWORDS_ABRIR_TICKET = ["aprobación", "aprobar", "excepción", "liberación", "autorización", "abrir ticket", "acceso especial"]
    if any(keyword in state["pregunta"].lower() for keyword in KEYWORDS_ABRIR_TICKET):
        print(" -> El RAG falló, pero se detectaron términos clave de soporte. Derivando a Ticket.")
        return "ticket"
    else:
        print(" -> Información insuficiente en base de datos. Solicitando más datos.")
        return "info"
```

### 6.6 Ensamblado y Construcción del Grafo

Finalmente, unimos todos los nodos y las aristas para compilar el agente. `add_conditional_edges` es el "semáforo" que decide a dónde va el flujo basándose en la función de enrutamiento que definimos arriba.

```python
workflow = StateGraph(AgentState)

workflow.add_node("triaje", nodo_triaje)
workflow.add_node("auto_resolver", nodo_auto_resolver)
workflow.add_node("pedir_info", nodo_pedir_info)
workflow.add_node("abrir_ticket", nodo_abrir_ticket)

workflow.add_edge(START, "triaje")

workflow.add_conditional_edges(
    "triaje", 
    arista_decision_triaje, 
    {
        "rag": "auto_resolver", 
        "info": "pedir_info", 
        "ticket": "abrir_ticket"
    }
)

workflow.add_conditional_edges(
    "auto_resolver", 
    arista_decision_rag, 
    {
        "info": "pedir_info", 
        "ticket": "abrir_ticket", 
        "ok": END 
    }
)

grafo = workflow.compile()
```

### 6.7 Bloque de Ejecución Principal
El script incluye una ejecución de prueba con ejemplos, imprimiendo el resultado del triaje, la acción final tomada por el agente y las fuentes de información (citaciones) utilizadas.

```python
if __name__ == "__main__":
    try:
        graph_bytes = grafo.get_graph().draw_mermaid_png()
        with open("flujo_agente.png", "wb") as f:
            f.write(graph_bytes)
        print("✅ Flujo lógico del agente guardado con éxito como 'flujo_agente.png'.\n")
    except Exception as e:
        print(f"⚠️ Nota de renderizado: Se requiere 'pygraphviz' o dependencias de Mermaid: {e}\n")

    mensajes_de_prueba = [
        "¿Puedo solicitar mi reembolso por usar mi internet en home office?",
        "Tengo un problema con algo que compré.",
        "Necesito que me aprueben una excepción de seguridad para instalar un programa."
    ]

    for i, prueba in enumerate(mensajes_de_prueba, 1):
        print(f"============================================================")
        print(f"TEST NÚMERO {i}")
        print(f"PREGUNTA DEL USUARIO: '{prueba}'")
        print(f"============================================================")
        
        resultado_conversacion = grafo.invoke({"pregunta": prueba})
        
        print(f"\n--- 📊 RESULTADO DEL TRIAJE ---")
        print(f"Decisión inicial: {resultado_conversacion['triaje']['decision']}")
        print(f"Urgencia clasificada: {resultado_conversacion['triaje']['urgencia']}")
        
        print(f"\n--- 🤖 RESPUESTA FINAL DEL AGENTE ---")
        print(f"Resolución final adoptada: {resultado_conversacion.get('accion_final', 'PEDIR_INFO')}")
        print(f"Mensaje para el usuario: {resultado_conversacion.get('respuesta_RAG')}")
        
        citaciones = resultado_conversacion.get('citaciones')
        if citaciones:
            print(f"\n--- 📂 CITACIONES Y SOPORTE DOCUMENTAL ---")
            for j, citacion in enumerate(citaciones, 1):
                doc_path = citacion['metadata']['file_path'] if isinstance(citacion, dict) else citacion.metadata['file_path']
                content = citacion['page_content'] if isinstance(citacion, dict) else citacion.page_content
                print(f"  [{j}] Fuente: '{doc_path}' \n      Contenido exacto: \"{content}\"")
        print("\n")
```
