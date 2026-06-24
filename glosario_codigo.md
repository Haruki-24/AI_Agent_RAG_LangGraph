# 📘 Glosario y Guía Didáctica del Agente RAG

Este documento sirve como acompañamiento técnico para el archivo `RAG_optimizado_2026.py`. El objetivo es explicar las decisiones arquitectónicas y el funcionamiento interno para facilitar futuras modificaciones o el onboarding de nuevos colaboradores.

## 1. Módulos y dependencias clave

Entender qué importa nuestro código es el primer paso para dominar su lógica:

* **`langgraph.graph` (StateGraph):** Es el corazón del agente. A diferencia de una cadena lineal, permite crear un grafo donde el estado fluye entre nodos. Si quieres añadir un nuevo paso al proceso (ej. una etapa de revisión humana), solo tienes que añadir un nodo y conectar las aristas.

* **`pydantic` (BaseModel):** Vital para la estabilidad. Al definir `TriajeOut`, obligamos a Gemini a responder con un JSON que nuestro código puede procesar sin miedo a errores de formato.

* **`dotenv`:** Permite que tu configuración local (`.env`) sea invisible para el control de versiones (Git), protegiendo tus credenciales de seguridad.

## 2. El Ciclo de Vida del Estado (`AgentState`)

El `AgentState` es la "memoria compartida". Cualquier función (nodo) que definas en `RAG_optimizado_2026.py` puede leer lo que hicieron los nodos anteriores.

### ¿Cómo actualizar el estado?
Cuando una función termina, devuelve un diccionario con las claves que quieres actualizar.

*Ejemplo:*

```python
def nodo_triaje(state):
    # El nodo lee state["pregunta"]
    resultado = ejecutar_triaje(state["pregunta"])
    # Y escribe en state["triaje"]
    return {"triaje": resultado}
```

## 3. ¿Cómo realizar futuras modificaciones?

**A. Si deseas añadir un nuevo nodo (ej. una etapa de traducción):**
1. Define la función del nodo: `def nodo_traduccion(state): ...`
2. Regístrala en el grafo: `workflow.add_node("traduccion", nodo_traduccion)`
3. Ajusta las conexiones (aristas) en la sección 4.1.

**B. Si el modelo Gemini cambia:**
Solo debes actualizar la inicialización del objeto `llm` en la sección 1.1. Gracias a que usamos `with_structured_output`, el resto de la lógica (triaje, validación) debería mantenerse intacta.

## 4. Referencia visual

Recuerda que al ejecutar `RAG_optimizado_2026.py`, el sistema genera automáticamente el archivo `flujo_agente.png`. Este diagrama es tu mejor aliado para explicar el flujo a otros colaboradores. Si el diagrama se vuelve complejo, considera dividir el flujo en sub-grafos.

> 💡 **Tip para colaboradores:** Mantén siempre el archivo `.env.example` actualizado si añades nuevas variables de entorno para que el resto del equipo sepa qué llaves configurar.

---

## 🔍 Desglose del Código Paso a Paso

### 1. Configuración e Importaciones
A continuación se explican las librerías base utilizadas:

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

### 1.1 Comprobación de API Key e Inicialización
Se obtiene la API Key del entorno y se inicializa el modelo:

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


### 2. Lógica de Triaje (Clasificación e Intenciones)
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

### 3. Lógica RAG
* **Carga:** Se recorre `./Docs/` buscando PDFs.
* **Procesamiento:** `RecursiveCharacterTextSplitter` divide el texto en fragmentos.
* **Vectorización:** `GoogleGenerativeAIEmbeddings` crea vectores de búsqueda.
* **Búsqueda:** `FAISS` recupera documentos relevantes según la consulta.

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

### 4. Configuración del Estado y Nodos de LangGraph
El `AgentState` funciona como el bus de datos compartido por todos los nodos del grafo. La lógica de enrutamiento se define mediante funciones condicionales que evalúan la salida del triaje y el éxito del RAG.

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


4.1 Ensamblado y Construcción del Grafo

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

### 5. Bloque de Ejecución Principal
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
