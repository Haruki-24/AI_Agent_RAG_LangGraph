 =====================================================================
# 1. CONFIGURACIÓN E IMPORTACIONES (DIDÁCTICAS Y COMENTADAS) 
 =====================================================================
**import os** 
'os' es de la biblioteca estándar de Python; nos permite interactuar con el sistema operativo
(lo usamos específicamente para leer variables de entorno como API Keys sin exponerlas en el código).

**from pathlib import Path**
'Path' (de pathlib) nos permite manejar rutas de archivos y carpetas de manera orientada a objetos,
resolviendo automáticamente las diferencias de formato entre Windows (con '\') y macOS/Linux (con '/').

**from typing import TypedDict, Optional, Dict, Literal, List**
'typing' proporciona herramientas de tipado estático para hacer nuestro código más legible, seguro y robusto.
- 'TypedDict': Define un diccionario con claves fijas y tipos específicos, ideal para el "estado" de LangGraph.
- 'Optional': Indica que una variable puede tener un tipo de dato específico (ej. str) o ser 'None'.
- 'Dict': Representa un tipo de diccionario estándar con tipos de clave/valor declarados.
- 'Literal': Limita el valor de una variable a opciones exactas de texto (ej. "ALTA" | "BAJA").
- 'List': Define una lista estructurada que contendrá elementos de un tipo específico (ej. List[str]).

**from pydantic import BaseModel**
'pydantic' es la biblioteca estándar en la industria de IA para la validación de datos.
- 'BaseModel': Nos permite crear clases estructuradas ("esquemas"). Cuando el LLM responde,
   Pydantic valida que la respuesta de la IA tenga exactamente la estructura definida en esta clase.

**from google.colab import userdata** 
## NOTA IMPORTANTE PARA DESARROLLADORES:
*'google.colab.userdata'*  -> es un módulo exclusivo de Google Colab para leer claves en la nube.
Está comentado aquí porque en entornos locales de Python se utiliza 'os.getenv()' junto con '.env'.

**from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings **
'langchain_google_genai' contiene los conectores oficiales para interactuar con los modelos de Google.
- 'ChatGoogleGenerativeAI': Wrapper para llamar a los modelos de chat (como Gemini 2.5 Flash).
- 'GoogleGenerativeAIEmbeddings': Transforma palabras o textos enteros en vectores matemáticos (números)
  que representan su significado semántico, algo esencial para la base de datos vectorial (RAG).

**from langchain_community.document_loaders import PyMuPDFLoader**
'PyMuPDFLoader' es un lector de archivos PDF de alto rendimiento. Extrae texto, páginas y
metadatos para que LangChain pueda procesar y segmentar la documentación interna de tu empresa.

**from langchain_community.vectorstores import FAISS**
'FAISS' (Facebook AI Similarity Search) es una librería ultrarrápida para almacenar vectores de texto.
En este script, actúa como nuestra base de datos vectorial local (en memoria) para buscar información relevante.

**from langchain_text_splitters import RecursiveCharacterTextSplitter**
'RecursiveCharacterTextSplitter' es un segmentador de texto inteligente. Divide documentos largos
en fragmentos ("chunks") más pequeños de forma recursiva (respetando párrafos, oraciones y espacios)
para que no superemos la ventana de contexto de los modelos y mantengamos la coherencia semántica.

**from langchain_core.messages import SystemMessage, HumanMessage**
Mensajes estructurados para arquitecturas de chat basadas en roles:
- 'SystemMessage': Define las instrucciones del sistema, el rol, la personalidad y límites de la IA.
- 'HumanMessage': Representa la consulta o mensaje directo enviado por el usuario.

**from langchain_core.prompts import ChatPromptTemplate**
'ChatPromptTemplate' nos permite construir plantillas dinámicas de prompts, lo que facilita
inyectar variables (como contexto y preguntas) de forma ordenada antes de enviárselas al LLM.

**from langchain.chains.combine_documents import create_stuff_documents_chain**
'create_stuff_documents_chain' es un asistente de LangChain que junta (o "rellena") una lista de
documentos recuperados de tu base de datos directamente en el contexto del prompt del LLM.

**from langgraph.graph import START, END, StateGraph**
Componentes de 'langgraph' para construir flujos de trabajo basados en estados (Agentes):
- 'START': Nodo especial que marca el punto de inicio de nuestro flujo de decisiones.
- 'END': Nodo especial que indica que el flujo ha terminado con éxito y debe retornar la respuesta.
- 'StateGraph': El motor principal que une nodos (funciones) y aristas (caminos) mediante un estado común.

**from dotenv import load_dotenv**
'dotenv' carga automáticamente las claves guardadas en tu archivo local '.env' al entorno de tu sistema,
permitiendo que 'os.getenv' las lea de forma segura y transparente.

*load_dotenv()* -> Carga e inicialización de las variables de entorno locales (.env)*

=====================================================================
# --- 1.1 COMPROBACIÓN DE API KEY E INICIALIZACIÓN ---
=====================================================================

### Intentamos obtener la API Key de Gemini desde las variables del sistema
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

### Si no existe la API Key, lanzamos un error explícito para guiar al usuario a configurarla
if not GEMINI_API_KEY:
    raise ValueError(
        "⚠️ Error: No se encontró la variable GEMINI_API_KEY.\n"
        "Por favor, asegúrate de crear tu archivo '.env' con la clave correcta "
        "o configurarla en tu terminal."
    )

### Inicializamos el Modelo de Lenguaje Grande (LLM) de Google
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash', # Usamos la versión de Flash por ser rápida, económica y altamente inteligente
    temperature=0.1,          # Temperatura baja para asegurar respuestas lógicas, consistentes y poco creativas
    google_api_key=GEMINI_API_KEY,
)

 =====================================================================
# --- 2. LÓGICA DE TRIAJE (CLASIFICACIÓN E INTENCIONES) ---
=====================================================================

### Esquema de validación estricto para estructurar la respuesta de triaje
class TriajeOut(BaseModel):
    # La decisión del agente debe limitarse ÚNICAMENTE a estas tres opciones de texto
    decision: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_TICKET"]
    # La urgencia del ticket debe limitarse ÚNICAMENTE a estas tres prioridades
    urgencia: Literal["BAJA", "MEDIANA", "ALTA"]

### Prompt para instruir al Agente de Triaje en su toma de decisiones
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

### Vinculamos el LLM con el esquema estructurado de Pydantic para garantizar que la respuesta sea un objeto JSON limpio
chain_de_triaje = llm.with_structured_output(TriajeOut)

def ejecutar_triaje(mensaje: str) -> Dict:
    """
    Función didáctica que envía la consulta del usuario a la IA y
    devuelve un diccionario nativo de Python con la clasificación.
    """
    salida: TriajeOut = chain_de_triaje.invoke([
        SystemMessage(content=PROMPT_TRIAJE), # Enviamos el rol y reglas del sistema
        HumanMessage(content=mensaje)         # Enviamos el mensaje del usuario
    ])
    return salida.model_dump() # Convierte el objeto de Pydantic a un dict clásico de Python {"decision":..., "urgencia":...}

=====================================================================
# --- 3. LÓGICA DE RAG (RECUPERACIÓN EN BASE DE DATOS VECTORIAL) ---
=====================================================================

### Simulación didáctica de una base de datos vectorial para ilustrar la respuesta del RAG
def busqueda_respuesta_RAG(pregunta: str) -> Dict:
    """
    Simula el comportamiento del sistema RAG. En producción, esta función utiliza
    retriever.invoke(pregunta) para buscar fragmentos semánticos en FAISS y luego
    los envía con 'create_stuff_documents_chain' a Gemini para formular la respuesta.
    """
    # Lógica didáctica para emular el éxito de búsqueda semántica mediante palabras clave
    if "reembolso" in pregunta.lower() or "internet" in pregunta.lower() or "home office" in pregunta.lower():
        return {
            "respuesta": "Sí, según la política de Home Office de Carraro Desarrollo de Software, tienes derecho a un reembolso mensual de hasta el 50% de tu factura de internet previa presentación de la factura correspondiente.",
            "citaciones": [
                {
                    "metadata": {"file_path": "politicas_rrhh_home_office.pdf"},
                    "page_content": "Sección 4.2 - Reembolso de Servicios: La compañía cubrirá el 50% del costo del plan contratado de internet para uso en Home Office."
                }
            ],
            "documentos_encontrados": True
        }
    
    # En caso de no encontrar coincidencia en la base de datos simulada, retorna negativo
    return {
        "respuesta": "No lo sé",
        "citaciones": [],
        "documentos_encontrados": False
    }

=====================================================================
# --- 4. CONFIGURACIÓN DEL ESTADO Y NODOS DE LANGGRAPH ---
=====================================================================

### El 'AgentState' es la memoria compartida del grafo. Todos los nodos leen de aquí y escriben aquí.
class AgentState(TypedDict):
    pregunta: str                  # Consulta original realizada por el empleado
    triaje: Dict                   # Almacena el resultado del nodo de triaje (decision y urgencia)
    respuesta_RAG: Optional[str]   # Almacena la respuesta generada si se activa el RAG
    citaciones: Optional[List]     # Lista de documentos/fuentes que justifican la respuesta del RAG
    rag_exito: bool                # Booleano que indica si la base de datos vectorial encontró información útil
    accion_final: str              # Estado final adoptado por el agente ante el Service Desk

## --- DEFINICIÓN DE NODOS (FUNCIONES DE ACCIÓN) ---

def nodo_triaje(state: AgentState) -> AgentState:
    """Primer nodo del grafo. Analiza la consulta y determina la intención inicial."""
    print("\n[Nodo Triaje] Analizando la intención y prioridad de la consulta...")
    resultado = ejecutar_triaje(state["pregunta"])
    return {"triaje": resultado} # Actualiza la clave 'triaje' en el estado compartido

def nodo_auto_resolver(state: AgentState) -> AgentState:
    """Nodo encargado de activar el motor de búsqueda semántica (RAG) para resolver la duda."""
    print("[Nodo Auto-Resolver] Consultando la base de datos de políticas internas...")
    respuesta_RAG = busqueda_respuesta_RAG(state["pregunta"])
    
    # Preparamos la actualización del estado con lo obtenido en la base de datos
    update: AgentState = {
        "respuesta_RAG": respuesta_RAG["respuesta"],
        "citaciones": respuesta_RAG.get("citaciones", []),
        "rag_exito": respuesta_RAG["documentos_encontrados"],
    }
    
    # Si la información existía en el manual de políticas, declaramos la resolución exitosa
    if respuesta_RAG["documentos_encontrados"]:
        update["accion_final"] = "AUTO_RESOLVER"
        
    return update

def nodo_pedir_info(state: AgentState) -> AgentState:
    """Nodo de fallback que se activa cuando la consulta es confusa o faltan datos."""
    print("[Nodo Pedir Info] La consulta requiere aclaraciones por parte del empleado.")
    return {
        "respuesta_RAG": "Por favor, proporciónanos más detalles (ej. montos, fechas, o el servicio específico) para procesar tu solicitud adecuadamente.",
        "citaciones": [],
        "accion_final": "PEDIR_INFO"
    }

def nodo_abrir_ticket(state: AgentState) -> AgentState:
    """Nodo final cuando la solicitud requiere obligatoriamente aprobación de un gestor de RRHH."""
    print("[Nodo Abrir Ticket] Derivando caso al departamento humano de soporte...")
    return {
        "respuesta_RAG": f"Se ha abierto un ticket formal bajo la categoría de políticas internas. Un agente humano revisará tu solicitud de: '{state['pregunta']}'",
        "citaciones": [],
        "accion_final": "ABRIR_TICKET"
    }

## --- DEFINICIÓN DE ARISTAS CONDICIONALES (CEREBRO DE ENRUTAMIENTO) ---

def arista_decision_triaje(state: AgentState) -> str:
    """Arista condicional que decide hacia qué nodo enviar la ejecución tras el triaje."""
    decision = state["triaje"]["decision"]
    print(f"[Enrutador Triaje] Enrutando flujo hacia: {decision}")
    
    if decision == "AUTO_RESOLVER": 
        return "rag"      # Envía al nodo 'auto_resolver'
    elif decision == "PEDIR_INFO": 
        return "info"     # Envía al nodo 'pedir_info'
    elif decision == "ABRIR_TICKET": 
        return "ticket"   # Envía al nodo 'abrir_ticket'
        
    raise ValueError(f"Decisión inesperada en el triaje: {decision}")

def arista_decision_rag(state: AgentState) -> str:
    """Arista condicional que analiza si el RAG resolvió con éxito o si debe escalar la solicitud."""
    print("[Enrutador RAG] Evaluando resultados de la consulta vectorial...")
    
    # 1. Si el RAG encontró la respuesta en las políticas, el flujo finaliza felizmente
    if state["rag_exito"]:
        print(" -> Éxito total. Cerrando flujo.")
        return "ok"

    # 2. Si el RAG falló, pero el usuario usó palabras críticas, abrimos un ticket directamente
    KEYWORDS_ABRIR_TICKET = ["aprobación", "aprobar", "excepción", "liberación", "autorización", "abrir ticket", "acceso especial"]
    if any(keyword in state["pregunta"].lower() for keyword in KEYWORDS_ABRIR_TICKET):
        print(" -> El RAG falló, pero se detectaron términos clave de soporte. Derivando a Ticket.")
        return "ticket"
        
    # 3. Si falló y es una pregunta ambigua o general, le pedimos información extra
    else:
        print(" -> Información insuficiente en base de datos. Solicitando más datos.")
        return "info"

=====================================================================
# --- 4.1 ENSAMBLADO Y CONSTRUCCIÓN DEL GRAFO ---
=====================================================================

### Inicializamos la estructura del grafo pasándole nuestro esquema de estado
workflow = StateGraph(AgentState)

### Agregamos los nodos de acción
workflow.add_node("triaje", nodo_triaje)
workflow.add_node("auto_resolver", nodo_auto_resolver)
workflow.add_node("pedir_info", nodo_pedir_info)
workflow.add_node("abrir_ticket", nodo_abrir_ticket)

### Definimos el punto de entrada del flujo
workflow.add_edge(START, "triaje")

### Conectamos el triaje con los nodos correspondientes según la decisión de la arista
workflow.add_conditional_edges(
    "triaje", 
    arista_decision_triaje, 
    {
        "rag": "auto_resolver", 
        "info": "pedir_info", 
        "ticket": "abrir_ticket"
    }
)

### Conectamos el RAG con las alternativas de escape (escalado) o cierre si tuvo éxito
workflow.add_conditional_edges(
    "auto_resolver", 
    arista_decision_rag, 
    {
        "info": "pedir_info", 
        "ticket": "abrir_ticket", 
        "ok": END # Finaliza el grafo
    }
)

### Compilamos el grafo para que esté listo para ejecutarse
grafo = workflow.compile()

=====================================================================
# --- 5. BLOQUE DE EJECUCIÓN PRINCIPAL ---
=====================================================================
if __name__ == "__main__":
    # Renderizar y guardar el grafo de forma local en una imagen para documentar el proyecto
    try:
        graph_bytes = grafo.get_graph().draw_mermaid_png()
        with open("flujo_agente.png", "wb") as f:
            f.write(graph_bytes)
        print("✅ Flujo lógico del agente guardado con éxito como 'flujo_agente.png'.\n")
    except Exception as e:
        print(f"⚠️ Nota de renderizado: Se requiere 'pygraphviz' o dependencias de Mermaid para guardar el mapa visual: {e}\n")

    # Lista ordenada de escenarios de prueba para demostrar el comportamiento del agente
    mensajes_de_prueba = [
        "¿Puedo solicitar mi reembolso por usar mi internet en home office?",
        "Tengo un problema con algo que compré.",
        "Necesito que me aprueben una excepción de seguridad para instalar un programa."
    ]

    # Ejecución iterativa de pruebas en consola
    for i, prueba in enumerate(mensajes_de_prueba, 1):
        print(f"============================================================")
        print(f"TEST NÚMERO {i}")
        print(f"PREGUNTA DEL USUARIO: '{prueba}'")
        print(f"============================================================")
        
        # Ejecutamos el agente enviando el estado inicial con la pregunta
        resultado_conversacion = grafo.invoke({"pregunta": prueba})
        
        # Desglosamos los resultados obtenidos de la memoria del estado finalizado
        print(f"\n--- 📊 RESULTADO DEL TRIAJE ---")
        print(f"Decisión inicial: {resultado_conversacion['triaje']['decision']}")
        print(f"Urgencia clasificada: {resultado_conversacion['triaje']['urgencia']}")
        
        print(f"\n--- 🤖 RESPUESTA FINAL DEL AGENTE ---")
        print(f"Resolución final adoptada: {resultado_conversacion.get('accion_final', 'PEDIR_INFO')}")
        print(f"Mensaje para el usuario: {resultado_conversacion.get('respuesta_RAG')}")
        
        # Mostramos citaciones de soporte (si existieron en la búsqueda del RAG)
        citaciones = resultado_conversacion.get('citaciones')
        if citaciones:
            print(f"\n--- 📂 CITACIONES Y SOPORTE DOCUMENTAL ---")
            for j, citacion in enumerate(citaciones, 1):
                doc_path = citacion['metadata']['file_path'] if isinstance(citacion, dict) else citacion.metadata['file_path']
                content = citacion['page_content'] if isinstance(citacion, dict) else citacion.page_content
                print(f"  [{j}] Fuente: '{doc_path}' \n      Contenido exacto: \"{content}\"")
        print("\n")