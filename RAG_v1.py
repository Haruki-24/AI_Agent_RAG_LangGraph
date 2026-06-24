# --- 1. CONFIGURACIÓN E IMPORTACIONES ---
import os
from pathlib import Path
from typing import TypedDict, Optional, Dict, Literal, List
from pydantic import BaseModel
# from google.colab import userdata  --> Aplica solo para Google Colab

# LangChain y componentes específicos
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings 
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langgraph.graph import START, END, StateGraph

# --- OBTENCIÓN DE API KEY LOCAL ---
# Obtenemos la key de las variables de entorno de tu sistema
from dotenv import load_dotenv
load_dotenv() # Esto busca el archivo .env y carga las variables en os.environ

# Obtener la API Key de Gemini desde las variables del sistema
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Si no existe la API Key, muestra un error para configurarla
if not GEMINI_API_KEY:
    raise ValueError("⚠️ Error: No se encontró GEMINI_API_KEY. Asegúrate de configurarla en tu entorno.")

# Inicialización del modelo LLM (Gemini 2.5 Flash)
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0.1,
    google_api_key=GEMINI_API_KEY,
)

print("✅ Modelos configurados correctamente")
print(f"📊 Gemini: gemini-2.5-flash")


# --- 2. LÓGICA DE TRIAJE (CLASIFICACIÓN) ---

# Esquema de validación para estructurar la respuesta de triaje
class TriajeOut(BaseModel):
    decision: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_TICKET"]
    urgencia: Literal["BAJA", "MEDIANA", "ALTA"]

# Prompt para instruir al Agente de Triaje en su toma de decisiones
PROMPT_TRIAJE = """
Eres un Agente de Triaje del Service Desk. Analiza el mensaje del usuario y devuelve SOLO un objeto JSON con `decision` y `urgencia`.
Reglas de decisión (`decision`):
- "AUTO_RESOLVER": Consultas claras, políticas, FAQs o procedimientos estándar.
- "PEDIR_INFO": Mensajes ambiguos, imprecisos o que requieren contexto adicional para procesarse.
- "ABRIR_TICKET": Solicitudes de excepciones, aprobaciones de acceso, autorizaciones o situaciones que requieren intervención humana especializada.
Reglas de urgencia (`urgencia`):
- "BAJA": Consultas generales, información no crítica, sin impacto en la productividad.
- "MEDIANA": Problemas con impacto parcial o que requieren atención en el día, pero sin bloqueo total.
- "ALTA": Bloqueo total de trabajo, incidentes de seguridad, impacto crítico en negocio o plazos de tiempo muy ajustados.
Instrucciones:
1. No incluyas texto extra, solo el JSON.
2. Prioriza precisión en la clasificación.
"""

# Vinculamos el LLM con Pydantic para garantizar una respuesta de tipo JSON
chain_de_triaje = llm.with_structured_output(TriajeOut)

def ejecutar_triaje(mensaje: str) -> Dict:
    salida: TriajeOut = chain_de_triaje.invoke([
        SystemMessage(content=PROMPT_TRIAJE),
        HumanMessage(content=mensaje)
    ])
    return salida.model_dump()

# --- 3. LÓGICA DE RAG (RECUPERACIÓN EN BASE DE DATOS VECTORIAL) ---

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

# 3. Inicializar el motor de búsqueda vectorial (Solo si hay documentos)
retriever = None
if docs:
    print(f"Total de documentos cargados en memoria: {len(docs)}")
    
    # Segmentamos los textos
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs_splits = splitter.split_documents(docs)

    # Creamos los embeddings
    modelo_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    
    # Creamos la base de datos vectorial
    vectorstore = FAISS.from_documents(docs_splits, modelo_embeddings)

    # Configuramos el recuperador (retriever)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 3}
    )

# 4. Configurar la Cadena RAG de LangChain
prompt_rag = ChatPromptTemplate([
    ("system",
     """
     Eres el especialista en RRHH de la empresa Carraro Desarrollo de Software.
     Responde siempre utilizando conocimientos de las bases de datos pasadas a ti.
     Si no hay informacion sobre la pregunta en los datos, responde solo "No lo sé"
     """),
    ("human", "Contexto : {context}. \nPregunta del empleado: {input}")
])

# Importante: Usamos la variable 'llm' que definimos en la sección 1
document_chain = create_stuff_documents_chain(llm, prompt=prompt_rag)

def busqueda_respuesta_RAG(pregunta: str) -> Dict:
    """Función que ejecuta el motor de búsqueda vectorial y formula la respuesta."""
    
    # Si no hay retriever (ej. carpeta vacía), no podemos buscar
    if not retriever:
        return {"respuesta": "No lo sé", "citaciones": [], "documentos_encontrados": False}

    # Buscar fragmentos relevantes en FAISS
    documentos_relacionados = retriever.invoke(pregunta)

    if not documentos_relacionados:
        return {
            "respuesta": "No lo sé",
            "citaciones": [],
            "documentos_encontrados": False
        }

    # Enviar los fragmentos encontrados a Gemini para que redacte la respuesta
    answer = document_chain.invoke({
        "input": pregunta,
        "context": documentos_relacionados
    })
    
    # Si la IA determina que la información no es suficiente, devuelve "No lo sé"
    if "no lo sé" in answer.lower():
        return {
            "respuesta": "No lo sé",
            "citaciones": [],
            "documentos_encontrados": False
        }

    # Retorno exitoso
    return {
        "respuesta": answer,
        "citaciones": documentos_relacionados,
        "documentos_encontrados": True
    }


# --- 4. DEFINICIÓN DEL GRAFO (LANGGRAPH) ---
# El 'AgentState' es la memoria compartida del grafo. Todos los nodos leen de aquí y escriben aquí.
class AgentState(TypedDict):
    pregunta: str
    triaje: Dict
    respuesta_RAG: Optional[str]
    citaciones: Optional[List]
    rag_exito: bool
    accion_final: str

# --- DEFINICIÓN DE NODOS (FUNCIONES DE ACCIÓN) ---

def nodo_triaje(state: AgentState) -> AgentState:
    print("Ejecutando nodo 'triaje'...")
    return {"triaje": ejecutar_triaje(state["pregunta"])}

def nodo_auto_resolver(state: AgentState) -> AgentState:
    print("Ejecutando nodo 'auto_resolver'...")
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
    print("Ejecutando nodo 'pedir_info'...")
    return {"respuesta_RAG": "Necesito más informacion sobre tu solicitud.", "citaciones": [], "accion_final": "PEDIR_INFO"}

def nodo_abrir_ticket(state: AgentState) -> AgentState:
    print("Ejecutando nodo 'abrir_ticket'...")
    return {"respuesta_RAG": f"Se ha abierto un ticket para: {state['pregunta']}", "citaciones": [], "accion_final": "ABRIR_TICKET"}

# --- DEFINICIÓN DE ARISTAS CONDICIONALES (CEREBRO DE ENRUTAMIENTO) ---
def arista_decision_triaje(state: AgentState) -> str:
    print("Ejecutando arista 'decision_triaje'...")
    decision = state["triaje"]["decision"]
    if decision == "AUTO_RESOLVER": return "rag"
    elif decision == "PEDIR_INFO": return "info"
    elif decision == "ABRIR_TICKET": return "ticket"
    raise ValueError(f"Decision no valida: {decision}")

def arista_decision_rag(state: AgentState) -> str:
    print("Ejecutando arista 'decision_rag'...")
   
   # 1. Si el RAG encontró la respuesta en las políticas, el flujo finaliza felizmente
   if state["rag_exito"]:
        print("RAG con éxito, finalizando flujo")
        return "ok"

    # 2. Si el RAG falló, pero el usuario usó palabras críticas, abrimos un ticket directamente
    KEYWORDS_ABRIR_TICKET = ["aprobación", "aprobar", "excepción", "liberación", "autorización", "abrir ticket", "acceso especial"]
    if any(keyword in state["pregunta"].lower() for keyword in KEYWORDS_ABRIR_TICKET):
        print("RAG falló, pero requiere ticket.")
        return "ticket"
    # 3. Si falló y es una pregunta ambigua o general, le pedimos información extra
    else:
        print("RAG falló, pedir información.")
        return "info"

# --- CONSTRUCCIÓN DEL GRAFO ---
# Inicializamos la estructura del grafo pasándole nuestro esquema de estado
workflow = StateGraph(AgentState)

# Agregamos los nodos de acción
workflow.add_node("triaje", nodo_triaje)
workflow.add_node("auto_resolver", nodo_auto_resolver)
workflow.add_node("pedir_info", nodo_pedir_info)
workflow.add_node("abrir_ticket", nodo_abrir_ticket)

# Definimos el punto de entrada del flujo
workflow.add_edge(START, "triaje")

# Conectamos el triaje con los nodos correspondientes según la decisión de la arista
workflow.add_conditional_edges("triaje",
                               arista_decision_triaje,
                               {
                                   "rag": "auto_resolver",
                                   "info": "pedir_info",
                                   "ticket": "abrir_ticket"
                               }
                              )
# Conectamos el RAG con las alternativas de escape (escalado) o cierre si tuvo éxito
workflow.add_conditional_edges("auto_resolver",
                               arista_decision_rag,
                               {
                                   "info": "pedir_info",
                                   "ticket": "abrir_ticket",
                                   "ok": END
                               }
                              )

# Compilamos el grafo para que esté listo para ejecutarse
grafo = workflow.compile()


# --- 5. EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    # 1. Guardar la imagen del grafo localmente (reemplaza IPython.display)
    try:
        graph_bytes = grafo.get_graph().draw_mermaid_png()
        with open("flujo_agente.png", "wb") as f:
            f.write(graph_bytes)
        print("✅ Imagen del grafo guardada como 'flujo_agente.png' en esta carpeta.\n")
    except Exception as e:
        print(f"⚠️ No se pudo renderizar la imagen del grafo (requiere dependencias adicionales de Mermaid): {e}\n")

    # 2. Pruebas
    mensajes_de_prueba = [
        "¿Puedo solicitar mi reembolso por usar mi internet en home office?",
        "Mi ordenador no funciona bien. Necesito ayuda.",
        "Necesito que me aprueben una excepción de seguridad para instalar un programa."
    ]

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

