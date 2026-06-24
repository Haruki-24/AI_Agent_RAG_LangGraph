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

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

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
class TriajeOut(BaseModel):
    decision: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_TICKET"]
    urgencia: Literal["BAJA", "MEDIANA", "ALTA"]

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
chain_de_triaje = llm.with_structured_output(TriajeOut)

def ejecutar_triaje(mensaje: str) -> Dict:
    salida: TriajeOut = chain_de_triaje.invoke([
        SystemMessage(content=PROMPT_TRIAJE),
        HumanMessage(content=mensaje)
    ])
    return salida.model_dump()


# --- 3. LÓGICA DE RAG (RECUPERACIÓN DE INFORMACIÓN) ---

docs = []

for n in Path("/content/drive/MyDrive/Colab Notebooks/RAG Agentes IA/Docs/").glob("*.pdf"):
    try:
        loader = PyMuPDFLoader(str(n))
        docs.extend(loader.load())
        print(f"Archivo cargado: {n.name}")
    except Exception as e:
        print(f"Error cargando archivo: {n.name}: {e}")

print(f"Total de documentos cargados: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
docs_splits = splitter.split_documents(docs)

modelo_embedings = GoogleGenerativeAI(
    model = 'models/gemini-embeding-001',
    google_api_key = GEMINI_API_KEY,
)

from langchain_community.vectorstores import FAISS
from google.colab import userdata
from langchain_google_genai import GoogleGenerativeAIEmbeddings


modelo_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)
vectorstore = FAISS.from_documents(docs_splits, modelo_embeddings)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.3, "k": 3}
)


prompt_rag = ChatPromptTemplate(
    [
        ("system",
         """
        Eres el especialista en RRHH de al empresa Carraro Desarrollo de Software.
        Responde siempre utilizando conocimientos  de las vbases de datos passadas a ti.
        Si no hat informacion sobre la pregunta en los datos, responde solo "no lo sé"
        """),
        ("human","Contexto : {context}. \nPregunta del empleado:{input}")
    ]
)

document_chain = create_stuff_documents_chain(llm_gemini, prompt=prompt_rag)

def busqueda_respuesta_RAG(pregunta) -> Dict:
    documentos_relacionados = retriever.invoke(pregunta)

    if not documentos_relacionados:
        return {
            "respuesta": "No lo sé",
            "citaciones": [],
            "documentos_encontrados": False
        }

    answer = document_chain.invoke({
            "input": pregunta,
            "context": documentos_relacionados
            })
    if answer.rstrip(".!?") == "No lo sé":
        return {
            "respuesta": "No lo sé",
            "citaciones": [],
            "documentos_encontrados": False
        }

    return {
    "respuesta": answer,
    "citaciones": documentos_relacionados,
    "documentos_encontrados": True
    }


Definicion de estructura de una funcion

def busqueda_respuesta_RAG(pregunta) -> Dict:
  {
      "respuesta": str,
      "fuente": [],
      "doc_encontrados": bool
  }

# Agente con LangGraph"""

!pip install langgraph

from typing import TypedDict, Optional

class AgentState(TypedDict, total = False):
    pregunta: str
    triaje: dict
    respuesta_RAG: Optional[str]
    citaciones: Optional[list]
    rag_exito: bool
    accion_final: str

def nodo_triaje(state: AgentState) -> AgentState:
    print("Ejecutando nodo 'triaje'...")
    # Extraemos el mensaje del estado
    mensaje = state["pregunta"]
    return {"triaje": triaje(mensaje)}

def nodo_auto_resolver(state: AgentState) -> AgentState:
    print("Ejecutando nodo 'auto_resolver'...")
    mensaje = state["pregunta"]
    respuesta_RAG = busqueda_respuesta_RAG(mensaje)

    update: AgentState = {
        "respuesta_RAG": respuesta_RAG["respuesta"],
        "citaciones": respuesta_RAG["citaciones"],
        "rag_exito": respuesta_RAG["documentos_encontrados"],
    }

    if respuesta_RAG["documentos_encontrados"]:
        update["accion_final"] = "AUTO_RESOLVER"

    return update

def nodo_pedir_info(state: AgentState) -> AgentState:
    print("Ejecutando nodo 'pedir_info'...")
    return {
        "respuesta_RAG": "Necesito más informacion sobre tu pedido",
        "citaciones": [],
        "accion_final": "PEDIR_INFO"
    }

def nodo_abrir_ticket(state: AgentState) -> AgentState:
    print("Ejecutando nodo 'abrir_ticket'...")
    mensaje = state["pregunta"]
    return {
        "respuesta_RAG": f"Abrir ticket para el pedido: {mensaje}",
        "citaciones": [],
        "accion_final": "ABRIR_TICKET"
    }

def arista_decision_triaje(state: AgentState) -> str:
    print("Ejecutnado 'decision_triaje'...")
    tri = state["triaje"]

    if tri["decision"] == "AUTO_RESOLVER":
        return "rag"
    elif tri["decision"] == "PEDIR_INFO":
        return "info"
    elif tri["decision"] == "ABRIR_TICKET":
        return "ticket"
    else:
        raise ValueError(f"Decision no valida: {tri['decision']}")

def arista_decision_rag(state: AgentState) -> str:
    print("Ejecutnado 'decision_auto_resolver'...")
    if state["rag_exito"]:
        print("RAG con éxito, finalizando flujo")
        return "ok"

    KEYWORDS_ABRIR_TICKET = ["aprobación", "aprobar", "excepción", "liberación", "autorización",
                           "autorizar", "abrir ticket", "acceso especial"]

    if any(keyword in state["pregunta"].lower() for keyword in KEYWORDS_ABRIR_TICKET):
      print("RAG ha fallado, pero hay palabras relacionadas con abrir ticket.")
      return "ticket"

    else:
      print("RAG ha fallado, pediré mas informacion al usuario.")
      return "info"

from langgraph.graph import START, END, StateGraph

workflow = StateGraph(AgentState)

# Añadimos los nodos de forma estándar para evitar errores de validación de esquema
workflow.add_node("triaje", nodo_triaje)
workflow.add_node("auto_resolver", nodo_auto_resolver)
workflow.add_node("pedir_info", nodo_pedir_info)
workflow.add_node("abrir_ticket", nodo_abrir_ticket)

workflow.add_edge(START, "triaje")

workflow.add_conditional_edges("triaje", arista_decision_triaje, {
    "rag": "auto_resolver",
    "info": "pedir_info",
    "ticket": "abrir_ticket"
})

workflow.add_conditional_edges("auto_resolver", arista_decision_rag, {
    "info": "pedir_info",
    "ticket": "abrir_ticket",
    "ok": END,
})

grafo = workflow.compile()

from IPython.display import display, Image

graph_bytes = grafo.get_graph().draw_mermaid_png()
display(Image(graph_bytes))



PREGUNTA = "¿Puedo solicitar mi reembolso por usar mi internet en home office?"

temp = grafo.invoke({"pregunta": PREGUNTA})

print(f"PREGUNTA: {PREGUNTA}")
print("")
print(f"DECISION: {temp['triaje']['decision']} | URGENCIA: {temp['triaje']['urgencia']} | ACCIÓN FINAL: {respuesta_RAG['accion_final']}")
print(f"RESPUESTA: {respuesta_RAG['respuesta']}")

if temp['citaciones']:
    for i, citacion in enumerate(temp['citaciones']):
        print(f"  - CITACION {i + 1}:")
        print(f"    Camino del documento: {citacion.metadata['file_path']}")
        print(f"    Contenido: {citacion.page_content.replace('\n', ' ')}")

for prueba in mensajes_de_prueba:
    respuesta = grafo.invoke({"pregunta": prueba})
    print('\n')
    print(f"PREGUNTA: {prueba}")
    print(f"DECISION DE TRIAJE: {respuesta['triaje']['decision']} | URGENCIA: {respuesta['triaje']['urgencia']} | ACCIÓN FINAL: {respuesta['accion_final']}")
    print(f"RESPUESTA: {respuesta['respuesta']}")
    if respuesta['citaciones']:
        for i, citacion in enumerate(respuesta['citaciones']):
            print(f"  - CITACION {i + 1}:")
            print(f"    Camino del documento: {citacion.metadata['file_path']}")
            print(f"    Contenido: {citacion.page_content.replace('\n', ' ')}")
    print("--------------------")

