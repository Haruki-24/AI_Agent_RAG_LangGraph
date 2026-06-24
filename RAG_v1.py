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


def busqueda_respuesta_RAG(pregunta) -> Dict:
  {
      "respuesta": str,
      "fuente": [],
      "doc_encontrados": bool
  }

# --- 4. DEFINICIÓN DEL GRAFO (LANGGRAPH) ---
class AgentState(TypedDict):
    pregunta: str
    triaje: Dict
    respuesta_RAG: Optional[str]
    citaciones: Optional[List]
    rag_exito: bool
    accion_final: str

def nodo_triaje(state: AgentState) -> AgentState:
    print("Ejecutando nodo 'triaje'...")
    return {"triaje": ejecutar_triaje(state["pregunta"])}

def nodo_auto_resolver(state: AgentState) -> AgentState:
    print("Ejecutando nodo 'auto_resolver'...")
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
    print("Ejecutando nodo 'pedir_info'...")
    return {"respuesta_RAG": "Necesito más informacion sobre tu solicitud.", "citaciones": [], "accion_final": "PEDIR_INFO"}

def nodo_abrir_ticket(state: AgentState) -> AgentState:
    print("Ejecutando nodo 'abrir_ticket'...")
    return {"respuesta_RAG": f"Se ha abierto un ticket para: {state['pregunta']}", "citaciones": [], "accion_final": "ABRIR_TICKET"}

def arista_decision_triaje(state: AgentState) -> str:
    print("Ejecutando arista 'decision_triaje'...")
    decision = state["triaje"]["decision"]
    if decision == "AUTO_RESOLVER": return "rag"
    elif decision == "PEDIR_INFO": return "info"
    elif decision == "ABRIR_TICKET": return "ticket"
    raise ValueError(f"Decision no valida: {decision}")

def arista_decision_rag(state: AgentState) -> str:
    print("Ejecutando arista 'decision_rag'...")
    if state["rag_exito"]:
        print("RAG con éxito, finalizando flujo")
        return "ok"

    KEYWORDS_ABRIR_TICKET = ["aprobación", "aprobar", "excepción", "liberación", "autorización", "abrir ticket", "acceso especial"]
    if any(keyword in state["pregunta"].lower() for keyword in KEYWORDS_ABRIR_TICKET):
        print("RAG falló, pero requiere ticket.")
        return "ticket"
    else:
        print("RAG falló, pedir información.")
        return "info"

# --- CONSTRUCCIÓN DEL GRAFO ---
workflow = StateGraph(AgentState)

workflow.add_node("triaje", nodo_triaje)
workflow.add_node("auto_resolver", nodo_auto_resolver)
workflow.add_node("pedir_info", nodo_pedir_info)
workflow.add_node("abrir_ticket", nodo_abrir_ticket)

workflow.add_edge(START, "triaje")
workflow.add_conditional_edges("triaje", arista_decision_triaje, {"rag": "auto_resolver", "info": "pedir_info", "ticket": "abrir_ticket"})
workflow.add_conditional_edges("auto_resolver", arista_decision_rag, {"info": "pedir_info", "ticket": "abrir_ticket", "ok": END})

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

    for prueba in mensajes_de_prueba:
        print(f"=====================================")
        print(f"PREGUNTA: {prueba}")
        respuesta = grafo.invoke({"pregunta": prueba})
        
        print(f"DECISION TRIAJE: {respuesta['triaje']['decision']} | URGENCIA: {respuesta['triaje']['urgencia']}")
        print(f"ACCIÓN FINAL DEL AGENTE: {respuesta.get('accion_final', 'NO DEFINIDA')}")
        print(f"RESPUESTA FINAL: {respuesta.get('respuesta_RAG', '')}")
        
        citaciones = respuesta.get('citaciones')
        if citaciones:
            for i, citacion in enumerate(citaciones):
                # La simulación usa diccionarios, pero un Document real tiene propiedades
                doc_path = citacion['metadata']['file_path'] if isinstance(citacion, dict) else citacion.metadata['file_path']
                content = citacion['page_content'] if isinstance(citacion, dict) else citacion.page_content
                print(f"  - CITACION {i + 1}: {doc_path} -> {content}")
        print("\n")

