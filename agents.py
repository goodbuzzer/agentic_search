import functools, operator
from typing import Annotated, Sequence, TypedDict, Dict, List, Any, Optional
from datetime import datetime

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

from tools import get_tools

load_dotenv()

# llm = ChatOpenAI(
#   model="gpt-4-turbo-preview",
#   temperature=0,
#   verbose=True
# )
llm = ChatOpenAI(
  model="gpt-4.1",
  temperature=0,
  verbose=True
)
tools = get_tools()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
  prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
  ])
  agent = create_openai_tools_agent(llm, tools, prompt)
  executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

  return executor

def agent_node(state, agent, name):
  result = agent.invoke(state)
  return {"messages": [HumanMessage(content=result["output"], name=name)]}

def get_members():
  return ["Web_Searcher", "Insight_Researcher"]

def create_supervisor():
  members = get_members()
  system_prompt = (
    f"""As a supervisor, your role is to oversee a dialogue between these"
    " workers: {members}. Based on the user's request,"
    " determine which worker should take the next action. Each worker is responsible for"
    " executing a specific task and reporting back their findings and progress. Once all tasks are complete,"
    " indicate with 'FINISH'.
    """
  )
  options = ["FINISH"] + members

  function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {"next": {"title": "Next", "anyOf": [{"enum": options}] }},
        "required": ["next"],
    },
  }

  prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),
  ]).partial(options=str(options), members=", ".join(members))

  supervisor_chain = (prompt | llm.bind_functions(functions=[function_def], function_call="route") | JsonOutputFunctionsParser())

  return supervisor_chain

def create_search_agent():
  search_agent = create_agent(llm, tools, "You are a web searcher. Search the internet for information. We are in 2025")
  search_node = functools.partial(agent_node, agent=search_agent, name="Web_Searcher")

  return search_node

def create_insights_researcher_agent():
  insights_research_agent = create_agent(llm, tools,
    """You are a Insight Researcher. Do step by step.
    Based on the provided content first identify the list of topics,
    then search internet for each topic one by one
    and finally find insights for each topic one by one.
    Include the insights and sources in the final response
    """)
  insights_researcher_node = functools.partial(agent_node, agent=insights_research_agent, name="Insight_Researcher")

  return insights_researcher_node


# Mettre à jour la classe SuperscriptRetrievalChain pour utiliser des exposants Unicode
class SuperscriptRetrievalChain(ConversationalRetrievalChain):
    """Chain qui ajoute des références avec numéros en exposant aux documents."""
    
    def _format_superscript(self, num):
        """Convertit un nombre en caractères exposants Unicode."""
        superscript_map = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", 
                          "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"}
        return ''.join(superscript_map[digit] for digit in str(num))
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Obtenir la réponse standard de la classe parente
        response = super()._call(inputs)
        
        # Obtenir les documents sources utilisés dans la réponse
        source_docs = response.get("source_documents", [])
        
        # Créer les notes de bas de page
        footnotes = []
        doc_map = {}  # Map pour éviter les références en double
        
        for i, doc in enumerate(source_docs):
            source = doc.metadata.get("source", f"Document {i+1}")
            page = doc.metadata.get("page", None)
            date = doc.metadata.get("date", None)
            
            # Créer une clé unique pour chaque combinaison source+page
            source_key = f"{source}_{page}" if page else source
            
            if source_key not in doc_map:
                doc_map[source_key] = len(footnotes) + 1
                ref_num = self._format_superscript(doc_map[source_key])
                
                # Formatter la référence avec le numéro de page si disponible
                ref_text = f"[{doc_map[source_key]}] {source}"
                if page:
                    ref_text += f", page {page}"
                if date:
                    ref_text += f", {date}"
                    
                footnotes.append(ref_text)
        
        # Ajouter les notes de bas de page à la réponse
        answer = response["answer"]
        if footnotes:
            answer += "\n\n**Références:**\n" + "\n".join(footnotes)
        
        response["answer"] = answer
        return response


def load_data(path):
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()
    return documents

load_dotenv()

NEWS_SOURCES = {
    "Ecofin": "https://www.agenceecofin.com/",
    "Sika Finance": "https://www.sikafinance.com",
    "Seneweb": "https://www.seneweb.com",
    "SeneNews": "https://www.senenews.com",
    "PressAfrik": "https://www.pressafrik.com",
    "Senego": "https://www.senego.com",
    "SenePlus": "https://www.seneplus.com",
    "Studio Tamani": "https://www.studiotamani.org",
    "Financial Afrik": "https://www.financialafrik.com"
}

def get_tavily_search_tool(start_date=None, end_date=None):
    """Crée un outil de recherche Tavily avec filtrage par date si spécifié."""
    
    # Formater les dates au format YYYY-MM-DD si elles sont fournies
    date_filters = {}
    if start_date:
        if isinstance(start_date, str):
            date_filters["start_date"] = start_date
        else:
            date_filters["start_date"] = start_date.strftime("%Y-%m-%d")
            
    if end_date:
        if isinstance(end_date, str):
            date_filters["end_date"] = end_date
        else:
            date_filters["end_date"] = end_date.strftime("%Y-%m-%d")
    
    # Créer l'outil de recherche Tavily
    search_tool = TavilySearchResults(
        max_results=5,
        include_domains=list(NEWS_SOURCES.values()),
        **date_filters
    )
    
    return Tool(
        name="Web Search",
        description="Search the web for up-to-date information",
        func=search_tool.invoke
    )

def get_web_documents(sources, start_date=None, end_date=None):
    """Charge le contenu des pages web sélectionnées en utilisant Tavily."""
    urls = [NEWS_SOURCES[source] for source in sources if source in NEWS_SOURCES]
    if not urls:
        return []
    
    try:
        search_tool = get_tavily_search_tool(start_date, end_date)
        
        # Effectuer une recherche pour chaque source
        all_docs = []
        for source in sources:
            if source in NEWS_SOURCES:
                # Rechercher des nouvelles récentes sur cette source
                query = f"recent news from {source}"
                results = search_tool.func(query)
                
                # Convertir les résultats en documents avec métadonnées
                for result in results:
                    content = f"Titre: {result.get('title', 'Sans titre')}\n\n{result.get('content', '')}"
                    all_docs.append({
                        "page_content": content,
                        "metadata": {
                            "source": result.get('url', source),
                            "date": result.get('published_date', None),
                        }
                    })
        return all_docs
    except Exception as e:
        print(f"Erreur lors du chargement des pages web: {e}")
        return []


def get_pdf_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        documents = []
        source_name = pdf_path.name if hasattr(pdf_path, 'name') else str(pdf_path)
        
        # Extraire le texte page par page avec le numéro de page
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():  # Ignorer les pages vides
                documents.append({
                    "page_content": text,
                    "metadata": {
                        "source": source_name,
                        "page": i + 1  # Numéroter les pages à partir de 1
                    }
                })
        
        # Si aucune page n'a été extraite, renvoyer un document vide
        if not documents:
            return {"page_content": "", "metadata": {"source": source_name, "page": 0}}
        
        # Si une seule page, renvoyer directement le document
        if len(documents) == 1:
            return documents[0]
        
        # Sinon, renvoyer la liste des documents
        return documents
    except Exception as e:
        print(f"Erreur lors de la lecture du PDF {pdf_path}: {e}")
        return {"page_content": "", "metadata": {"source": "Fichier non lisible", "page": 0}}


def get_text_chunks(documents):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    all_chunks = []
    
    # Si documents est une liste de documents déjà formatés
    if isinstance(documents, list) and all(isinstance(doc, dict) for doc in documents):
        for doc in documents:
            chunks = text_splitter.split_text(doc["page_content"])
            for chunk in chunks:
                all_chunks.append({
                    "page_content": chunk,
                    "metadata": doc["metadata"]
                })
    # Si documents est un seul document formaté
    elif isinstance(documents, dict) and "page_content" in documents:
        chunks = text_splitter.split_text(documents["page_content"])
        for chunk in chunks:
            all_chunks.append({
                "page_content": chunk,
                "metadata": documents["metadata"]
            })
    # Si documents est un texte brut
    elif isinstance(documents, str):
        chunks = text_splitter.split_text(documents)
        for chunk in chunks:
            all_chunks.append({
                "page_content": chunk,
                "metadata": {"source": "Texte inconnu", "page": 0}
            })
    
    return all_chunks


def get_vectorstore(chunks):
    if not chunks:
        return None
        
    try:
        embeddings = OpenAIEmbeddings()
        texts = [chunk["page_content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        return vectorstore
    except Exception as e:
        print(f"Erreur lors de la création du vectorstore: {e}")
        return None


class SuperscriptRetrievalChain(ConversationalRetrievalChain):
    """Chain qui ajoute des références avec numéros en exposant aux documents."""
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Obtenir la réponse standard de la classe parente
        response = super()._call(inputs)
        
        # Obtenir les documents sources utilisés dans la réponse
        source_docs = response.get("source_documents", [])
        
        # Créer les notes de bas de page
        footnotes = []
        doc_map = {}  # Map pour éviter les références en double
        
        for i, doc in enumerate(source_docs):
            source = doc.metadata.get("source", f"Document {i+1}")
            page = doc.metadata.get("page", None)
            
            # Créer une clé unique pour chaque combinaison source+page
            source_key = f"{source}_{page}" if page else source
            
            if source_key not in doc_map:
                doc_map[source_key] = len(footnotes) + 1
                
                # Formatter la référence avec le numéro de page si disponible
                if page:
                    footnotes.append(f"[{doc_map[source_key]}] {source}, page {page}")
                else:
                    footnotes.append(f"[{doc_map[source_key]}] {source}")
            
            # Ajouter le numéro de référence au texte (à implémenter si nécessaire)
        
        # Ajouter les notes de bas de page à la réponse
        answer = response["answer"]
        if footnotes:
            answer += "\n\n**Références:**\n" + "\n".join(footnotes)
        
        response["answer"] = answer
        return response


def get_conversation_chain(vectorstore):
    if not vectorstore:
        return None
        
    try:
        # Instructions système pour guider l'IA sur la façon de répondre
        system_template = """
        Vous êtes un assistant IA spécialisé en veille stratégique et analyse documentaire.
        
        Pour chaque question, suivez ces instructions :
        1. Analysez soigneusement la question de l'utilisateur
        2. Recherchez les informations pertinentes dans les documents fournis
        3. Structurez votre réponse de façon claire et concise
        4. Citez précisément les sources utilisées
        5. Si les documents ne contiennent pas l'information demandée, indiquez-le clairement
        
        Important :
        - Basez vos réponses UNIQUEMENT sur le contenu des documents fournis
        - N'inventez pas d'informations qui ne sont pas dans les documents
        - Utilisez des formulations comme "Selon le document X..." ou "D'après les données disponibles..."
        - Présentez les informations de manière objective et factuelle
        - Si pertinent, indiquez les limites des informations disponibles
        
        Votre réponse sera automatiquement complétée avec les références précises des documents utilisés.
        
        Question: {question}
        
        Chat History: {chat_history}
        
        Documents pertinents:
        {context}
        
        Réponse:
        """
        
        SYSTEM_PROMPT = PromptTemplate(
            input_variables=["question", "chat_history", "context"],
            template=system_template
        )
        
        llm = ChatOpenAI(temperature=0.2)  # Température basse pour des réponses plus précises
        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'  # Spécifie quelle clé stocker dans la mémoire
        )
        
        conversation_chain = SuperscriptRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={
                    "k": 5,  # Récupère les 5 documents les plus pertinents
                    "score_threshold": 0.5  # Ne considère que les documents avec un score > 0.5
                }
            ),
            memory=memory,
            return_source_documents=True,  # Important pour obtenir les documents sources
            verbose=True,
            combine_docs_chain_kwargs={"prompt": SYSTEM_PROMPT}  # Inclut les instructions système
        )
        
        return conversation_chain
    except Exception as e:
        print(f"Erreur lors de la création de la chaîne de conversation: {e}")
        return None