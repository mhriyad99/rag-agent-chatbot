from app.schemas.schemas import GraphState
from app.core.settings import COLLECTION_SCOPE
from app.services.question_router import get_question_router
from app.services.generator import get_response_generator

from langchain_core.documents import Document
from langchain_core.retrievers import RetrieverLike
from langchain_community.tools.tavily_search import TavilySearchResults


# noinspection PyTypeChecker
def route_question(state: GraphState):
    if not COLLECTION_SCOPE:
        return "websearch"

    router = get_question_router()

    question = state["question"]
    source = router.invoke({"question": question})

    if source["datasource"] == "vectorstore":
        return "vectorstore"
    else:
        return "websearch"


def retrieve(state: GraphState, retriever: RetrieverLike):
    question = state["question"]
    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}


def web_search(state: GraphState):
    question = state["question"]
    documents = state["documents"]

    search = TavilySearchResults(k=3)
    web_text = search.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in web_text])
    web_results = Document(page_content=web_results)

    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"question": question, "documents": documents}


# noinspection PyTypeChecker
def generate(state: GraphState):
    question = state["question"]
    documents = state["documents"]

    rag_chain = get_response_generator()

    context = [d.page_content for d in documents]
    generation = rag_chain.invoke({"question": question, "context": context})

    return {"question": question, "documents": context, "generation": generation}


def decide_to_generate(state: GraphState):
    question = state["question"]
    search = state["websearch"]
    documents = state["documents"]

    if search == "Yes":
        return "websearch"
    else:
        return "generate"







