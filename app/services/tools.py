from langchain_core.documents import Document

from app.schemas.schemas import GraphState
from app.core.settings import COLLECTION_SCOPE

from langchain_core.retrievers import RetrieverLike
from langchain_community.tools.tavily_search import TavilySearchResults


# noinspection PyTypeChecker
def route_question(state: GraphState, router: RetrieverLike):
    if not COLLECTION_SCOPE:
        return "web_search"

    question = state["question"]
    source = router.invoke({"question": question})

    if source["datasource"] == "vectorstore":
        return "vectorstore"
    else:
        return "web_search"


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
def generator(state: GraphState, rag_chain: RetrieverLike):
    question = state["question"]
    documents = state["documents"]

    context = [d.page_content for d in documents]
    generation = rag_chain.invoke({"question": question, "context": context})

    return {"question": question, "context": context, "generation": generation}




