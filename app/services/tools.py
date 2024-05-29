from app.schemas.schemas import GraphState
from app.core.settings import COLLECTION_SCOPE
from app.services.question_router import get_question_router, get_question_rephraser
from app.services.generator import get_response_generator
from app.services.vectorstore import DocVectorStore
from app.services.global_state import history

from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults


def question_rephraser(state: GraphState):
    question = state["question"]

    if len(history) > 0:
        chat_history = history
        chain = get_question_rephraser()
        rephrased_question = chain.invoke({"chat_history": chat_history, "question": question})
        question = rephrased_question["output"]
        return {"question": question, "chat_history": chat_history}

    return {"question": question}


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


def retrieve(state: GraphState):
    question = state["question"]
    retriever = DocVectorStore.load_vector_store()
    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}


def web_search(state: GraphState):
    question = state["question"]
    documents = state["documents"]

    search = TavilySearchResults(max_results=3)
    web_text = search.invoke({"query": question})
    web_text = "\n".join([d["content"] for d in web_text])
    web_results = Document(page_content=web_text)

    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"question": question, "documents": documents}


# noinspection PyTypeChecker
def generate(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    iteration = state["iteration"]

    if iteration is None:
        iteration = 1
    else:
        iteration += 1

    rag_chain = get_response_generator()
    context = [d.page_content for d in documents]
    generation = rag_chain.invoke({"question": question, "context": context})

    return {"question": question, "documents": documents, "generation": generation,
            "iteration": iteration}


def decide_to_generate(state: GraphState):
    question = state["question"]
    search = state["web_search"]
    documents = state["documents"]

    if search == "Yes":
        return "websearch"
    else:
        return "generate"


def store_history(state: GraphState):
    question = state["question"]
    generation = state["generation"]
    chat_history = state["chat_history"]
    print(chat_history)
    human = "Human: " + question + "\n"
    ai = "AI Assistant: " + generation + "\n"
    if not chat_history:
        chat_history = [human, ai]
    else:
        chat_history.append(human)
        chat_history.append(ai)
    print("chat_history", chat_history)
    return {"chat_history": chat_history}
