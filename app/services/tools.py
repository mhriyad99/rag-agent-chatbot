from app.schemas.schemas import GraphState

from langchain_core.retrievers import RetrieverLike


def retrieve(state:GraphState, retriever: RetrieverLike):
    question = state["question"]
    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}


def generator(state: GraphState):
    pass

