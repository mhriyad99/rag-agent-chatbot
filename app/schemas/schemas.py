from typing import List, TypedDict

from pydantic import BaseModel


class Query(BaseModel):
    query: str


class GraphState(TypedDict):
    """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            web_search: whether to add search
            documents: list of documents
    """
    question: str
    generation: str
    web_search: str
    documents: List[str]
