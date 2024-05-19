from typing import List

from fastapi import UploadFile
from pydantic import BaseModel


class Query(BaseModel):
    query: str


class GraphState(BaseModel):
    """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            web_search: whether to add search
            documents: list of documents
    """
    question: str
    generation: str = None
    web_search: str = None
    documents: List[str] = None
