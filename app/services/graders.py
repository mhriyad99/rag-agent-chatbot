from app.promt_templates.templates import temp_llama_retrival_grader, \
    temp_llama_hallucination_grader , temp_llama_answer_grader
from app.core.settings import MODEL
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser


def get_retrieval_grader():
    llm = ChatOllama(model=MODEL, fomat="json", temperature=0)
    prompt = temp_llama_retrival_grader
    retrieval_grader = prompt | llm | JsonOutputParser()

    return retrieval_grader


def get_hallucination_grader():
    llm = ChatOllama(model=MODEL, format="json", temperature=0)
    prompt = temp_llama_hallucination_grader
    hallucination_grader = prompt | llm | JsonOutputParser()

    return hallucination_grader


def get_answer_grader():
    llm = ChatOllama(model=MODEL, format="json", temperature=0)
    prompt = temp_llama_answer_grader
    answer_grader = prompt | llm | JsonOutputParser()

    return answer_grader
