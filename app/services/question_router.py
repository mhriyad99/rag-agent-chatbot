from app.promt_templates.templates import temp_llama_question_router, temp_llama_question_rephrase
from app.services.llm import llm

from langchain_core.output_parsers import JsonOutputParser


def get_question_router():
    prompt = temp_llama_question_router
    chain = prompt | llm | JsonOutputParser()

    return chain


def get_question_rephraser():
    prompt = temp_llama_question_rephrase
    chain = prompt | llm | JsonOutputParser()

    return chain


