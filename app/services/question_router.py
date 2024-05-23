from app.promt_templates.templates import temp_llama_question_router
from app.services.llm import llm

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser


def get_question_router():
    prompt = temp_llama_question_router
    chain = prompt | llm | JsonOutputParser()

    return chain
