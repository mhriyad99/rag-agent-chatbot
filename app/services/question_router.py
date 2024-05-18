from app.promt_templates.templates import temp_llama_question_router
from app.core.settings import MODEL

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser


def get_question_router():
    llm = ChatOllama(model=MODEL, fomat="json", temperature=0)
    prompt = temp_llama_question_router
    chain = prompt | llm | JsonOutputParser()

    return chain
