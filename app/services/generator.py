from app.promt_templates.templates import temp_llama_generation
from app.core.settings import MODEL

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser


def get_response_generator():
    prompt = temp_llama_generation
    llm = ChatOllama(model=MODEL, temperature=0)
    chain = prompt | llm | StrOutputParser()

    return chain

