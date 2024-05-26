from app.promt_templates.templates import temp_llama_generation
from app.services.llm import gen_llm

from langchain_core.output_parsers import StrOutputParser


def get_response_generator():
    prompt = temp_llama_generation
    chain = prompt | gen_llm | StrOutputParser()

    return chain

