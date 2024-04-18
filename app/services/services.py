from langchain_community.llms import Ollama
from app.core.settings import MODEL

model = Ollama(model=MODEL)


def get_response(query):
    response = model.invoke(query)
    return response
