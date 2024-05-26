from app.core.settings import MODEL, BASE_URL
from langchain_community.chat_models import ChatOllama


if BASE_URL:
    llm = ChatOllama(model=MODEL, fomat="json", base_url=BASE_URL, temperature=0)
    gen_llm = ChatOllama(model=MODEL, base_url=BASE_URL, temperature=0)
else:
    llm = ChatOllama(model=MODEL, fomat="json", temperature=0)
    gen_llm = ChatOllama(model=MODEL, temperature=0)
