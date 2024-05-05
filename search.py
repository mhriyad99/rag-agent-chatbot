from uuid import uuid4

from elasticsearch import Elasticsearch
from langchain import hub
from langchain.memory import ElasticsearchChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_elasticsearch import ElasticsearchStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = OllamaEmbeddings(model='llama3', base_url="http://35.209.146.25")

ELASTIC_API_KEY = "T0w3blBJOEJVcjdaV3hZT3BtM1A6bXg1QjFjb1VSWmFEVGgycF9SMnlIZw=="

client = Elasticsearch(cloud_id="test_rag:YXNpYS1zb3V0aDEuZ2NwLmVsYXN0aWMtY2xvdWQuY29tOjQ0MyQwMzQ1MmQ3MzY4YWQ0Nzk1OTJmNGM0NTgwODY4YzkyYSQ0ZDhlYmJmMzVkOTA0NzIxYjRjOGQxNDdlZTU2N2YyMw==",
                       api_key=ELASTIC_API_KEY)


llm = Ollama(model="llama3", base_url="http://35.209.146.25", stop=['<|eot_id|>'])

session_id = str(uuid4())
chat_history = ElasticsearchChatMessageHistory(
    es_connection=client,
    session_id=session_id,
    index="workplace-docs-chat-history",
)

loader = TextLoader(file_path='./sbd.txt')
text = loader.load()

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
# vector_store = ElasticsearchStore.from_documents(
vector_store = ElasticsearchStore(
    es_connection=client,
    index_name="workplace-docs",
    embedding=embeddings,
)

retriever = vector_store.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

async def chat(msg, func=None):
    print(msg)
    output = {}
    curr_key = None
    for chunk in rag_chain_with_source.stream(msg):
        for key in chunk:
            print(key)
            if key not in output:
                output[key] = chunk[key]
            else:
                output[key] += chunk[key]
            if  key == 'answer':
                if key != curr_key:
                    await func(chunk[key])
                else:
                    await func(chunk[key])
            curr_key = key
    await func("", active=False)
