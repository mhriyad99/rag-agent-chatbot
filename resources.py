from elasticsearch import Elasticsearch
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama

ELASTIC_API_KEY = "T0w3blBJOEJVcjdaV3hZT3BtM1A6bXg1QjFjb1VSWmFEVGgycF9SMnlIZw=="
ELASTIC_CLOUD_ID = "test_rag:YXNpYS1zb3V0aDEuZ2NwLmVsYXN0aWMtY2xvdWQuY29tOjQ0MyQwMzQ1MmQ3MzY4YWQ0Nzk1OTJmNGM0NTgwODY4YzkyYSQ0ZDhlYmJmMzVkOTA0NzIxYjRjOGQxNDdlZTU2N2YyMw=="
MODEL = "llama3"
BASE_URL = "http://35.209.146.25"

embeddings = OllamaEmbeddings(model='llama3', base_url=BASE_URL)
llm = Ollama(model="llama3", base_url=BASE_URL, stop=['<|eot_id|>'])
client = Elasticsearch(cloud_id=ELASTIC_CLOUD_ID,
                       api_key=ELASTIC_API_KEY)
