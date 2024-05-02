import json
from pprint import pprint
import os

import urllib3
from elasticsearch import Elasticsearch
from langchain_community.embeddings import OllamaEmbeddings
from app.core.settings import DOC_PATH, MODEL
# from sentence_transformers import SentenceTransformer #"all-MiniLM-L6-v2"


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class Search:
    def __init__(self):
        self.model = OllamaEmbeddings(model=MODEL)
        self.es = Elasticsearch("https://192.168.13.247:9200/",
                                basic_auth=("elastic", "atkg49qNOB-0mvKyowqQ"),
                                ca_certs="./http_ca.crt")
        client_info = self.es.info()
        print("Connected to Elasticsearch!")
        pprint(client_info.body)

    def create_index(self):
        self.es.indices.delete(index="aws_book", ignore_unavailable=True)
        self.es.indices.create(
            index="context",
            mappings={
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                    }
                }
            },
        )

    def get_embedding(self, text):
        return self.model.encode(text)

    def insert_document(self, document):
        return self.es.index(
            index="context",
            document={
                **document,
                "embedding": self.get_embedding(document["description"]),
            },
        )

    def insert_document_json(self, documents):
        for document in documents:
            self.insert_document(document)
        print("done")

    def insert_documents(self, documents):
        operations = []
        for document in documents:
            operations.append({"index": {"_index": "context"}})
            operations.append(
                {
                    **document,
                    "embedding": self.get_embedding(document["description"]),
                }
            )
        return self.es.bulk(index="context", operations=operations, refresh=True)

    def reindex(self):
        self.create_index()
        with open("data.json", "rt") as f:
            documents = json.loads(f.read())
        return self.insert_documents(documents)

    def reindex_json(self):
        self.create_index()
        with open("data1.json", "rt") as f:
            documents = json.loads(f.read())
        return self.insert_document_json(documents)

    def search(self, **query_args):
        return self.es.search(index="context", **query_args)

    def retrieve_document(self, id):
        return self.es.get(index="context", id=id)
