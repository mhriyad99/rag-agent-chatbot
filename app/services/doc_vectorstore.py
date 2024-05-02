from typing import Union, Callable, Any

from app.core.settings import DOC_PATH, MODEL
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


class DocVectorStore:
    doc_store: Union[Callable[..., Chroma], None] = None

    def __init__(self, path=DOC_PATH, chunk_size=500, chunk_overlap=50, embedding_model=MODEL):
        self.path = path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model

    def create_store(self):
        loader = TextLoader(file_path=self.path)
        text = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                       chunk_overlap=self.chunk_overlap)
        splits = text_splitter.split_documents(text)
        embeddings = OllamaEmbeddings(model=self.embedding_model)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        self.docstore = vectorstore


