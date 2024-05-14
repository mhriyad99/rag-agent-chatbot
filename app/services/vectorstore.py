from typing import Union, Callable, Any

from app.core.settings import DOC_PATH, MODEL
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


class DocVectorStore:
    vector_store: Union[Callable[..., Chroma], None] = None

    @classmethod
    def load_store(cls, path=DOC_PATH, chunk_size=500, chunk_overlap=50) -> Chroma:
        embedding = OllamaEmbeddings(model=MODEL)
        if cls.vector_store is None:
            cls.vector_store = Chroma(persist_directory="./chroma",
                                      embedding_function=embedding)

    @classmethod
    def create_store(cls, path=DOC_PATH, chunk_size=500, chunk_overlap=50, embedding_model=MODEL):
        loader = TextLoader(file_path=path)
        text = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(text)
        embeddings = OllamaEmbeddings(model=embedding_model)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        cls.vector_store = vectorstore


