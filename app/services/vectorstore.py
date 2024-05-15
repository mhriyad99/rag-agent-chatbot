from langchain_core.retrievers import RetrieverLike

from app.core.settings import DOC_PATH, MODEL, VECTOR_DB_PATH
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


class DocVectorStore:

    @staticmethod
    def load_vector_store(path=VECTOR_DB_PATH) -> RetrieverLike:
        embedding = OllamaEmbeddings(model=MODEL)
        vector_store = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding)

        return vector_store.as_retriever()

    @staticmethod
    def build_vector_store(path=DOC_PATH, chunk_size=500, chunk_overlap=50,
                           embedding_model=MODEL):
        loader = TextLoader(file_path=path)
        text = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(text)
        embeddings = OllamaEmbeddings(model=embedding_model)
        Chroma.from_documents(documents=splits, embedding=embeddings,
                              persist_directory=VECTOR_DB_PATH)
