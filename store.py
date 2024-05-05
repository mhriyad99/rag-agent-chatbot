from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_elasticsearch import ElasticsearchStore

from resources import client, embeddings, ELASTIC_CLOUD_ID, ELASTIC_API_KEY

vector_store = ElasticsearchStore(
    es_connection=client,
    index_name="softbd-context",
    embedding=embeddings,
)


def _metadata_func(record: dict, metadata: dict) -> dict:
    metadata["name"] = record.get("name")
    metadata["summary"] = record.get("summary")
    metadata["url"] = record.get("url")
    metadata["category"] = record.get("category")
    metadata["updated_at"] = record.get("updated_at")

    return metadata


def load_json_context(file):
    loader = JSONLoader(
        file_path=file,
        jq_schema=".[]",
        content_key="content",
        metadata_func=_metadata_func,
    )

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=256
    )
    docs = loader.load_and_split(text_splitter=text_splitter)
    _store_document(docs)


# chat_history = ElasticsearchChatMessageHistory(
#     es_connection=client,
#     session_id=session_id,
#     index="workplace-docs-chat-history",
# )

def _store_document(docs):
    vector_store.from_documents(
        docs,
        embeddings,
        index_name="softbd-context",
        es_cloud_id=ELASTIC_CLOUD_ID,
        es_api_key=ELASTIC_API_KEY,
    )


retriever = vector_store.as_retriever()
