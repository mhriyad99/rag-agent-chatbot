from io import BytesIO
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
import pypdf
from app.core.settings import MODEL
from app.promt_template.template import template
from app.services.doc_vectorstore import DocVectorStore
from app.services.utils import combine_docs

model = Ollama(model=MODEL, stop=['<|eot_id|>'])


def get_response(query):
    response = model.invoke(query)
    return response


def pdf_reader(pdf_file):
    reader = pypdf.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text


def summarizer(context, question):
    prompt = PromptTemplate(input_variables=['context', 'question'], template=template)
    prompt.format(context="Here is the context", question="Here is a question")
    chain = LLMChain(llm=model, prompt=prompt)
    response = chain.invoke({"context": context, "question":
        question})
    return response


def ielts_qa(question):
    retriever = DocVectorStore.vector_store.as_retriever()
    retrieved_docs = retriever.invoke(question)
    context = combine_docs(retrieved_docs)
    prompt = PromptTemplate(input_variables=['context', 'question'], template=template)
    prompt.format(context="Here is the context", question="Here is a question")
    chain = LLMChain(llm=model, prompt=prompt)
    response = chain.invoke({"context": context, "question":
        question})
    return response
