from typing import Annotated
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form
from app.schemas.schemas import Query
from app.services.services import get_response, summarizer, pdf_reader

app = FastAPI()


@app.get("/")
def hello():
    return {"message": "hello world!"}


@app.post("/query")
def search(payload: Query):
    query = payload.query
    response = get_response(query)
    return {"query": query, "response": response}


@app.post("/pdf_search")
def search(file: UploadFile = File(description="upload pdf file"), query: str = Form(...)):
    pdf_file = BytesIO(file.file.read())
    context = pdf_reader(pdf_file)
    question = query
    response = summarizer(context=context, question=question)
    return {"response": response}