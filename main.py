from fastapi import FastAPI
from app.schemas.schemas import Query
from app.services.services import get_response

app = FastAPI()


@app.get("/")
def hello():
    return {"message": "hello world!"}


@app.post("/query")
def search(payload: Query):
    query = payload.query
    response = get_response(query)
    return {"query": query, "response": response}