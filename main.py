import json
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from store import vector_store, load_json_context
from app.schemas.schemas import Query
from app.services.services import get_response, summarizer, pdf_reader, ielts_qa
from QA import chat
import tempfile

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.post("/query")
def search(payload: Query):
    query = payload.query
    response = get_response(query)
    return {"query": query, "response": response}


@app.post("/ielts-query")
def search(payload: Query):
    query = payload.query
    response = ielts_qa(query)
    return {"query": query, "response": response}


@app.post("/pdf_search")
def search(file: UploadFile = File(description="upload pdf file"), query: str = Form(...)):
    pdf_file = BytesIO(file.file.read())
    context = pdf_reader(pdf_file)
    question = query
    response = summarizer(context=context, question=question)
    return {"response": response}


class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("simple.html", context={"request": request})


@app.post("/clear-context")
async def clear_context():
    vector_store.client.indices.delete(index="softbd-context")


@app.post("/upload")
async def upload_json(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name
            background_tasks.add_task(load_json_context, tmp_file_path)
        return {"message": "File uploaded successfully", "filename": file.filename, "tmp_file_path": tmp_file_path}
    except Exception as e:
        return {"error": str(e)}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            # await manager.broadcast(json.dumps({"client_id": client_id, "text": data}))
            # print(data)
            await chat(data, broadcast)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"disconnected {client_id}")


async def broadcast(message: str, active=True):
    await manager.broadcast(json.dumps({"client_id": 'llm', "text": message, "active": active}))
