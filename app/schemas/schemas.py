from fastapi import UploadFile
from pydantic import BaseModel


class Query(BaseModel):
    query: str
