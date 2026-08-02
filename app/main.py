import os
import shutil
import tempfile

import anthropic
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.ingest import ingest_file
from app.rag import answer_question

app = FastAPI(title="RAG Chatbot")

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


class Question(BaseModel):
    question: str


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunk_count = ingest_file(tmp_path, source_name=file.filename)
    finally:
        os.remove(tmp_path)

    if chunk_count == 0:
        raise HTTPException(400, "No extractable text found in the file.")

    return {"filename": file.filename, "chunks_added": chunk_count}


@app.post("/ask")
async def ask(payload: Question):
    if not payload.question.strip():
        raise HTTPException(400, "Question must not be empty.")
    try:
        return answer_question(payload.question)
    except anthropic.APIError as e:
        raise HTTPException(502, f"Claude API error: {e.message}") from e


@app.get("/")
async def root():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
