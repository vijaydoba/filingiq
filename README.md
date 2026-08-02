# RAG Chatbot

A minimal Retrieval-Augmented Generation (RAG) chatbot: upload a document, ask questions about it, and get answers grounded in the document's content with cited sources — instead of relying on the model's general knowledge.

## How it works

1. **Ingest** — an uploaded PDF/TXT/MD file is split into overlapping chunks and embedded locally with `sentence-transformers` (`all-MiniLM-L6-v2`).
2. **Store** — chunk embeddings are persisted in a local [ChromaDB](https://www.trychroma.com/) collection.
3. **Retrieve** — a question is embedded and matched against the stored chunks via vector similarity search.
4. **Generate** — the top matching chunks are passed as context to the Claude API, which answers strictly from that context and cites its sources.

## Stack

Python · FastAPI · Anthropic Claude API · ChromaDB · sentence-transformers · pypdf

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY

uvicorn app.main:app --reload
```

Open http://localhost:8000, upload a document, then ask a question about it.

## Deploy with Docker

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY

docker compose up -d --build
```

The app is served on `http://<server-ip>:8000`. The vector store persists in a named Docker volume (`chroma_data`) across restarts/redeploys.

To update after a code change:

```bash
git pull
docker compose up -d --build
```

## API

- `POST /upload` — multipart file upload (`.pdf`, `.txt`, `.md`), ingests the document into the vector store.
- `POST /ask` — `{"question": "..."}`, returns `{"answer": "...", "sources": [...]}`.

## Project structure

```
app/
  main.py     # FastAPI routes
  ingest.py   # text extraction, chunking, embedding, vector store
  rag.py      # retrieval + Claude API call
  config.py   # settings
static/
  index.html  # minimal single-page UI
```

## Possible extensions

- Swap ChromaDB for PostgreSQL + pgvector for a production-shaped setup.
- Add streaming responses from the Claude API.
- Support multi-turn conversation with chat history.
- Add re-ranking of retrieved chunks before generation.
