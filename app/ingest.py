import uuid

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, EMBEDDING_MODEL

_embedder = SentenceTransformer(EMBEDDING_MODEL)
_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection(COLLECTION_NAME)


def extract_text(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def ingest_file(file_path: str, source_name: str) -> int:
    text = extract_text(file_path)
    chunks = chunk_text(text)
    if not chunks:
        return 0

    embeddings = _embedder.encode(chunks).tolist()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

    _collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return len(chunks)


def get_collection():
    return _collection


def get_embedder():
    return _embedder
