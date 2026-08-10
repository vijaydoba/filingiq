import uuid

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, EMBEDDING_MODEL

_embedder = None
_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection(COLLECTION_NAME)

_company_cache: dict[str, dict] | None = None


def _build_company_cache() -> dict[str, dict]:
    data = _collection.get(include=["metadatas"])
    cache: dict[str, dict] = {}
    for meta in data["metadatas"]:
        company = meta.get("company", "Unknown")
        entry = cache.setdefault(
            company, {"chunks": 0, "country": meta.get("country", "") or "", "source_url": meta.get("source_url", "") or ""}
        )
        entry["chunks"] += 1
        if not entry["country"] and meta.get("country"):
            entry["country"] = meta["country"]
        if not entry["source_url"] and meta.get("source_url"):
            entry["source_url"] = meta["source_url"]
    return cache


def _get_company_cache() -> dict[str, dict]:
    global _company_cache
    if _company_cache is None:
        _company_cache = _build_company_cache()
    return _company_cache


def extract_text(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_pages(file_path: str) -> list[tuple[int | None, str]]:
    """Return text grouped by page where the input format supports pages."""
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        return [(index + 1, page.extract_text() or "") for index, page in enumerate(reader.pages)]
    return [(None, extract_text(file_path))]


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


def ingest_text(
    text: str,
    source_name: str,
    company: str,
    country: str = "",
    page_number: int | None = None,
    source_url: str = "",
) -> int:
    chunks = chunk_text(text)
    if not chunks:
        return 0

    embeddings = get_embedder().encode(chunks).tolist()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {
            "source": source_name,
            "chunk_index": i,
            "company": company,
            "country": country,
            "page": page_number or 0,
            "source_url": source_url,
        }
        for i in range(len(chunks))
    ]

    _collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    cache = _get_company_cache()
    entry = cache.setdefault(company, {"chunks": 0, "country": country, "source_url": source_url})
    entry["chunks"] += len(chunks)
    if country:
        entry["country"] = country
    if source_url:
        entry["source_url"] = source_url

    return len(chunks)


def ingest_file(file_path: str, source_name: str, company: str, source_url: str = "") -> int:
    total = 0
    for page_number, text in extract_pages(file_path):
        total += ingest_text(
            text,
            source_name=source_name,
            company=company,
            page_number=page_number,
            source_url=source_url,
        )
    return total


def delete_company(company: str) -> int:
    existing = _collection.get(where={"company": company}, include=[])
    ids = existing["ids"]
    if ids:
        _collection.delete(ids=ids)
    _get_company_cache().pop(company, None)
    return len(ids)


def list_companies() -> list[dict]:
    cache = _get_company_cache()
    return [
        {
            "company": company,
            "chunks": info["chunks"],
            "country": info["country"],
            "source_url": info.get("source_url", ""),
        }
        for company, info in sorted(cache.items())
    ]


def get_collection():
    return _collection


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder
