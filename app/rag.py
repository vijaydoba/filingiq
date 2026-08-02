import anthropic

from app.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, TOP_K
from app.ingest import get_collection, get_embedder

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    "If the context does not contain the answer, say you don't know based on the given documents. "
    "Cite the source file for each claim you make."
)


def answer_question(question: str, top_k: int = TOP_K) -> dict:
    embedder = get_embedder()
    collection = get_collection()

    query_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    if not documents:
        return {"answer": "No documents have been ingested yet. Upload a document first.", "sources": []}

    context = "\n\n---\n\n".join(
        f"[Source: {meta['source']}]\n{doc}" for doc, meta in zip(documents, metadatas)
    )

    message = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }
        ],
    )

    answer_text = "".join(block.text for block in message.content if block.type == "text")
    sources = sorted({meta["source"] for meta in metadatas})

    return {"answer": answer_text, "sources": sources}
