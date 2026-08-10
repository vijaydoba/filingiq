import json
import re
import time
from hashlib import sha256
from collections.abc import Iterator

import anthropic

from app.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, TOP_K
from app.ingest import get_collection, get_embedder

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
_answer_cache: dict[str, tuple[float, dict]] = {}
_summary_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL_SECONDS = 900
_CACHE_LIMIT = 256

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    "If the context does not contain the answer, say you don't know based on the given documents. "
    "Cite the supporting context number for each claim you make, like [1] or [2]."
    " Preserve exact numbers, currencies, units, and reporting periods. Do not calculate a "
    "risk/reward ratio or investment recommendation unless the document explicitly provides it. "
    "If a requested metric is absent, say that directly and do not substitute unrelated risk details."
)

SUMMARY_SYSTEM_PROMPT = (
    "You extract structured facts from SEC filing excerpts (10-K/20-F). Use ONLY the provided "
    "context. You are summarizing what the company itself states in its filing — you are not "
    "giving investment advice or predicting stock performance. Respond with ONLY a single JSON "
    "object, no markdown fences, no commentary, matching this shape exactly:\n"
    "{\n"
    '  "fiscal_period": string,\n'
    '  "revenue_latest": string | null,\n'
    '  "revenue_prior": string | null,\n'
    '  "revenue_change_pct": string | null,\n'
    '  "net_income_latest": string | null,\n'
    '  "net_income_prior": string | null,\n'
    '  "net_income_change_pct": string | null,\n'
    '  "growth_drivers": [{"point": string, "source": string}],\n'
    '  "key_risks": [{"point": string, "source": string}]\n'
    "}\n"
    "growth_drivers and key_risks should each have 2-4 items, quoting or closely paraphrasing "
    "what the filing itself says (e.g. under Business, MD&A, or Risk Factors). Use null for any "
    "numeric field not present in the context. Keep each point under 20 words."
)


def _retrieve(query: str, company: str | None, top_k: int) -> tuple[list[str], list[dict], list[float]]:
    embedder = get_embedder()
    collection = get_collection()

    if collection.count() == 0:
        return [], [], []

    query_embedding = embedder.encode([query]).tolist()
    where = {"company": company} if company else None
    results = collection.query(query_embeddings=query_embedding, n_results=max(top_k * 3, 12), where=where,
                               include=["documents", "metadatas", "distances"])

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results.get("distances", [[]])[0] if results.get("distances") else []
    query_terms = {term.lower() for term in re.findall(r"[a-zA-Z0-9]{3,}", query)}
    query_lower = query.lower()
    strategy_query = any(term in query_lower for term in ("strategy", "outlook", "priorit", "future", "growth driver"))
    risk_query = any(term in query_lower for term in ("risk", "threat", "uncertainty", "challenge"))
    ranked = []
    for index, (document, metadata) in enumerate(zip(documents, metadatas)):
        document_lower = document.lower()
        words = set(re.findall(r"[a-zA-Z0-9]{3,}", document_lower))
        overlap = len(query_terms & words) / max(len(query_terms), 1)
        distance = distances[index] if index < len(distances) else 1.0
        intent_boost = 0.0
        if strategy_query:
            intent_boost += 0.16 * sum(term in document_lower for term in (
                "business overview", "strategy", "strategic priorities", "outlook",
                "management's discussion", "transformation", "growth drivers",
            ))
            intent_boost -= 0.14 * sum(term in document_lower for term in (
                "board of directors", "committee", "director profile", "remuneration",
                "compensation", "corporate governance",
            ))
        if risk_query:
            intent_boost += 0.12 * sum(term in document_lower for term in (
                "risk factors", "principal risks", "risk management", "could adversely",
            ))
        ranked.append((overlap + intent_boost - float(distance) * 0.15, document, metadata, distance))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = []
    seen = set()
    for item in ranked:
        # Chroma can return the same passage more than once when adjacent chunks
        # overlap. Keep the evidence set compact and avoid duplicate citations.
        fingerprint = sha256(item[1].strip().encode("utf-8")).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(item)
        if len(selected) == top_k:
            break
    return ([item[1] for item in selected], [item[2] for item in selected], [item[3] for item in selected])


def answer_question(question: str, company: str | None = None, top_k: int = TOP_K) -> dict:
    cache_key = f"{company or '*'}|{question.strip().lower()}|{get_collection().count()}"
    cached = _answer_cache.get(cache_key)
    if cached and time.time() - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]
    documents, metadatas, distances = _retrieve(question, company, top_k)

    if not documents:
        scope = f' for "{company}"' if company else ""
        result = {"answer": f"No documents found{scope}. Upload a document first.", "sources": []}
        _answer_cache[cache_key] = (time.time(), result)
        return result

    context = "\n\n---\n\n".join(
        f"[{index}] Source: {meta['source']}"
        f"{', page ' + str(meta.get('page')) if meta.get('page') else ''}\n{doc}"
        for index, (doc, meta) in enumerate(zip(documents, metadatas), start=1)
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

    citations = [
        {
            "id": index,
            "source": meta["source"],
            "page": meta.get("page") or None,
            "excerpt": doc[:360].strip(),
            "distance": round(float(distances[index - 1]), 4) if index - 1 < len(distances) else None,
        }
        for index, (doc, meta) in enumerate(zip(documents, metadatas), start=1)
    ]
    result = {"answer": answer_text, "sources": sources, "citations": citations}
    _answer_cache[cache_key] = (time.time(), result)
    if len(_answer_cache) > _CACHE_LIMIT:
        _answer_cache.pop(next(iter(_answer_cache)))
    return result


def stream_answer_question(question: str, company: str | None = None, top_k: int = TOP_K) -> Iterator[dict]:
    """Yield answer deltas followed by evidence metadata for a streaming client."""
    cache_key = f"{company or '*'}|{question.strip().lower()}|{get_collection().count()}"
    cached = _answer_cache.get(cache_key)
    if cached and time.time() - cached[0] < _CACHE_TTL_SECONDS:
        result = cached[1]
        yield {"type": "delta", "text": result.get("answer", "")}
        yield {"type": "done", "sources": result.get("sources", []), "citations": result.get("citations", [])}
        return
    documents, metadatas, distances = _retrieve(question, company, top_k)
    if not documents:
        yield {"type": "delta", "text": "No documents found. Upload a document first."}
        yield {"type": "done", "sources": [], "citations": []}
        return
    context = "\n\n---\n\n".join(
        f"[{index}] Source: {meta['source']}"
        f"{', page ' + str(meta.get('page')) if meta.get('page') else ''}\n{doc}"
        for index, (doc, meta) in enumerate(zip(documents, metadatas), start=1)
    )
    citations = [
        {"id": index, "source": meta["source"], "page": meta.get("page") or None,
         "excerpt": doc[:360].strip(),
         "distance": round(float(distances[index - 1]), 4) if index - 1 < len(distances) else None}
        for index, (doc, meta) in enumerate(zip(documents, metadatas), start=1)
    ]
    with _client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    ) as stream:
        answer_parts = []
        for text in stream.text_stream:
            answer_parts.append(text)
            yield {"type": "delta", "text": text}
    result = {"answer": "".join(answer_parts), "sources": sorted({m["source"] for m in metadatas}), "citations": citations}
    _answer_cache[cache_key] = (time.time(), result)
    if len(_answer_cache) > _CACHE_LIMIT:
        _answer_cache.pop(next(iter(_answer_cache)))
    yield {"type": "done", "sources": result["sources"], "citations": citations}


def summarize_company(company: str) -> dict:
    cache_key = f"{company.strip().lower()}|{get_collection().count()}"
    cached = _summary_cache.get(cache_key)
    if cached and time.time() - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]
    fin_docs, fin_meta, _ = _retrieve(
        "total revenue net income profit loss financial results fiscal year", company, top_k=6
    )
    risk_docs, risk_meta, _ = _retrieve(
        "risk factors competition growth drivers outlook strategy future demand", company, top_k=6
    )

    documents = []
    metadatas = []
    seen = set()
    for doc, meta in zip(fin_docs + risk_docs, fin_meta + risk_meta):
        fingerprint = sha256(doc.strip().encode("utf-8")).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        documents.append(doc)
        metadatas.append(meta)

    if not documents:
        raise ValueError(f'No documents found for "{company}".')

    context = "\n\n---\n\n".join(
        f"[Source: {meta['source']}]\n{doc}" for doc, meta in zip(documents, metadatas)
    )

    message = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=SUMMARY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}"}],
    )

    raw_text = "".join(block.text for block in message.content if block.type == "text")
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()

    try:
        summary = json.loads(cleaned)
    except json.JSONDecodeError:
        summary = {"parse_error": True, "raw": raw_text}

    summary["sources"] = sorted({meta["source"] for meta in metadatas})
    _summary_cache[cache_key] = (time.time(), summary)
    if len(_summary_cache) > _CACHE_LIMIT:
        _summary_cache.pop(next(iter(_summary_cache)))
    return summary
