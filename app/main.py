import os
import shutil
import tempfile
import json
import time
import urllib.parse
import urllib.request
from collections import defaultdict, deque

import anthropic
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import annual_reports, edgar, nse, xetra
from app.ingest import delete_company, ingest_file, ingest_text, list_companies
from app.rag import answer_question, stream_answer_question, summarize_company

app = FastAPI(title="RAG Chatbot")

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

# Claude API calls cost real money per request, so the LLM-backed endpoints get a
# per-IP rate limit. This is a public demo deployment with no auth in front of it.
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 10
_request_log: dict[str, deque[float]] = defaultdict(deque)


def enforce_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    log = _request_log[client_ip]
    while log and now - log[0] > RATE_LIMIT_WINDOW_SECONDS:
        log.popleft()
    if len(log) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(429, "Too many requests. Please wait a minute and try again.")
    log.append(now)


class Question(BaseModel):
    question: str
    company: str | None = None


class AutoIngestRequest(BaseModel):
    company: str


class UrlImportRequest(BaseModel):
    company: str
    url: str


class CompareRequest(BaseModel):
    question: str
    companies: list[str]


@app.post("/upload")
async def upload_document(file: UploadFile = File(...), company: str = Form(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    if not company.strip():
        raise HTTPException(400, "Company name must not be empty.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunk_count = ingest_file(tmp_path, source_name=file.filename, company=company.strip())
    finally:
        os.remove(tmp_path)

    if chunk_count == 0:
        raise HTTPException(400, "No extractable text found in the file.")

    return {"filename": file.filename, "company": company.strip(), "chunks_added": chunk_count}


@app.post("/import-url")
async def import_document_url(payload: UrlImportRequest):
    """Download and index a user-provided annual-report URL."""
    company = payload.company.strip()
    url = payload.url.strip()
    parsed = urllib.parse.urlparse(url)
    if not company:
        raise HTTPException(400, "Company name must not be empty.")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(400, "Enter a valid http(s) PDF URL.")

    suffix = os.path.splitext(parsed.path)[1].lower()
    if suffix not in ALLOWED_EXTENSIONS:
        suffix = ".pdf"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "FilingIQ/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            content_length = int(response.headers.get("Content-Length") or 0)
            if content_length > 25 * 1024 * 1024:
                raise HTTPException(413, "The document is larger than the 25 MB limit.")
            data = response.read(25 * 1024 * 1024 + 1)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Could not download the document: {exc}") from exc

    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "The document is larger than the 25 MB limit.")
    source_name = os.path.basename(parsed.path) or f"{company}-annual-report{suffix}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        chunk_count = ingest_file(tmp_path, source_name=source_name, company=company, source_url=url)
    finally:
        os.remove(tmp_path)
    if chunk_count == 0:
        raise HTTPException(400, "No extractable text found in the downloaded document.")
    return {"filename": source_name, "company": company, "chunks_added": chunk_count, "url": url}


@app.get("/companies")
async def companies():
    result = []
    for c in list_companies():
        info = edgar.market_info(c["country"])
        result.append({**c, **info})
    return {"companies": result}


@app.delete("/companies/{company}")
async def remove_company(company: str):
    deleted = delete_company(company)
    if deleted == 0:
        raise HTTPException(404, f'No documents found for "{company}".')
    return {"company": company, "chunks_deleted": deleted}


@app.get("/search-companies")
async def search_companies_endpoint(q: str):
    if not q.strip():
        return {"results": []}
    query = q.strip()
    query_lower = query.lower()
    # Prefer an exact exchange symbol before fuzzy SEC suggestions. This avoids
    # a short NSE symbol such as TCS being mistaken for a similar SEC ticker.
    nse_results = nse.search_companies(query, limit=5)
    xetra_results = xetra.search_companies(query, limit=5)
    exact = [{"title": c["title"], "ticker": c["symbol"], "source": "NSE"} for c in nse_results
             if c["symbol"].lower() == query_lower]
    exact += [{"title": c["title"], "ticker": c["symbol"], "source": "XETRA"} for c in xetra_results
              if c["symbol"].lower() == query_lower]
    if exact:
        return {"results": exact[:5]}

    sec_results = edgar.search_companies(query, limit=5)

    results = []
    seen_titles = set()

    def _add(items, source):
        for c in items:
            if c["title"] not in seen_titles and len(results) < 5:
                results.append({"title": c["title"], "ticker": c["symbol"], "source": source})
                seen_titles.add(c["title"])

    # A direct name match on a curated exchange list is a stronger signal than
    # SEC's fuzzy ratio match, so it goes first and isn't crowded out by noise.
    _add([c for c in nse_results if query_lower in c["title"].lower()], "NSE")
    _add([c for c in xetra_results if query_lower in c["title"].lower()], "XETRA")

    for c in sec_results:
        if c["title"] not in seen_titles and len(results) < 5:
            results.append({"title": c["title"], "ticker": c["ticker"], "source": "SEC"})
            seen_titles.add(c["title"])

    _add(nse_results, "NSE")
    _add(xetra_results, "XETRA")

    return {"results": results}


@app.post("/auto-ingest")
async def auto_ingest(payload: AutoIngestRequest):
    if not payload.company.strip():
        raise HTTPException(400, "Company name must not be empty.")

    try:
        filing = edgar.fetch_latest_filing_text(payload.company)
    except ValueError as sec_error:
        # SEC is not a universal source. For an exact NSE or XETRA symbol,
        # discover the company's public annual-report PDF and use the same
        # ingestion path.
        query = payload.company.strip().lower()
        candidates = [(item, "NSE") for item in nse.search_companies(payload.company.strip(), limit=5)]
        candidates += [(item, "XETRA") for item in xetra.search_companies(payload.company.strip(), limit=5)]
        exact = next(
            (
                item
                for item, _source in candidates
                if item["symbol"].lower() == query
                or item["title"].lower() == query
                or query in item["title"].lower()
            ),
            None,
        )
        if not exact:
            raise HTTPException(404, str(sec_error)) from sec_error
        report_url = annual_reports.discover_annual_report(
            exact["title"], ticker=exact["symbol"]
        )
        if not report_url:
            raise HTTPException(
                404,
                f'No public annual report was found automatically for "{exact["title"]}". '
                "Paste the official PDF URL or upload the file.",
            )
        return await import_document_url(
            UrlImportRequest(company=exact["title"], url=report_url)
        )
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch filing from SEC EDGAR: {e}") from e

    source_name = f"{filing['ticker']}-{filing['form']}-{filing['filing_date']}"
    chunk_count = ingest_text(
        filing["text"],
        source_name=source_name,
        company=filing["company"],
        country=filing["country"] or "",
        source_url=filing["source_url"],
    )

    if chunk_count == 0:
        raise HTTPException(400, "No extractable text found in the fetched filing.")

    return {
        "company": filing["company"],
        "ticker": filing["ticker"],
        "form": filing["form"],
        "filing_date": filing["filing_date"],
        "source_url": filing["source_url"],
        "chunks_added": chunk_count,
        "country": filing["country"],
    }


@app.get("/summary", dependencies=[Depends(enforce_rate_limit)])
async def summary(company: str):
    if not company.strip():
        raise HTTPException(400, "Company must not be empty.")
    try:
        return summarize_company(company.strip())
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    except anthropic.APIError as e:
        raise HTTPException(502, f"Claude API error: {e.message}") from e


@app.post("/ask", dependencies=[Depends(enforce_rate_limit)])
async def ask(payload: Question):
    if not payload.question.strip():
        raise HTTPException(400, "Question must not be empty.")
    try:
        return answer_question(payload.question, company=payload.company)
    except anthropic.APIError as e:
        raise HTTPException(502, f"Claude API error: {e.message}") from e


@app.post("/compare", dependencies=[Depends(enforce_rate_limit)])
async def compare(payload: CompareRequest):
    """Answer one research question for two indexed companies.

    This intentionally reuses the verified, company-scoped RAG path. It is a
    small comparison workflow for the focused demo, not an investment signal.
    """
    question = payload.question.strip()
    companies = [c.strip() for c in payload.companies if c and c.strip()]
    if not question:
        raise HTTPException(400, "Question must not be empty.")
    if len(companies) != 2 or companies[0].lower() == companies[1].lower():
        raise HTTPException(400, "Choose two different companies.")
    try:
        answers = [answer_question(question, company=name) for name in companies]
        return {"question": question, "results": answers}
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    except anthropic.APIError as e:
        raise HTTPException(502, f"Claude API error: {e.message}") from e


@app.post("/ask/stream", dependencies=[Depends(enforce_rate_limit)])
async def ask_stream(payload: Question):
    if not payload.question.strip():
        raise HTTPException(400, "Question must not be empty.")

    def events():
        for event in stream_answer_question(payload.question, company=payload.company):
            yield json.dumps(event) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/")
async def root():
    return FileResponse("static/index.html", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory="static"), name="static")
