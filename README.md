# FilingIQ — Financial Document Intelligence

A multi-company Retrieval-Augmented Generation (RAG) workspace: pull a public company's SEC filing (or upload your own document), ask questions about it, and inspect answers grounded in source evidence instead of relying on the model's general knowledge.

## How it works

1. **Ingest** — a document (auto-fetched 10-K/20-F, or an uploaded PDF/TXT/MD) is split into overlapping chunks and embedded locally with `sentence-transformers` (`all-MiniLM-L6-v2`).
2. **Store** — chunk embeddings are persisted in a local [ChromaDB](https://www.trychroma.com/) collection, tagged by company and country.
3. **Retrieve** — a question is embedded and matched against the stored chunks via vector similarity search, optionally scoped to one company.
4. **Generate** — the top matching chunks are passed as context to the Claude API, which answers strictly from that context and cites its sources.

## Features

- **Auto-fetch from SEC EDGAR** — type a company name or ticker and it resolves to the company's latest 10-K/20-F/40-F filing, downloads it, strips XBRL/HTML noise, and ingests it automatically. Works for any SEC-registered company (US-listed, or foreign companies with a US ADR — many large German companies like SAP still file a 20-F this way).
- **India (NSE) and Germany (XETRA) coverage** — for companies with no SEC filing, auto-ingest falls back to a curated NSE/XETRA company directory and attempts to discover the company's official public annual-report PDF automatically; if discovery fails, you paste the PDF URL or upload the file directly.
- **Multi-company, scoped Q&A** — every document is tagged by company; questions can be scoped to one company so answers never mix data across companies.
- **Compare two companies** — ask the same research question of two indexed companies side by side, each with its own scoped, cited answer.
- **Auto-generated company summary** — selecting a company immediately shows revenue/net income (with YoY change), plus "Pros" (growth signals) and "Cons" (key risks) extracted directly from the filing's MD&A and Risk Factors sections — no need to ask a question first.
- **Home page company browser** — a searchable grid of every ingested company, each showing its exchange and current local market time (computed from the company's registered country, via SEC's own address data).
- **Market/country filter** — a dropdown to narrow the company list to a specific country's market.
- **Smart search suggestions** — searching for a company that isn't loaded yet checks the full SEC EDGAR universe (~8,000 companies), NSE's Indian equity list (~2,400 companies), and a curated DAX 40 (Germany) list for close name/ticker matches, so typos and partial names still find the right company.
- **Delete companies** — remove a bad or unwanted entry directly from the UI (hover a company row → `×` → confirm).
- **Manual upload** — index `.pdf` / `.txt` / `.md` files from the sidebar or API for companies with no bulk filing source, or when automatic PDF discovery doesn't find one.
- **Evidence-first answers** — retrieved passages are reranked and returned with page metadata, excerpts, and similarity distances so every answer can be inspected.
- **Streaming chat** — answers stream over NDJSON while the model is generating instead of waiting for the entire response.

### A note on India / Germany / non-US companies

There's no single bulk filing API for Indian, German, or most non-US company filings the way SEC EDGAR provides for the US. Searching for a company from NSE's or XETRA's curated directory (e.g. "Tata Motors", "Volkswagen") will find it for discovery and auto-ingest purposes; auto-ingest then does a best-effort web search for the company's official annual-report PDF. That discovery step depends on the target site allowing automated requests — some investor-relations sites block it, in which case ingest falls back to a clear error asking you to paste the PDF URL or upload the file yourself via `/upload` or `/import-url`.

## Stack

Python · FastAPI · Anthropic Claude API · ChromaDB · sentence-transformers · pypdf · SEC EDGAR API

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY

uvicorn app.main:app --reload
```

Open http://localhost:8000. Use "auto-fetch" in the sidebar to pull a company's filing by name or ticker (e.g. `tesla`, `infosys`, `sap`), then ask a question — or pick a company from the home page to see its auto-generated summary first.

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

- `POST /auto-ingest` — `{"company": "tesla"}`, resolves via SEC EDGAR and ingests the latest 10-K/20-F/40-F automatically.
- `POST /upload` — multipart file upload (`.pdf`, `.txt`, `.md`) + `company` form field, ingests a document manually.
- `POST /import-url` — `{"company": "...", "url": "https://.../annual-report.pdf"}`, downloads and indexes a public document (25 MB limit).
- `GET /companies` — list all ingested companies with chunk counts, country, exchange, and current local market time.
- `DELETE /companies/{company}` — remove a company and all its chunks.
- `GET /search-companies?q=...` — fuzzy search SEC + NSE + XETRA company directories for name/ticker suggestions.
- `GET /summary?company=...` — auto-generated financial summary (revenue, net income, growth signals, risks) for a company.
- `POST /ask` — `{"question": "...", "company": "..."}` (company optional), returns `{"answer": "...", "sources": [...]}`.
- `POST /ask/stream` — same request body, returns newline-delimited streaming events (`delta`, then `done` with citations).
- `POST /compare` — `{"question": "...", "companies": ["A", "B"]}`, runs the same scoped question against two indexed companies and returns both results.

## Project structure

```
app/
  main.py            # FastAPI routes
  ingest.py          # text extraction, chunking, embedding, vector store, company cache
  rag.py             # retrieval + Claude API calls (Q&A, summary, compare)
  edgar.py           # SEC EDGAR lookup, filing fetch/clean, country/timezone resolution
  nse.py             # NSE (India) company directory search
  xetra.py           # XETRA (Germany, DAX 40) company directory search
  annual_reports.py  # best-effort public annual-report PDF discovery for non-SEC companies
  config.py          # settings
static/
  index.html  # responsive dashboard, library, upload, compare, and chat UI
tests/
  test_ingest.py     # unit tests for chunking
evaluation/
  run_eval.py     # smoke evaluation against a running instance
  questions.json  # sample question set with expected sources
```

## Possible extensions

- Swap ChromaDB for PostgreSQL + pgvector for a production-shaped setup.
- Support multi-turn conversation with chat history.
- Add re-ranking of retrieved chunks before generation.
- Build a proper bulk filing-document integration for more non-US markets (e.g. Japan's EDINET API, UK's Companies House).
- Add hardcoded official investor-relations fallback URLs for more curated NSE/XETRA companies, for when automated PDF discovery is blocked.

## Evaluation and limitations

The retrieval layer uses a wider vector candidate set followed by lightweight lexical reranking. PDF uploads retain page numbers; SEC HTML filings retain filing-level source metadata because HTML does not have stable PDF pages. The system is designed for grounded research assistance, not investment advice.

For a portfolio deployment, record a small question set per filing and report retrieval hit rate, citation coverage, and grounded-answer review results in the project README. Also add authentication, upload size limits, background indexing jobs, and rate limiting before exposing the API publicly.

Run the included smoke evaluation against a running instance with:

```bash
python evaluation/run_eval.py --base-url http://localhost:8000
```
