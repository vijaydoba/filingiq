import csv
import difflib
import io
import urllib.request

NSE_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
NSE_EQUITY_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

_nse_cache: list[dict] | None = None


def _load_nse_companies() -> list[dict]:
    global _nse_cache
    if _nse_cache is not None:
        return _nse_cache

    req = urllib.request.Request(NSE_EQUITY_LIST_URL, headers={"User-Agent": NSE_USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        text = resp.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(text))
    companies = []
    for row in reader:
        symbol = row.get("SYMBOL", "").strip()
        name = row.get("NAME OF COMPANY", "").strip()
        if symbol and name:
            companies.append({"symbol": symbol, "title": name})

    _nse_cache = companies
    return companies


def search_companies(query: str, limit: int = 5) -> list[dict]:
    """Find NSE-listed companies whose name or symbol is close to the query."""
    try:
        companies = _load_nse_companies()
    except Exception:
        return []

    query_lower = query.strip().lower()
    if not query_lower:
        return []

    symbol_hits = [c for c in companies if c["symbol"].lower() == query_lower]
    substring_hits = sorted(
        (c for c in companies if query_lower in c["title"].lower() and c not in symbol_hits),
        key=lambda c: len(c["title"]),
    )

    seen_titles = {c["title"] for c in symbol_hits + substring_hits}
    ranked = symbol_hits + substring_hits

    if not ranked:
        scored = []
        for c in companies:
            ratio = difflib.SequenceMatcher(None, query_lower, c["title"].lower()).ratio()
            if len(query_lower) >= 6:
                first_word = c["title"].split()[0] if c["title"].split() else c["title"]
                if abs(len(first_word) - len(query_lower)) <= 2:
                    ratio = max(ratio, difflib.SequenceMatcher(None, query_lower, first_word.lower()).ratio())
            if ratio >= 0.72:
                scored.append((ratio, c))
        scored.sort(key=lambda x: -x[0])
        for _, c in scored[:limit]:
            if c["title"] not in seen_titles:
                ranked.append(c)
                seen_titles.add(c["title"])

    return ranked[:limit]
