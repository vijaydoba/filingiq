import difflib
import json
import re
import urllib.request
from html.parser import HTMLParser

SEC_USER_AGENT = "RAG-Chatbot-Demo vijay83061@gmail.com"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
PREFERRED_FORMS = ["10-K", "20-F", "40-F"]

US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
    "VA", "WA", "WV", "WI", "WY", "DC", "PR", "VI", "GU", "AS", "MP",
}

COUNTRY_TIMEZONES = {
    "United States": {"tz": "America/New_York", "exchange": "NYSE/Nasdaq"},
    "India": {"tz": "Asia/Kolkata", "exchange": "NSE/BSE"},
    "Germany": {"tz": "Europe/Berlin", "exchange": "XETRA"},
    "United Kingdom": {"tz": "Europe/London", "exchange": "LSE"},
    "Japan": {"tz": "Asia/Tokyo", "exchange": "TSE"},
    "China": {"tz": "Asia/Shanghai", "exchange": "SSE/SZSE"},
    "Canada": {"tz": "America/Toronto", "exchange": "TSX"},
    "France": {"tz": "Europe/Paris", "exchange": "Euronext Paris"},
    "Switzerland": {"tz": "Europe/Zurich", "exchange": "SIX"},
    "Netherlands": {"tz": "Europe/Amsterdam", "exchange": "Euronext Amsterdam"},
    "Israel": {"tz": "Asia/Jerusalem", "exchange": "TASE"},
    "South Korea": {"tz": "Asia/Seoul", "exchange": "KRX"},
    "Brazil": {"tz": "America/Sao_Paulo", "exchange": "B3"},
    "Australia": {"tz": "Australia/Sydney", "exchange": "ASX"},
    "Singapore": {"tz": "Asia/Singapore", "exchange": "SGX"},
    "Ireland": {"tz": "Europe/Dublin", "exchange": "Euronext Dublin"},
    "Cayman Islands": {"tz": "America/Cayman", "exchange": "—"},
    "Bermuda": {"tz": "Atlantic/Bermuda", "exchange": "—"},
}

_ticker_cache: list[dict] | None = None


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": SEC_USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": SEC_USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _load_tickers() -> list[dict]:
    global _ticker_cache
    if _ticker_cache is None:
        data = _fetch_json(TICKERS_URL)
        _ticker_cache = list(data.values())
    return _ticker_cache


def market_info(country: str | None) -> dict:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    entry = COUNTRY_TIMEZONES.get(country or "")
    if not entry:
        return {"country": country or "Unknown", "exchange": None, "local_time": None, "tz_abbr": None}

    now = datetime.now(ZoneInfo(entry["tz"]))
    return {
        "country": country,
        "exchange": entry["exchange"],
        "local_time": now.strftime("%H:%M"),
        "tz_abbr": now.strftime("%Z"),
    }


def find_company(name: str) -> dict | None:
    """Resolve a free-text company name or ticker to a SEC-registered company."""
    companies = _load_tickers()
    query = name.strip().lower()
    if not query:
        return None

    for c in companies:
        if c["ticker"].lower() == query:
            return c

    exact = [c for c in companies if c["title"].lower() == query]
    if exact:
        return exact[0]

    if len(query) < 6:
        contains = [
            c for c in companies
            if re.search(rf"\b{re.escape(query)}\b", c["title"], flags=re.IGNORECASE)
        ]
    else:
        contains = [c for c in companies if query in c["title"].lower()]
    if contains:
        contains.sort(key=lambda c: len(c["title"]))
        return contains[0]

    return None


def search_companies(query: str, limit: int = 5) -> list[dict]:
    """Find SEC-registered companies whose name or ticker is close to the query."""
    companies = _load_tickers()
    query_lower = query.strip().lower()
    if not query_lower:
        return []

    by_title: dict[str, dict] = {}
    for c in companies:
        by_title.setdefault(c["title"], c)
    unique_companies = list(by_title.values())

    ticker_hits = [c for c in unique_companies if c["ticker"].lower() == query_lower]
    if len(query_lower) < 6:
        substring_candidates = (
            c for c in unique_companies
            if re.search(rf"\b{re.escape(query_lower)}\b", c["title"], flags=re.IGNORECASE)
            and c not in ticker_hits
        )
    else:
        substring_candidates = (
            c for c in unique_companies if query_lower in c["title"].lower() and c not in ticker_hits
        )
    substring_hits = sorted(substring_candidates, key=lambda c: len(c["title"]))

    seen_titles = {c["title"] for c in ticker_hits + substring_hits}
    ranked = ticker_hits + substring_hits

    if not ranked:
        scored = []
        for title in by_title:
            ratio = difflib.SequenceMatcher(None, query_lower, title.lower()).ratio()
            if len(query_lower) >= 6:
                first_word = title.split()[0] if title.split() else title
                if abs(len(first_word) - len(query_lower)) <= 2:
                    ratio = max(ratio, difflib.SequenceMatcher(None, query_lower, first_word.lower()).ratio())
            # Short inputs are usually tickers; fuzzy company-title matches are
            # too noisy for them (e.g. "tcs" vs "BTCS INC.").
            threshold = 0.9 if len(query_lower) < 6 else 0.72
            if ratio >= threshold:
                scored.append((ratio, title))
        scored.sort(key=lambda x: -x[0])
        for _, title in scored[:limit]:
            if title not in seen_titles:
                ranked.append(by_title[title])
                seen_titles.add(title)

    if not ranked and len(query_lower) <= 6 and query_lower.isalpha():
        # Do not match a short ticker to a longer ticker (for example TCS -> BTCS).
        # Short ticker suggestions must be conservative because a false positive
        # sends the user to the wrong company's filing.
        tickers = [c["ticker"] for c in unique_companies if len(c["ticker"]) == len(query)]
        close_tickers = difflib.get_close_matches(query.upper(), tickers, n=limit, cutoff=0.9)
        by_ticker = {c["ticker"]: c for c in unique_companies}
        for ticker in close_tickers:
            candidate = by_ticker[ticker]
            if candidate["title"] not in seen_titles:
                ranked.append(candidate)
                seen_titles.add(candidate["title"])

    return ranked[:limit]


def find_latest_filing(cik: int) -> dict | None:
    """Return metadata for the most recent 10-K/20-F filing for a given CIK."""
    data = _fetch_json(SUBMISSIONS_URL.format(cik=cik))
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])

    business_address = data.get("addresses", {}).get("business", {}) or {}
    state_or_country = (business_address.get("stateOrCountry") or "").upper()
    state_or_country_desc = business_address.get("stateOrCountryDescription")

    if business_address.get("country"):
        country = business_address["country"]
    elif state_or_country and state_or_country not in US_STATE_CODES and state_or_country_desc:
        country = state_or_country_desc
    else:
        country = "United States"

    exchanges = data.get("exchanges") or []
    exchange = exchanges[0] if exchanges else None

    for wanted_form in PREFERRED_FORMS:
        for i, form in enumerate(forms):
            if form == wanted_form:
                accession = accession_numbers[i].replace("-", "")
                return {
                    "form": form,
                    "filing_date": filing_dates[i],
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_docs[i]}",
                    "country": country,
                    "exchange": exchange,
                }
    return None


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = False
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip = True
        if tag in ("br", "p", "div", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = False

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def clean_filing_text(html: str) -> str:
    html = re.sub(r"<ix:header.*?</ix:header>", "", html, flags=re.DOTALL | re.IGNORECASE)
    parser = _TextExtractor()
    parser.feed(html)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def fetch_latest_filing_text(company_name: str) -> dict:
    """Resolve a company name to its latest 10-K/20-F and return cleaned text + metadata."""
    company = find_company(company_name)
    if not company:
        raise ValueError(f'No SEC-registered company found matching "{company_name}".')

    filing = find_latest_filing(company["cik_str"])
    if not filing:
        raise ValueError(f'No 10-K or 20-F filing found for "{company["title"]}".')

    html = _fetch_text(filing["url"])
    text = clean_filing_text(html)

    return {
        "company": company["title"],
        "ticker": company["ticker"],
        "form": filing["form"],
        "filing_date": filing["filing_date"],
        "source_url": filing["url"],
        "text": text,
        "country": filing["country"],
        "exchange": filing["exchange"],
    }
