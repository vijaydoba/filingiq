"""Best-effort discovery of public annual-report PDFs for non-SEC companies."""

import html
import re
import urllib.parse
import urllib.request

KNOWN_OFFICIAL_REPORTS = {
    # Stable official investor-relations fallback when public search is blocked.
    "TCS": "https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2024-25/ar/annual-report-2024-2025.pdf",
}


def discover_annual_report(company: str, ticker: str = "") -> str | None:
    if ticker.upper() in KNOWN_OFFICIAL_REPORTS:
        return KNOWN_OFFICIAL_REPORTS[ticker.upper()]
    query = f'"{company}" annual report latest pdf'
    search_url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(search_url, headers={"User-Agent": "FilingIQ/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            page = response.read().decode("utf-8", errors="ignore")
    except Exception:
        return None

    candidates: list[str] = []
    for encoded in re.findall(r"uddg=([^&\"]+)", page):
        url = html.unescape(urllib.parse.unquote(encoded))
        if url.lower().endswith(".pdf") or ".pdf?" in url.lower():
            candidates.append(url)
    if not candidates:
        return None

    tokens = {token.lower() for token in re.findall(r"[a-z0-9]{3,}", f"{company} {ticker}")}

    def score(url: str) -> tuple[int, int]:
        lower = url.lower()
        host = urllib.parse.urlparse(url).netloc.lower()
        official = any(token in host for token in tokens)
        exchange = "nsearchives.nseindia.com" in host or "bseindia.com" in host
        annual = any(word in lower for word in ("annual", "ar202", "financial"))
        # Prefer official company/exchange documents over aggregators.
        return (4 if official else 3 if exchange else 1 if annual else 0, -len(url))

    return sorted(set(candidates), key=score, reverse=True)[0]
