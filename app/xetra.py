import difflib

# Deutsche Börse does not publish an open, unauthenticated listing CSV like NSE's
# EQUITY_L.csv, so the DAX 40 constituents are curated here instead of fetched.
_GERMAN_COMPANIES = [
    {"symbol": "SAP", "title": "SAP SE"},
    {"symbol": "SIE", "title": "Siemens AG"},
    {"symbol": "VOW3", "title": "Volkswagen AG"},
    {"symbol": "DBK", "title": "Deutsche Bank AG"},
    {"symbol": "ALV", "title": "Allianz SE"},
    {"symbol": "BAS", "title": "BASF SE"},
    {"symbol": "BAYN", "title": "Bayer AG"},
    {"symbol": "DTE", "title": "Deutsche Telekom AG"},
    {"symbol": "MBG", "title": "Mercedes-Benz Group AG"},
    {"symbol": "BMW", "title": "Bayerische Motoren Werke AG"},
    {"symbol": "ADS", "title": "Adidas AG"},
    {"symbol": "MUV2", "title": "Munich Re AG"},
    {"symbol": "IFX", "title": "Infineon Technologies AG"},
    {"symbol": "DHL", "title": "DHL Group"},
    {"symbol": "RWE", "title": "RWE AG"},
    {"symbol": "EOAN", "title": "E.ON SE"},
    {"symbol": "HEN3", "title": "Henkel AG & Co. KGaA"},
    {"symbol": "FRE", "title": "Fresenius SE & Co. KGaA"},
    {"symbol": "CON", "title": "Continental AG"},
    {"symbol": "DB1", "title": "Deutsche Börse AG"},
    {"symbol": "AIR", "title": "Airbus SE"},
    {"symbol": "DTG", "title": "Daimler Truck Holding AG"},
    {"symbol": "P911", "title": "Porsche AG"},
    {"symbol": "SHL", "title": "Siemens Healthineers AG"},
    {"symbol": "MRK", "title": "Merck KGaA"},
    {"symbol": "VNA", "title": "Vonovia SE"},
    {"symbol": "ZAL", "title": "Zalando SE"},
    {"symbol": "SRT3", "title": "Sartorius AG"},
    {"symbol": "1COV", "title": "Covestro AG"},
    {"symbol": "BEI", "title": "Beiersdorf AG"},
]


def search_companies(query: str, limit: int = 5) -> list[dict]:
    """Find XETRA-listed German companies whose name or symbol is close to the query."""
    query_lower = query.strip().lower()
    if not query_lower:
        return []

    companies = _GERMAN_COMPANIES
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
