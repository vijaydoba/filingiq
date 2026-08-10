"""Small, reproducible smoke evaluation for a running RAG instance.

Usage: python evaluation/run_eval.py --base-url http://localhost:8000
"""

import argparse
import json
import urllib.request
from pathlib import Path


def ask(base_url: str, item: dict) -> dict:
    body = json.dumps({"question": item["question"], "company": item["company"]}).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/ask", data=body,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    questions = json.loads((Path(__file__).with_name("questions.json")).read_text())
    grounded = 0
    cited = 0
    for item in questions:
        result = ask(args.base_url, item)
        citations = result.get("citations", [])
        answer = result.get("answer", "")
        has_expected_source = any(item["expected_source_contains"] in c.get("source", "") for c in citations)
        has_inline_citation = any(f"[{c.get('id')}]" in answer for c in citations)
        grounded += int(has_expected_source)
        cited += int(has_inline_citation)
        print(json.dumps({"question": item["question"], "grounded": has_expected_source, "cited": has_inline_citation}))
    total = len(questions) or 1
    print(json.dumps({"retrieval_hit_rate": grounded / total, "citation_coverage": cited / total}))


if __name__ == "__main__":
    main()
