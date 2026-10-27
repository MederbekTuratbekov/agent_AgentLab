"""Статьи Википедии для RAG. Берём через MediaWiki API."""
import re
import time
import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "agentlab/0.1 (learning project; your_email@example.com)"}

MAX_CHARS_PER_DOC = 60_000

DEFAULT_TITLES = [
    "Python (programming language)",
    "Machine learning",
    "R (programming language)",
    "Java (programming language)",
    "JavaScript",
    "Artificial intelligence",
    "Deep learning",
    "Neural network (machine learning)",
    "Data science",
    "Algorithm",
    "Programming language",
    "Computer science",
    "Natural language processing",
    "Statistics",
    "Database",
]


def _clean(text: str) -> str:
    text = re.sub(r"^=+.*?=+\s*$", "", text, flags=re.M)  # заголовки "== History =="
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def _get_with_retry(params: dict, tries: int = 5) -> requests.Response:
    for attempt in range(tries):
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        retry_after = r.headers.get("Retry-After", "")
        wait = int(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1)
        print(f"  [429] Википедия просит подождать. Жду {wait} сек.")
        time.sleep(wait)
    r.raise_for_status()
    return r


def load_wiki(titles: list[str] | None = None) -> list[dict]:
    docs = []
    for title in titles or DEFAULT_TITLES:
        r = _get_with_retry({
            "action": "query", "format": "json", "prop": "extracts",
            "explaintext": 1, "redirects": 1, "titles": title,
        })
        page = next(iter(r.json()["query"]["pages"].values()))
        text = _clean(page.get("extract", ""))
        if not text:
            print(f"  [WARN] не найдена статья: {title}")
            continue
        docs.append({"id": str(page["pageid"]), "title": page["title"], "text": text[:MAX_CHARS_PER_DOC]})
        time.sleep(1)
    return docs