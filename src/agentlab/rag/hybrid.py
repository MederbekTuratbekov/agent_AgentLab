"""Hybrid search: BM25 (слова) + vector (смысл), слияние через RRF."""
import re

from rank_bm25 import BM25Okapi

from . import store

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[dict]):
        if not chunks:
            raise RuntimeError("Таблица chunks пуста. Запусти: agentlab ingest")
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query: str, top_k: int) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {**self.chunks[i], "bm25_score": float(scores[i])} for i in order if scores[i] > 0
        ]


_index: BM25Index | None = None


def get_index() -> BM25Index:
    """Индекс строится один раз на процесс, а не на каждый запрос."""
    global _index
    if _index is None:
        _index = BM25Index(store.fetch_all_chunks())
    return _index


def rrf(result_lists: list[list[dict]], top_k: int = 5, k: int = 60, weights=None) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum(w / (k + rank)). Не зависит от шкал оценок."""
    weights = weights or [1.0] * len(result_lists)
    scores: dict[tuple, float] = {}
    items: dict[tuple, dict] = {}
    for w, results in zip(weights, result_lists):
        for rank, r in enumerate(results, start=1):
            key = (r["doc_id"], r["chunk_index"])
            scores[key] = scores.get(key, 0.0) + w / (k + rank)
            items.setdefault(key, r)
    ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [{**items[key], "score": scores[key]} for key in ranked]


def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.5, pool: int = 20) -> list[dict]:
    """alpha — вес векторного поиска. 1 - alpha — вес BM25."""
    vec = store.vector_search(query, pool)
    bm = get_index().search(query, pool)
    return rrf([vec, bm], top_k=top_k, weights=[alpha, 1 - alpha])


def search(query: str, mode: str = "hybrid", top_k: int = 5) -> list[dict]:
    if mode == "vector":
        return [{**r, "score": 1 - r["distance"]} for r in store.vector_search(query, top_k)]
    if mode == "bm25":
        return [{**r, "score": r["bm25_score"]} for r in get_index().search(query, top_k)]
    return hybrid_search(query, top_k=top_k)