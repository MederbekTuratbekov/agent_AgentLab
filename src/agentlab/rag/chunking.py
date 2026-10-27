"""Две стратегии разбивки текста на chunks."""
import re


def chunk_fixed_size(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    step = max(1, size - overlap)
    pieces = (text[i : i + size].strip() for i in range(0, len(text), step))
    return [p for p in pieces if p]


def chunk_sentence_based(text: str, max_size: int = 500) -> list[str]:
    """Склеиваем предложения, пока chunk не станет длиннее max_size."""
    chunks, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_size:
            current += " " + sentence
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def chunk_document(doc: dict, strategy: str = "sentence_based", min_len: int = 100) -> list[dict]:
    if strategy == "fixed_size":
        raw = chunk_fixed_size(doc["text"])
    elif strategy == "sentence_based":
        raw = chunk_sentence_based(doc["text"])
    else:
        raise ValueError(f"Неизвестная стратегия: {strategy}")
    return [
        {"doc_id": doc["id"], "title": doc["title"], "chunk_index": i, "text": t}
        for i, t in enumerate(raw)
        if len(t) >= min_len
    ]