"""Метрики. SQuAD: EM, F1, contains. GSM8K: сравнение чисел."""
import re
import string
from collections import Counter

_PUNCT = set(string.punctuation)


def normalize(s: str) -> str:
    s = "".join(ch for ch in s.lower() if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> float:
    return float(any(normalize(pred) == normalize(g) for g in golds))


def _f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def f1(pred: str, golds: list[str]) -> float:
    return max((_f1(pred, g) for g in golds), default=0.0)


def contains(pred: str, golds: list[str]) -> float:
    """Мягкая метрика: эталон входит в ответ как подстрока."""
    p = normalize(pred)
    return float(any(normalize(g) and normalize(g) in p for g in golds))


def extract_number(text) -> float | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return float(nums[-1]) if nums else None


def numbers_match(pred, truth, tol: float = 1e-2) -> bool:
    a, b = extract_number(pred), extract_number(truth)
    return a is not None and b is not None and abs(a - b) < tol