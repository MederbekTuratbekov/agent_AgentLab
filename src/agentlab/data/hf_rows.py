"""SQuAD и GSM8K через REST API HuggingFace. Без datasets и pyarrow. С кэшем на диске."""
import json
import random
import re
import time

import requests

from .. import settings
from ..utils import pmap

ROWS_API = "https://datasets-server.huggingface.co/rows"

# Размеры сплитов фиксированы: датасеты статичны.
SPLIT_SIZES = {
    ("rajpurkar/squad", "validation"): 10570,
    ("rajpurkar/squad", "train"): 87599,
    ("openai/gsm8k", "test"): 1319,
    ("openai/gsm8k", "train"): 7473,
}


def _fetch_row(dataset: str, config: str, split: str, offset: int) -> dict:
    params = {
        "dataset": dataset, "config": config, "split": split,
        "offset": offset, "length": 1,
    }
    for attempt in range(4):
        try:
            r = requests.get(ROWS_API, params=params, timeout=30)
            r.raise_for_status()
            return r.json()["rows"][0]["row"]
        except (requests.RequestException, KeyError, IndexError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def sample_rows(dataset: str, config: str, split: str, n: int, seed: int) -> list[dict]:
    """n случайных строк. Одинаковые (n, seed) дают одинаковую выборку."""
    cache = settings.DATA_DIR / "hf" / f"{dataset.replace('/', '_')}_{split}_{n}_{seed}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    total = SPLIT_SIZES[(dataset, split)]
    offsets = random.Random(seed).sample(range(total), n)
    rows = pmap(lambda off: _fetch_row(dataset, config, split, off), offsets, workers=4)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def load_squad(n: int, seed: int = 42, split: str = "validation") -> list[dict]:
    rows = sample_rows("rajpurkar/squad", "plain_text", split, n, seed)
    return [
        {"context": r["context"], "question": r["question"], "answers": r["answers"]["text"]}
        for r in rows
    ]


def _final_number(answer_field: str) -> str:
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_field)
    return m.group(1).replace(",", "") if m else answer_field.strip()


def load_gsm8k(n: int, seed: int = 42, split: str = "test") -> list[dict]:
    rows = sample_rows("openai/gsm8k", "main", split, n, seed)
    return [{"question": r["question"], "answer": _final_number(r["answer"])} for r in rows]