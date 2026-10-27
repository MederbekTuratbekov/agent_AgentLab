"""Эксперимент 1: base / few-shot / CoT и temperature на SQuAD."""
import json
from functools import partial
from statistics import mean

from pydantic import BaseModel, field_validator

from agentlab.data.hf_rows import load_squad
from ..llm import chat_json
from agentlab.observability import price
from ..utils import pmap, print_table, save_json
from src import settings
from .metrics import contains, exact_match, f1

BASE_PROMPT = """You answer questions about a text.
Find the exact answer in the context. Keep it short: a word, a date, or a short phrase copied from the context.
Do not invent an answer that is not in the context.
Reply with JSON only:
{"answer": "<short answer>", "confidence": "high" | "medium" | "low"}"""

COT_PROMPT = """You answer questions about a text.
First reason step by step: which part of the context is relevant and what exactly is asked.
Then give a short final answer copied from the context.
Reply with JSON only:
{"reasoning": "<your reasoning>", "answer": "<short answer>", "confidence": "high" | "medium" | "low"}"""

CONFIGS = [
    ("base t=0.0", {"temperature": 0.0, "few_shot": False, "cot": False}),
    ("base t=0.7", {"temperature": 0.7, "few_shot": False, "cot": False}),
    ("base t=1.2", {"temperature": 1.2, "few_shot": False, "cot": False}),
    ("few-shot t=0.0", {"temperature": 0.0, "few_shot": True, "cot": False}),
    ("cot t=0.0", {"temperature": 0.0, "few_shot": False, "cot": True}),
]


class QAAnswer(BaseModel):
    answer: str
    confidence: str = "medium"
    reasoning: str | None = None

    @field_validator("answer", mode="before")
    @classmethod
    def to_str(cls, v):
        return str(v)


def build_few_shot_prefix(n: int = 3, seed: int = 123) -> str:
    """Примеры из train. Тест идёт по validation, поэтому пересечения нет."""
    blocks = []
    for ex in load_squad(n, seed=seed, split="train"):
        reply = json.dumps({"answer": ex["answers"][0], "confidence": "high"}, ensure_ascii=False)
        blocks.append(f"Context:\n{ex['context']}\n\nQuestion: {ex['question']}\nReply: {reply}")
    return "Examples of correct replies:\n\n" + "\n\n---\n\n".join(blocks) + "\n\n---\n\nNow your task.\n\n"


def run_one(item: dict, cfg: dict, prefix: str) -> dict:
    system = COT_PROMPT if cfg["cot"] else BASE_PROMPT
    user = (prefix if cfg["few_shot"] else "") + f"Context:\n{item['context']}\n\nQuestion: {item['question']}"
    try:
        data, usage = chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=cfg["temperature"],
        )
        return {"pred": QAAnswer(**data).answer, "ok": True, "error": None, **usage}
    except Exception as e:  # ошибка API или плохой JSON: считаем неверным ответом
        return {"pred": "", "ok": False, "error": repr(e), "input": 0, "output": 0}


def run(n: int = 50, workers: int = 8) -> dict:
    questions = load_squad(n, seed=42, split="validation")
    golds = [q["answers"] for q in questions]
    prefix = build_few_shot_prefix()

    report, rows = {}, []
    for name, cfg in CONFIGS:
        results = pmap(partial(run_one, cfg=cfg, prefix=prefix), questions, workers)
        failed = [r for r in results if not r["ok"]]
        if failed:
            print(f"[WARN] {name}: {len(failed)} ошибок. Первая: {failed[0]['error']}")
        in_tok = sum(r["input"] for r in results)
        out_tok = sum(r["output"] for r in results)
        report[name] = {
            "em": round(mean(exact_match(r["pred"], g) for r, g in zip(results, golds)), 3),
            "f1": round(mean(f1(r["pred"], g) for r, g in zip(results, golds)), 3),
            "contains": round(mean(contains(r["pred"], g) for r, g in zip(results, golds)), 3),
            "failed": len(failed),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": round(price(settings.OPENAI_MODEL, in_tok, out_tok), 5),
        }
        rows.append({"config": name, **report[name]})

    print(f"\n=== SQuAD: {n} вопросов ===")
    print_table(rows, [("config", "config", 16), ("em", "EM", 6), ("f1", "F1", 6),
                       ("contains", "contains", 9), ("failed", "failed", 7),
                       ("input_tokens", "in_tok", 8), ("cost_usd", "$", 8)])
    print("Отчёт:", save_json("prompting", report))
    return report