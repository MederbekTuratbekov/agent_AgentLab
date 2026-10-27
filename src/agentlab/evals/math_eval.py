"""Эксперимент 2: GSM8K. Три системы: direct, cot, agent (LangGraph + калькулятор)."""
import uuid
from functools import partial
from statistics import mean

from src import settings
from agentlab.agent.graph import ask, build_agent
from agentlab.agent.tools import calculator
from agentlab.data.hf_rows import load_gsm8k
from ..llm import chat_json
from agentlab.observability import price
from agentlab.observability.tracing import TraceHandler
from ..utils import pmap, print_table, save_json
from .metrics import numbers_match

DIRECT_PROMPT = """Solve the math word problem. Give the final answer immediately. No explanation.
Reply with JSON only: {"answer": <number>}"""

COT_PROMPT = """Solve the math word problem step by step. Do the arithmetic carefully.
Reply with JSON only: {"reasoning": "<steps>", "answer": <number>}"""

AGENT_SUFFIX = "\n\nUse the calculator for all arithmetic. End with the final number only."


def run_llm(item: dict, system: str) -> dict:
    try:
        data, u = chat_json([{"role": "system", "content": system},
                             {"role": "user", "content": item["question"]}])
        return {"pred": str(data["answer"]), "llm_calls": 1, "tool_calls": 0,
                "input": u["input"], "output": u["output"], "error": None}
    except Exception as e:
        return {"pred": "", "llm_calls": 0, "tool_calls": 0, "input": 0, "output": 0, "error": repr(e)}


def run_agent(item: dict, app) -> dict:
    handler = TraceHandler("gsm8k")
    try:
        pred = ask(app, item["question"] + AGENT_SUFFIX, session_id="eval-math",
                   thread_id=str(uuid.uuid4()), trace=handler)
        error = None
    except Exception as e:
        pred, error = "", repr(e)
    st = handler.stats()
    return {"pred": pred, "llm_calls": st["llm_calls"], "tool_calls": st["tool_calls"],
            "input": st["input_tokens"], "output": st["output_tokens"], "error": error}


def run(n: int = 30, split: str = "test", workers: int = 4) -> dict:
    items = load_gsm8k(n, seed=42, split=split)
    app = build_agent(tools=[calculator], checkpointer=False)
    systems = {
        "direct": partial(run_llm, system=DIRECT_PROMPT),
        "cot": partial(run_llm, system=COT_PROMPT),
        "agent": partial(run_agent, app=app),
    }

    report, rows, details = {}, [], {}
    for name, fn in systems.items():
        results = pmap(fn, items, workers)
        errors = [r["error"] for r in results if r["error"]]
        if errors:
            print(f"[WARN] {name}: {len(errors)} ошибок. Первая: {errors[0]}")
        correct = [numbers_match(r["pred"], it["answer"]) for r, it in zip(results, items)]
        in_tok = sum(r["input"] for r in results)
        out_tok = sum(r["output"] for r in results)
        report[name] = {
            "accuracy": round(mean(correct), 3),
            "correct": sum(correct),
            "total": n,
            "avg_llm_calls": round(mean(r["llm_calls"] for r in results), 2),
            "avg_tool_calls": round(mean(r["tool_calls"] for r in results), 2),
            "cost_usd": round(price(settings.OPENAI_MODEL, in_tok, out_tok), 5),
        }
        rows.append({"system": name, **report[name]})
        details[name] = [
            {"question": it["question"], "truth": it["answer"], "pred": r["pred"], "ok": ok}
            for it, r, ok in zip(items, results, correct)
        ]

    print(f"\n=== GSM8K ({split}): {n} задач ===")
    print_table(rows, [("system", "system", 8), ("accuracy", "acc", 7), ("correct", "ok", 4),
                       ("avg_llm_calls", "llm/задача", 11), ("avg_tool_calls", "tool/задача", 12),
                       ("cost_usd", "$", 8)])
    print("Отчёт:", save_json("math", {"summary": report, "details": details}))
    return report