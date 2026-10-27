"""Эксперимент 4: полный путь. Агент + RAG + LLM-судья."""
import time
import uuid
from collections import Counter

from agentlab.agent.graph import ask, build_agent
from agentlab.observability.tracing import TraceHandler
from ..utils import pmap, save_json
from .judge import judge_answer

QA = [
    {"question": "Who created the Python programming language?", "reference": "Guido van Rossum"},
    {"question": "Who created JavaScript?", "reference": "Brendan Eich"},
    {"question": "Who created the Java programming language at Sun Microsystems?", "reference": "James Gosling"},
    {"question": "Who created the R programming language?", "reference": "Ross Ihaka and Robert Gentleman"},
    {"question": "In which year was Python first released?", "reference": "1991"},
    {"question": "What does SQL stand for?", "reference": "Structured Query Language"},
    {"question": "Which 1956 workshop is seen as the birth of artificial intelligence as a field?",
     "reference": "The Dartmouth workshop"},
    {"question": "Who proposed the Turing test?", "reference": "Alan Turing"},
]


def run(workers: int = 4) -> dict:
    app = build_agent(checkpointer=False)

    def one(item: dict) -> dict:
        handler = TraceHandler("rag_e2e")
        t0 = time.time()
        try:
            answer = ask(app, item["question"], session_id="eval-rag",
                         thread_id=str(uuid.uuid4()), trace=handler)
            verdict = judge_answer(item["question"], answer, item["reference"])
            v, reasoning = verdict.verdict.value, verdict.reasoning
        except Exception as e:
            answer, v, reasoning = "", "incorrect", repr(e)
        st = handler.stats()
        return {**item, "answer": answer, "verdict": v, "reasoning": reasoning,
                "used_retrieval": "search_documents" in st["tools_used"],
                "llm_calls": st["llm_calls"], "latency_s": round(time.time() - t0, 2)}

    results = pmap(one, QA, workers)
    counts = Counter(r["verdict"] for r in results)
    n = len(results)
    summary = {
        "total": n,
        "correct": counts["correct"],
        "partially_correct": counts["partially_correct"],
        "incorrect": counts["incorrect"],
        "used_retrieval_rate": round(sum(r["used_retrieval"] for r in results) / n, 3),
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / n, 2),
    }
    print("\n=== RAG end-to-end (LLM-судья) ===")
    for r in results:
        print(f"[{r['verdict']:<17}] retrieval={r['used_retrieval']} | {r['question']}")
    print(summary)
    print("Отчёт:", save_json("rag_e2e", {"summary": summary, "details": results}))
    return summary