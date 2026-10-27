"""Эксперимент 3: качество поиска. hit_rate@k и MRR для vector / bm25 / hybrid."""
from agentlab.rag.hybrid import search
from ..utils import print_table, save_json

PY, ML = "Python (programming language)", "Machine learning"
JAVA, R = "Java (programming language)", "R (programming language)"

# Расширь набор до 30-50 запросов вручную. Тогда цифры станут надёжнее.
EVAL_SET = [
    {"query": "Who created Python and when was it first released?", "relevant": {PY}},
    {"query": "What is the difference between supervised and unsupervised learning?", "relevant": {ML}},
    {"query": "Which language do statisticians use for data analysis?", "relevant": {R}},
    {"query": "What does the Java virtual machine do?", "relevant": {JAVA}},
    {"query": "Which language runs in web browsers and was created by Brendan Eich?", "relevant": {"JavaScript"}},
    {"query": "What are convolutional layers used for in image recognition?",
     "relevant": {"Deep learning", "Neural network (machine learning)"}},
    {"query": "What are tokenization and part-of-speech tagging?", "relevant": {"Natural language processing"}},
    {"query": "What is a relational database and SQL?", "relevant": {"Database"}},
    {"query": "What is the definition of an algorithm and who was Al-Khwarizmi?", "relevant": {"Algorithm"}},
    {"query": "What is a p-value in hypothesis testing?", "relevant": {"Statistics"}},
    {"query": "Who coined the term artificial intelligence at the Dartmouth workshop?",
     "relevant": {"Artificial intelligence"}},
    {"query": "What does a data scientist do with big data?", "relevant": {"Data science"}},
    {"query": "What is the difference between compiled and interpreted languages?",
     "relevant": {"Programming language"}},
    {"query": "What is a Turing machine?", "relevant": {"Computer science", "Algorithm"}},
]


def run(k: int = 5) -> dict:
    report, rows = {}, []
    for mode in ("vector", "bm25", "hybrid"):
        hits, rr, misses = 0, 0.0, []
        for item in EVAL_SET:
            titles = [r["title"] for r in search(item["query"], mode=mode, top_k=k)]
            rank = next((i for i, t in enumerate(titles, 1) if t in item["relevant"]), None)
            if rank:
                hits += 1
                rr += 1 / rank
            else:
                misses.append(item["query"])
        n = len(EVAL_SET)
        report[mode] = {"hit_rate": round(hits / n, 3), "mrr": round(rr / n, 3), "misses": misses}
        rows.append({"mode": mode, f"hit@{k}": report[mode]["hit_rate"], "mrr": report[mode]["mrr"]})

    print(f"\n=== Retrieval: {len(EVAL_SET)} запросов, k={k} ===")
    print_table(rows, [("mode", "mode", 8), (f"hit@{k}", f"hit@{k}", 8), ("mrr", "MRR", 8)])
    print("Отчёт:", save_json("retrieval", report))
    return report