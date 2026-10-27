import argparse
import sys

from .observability.cost import METER

DEFAULT_TOPIC = "Ideas to increase student engagement in an online programming course"


def cmd_ingest(a):
    from .data.wiki import DEFAULT_TITLES, load_wiki
    from .rag import store
    from .rag.chunking import chunk_document

    store.setup_database()
    docs = load_wiki(DEFAULT_TITLES)
    chunks = [c for d in docs for c in chunk_document(d, a.strategy)]
    print(f"Статей: {len(docs)}. Chunks: {len(chunks)} ({a.strategy})")
    store.reset_chunks()
    store.insert_chunks(chunks)


def cmd_search(a):
    from .rag.hybrid import search

    for r in search(a.query, mode=a.mode, top_k=a.k):
        print(f"[{r['score']:.4f}] {r['title']} (chunk {r['chunk_index']})\n    {r['text'][:160]}...")


def cmd_ask(a):
    from .agent.graph import ask, build_agent
    from .observability.tracing import TraceHandler

    trace = TraceHandler("ask")
    print(ask(build_agent(), a.question, session_id=a.session, trace=trace))
    trace.print_summary()
    print("Трейс:", trace.save())


def cmd_chat(a):
    from .agent.graph import ask, build_agent
    from .observability.tracing import TraceHandler

    app, trace = build_agent(), TraceHandler("chat")
    print(f"Чат. Сессия: {a.session}. Выход: exit")
    while True:
        try:
            q = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            print("Агент:", ask(app, q, session_id=a.session, trace=trace))
    trace.print_summary()
    print("Трейс:", trace.save())


def cmd_forget(a):
    from .agent.memory import get_memory

    get_memory().clear(a.session)
    print("Долгая память сессии очищена.")


def cmd_eval_prompting(a):
    from .evals import prompting
    prompting.run(n=a.n, workers=a.workers)


def cmd_eval_math(a):
    from .evals import math_eval
    math_eval.run(n=a.n, split=a.split, workers=a.workers)


def cmd_eval_retrieval(a):
    from .evals import retrieval
    retrieval.run(k=a.k)


def cmd_eval_rag(a):
    from .evals import rag_e2e
    rag_e2e.run()


def cmd_brainstorm(a):
    from .multi_agent import run_brainstorm
    run_brainstorm(a.topic)


def main():
    if hasattr(sys.stdout, "reconfigure"):  # Windows: русский текст и спецсимволы
        sys.stdout.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(prog="agentlab", description="RAG-агент + eval-стенд")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="загрузить статьи, разбить, посчитать эмбеддинги")
    s.add_argument("--strategy", choices=["sentence_based", "fixed_size"], default="sentence_based")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("search", help="поиск по базе")
    s.add_argument("query")
    s.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    s.add_argument("-k", type=int, default=5)
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("ask", help="один вопрос агенту")
    s.add_argument("question")
    s.add_argument("--session", default="default")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("chat", help="диалог с агентом")
    s.add_argument("--session", default="default")
    s.set_defaults(fn=cmd_chat)

    s = sub.add_parser("forget", help="очистить долгую память сессии")
    s.add_argument("--session", default="default")
    s.set_defaults(fn=cmd_forget)

    s = sub.add_parser("eval-prompting", help="SQuAD: base / few-shot / CoT / temperature")
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(fn=cmd_eval_prompting)

    s = sub.add_parser("eval-math", help="GSM8K: direct / cot / agent")
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--split", choices=["train", "test"], default="test")
    s.add_argument("--workers", type=int, default=4)
    s.set_defaults(fn=cmd_eval_math)

    s = sub.add_parser("eval-retrieval", help="vector / bm25 / hybrid: hit@k и MRR")
    s.add_argument("-k", type=int, default=5)
    s.set_defaults(fn=cmd_eval_retrieval)

    s = sub.add_parser("eval-rag", help="агент + RAG + LLM-судья")
    s.set_defaults(fn=cmd_eval_rag)

    s = sub.add_parser("brainstorm", help="orchestrator-worker + sampling")
    s.add_argument("--topic", default=DEFAULT_TOPIC)
    s.set_defaults(fn=cmd_brainstorm)

    args = p.parse_args()
    args.fn(args)
    if METER.calls:
        print("\n" + METER.line())