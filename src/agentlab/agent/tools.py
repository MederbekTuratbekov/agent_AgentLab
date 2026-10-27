"""Инструменты агента. Без `from __future__ import annotations`: иначе ломается инъекция config."""
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from agentlab.rag.hybrid import hybrid_search
from .calc import safe_eval
from .memory import get_memory


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, for example '200 * 0.15'.
    Allowed: numbers, + - * / // % **, parentheses, abs, round, min, max, sqrt."""
    try:
        result = safe_eval(expression)
    except Exception as e:  # модель увидит текст ошибки и исправит выражение
        return f"Error: {e}"
    if isinstance(result, float):
        result = round(result, 10)
        if result.is_integer():
            result = int(result)
    return str(result)


@tool
def search_documents(query: str) -> str:
    """Search the document base (Wikipedia articles about programming and AI).
    Use it for factual questions. Returns the top passages with source titles."""
    try:
        results = hybrid_search(query, top_k=3)
    except Exception as e:
        return f"Search error: {e}"
    if not results:
        return "Nothing found."
    return "\n\n---\n\n".join(f"[Source: {r['title']}]\n{r['text']}" for r in results)


@tool
def remember_fact(fact: str, config: RunnableConfig) -> str:
    """Save a lasting fact about the user (name, country, preferences) to long-term memory."""
    sid = config.get("configurable", {}).get("session_id", "default")
    get_memory().add_fact(sid, fact[:300])
    return "Saved."


TOOLS = [calculator, search_documents, remember_fact]

TOOL_HINTS = {
    "calculator": "Use calculator for any arithmetic. Do not compute in your head.",
    "search_documents": "Use search_documents for factual questions. Name the source title in the answer.",
    "remember_fact": "If the user states a lasting fact about themselves, call remember_fact.",
}