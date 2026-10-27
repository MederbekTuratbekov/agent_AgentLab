"""Токены и бюджет контекста. Окно обрезает историю ПЕРЕД вызовом LLM."""
import json

import tiktoken
from langchain_core.messages import BaseMessage, HumanMessage

_ENC = tiktoken.get_encoding("o200k_base")  # кодировка gpt-4o и gpt-4o-mini


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


def message_tokens(m: BaseMessage) -> int:
    text = m.content if isinstance(m.content, str) else str(m.content)
    calls = getattr(m, "tool_calls", None)
    if calls:
        text += json.dumps(calls, ensure_ascii=False)
    return count_tokens(text) + 4  # +4: служебные токены роли


def trim_history(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    """Оставляем свежие сообщения. История всегда начинается с HumanMessage:
    API не принимает ToolMessage без AIMessage с tool_calls перед ним."""
    kept, used = [], 0
    for m in reversed(messages):
        t = message_tokens(m)
        if used + t > max_tokens:
            break
        kept.append(m)
        used += t
    kept.reverse()

    while kept and not isinstance(kept[0], HumanMessage):
        kept.pop(0)
    if kept:
        return kept

    # Бюджет мал. Берём всё с последнего вопроса пользователя.
    human_idx = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    return messages[human_idx[-1]:] if human_idx else messages[-1:]