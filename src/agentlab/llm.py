"""Тонкая обёртка над OpenAI. Все вызовы идут через неё: так считаются токены и деньги."""
import json

from openai import OpenAI

from . import settings
from .observability.cost import METER

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.require_key(), max_retries=5, timeout=60)
    return _client


def chat_json(messages, *, temperature=0.0, top_p=1.0, model=None) -> tuple[dict, dict]:
    """Возвращает (разобранный JSON, {"input": int, "output": int})."""
    model = model or settings.OPENAI_MODEL
    resp = client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        response_format={"type": "json_object"},
    )
    usage = {"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens}
    METER.add(model, usage["input"], usage["output"])
    return json.loads(resp.choices[0].message.content), usage


def chat_text(messages, *, temperature=0.0, model=None) -> tuple[str, dict]:
    model = model or settings.OPENAI_MODEL
    resp = client().chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    usage = {"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens}
    METER.add(model, usage["input"], usage["output"])
    return resp.choices[0].message.content, usage


def embed(texts: list[str], batch: int = 64) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        resp = client().embeddings.create(
            model=settings.EMBEDDING_MODEL, input=texts[i : i + batch]
        )
        out.extend(d.embedding for d in resp.data)
    return out