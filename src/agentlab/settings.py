"""Настройки проекта. Всё читается из .env в корне."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# USD за 1M токенов: (input, output). Проверь цены на сайте OpenAI перед отчётом.
PRICES_PER_1M = {"gpt-4o-mini": (0.15, 0.60)}

MAX_HISTORY_TOKENS = 4000

PG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": int(os.getenv("PG_PORT", "5433")),
    "dbname": os.getenv("PG_DBNAME", "agentlab"),
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", "postgres"),
}
REDIS = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "db": int(os.getenv("REDIS_DB", "0")),
}

DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"


def require_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or "your" in key:
        raise RuntimeError(
            "OPENAI_API_KEY не найден. Скопируй .env.example в .env и впиши ключ."
        )
    return key