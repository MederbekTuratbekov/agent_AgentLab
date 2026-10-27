"""Долгая память: факты о пользователе. Redis. Если Redis недоступен, память в процессе."""
from collections import defaultdict

import redis

from src import settings

PREFIX = "agentlab:facts:"


class InMemoryStore:
    def __init__(self):
        self._d = defaultdict(list)

    def get_facts(self, sid: str) -> list[str]:
        return list(self._d[sid])

    def add_fact(self, sid: str, fact: str) -> None:
        if fact not in self._d[sid]:
            self._d[sid].append(fact)

    def clear(self, sid: str) -> None:
        self._d.pop(sid, None)


class RedisStore:
    def __init__(self, r: redis.Redis):
        self.r = r

    def get_facts(self, sid: str) -> list[str]:
        return self.r.lrange(PREFIX + sid, 0, -1)

    def add_fact(self, sid: str, fact: str) -> None:
        if fact not in self.get_facts(sid):
            self.r.rpush(PREFIX + sid, fact)

    def clear(self, sid: str) -> None:
        self.r.delete(PREFIX + sid)


_store = None


def get_memory():
    global _store
    if _store is None:
        try:
            r = redis.Redis(**settings.REDIS, decode_responses=True, socket_connect_timeout=1)
            r.ping()
            _store = RedisStore(r)
        except redis.exceptions.RedisError:
            print("[memory] Redis недоступен. Использую память в процессе (docker compose up -d).")
            _store = InMemoryStore()
    return _store