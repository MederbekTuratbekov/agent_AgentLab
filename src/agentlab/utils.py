import json
from concurrent.futures import ThreadPoolExecutor

from . import settings


def pmap(fn, items, workers: int = 8) -> list:
    """map в потоках. Порядок результатов сохраняется."""
    items = list(items)
    if workers <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def save_json(name: str, data) -> str:
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = settings.REPORTS_DIR / f"{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def print_table(rows: list[dict], columns: list[tuple[str, str, int]]) -> None:
    """columns: (ключ, заголовок, ширина)."""
    print("  ".join(f"{title:<{w}}" for _, title, w in columns))
    for r in rows:
        print("  ".join(f"{str(r.get(key, '')):<{w}}" for key, _, w in columns))