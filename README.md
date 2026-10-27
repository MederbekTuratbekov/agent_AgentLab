# AgentLab

RAG-агент на LangGraph с памятью, трассировкой и стендом оценки качества.
Каждое решение в проекте проверено цифрами.

## Что внутри

- Hybrid search: BM25 + pgvector (HNSW), слияние через RRF.
- Агент на LangGraph: калькулятор, поиск по документам, запись фактов в память.
- Память: короткая (checkpointer) и долгая (Redis).
- Наблюдаемость: трейс каждого шага, токены, стоимость, лимит контекста.
- Eval-стенд: SQuAD, GSM8K, retrieval, end-to-end с LLM-судьёй.
- Мультиагент: orchestrator-worker + эксперимент по sampling.

## Архитектура

```
              ┌─────────────── CLI (agentlab) ───────────────┐
              │                                              │
   ask / chat │                                  eval-* / brainstorm
              ▼                                              ▼
      ┌───────────────┐   tool calls   ┌─────────────┐   ┌──────────────┐
      │ LangGraph     │ ─────────────▶ │ calculator  │   │ evals/       │
      │ agent         │                │ search_docs │   │ metrics,     │
      │ (trim history)│ ◀───────────── │ remember    │   │ judge        │
      └──────┬────────┘                └──────┬──────┘   └──────────────┘
             │ callbacks                      │
             ▼                                ▼
      ┌───────────────┐               ┌─────────────────────────┐
      │ TraceHandler  │               │ hybrid_search           │
      │ + CostMeter   │               │ BM25 ─┐                 │
      └───────────────┘               │       ├─ RRF ─▶ top-k   │
             Redis ◀── facts          │ pgvector (HNSW) ─┘      │
                                      └─────────────────────────┘
```

## Запуск

См. команды в конце этого файла.

## Результаты

Цифры получены в моём прогоне на `gpt-4o-mini`. Полные отчёты лежат в папке `reports\`.

| Эксперимент | Вариант | Метрика | Значение | Стоимость |
|---|---|---|---|---|
| SQuAD, n=50 | base, t=0.0 | EM / F1 | 0.86 / 0.916 | $0.0023 |
| SQuAD, n=50 | base, t=0.7 | EM / F1 | 0.88 / 0.931 | $0.0024 |
| SQuAD, n=50 | base, t=1.2 | EM / F1 | 0.86 / 0.912 | $0.0023 |
| SQuAD, n=50 | few-shot, t=0.0 | EM / F1 | 0.90 / 0.930 | $0.0066 |
| SQuAD, n=50 | CoT, t=0.0 | EM / F1 | 0.68 / 0.810 | $0.0041 |
| GSM8K, n=30 | direct / cot / agent | accuracy | ... | $... |
| Retrieval, 14 запросов | vector | hit@5 / MRR | 1.0 / 0.857 | — |
| Retrieval, 14 запросов | bm25 | hit@5 / MRR | 0.929 / 0.72 | — |
| Retrieval, 14 запросов | hybrid | hit@5 / MRR | 1.0 / 0.911 | — |
| RAG e2e, 8 вопросов | агент + судья | correct | 8 из 8 | $0.0015 |
| Sampling 3x3 | conservative | distinct-1 | 0.549 | $0.0011 за весь запуск |
| Sampling 3x3 | balanced | distinct-1 | 0.579 | |
| Sampling 3x3 | creative | distinct-1 | 0.569 | |


### Что видно в цифрах

- Few-shot даёт лучший EM (0.90). Он тратит втрое больше входных токенов, чем base.
- CoT показал худший EM (0.68), но `contains` у него 0.92. Гипотеза: ответы длинные, точное совпадение строк им не подходит. Я её ещё не проверил.
- Три температуры отличаются на 1-2 пункта EM. При n=50 это шум.
- Hybrid лучше vector по MRR (0.911 против 0.857). Запросов всего 14, поэтому разница ненадёжна.
- Самое высокое distinct-1 у `balanced`, а не у `creative`. Это один запуск.
## Решения и компромиссы

- **RRF вместо взвешенной суммы.** Оценки BM25 и косинусной близости имеют разные шкалы. RRF смотрит только на ранги.
- **HNSW вместо ivfflat.** Работает на пустой таблице. Не нужно обучать индекс.
- **BM25 в памяти процесса.** Быстро для сотен и тысяч chunks. Для миллионов нужен полнотекстовый поиск в БД или OpenSearch.
- **AST-калькулятор вместо eval.** Модель — недоверенный источник входа.
- **Ошибки считаются неверными ответами.** Иначе точность завышена.
- **Сетка 3x3 в sampling.** Каждая подзадача идёт через каждую конфигурацию. Влияние sampling отделено от влияния темы.
- **Окно токенов не режет пары tool_call и tool.** Иначе API вернёт ошибку.
- **Тест не используется для подбора промптов.** Промпты настраиваются на `--split train`. Финальный замер на test делается один раз.

## Ограничения

- Выборки маленькие (n = 30-50). Разница в 2-3 пункта находится в пределах шума.
- LLM-судья и агент — одна и та же модель. У судьи есть риск симпатии к своим ответам. Нужна калибровка на ручной разметке.
- Факты из долгой памяти попадают в системный промпт. Это риск отравления памяти.
- Тексты из базы попадают в контекст. Это риск prompt injection через документы.
- Цены токенов лежат в `settings.py`. Проверяй их перед отчётом.
- Нет реранкера и нет асинхронности.

## Что улучшить дальше

- Реранкер (cross-encoder) поверх top-20.
- Оценка chunking на большем наборе запросов.
- Langfuse или OpenTelemetry вместо JSON-трейсов.
- Eval в CI с порогом точности.
- Кэш эмбеддингов и prompt caching.

## Команды

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env # впиши OPENAI_API_KEY в .env.
docker compose up -d
pytest -q

agentlab ingest
agentlab search "who created javascript"
agentlab ask "Who created Python? Answer using the documents."
agentlab chat --session kg
agentlab eval-retrieval
agentlab eval-prompting --n 50
agentlab eval-math --n 30
agentlab eval-rag
agentlab brainstorm
```

## Структура проекта
```
agentlab/
├── README.md
├── pyproject.toml
├── docker-compose.yml
├── .env.example
├── .gitignore
├── .github/workflows/ci.yml
├── tests/
│   ├── test_calc.py
│   ├── test_metrics.py
│   ├── test_chunking.py
│   ├── test_hybrid.py
│   └── test_tokens.py
└── src/agentlab/
    ├── __init__.py
    ├── __main__.py
    ├── cli.py
    ├── settings.py
    ├── utils.py
    ├── llm.py
    ├── multi_agent.py
    ├── data/
    │   ├── __init__.py
    │   ├── hf_rows.py
    │   └── wiki.py
    ├── rag/
    │   ├── __init__.py
    │   ├── chunking.py
    │   ├── store.py
    │   └── hybrid.py
    ├── agent/
    │   ├── __init__.py
    │   ├── calc.py
    │   ├── memory.py
    │   ├── tools.py
    │   └── graph.py
    ├── observability/
    │   ├── __init__.py
    │   ├── cost.py
    │   ├── tokens.py
    │   └── tracing.py
    └── evals/
        ├── __init__.py
        ├── metrics.py
        ├── judge.py
        ├── prompting.py
        ├── math_eval.py
        ├── retrieval.py
        └── rag_e2e.py
```
