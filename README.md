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

Заполни после своего прогона. Не вписывай чужие цифры.

| Эксперимент | Вариант | Метрика | Значение | Стоимость |
|---|---|---|---|---|
| SQuAD, n=50 | base / few-shot / CoT | F1 | ... | $... |
| GSM8K, n=30 | direct / cot / agent | accuracy | ... | $... |
| Retrieval, 14 запросов | vector / bm25 / hybrid | hit@5, MRR | ... | — |
| RAG e2e, 8 вопросов | агент + судья | correct | ... | $... |
| Sampling 3x3 | conservative / balanced / creative | distinct-1 | ... | $... |

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
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
docker compose up -d
pytest -q
agentlab ingest
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
          │
          ├── __init__.py     
          ├── __main__.py
          ├── cli.py
          ├── settings.py
          ├── utils.py
          ├── llm.py
          ├── multi_agent.py
          │
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
