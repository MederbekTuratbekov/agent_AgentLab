"""Orchestrator-worker + sampling. Каждая подзадача идёт через КАЖДУЮ конфигурацию (сетка 3x3).
Так влияние sampling не смешивается с влиянием подзадачи."""
import re

from pydantic import BaseModel, Field

from .llm import chat_json, chat_text
from .utils import pmap, save_json

ORCH_PROMPT = """You split a broad topic into exactly 3 narrow, non-overlapping aspects.
Each aspect is a concrete angle a worker can brainstorm on. Write in the language of the topic.
Reply with JSON only: {"subtasks": ["<aspect 1>", "<aspect 2>", "<aspect 3>"]}"""

WORKER_PROMPT = """Generate exactly 5 concrete ideas for the given subtask.
Each idea is 1-2 sentences. No filler. Do not repeat the wording of the subtask.
Write in the language of the subtask.
Reply with JSON only: {"ideas": ["<idea 1>", "<idea 2>", "<idea 3>", "<idea 4>", "<idea 5>"]}"""

SUMMARY_PROMPT = """You are the orchestrator. You get a topic and ideas from workers.
The workers used different sampling settings. Write 3-5 sentences in the language of the topic:
which ideas look most useful, and whether the settings changed the variety of ideas. Plain text."""


class SamplingConfig(BaseModel):
    name: str
    temperature: float
    top_p: float = 1.0  # top_k нет: OpenAI API его не поддерживает


CONFIGS = [
    SamplingConfig(name="conservative", temperature=0.1, top_p=1.0),
    SamplingConfig(name="balanced", temperature=0.8, top_p=0.9),
    SamplingConfig(name="creative", temperature=1.3, top_p=0.95),
]


class Plan(BaseModel):
    subtasks: list[str] = Field(min_length=3)


class WorkerOutput(BaseModel):
    ideas: list[str]


def distinct_1(texts: list[str]) -> float:
    """Доля уникальных слов. Грубая метрика разнообразия."""
    words = re.findall(r"\w+", " ".join(texts).lower())
    return round(len(set(words)) / len(words), 3) if words else 0.0


def run_worker(job: tuple[str, SamplingConfig]) -> dict:
    subtask, cfg = job
    data, _ = chat_json(
        [{"role": "system", "content": WORKER_PROMPT},
         {"role": "user", "content": f"Subtask: {subtask}"}],
        temperature=cfg.temperature, top_p=cfg.top_p,
    )
    return {"subtask": subtask, "config": cfg.name, "ideas": WorkerOutput(**data).ideas}


def run_brainstorm(topic: str) -> dict:
    print(f"=== Тема: {topic} ===\n\nШаг 1/3: orchestrator делит тему...")
    data, _ = chat_json(
        [{"role": "system", "content": ORCH_PROMPT}, {"role": "user", "content": f"Topic: {topic}"}],
        temperature=0.3,
    )
    subtasks = Plan(**data).subtasks[:3]
    for i, s in enumerate(subtasks, 1):
        print(f"  {i}. {s}")

    print("\nШаг 2/3: 9 worker-запусков параллельно (3 подзадачи x 3 конфигурации)...")
    runs = pmap(run_worker, [(s, c) for s in subtasks for c in CONFIGS], workers=9)

    diversity = {
        c.name: distinct_1([idea for r in runs if r["config"] == c.name for idea in r["ideas"]])
        for c in CONFIGS
    }
    print("Разнообразие (distinct-1, больше = разнообразнее):", diversity)

    print("\nШаг 3/3: orchestrator пишет вывод...")
    blocks = [f"Topic: {topic}"]
    for r in runs:
        ideas = "\n".join(f"  - {i}" for i in r["ideas"])
        blocks.append(f"Subtask: {r['subtask']}\nSampling: {r['config']}\nIdeas:\n{ideas}")
    summary, _ = chat_text(
        [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": "\n\n".join(blocks)}],
        temperature=0.3,
    )
    print(f"\n=== Итог ===\n{summary}")

    report = {"topic": topic, "subtasks": subtasks, "diversity": diversity, "runs": runs, "summary": summary}
    print("\nОтчёт:", save_json("brainstorm", report))
    return report