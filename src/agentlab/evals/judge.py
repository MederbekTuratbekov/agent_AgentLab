"""LLM-as-judge: оценка ответа по смыслу, когда точное сравнение строк не работает."""
from enum import Enum

from pydantic import BaseModel

from ..llm import chat_json


class Verdict(str, Enum):
    CORRECT = "correct"
    PARTIAL = "partially_correct"
    INCORRECT = "incorrect"


class JudgeResult(BaseModel):
    verdict: Verdict
    reasoning: str


JUDGE_PROMPT = """You judge an AI answer. You get a question, the answer, and a reference answer.
Compare by meaning, not by exact text. Different wording of the same fact is correct.
- correct: the answer contains the reference fact.
- partially_correct: some of it is right, but something important is missing or wrong.
- incorrect: the answer is wrong or does not answer the question.
Reply with JSON only:
{"verdict": "correct" | "partially_correct" | "incorrect", "reasoning": "<one or two sentences>"}"""


def judge_answer(question: str, answer: str, reference: str) -> JudgeResult:
    data, _ = chat_json([
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}\n\nReference: {reference}"},
    ])
    return JudgeResult(**data)