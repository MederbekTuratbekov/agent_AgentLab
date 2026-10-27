import threading

from .. import settings


def price(model: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = settings.PRICES_PER_1M.get(model, (0.0, 0.0))
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


class CostMeter:
    """Общий счётчик токенов и денег за один запуск CLI. Потокобезопасный."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0

    def add(self, model: str, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cost_usd += price(model, input_tokens, output_tokens)

    def line(self) -> str:
        return (f"LLM-вызовов: {self.calls} | токены: {self.input_tokens} in / "
                f"{self.output_tokens} out | ~${self.cost_usd:.4f}")


METER = CostMeter()