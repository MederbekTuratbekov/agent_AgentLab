"""Трассировка агента через callbacks LangChain. Пишет каждый вызов LLM и инструмента."""
import json
import time
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler

from .. import settings
from .cost import METER, price


class TraceHandler(BaseCallbackHandler):
    def __init__(self, name: str):
        self.name = name
        self.trace_id = f"{name}_{int(time.time())}"
        self.events: list[dict] = []
        self._open: dict = {}
        self._t0 = time.time()

    def _begin(self, run_id, **meta):
        self._open[run_id] = {"t": time.time(), **meta}

    def _finish(self, run_id, event_type: str, name: str, **fields):
        meta = self._open.pop(run_id, {})
        started = meta.get("t", time.time())
        self.events.append({
            "time": datetime.now().isoformat(timespec="seconds"),
            "type": event_type,
            "name": name,
            "input": str(meta.get("input", ""))[:300],
            "duration_ms": round((time.time() - started) * 1000, 1),
            **fields,
        })

    # --- LLM ---
    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        last = str(messages[0][-1].content)[:300] if messages and messages[0] else ""
        self._begin(run_id, input=last)

    def on_llm_end(self, response, *, run_id, **kwargs):
        usage, text, tool_names = {}, "", []
        try:
            msg = response.generations[0][0].message
            usage = msg.usage_metadata or {}
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            tool_names = [c["name"] for c in (msg.tool_calls or [])]
        except (AttributeError, IndexError):
            pass
        i, o = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        METER.add(settings.OPENAI_MODEL, i, o)
        self._finish(run_id, "llm_call", settings.OPENAI_MODEL,
                     output=(text or f"tool_calls={tool_names}")[:500],
                     input_tokens=i, output_tokens=o)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, "llm_call", settings.OPENAI_MODEL, error=str(error)[:300])

    # --- Инструменты ---
    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = kwargs.get("name") or (serialized or {}).get("name", "tool")
        self._begin(run_id, input=input_str, name=name)

    def on_tool_end(self, output, *, run_id, **kwargs):
        name = self._open.get(run_id, {}).get("name", "tool")
        output = getattr(output, "content", output)
        self._finish(run_id, "tool_call", name, output=str(output)[:500])

    def on_tool_error(self, error, *, run_id, **kwargs):
        name = self._open.get(run_id, {}).get("name", "tool")
        self._finish(run_id, "tool_call", name, error=str(error)[:300])

    # --- Итоги ---
    def stats(self) -> dict:
        llm = [e for e in self.events if e["type"] == "llm_call"]
        tools = [e for e in self.events if e["type"] == "tool_call"]
        i = sum(e.get("input_tokens", 0) for e in llm)
        o = sum(e.get("output_tokens", 0) for e in llm)
        return {
            "llm_calls": len(llm),
            "tool_calls": len(tools),
            "tools_used": sorted({e["name"] for e in tools}),
            "input_tokens": i,
            "output_tokens": o,
            "cost_usd": price(settings.OPENAI_MODEL, i, o),
        }

    def print_summary(self) -> None:
        s = self.stats()
        print(f"\n=== Trace: {self.trace_id} ===")
        print(f"LLM: {s['llm_calls']} | инструменты: {s['tool_calls']} {s['tools_used']} "
              f"| {s['input_tokens']} in / {s['output_tokens']} out | ~${s['cost_usd']:.4f}")
        for e in self.events:
            print(f"  [{e['type']}] {e['name']} — {e['duration_ms']} ms")

    def save(self) -> str:
        path = settings.REPORTS_DIR / "traces" / f"{self.trace_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"trace_id": self.trace_id, "total_ms": round((time.time() - self._t0) * 1000, 1),
                "stats": self.stats(), "events": self.events}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)