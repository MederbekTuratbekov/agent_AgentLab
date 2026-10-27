"""LangGraph-агент: agent -> tools -> agent. Короткая память (checkpointer) + долгая (Redis)."""
import uuid
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from .. import settings
from agentlab.observability.tokens import trim_history
from .memory import get_memory
from .tools import TOOL_HINTS, TOOLS

BASE_PROMPT = "You are a careful assistant. Answer in the user's language. Keep answers short."


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def build_system_prompt(tools) -> str:
    hints = [f"- {TOOL_HINTS[t.name]}" for t in tools if t.name in TOOL_HINTS]
    return "\n".join([BASE_PROMPT, *hints])


def build_agent(tools=None, checkpointer: bool = True):
    tools = tools or TOOLS
    use_memory = any(t.name == "remember_fact" for t in tools)
    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL, temperature=0,
        api_key=settings.require_key(), max_retries=5,
    ).bind_tools(tools)
    base_prompt = build_system_prompt(tools)

    def agent_node(state: AgentState, config: RunnableConfig):
        system = base_prompt
        if use_memory:
            sid = config.get("configurable", {}).get("session_id", "default")
            facts = get_memory().get_facts(sid)
            if facts:
                system += "\n\nKnown facts about the user:\n" + "\n".join(f"- {f}" for f in facts)
        history = trim_history(state["messages"], settings.MAX_HISTORY_TOKENS)
        # config пробрасываем явно: так callbacks (трейс) доходят до вызова LLM
        response = llm.invoke([SystemMessage(content=system)] + history, config)
        return {"messages": [response]}

    def route(state: AgentState):
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver() if checkpointer else None)


def ask(app, question: str, session_id: str = "default", thread_id: str | None = None,
        trace=None, recursion_limit: int = 12) -> str:
    """thread_id — короткая память диалога. session_id — долгая память (Redis)."""
    cfg = {
        "configurable": {"session_id": session_id, "thread_id": thread_id or session_id or str(uuid.uuid4())},
        "recursion_limit": recursion_limit,  # защита от бесконечного цикла
    }
    if trace is not None:
        cfg["callbacks"] = [trace]
    try:
        out = app.invoke({"messages": [HumanMessage(content=question)]}, cfg)
    except GraphRecursionError:
        return "[агент не уложился в лимит шагов]"
    return out["messages"][-1].content