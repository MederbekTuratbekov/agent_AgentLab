from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentlab.observability.tokens import trim_history


def test_trim_never_starts_with_tool_message():
    msgs = [
        HumanMessage(content="a" * 400),
        AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "1"}]),
        ToolMessage(content="2", tool_call_id="1"),
        AIMessage(content="done"),
    ]
    out = trim_history(msgs, max_tokens=30)
    assert isinstance(out[0], HumanMessage)