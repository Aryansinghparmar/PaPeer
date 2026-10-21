"""Convert LangGraph checkpoint messages into visible chat history."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage


def is_internal_message(message: Any) -> bool:
    """Return True for messages used by the graph but not shown to a user."""

    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    return bool(additional_kwargs.get("internal"))


def _content(message: Any) -> str:
    value = getattr(message, "content", "")
    return value if isinstance(value, str) else str(value)


def clean_history_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Keep one user message and one final assistant message per turn.

    Tool calls, tool results, empty AI messages, and internal query rewrites are
    graph execution details. They must not become visible chat bubbles.
    """

    visible: list[dict[str, Any]] = []
    pending_assistant: str | None = None

    for message in messages:
        if is_internal_message(message):
            continue

        if isinstance(message, HumanMessage):
            if pending_assistant:
                visible.append({"role": "assistant", "content": pending_assistant})
                pending_assistant = None
            content = _content(message).strip()
            if content:
                visible.append({"role": "user", "content": content})
            continue

        if isinstance(message, AIMessage):
            # AI tool-call messages are not final answers. They are paired with
            # ToolMessage objects and must never be rendered as assistant text.
            if getattr(message, "tool_calls", None):
                continue
            content = _content(message).strip()
            if content:
                pending_assistant = content

    if pending_assistant:
        visible.append({"role": "assistant", "content": pending_assistant})

    turn = 0
    for item in visible:
        if item["role"] == "assistant":
            turn += 1
            item.update({"turn": turn, "graph_state": {}})
    return visible
