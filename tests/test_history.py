from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.history import clean_history_messages


def test_history_keeps_only_user_and_final_assistant_messages() -> None:
    messages = [
        HumanMessage(content="What is attention?"),
        AIMessage(content="", tool_calls=[{"name": "retrieve", "args": {}, "id": "1"}]),
        ToolMessage(content="Retrieved context", tool_call_id="1"),
        HumanMessage(
            content="attention mechanism",
            additional_kwargs={"internal": True, "message_role": "query_rewrite"},
        ),
        AIMessage(content="The final answer."),
    ]

    assert clean_history_messages(messages) == [
        {"role": "user", "content": "What is attention?"},
        {
            "role": "assistant",
            "content": "The final answer.",
            "turn": 1,
            "graph_state": {},
        },
    ]


def test_history_flushes_previous_answer_before_next_user_turn() -> None:
    messages = [
        HumanMessage(content="First"),
        AIMessage(content="First answer"),
        HumanMessage(content="Second"),
        AIMessage(content="Second answer"),
    ]

    visible = clean_history_messages(messages)
    assert [item["content"] for item in visible] == [
        "First",
        "First answer",
        "Second",
        "Second answer",
    ]
