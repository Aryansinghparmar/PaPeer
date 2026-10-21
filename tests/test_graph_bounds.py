"""Offline proofs that one graph turn always terminates within its stated bounds.

Every chat model, the router chain and the vector store are replaced by small
counting fakes, so these tests are deterministic and make no network calls.
"""

from __future__ import annotations

import asyncio
import itertools
from collections import Counter
from typing import Any, Callable

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

import backend.rag_graph as rag_graph
from backend.history import is_internal_message
from backend.models import RelevancyDecision, RouterDecision


class _Fake:
    def __init__(self, fn: Callable[[Any], Any]) -> None:
        self.fn = fn

    def invoke(self, value: Any, config: Any = None, **kwargs: Any) -> Any:
        return self.fn(value)


def _cycle_position(messages: list[Any]) -> tuple[int, int]:
    """(tool rounds in the current retrieval cycle, rewrites so far this turn)."""

    rounds = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        rounds += isinstance(message, ToolMessage)
    rewrites = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            if not is_internal_message(message):
                break
            rewrites += 1
    return rounds, rewrites


class Harness:
    """Patches every LLM / retrieval dependency of backend.rag_graph with fakes.

    agent_policy:
      "always" — the tool-bound agent emits a tool call on every call.
      "never"  — it never calls a tool.
      "worst"  — scripted adversary that drives the longest legal path: in each
                 cycle that can still be rewritten it stops one round short of the
                 cap (so relevancy_check -> query_rewrite runs), and in the final
                 cycle it runs to the cap.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        max_attempts: int = rag_graph.MAX_RETRIEVAL_ATTEMPTS,
        max_rewrites: int = rag_graph.MAX_QUERY_REWRITES,
        route: str = "retrieve",
        relevant: bool = False,
        agent_policy: str = "always",
        plain_emits_tool_calls: bool = False,
        docs: list[Document] | None = None,
    ) -> None:
        self.max_attempts = max_attempts
        self.max_rewrites = max_rewrites
        self.route = route
        self.relevant = relevant
        self.agent_policy = agent_policy
        self.plain_emits_tool_calls = plain_emits_tool_calls
        self.docs = docs if docs is not None else [
            Document(page_content="Attention lets tokens attend to each other.", metadata={"title": "Paper"})
        ]
        self.calls: Counter[str] = Counter()
        self.agent_inputs: list[list[Any]] = []
        self.router_inputs: list[str] = []
        self.answer_inputs: list[list[Any]] = []
        self._ids = itertools.count()

        monkeypatch.setattr(rag_graph, "MAX_RETRIEVAL_ATTEMPTS", max_attempts)
        monkeypatch.setattr(rag_graph, "MAX_QUERY_REWRITES", max_rewrites)
        monkeypatch.setattr(rag_graph, "router_chain", _Fake(self._router))
        monkeypatch.setattr(rag_graph, "retrieval_llm", _Fake(self._agent))
        monkeypatch.setattr(rag_graph, "llm", _Fake(self._plain))
        monkeypatch.setattr(rag_graph, "answer_llm", _Fake(self._plain))
        monkeypatch.setattr(rag_graph, "relevancy_llm", _Fake(self._relevancy))
        monkeypatch.setattr(rag_graph, "vs_retrieve", self._retrieve)

    @property
    def llm_calls(self) -> int:
        return sum(count for name, count in self.calls.items() if name != "vs_retrieve")

    def _tool_call(self) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "retrieve_from_vectorstore",
                "args": {"query": "attention", "k": 4},
                "id": f"call_{next(self._ids)}",
                "type": "tool_call",
            }],
        )

    def _router(self, inputs: dict) -> RouterDecision:
        self.calls["router"] += 1
        self.router_inputs.append(inputs["query"])
        return RouterDecision(route=self.route)

    def _agent(self, messages: list[Any]) -> AIMessage:
        self.calls["agent"] += 1
        self.agent_inputs.append(messages)
        if self.agent_policy == "always":
            return self._tool_call()
        if self.agent_policy == "never":
            return AIMessage(content="no tools needed")
        rounds, rewrites = _cycle_position(messages[1:])
        if rewrites < self.max_rewrites and rounds >= self.max_attempts - 1:
            return AIMessage(content="stopping early")
        return self._tool_call()

    def _plain(self, messages: list[Any]) -> AIMessage:
        system = messages[0]["content"] if isinstance(messages[0], dict) else ""
        if system == rag_graph.RETRIEVE_SYSTEM:
            self.calls["agent_at_cap"] += 1
            self.agent_inputs.append(messages)
            response = AIMessage(content="at the cap")
        elif system == rag_graph.QUERY_REWRITE_SYSTEM:
            self.calls["rewrite"] += 1
            response = AIMessage(content="rewritten attention query")
        else:
            self.calls["answer"] += 1
            self.answer_inputs.append(messages)
            response = AIMessage(content="final answer")
        if self.plain_emits_tool_calls:
            response = AIMessage(content=response.content, tool_calls=self._tool_call().tool_calls)
        return response

    def _relevancy(self, messages: list[Any]) -> RelevancyDecision:
        self.calls["relevancy"] += 1
        return RelevancyDecision(is_relevant=self.relevant, reason="fake")

    def _retrieve(self, query: str, session_id: str, k: int = 4) -> list[Document]:
        self.calls["vs_retrieve"] += 1
        return list(self.docs)


def turn_input(text: str, session_id: str = "s1") -> dict:
    """Same initial state the API and Streamlit send for each user turn."""

    return {
        "messages": [HumanMessage(content=text)],
        "session_id": session_id,
        "query": text,
        "route": None,
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "claim_verdict": None,
        "claim_source": None,
        "superseding_papers": [],
        "answer": None,
        "is_relevant": None,
        "rewrite_count": 0,
        "force_retrieval": False,
    }


def run_turn(graph: Any, text: str, thread: str = "t1") -> tuple[list[str], dict]:
    """Run one turn; return the executed node names (one per superstep) and final state.

    LangGraph also spends one superstep on ``__start__``, which "updates" omits.
    """

    config = {"configurable": {"thread_id": thread}}
    nodes: list[str] = []
    for update in graph.stream(turn_input(text), config, stream_mode="updates"):
        nodes.extend(update.keys())
    return nodes, graph.get_state(config).values


def orphan_tool_calls(messages: list[Any]) -> list[str]:
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    return [
        call["id"]
        for m in messages
        if isinstance(m, AIMessage)
        for call in (m.tool_calls or [])
        if call["id"] not in answered
    ]


# ── Formulas ──────────────────────────────────────────────────────────────────

def test_default_bounds_and_recursion_limit() -> None:
    attempts, rewrites = rag_graph.MAX_RETRIEVAL_ATTEMPTS, rag_graph.MAX_QUERY_REWRITES
    worst = rag_graph.worst_case_supersteps(attempts, rewrites)
    assert worst == (rewrites + 1) * (2 * attempts + 1) + 3
    assert rag_graph.worst_case_llm_calls(attempts, rewrites) == rewrites * (attempts + 2) + attempts + 3
    graph = rag_graph.compile_graph(InMemorySaver())
    assert graph.config["recursion_limit"] == rag_graph.graph_recursion_limit() >= worst


def test_configured_limit_below_worst_case_is_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag_graph, "GRAPH_RECURSION_LIMIT", 3)
    worst = rag_graph.worst_case_supersteps(rag_graph.MAX_RETRIEVAL_ATTEMPTS, rag_graph.MAX_QUERY_REWRITES)
    assert rag_graph.graph_recursion_limit() == worst
    monkeypatch.setattr(rag_graph, "GRAPH_RECURSION_LIMIT", worst + 50)
    assert rag_graph.graph_recursion_limit() == worst + 50


# ── Adversarial termination ───────────────────────────────────────────────────

def test_always_tool_calling_agent_terminates(monkeypatch: pytest.MonkeyPatch) -> None:
    # Agent always wants another tool call, relevancy always fails, and even the
    # tool-less model misbehaves by emitting tool calls.
    harness = Harness(monkeypatch, agent_policy="always", relevant=False, plain_emits_tool_calls=True)
    graph = rag_graph.compile_graph(InMemorySaver())
    attempts, rewrites = harness.max_attempts, harness.max_rewrites

    nodes, state = run_turn(graph, "What is attention?")

    assert state["answer"] == "final answer"
    assert nodes[-1] == "generate_answer"
    assert harness.calls["vs_retrieve"] <= (rewrites + 1) * attempts
    assert state["rewrite_count"] <= rewrites
    assert len(nodes) + 1 <= rag_graph.worst_case_supersteps(attempts, rewrites)
    assert harness.llm_calls <= rag_graph.worst_case_llm_calls(attempts, rewrites)
    assert orphan_tool_calls(state["messages"]) == []


@pytest.mark.parametrize("attempts,rewrites", [(1, 0), (1, 1), (2, 1), (3, 1), (3, 2), (5, 3)])
def test_worst_case_adversary_hits_bounds_exactly(
    monkeypatch: pytest.MonkeyPatch, attempts: int, rewrites: int
) -> None:
    harness = Harness(
        monkeypatch, max_attempts=attempts, max_rewrites=rewrites, agent_policy="worst", relevant=False
    )
    graph = rag_graph.compile_graph(InMemorySaver())
    worst = rag_graph.worst_case_supersteps(attempts, rewrites)

    nodes, state = run_turn(graph, "What is attention?")

    assert state["answer"] == "final answer"
    assert state["rewrite_count"] == rewrites
    assert nodes.count("query_rewrite") == rewrites
    assert harness.calls["vs_retrieve"] <= (rewrites + 1) * attempts
    assert orphan_tool_calls(state["messages"]) == []
    # The formulas are tight: this adversary reaches them exactly. (With A=1 a
    # rewrite cycle retrieves nothing, so its relevancy check skips the LLM.)
    assert len(nodes) + 1 == worst
    assert harness.llm_calls <= rag_graph.worst_case_llm_calls(attempts, rewrites)
    if attempts >= 2:
        assert harness.llm_calls == rag_graph.worst_case_llm_calls(attempts, rewrites)
    assert graph.config["recursion_limit"] >= worst


@pytest.mark.parametrize("attempts,rewrites", [(1, 0), (3, 1), (5, 3)])
def test_recursion_limit_matches_langgraph_step_counting(
    monkeypatch: pytest.MonkeyPatch, attempts: int, rewrites: int
) -> None:
    worst = rag_graph.worst_case_supersteps(attempts, rewrites)
    Harness(monkeypatch, max_attempts=attempts, max_rewrites=rewrites, agent_policy="worst")
    run_turn(rag_graph.compile_graph(InMemorySaver(), recursion_limit=worst), "q")

    Harness(monkeypatch, max_attempts=attempts, max_rewrites=rewrites, agent_policy="worst")
    with pytest.raises(GraphRecursionError):
        run_turn(rag_graph.compile_graph(InMemorySaver(), recursion_limit=worst - 1), "q")


def test_rewrite_loop_is_bounded_when_agent_never_retrieves(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = Harness(monkeypatch, agent_policy="never", relevant=False)
    nodes, state = run_turn(rag_graph.compile_graph(InMemorySaver()), "q")

    assert nodes.count("query_rewrite") == harness.max_rewrites
    assert state["answer"]  # canned "not found" answer, no LLM answer call
    assert harness.calls["vs_retrieve"] == 0


# ── Compiled-graph contract (with_config keeps the checkpointer) ─────────────

def test_with_config_graph_keeps_checkpointer_and_state_api(monkeypatch: pytest.MonkeyPatch) -> None:
    Harness(monkeypatch, route="direct_answer")
    saver = InMemorySaver()
    graph = rag_graph.compile_graph(saver)
    config = {"configurable": {"thread_id": "contract"}}

    assert graph.checkpointer is saver
    graph.invoke(turn_input("hello"), config)
    assert graph.get_state(config).values["answer"] == "final answer"

    async def run_async() -> list[Any]:
        chunks = [chunk async for chunk in graph.astream(turn_input("again"), config, stream_mode="updates")]
        state = await graph.aget_state(config)
        return [chunks, state]

    chunks, state = asyncio.run(run_async())
    assert chunks and state.values["answer"] == "final answer"
    # The checkpointer persisted both turns.
    assert [m.content for m in state.values["messages"] if isinstance(m, HumanMessage)] == ["hello", "again"]


def test_baked_recursion_limit_applies_when_caller_passes_only_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    Harness(monkeypatch, agent_policy="always")
    graph = rag_graph.compile_graph(InMemorySaver(), recursion_limit=3)
    with pytest.raises(GraphRecursionError):
        graph.invoke(turn_input("q"), {"configurable": {"thread_id": "limit"}})


def test_build_graph_sqlite_and_async(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    Harness(monkeypatch, route="direct_answer")
    expected = rag_graph.graph_recursion_limit()
    config = {"configurable": {"thread_id": "sqlite"}}

    graph = rag_graph.build_graph(db_path=str(tmp_path / "sync.db"))
    try:
        assert graph.config["recursion_limit"] == expected
        graph.invoke(turn_input("hi"), config)
        assert graph.get_state(config).values["answer"] == "final answer"
    finally:
        graph.checkpointer.conn.close()

    async def run_async() -> str:
        agraph = await rag_graph.build_graph_async(db_path=str(tmp_path / "async.db"))
        try:
            assert agraph.config["recursion_limit"] == expected
            await agraph.ainvoke(turn_input("hi"), config)
            return (await agraph.aget_state(config)).values["answer"]
        finally:
            await agraph.checkpointer.conn.close()

    assert asyncio.run(run_async()) == "final answer"
