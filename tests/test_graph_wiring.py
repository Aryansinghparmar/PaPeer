"""Paper-only retrieval during evaluation and the claim-verification answer format."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

import backend.rag_graph as rag_graph


class _Recorder:
    def __init__(self, name: str, calls: list[tuple[str, list]]) -> None:
        self.name = name
        self.calls = calls

    def invoke(self, messages, config=None, **kwargs):
        self.calls.append((self.name, messages))
        return AIMessage(content="done")


def _run_agent(monkeypatch, paper_only: bool) -> list[tuple[str, list]]:
    calls: list[tuple[str, list]] = []
    monkeypatch.setattr(rag_graph, "retrieval_llm", _Recorder("all_tools", calls))
    monkeypatch.setattr(rag_graph, "paper_retrieval_llm", _Recorder("paper_only", calls))
    rag_graph.agent_node({
        "session_id": "s",
        "messages": [HumanMessage(content="What does the paper say about BM25?")],
        "retrieval_attempts": 0,
        "paper_only": paper_only,
    })
    return calls


def test_paper_only_binds_only_the_vector_store_tool(monkeypatch):
    calls = _run_agent(monkeypatch, paper_only=True)
    assert [name for name, _ in calls] == ["paper_only"]
    assert "web_search is disabled" in calls[0][1][0]["content"]


def test_default_agent_keeps_both_tools(monkeypatch):
    calls = _run_agent(monkeypatch, paper_only=False)
    assert [name for name, _ in calls] == ["all_tools"]
    assert "web_search is disabled" not in calls[0][1][0]["content"]


def test_paper_only_llm_really_omits_web_search():
    tools = rag_graph.paper_retrieval_llm.kwargs["tools"]
    assert [t["function"]["name"] for t in tools] == ["retrieve_from_vectorstore"]


def test_claim_answer_labels_status_and_never_calls_challenged_work_superseding():
    papers = [{
        "title": "A newer result", "url": "https://arxiv.org/abs/2401.00001",
        "summary": "Fails to reproduce the gain.", "source": "arxiv", "published": "2024-01-02",
    }]
    text = rag_graph.format_claim_answer("X beats Y", "Contested.", papers, "challenged")
    assert "**Claim check: Challenged**" in text
    assert "Newer work that challenges it" in text
    assert "Superseding" not in text
    assert "(arxiv, 2024-01-02)" in text


def test_claim_answer_without_papers_for_unavailable_sources():
    text = rag_graph.format_claim_answer("X", "Could not reach the sources.", [], "sources_unavailable")
    assert text.startswith("**Claim check: Could not verify**")
    assert "Link:" not in text
