"""Side-channel (/btw) answers that bypass the RAG graph and its checkpointer."""

import os
from typing import Generator

from langchain_core.prompts import ChatPromptTemplate

from backend.models import BtwRouteDecision
from backend.config import TAVILY_MAX_RESULTS, TAVILY_SEARCH_DEPTH
from backend.llm import get_chat_model
from backend.tracing import traced_tavily_search

llm = get_chat_model()


def handle_btw(query: str) -> Generator[str, None, None]:
    """Off-topic side channel — never touches the vector store or checkpointer."""
    route_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Decide if answering this question requires a real-time web search (recent events, "
         "current prices, breaking news) or if your general knowledge is sufficient."),
        ("human", "{query}"),
    ])
    decision = (route_prompt | llm.with_structured_output(BtwRouteDecision)).invoke({"query": query})

    if decision.needs_web_search:
        results = traced_tavily_search(
            os.environ["TAVILY_API_KEY"],
            query,
            TAVILY_MAX_RESULTS,
            TAVILY_SEARCH_DEPTH,
        )
        context = "\n\n".join(r["content"] for r in results["results"])
        sources = "\n".join(f"- {r['url']}" for r in results["results"])

        answer_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Answer the question using the web search results below. Be concise.\n\n"
             f"Results:\n{context}\n\nSources:\n{sources}"),
            ("human", "{query}"),
        ])
    else:
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question concisely from your general knowledge."),
            ("human", "{query}"),
        ])

    for chunk in (answer_prompt | llm).stream({"query": query}):
        if chunk.content:
            yield chunk.content
