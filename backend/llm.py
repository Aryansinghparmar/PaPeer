"""Single factory for chat models.

Every LLM call in the app goes through ``get_chat_model`` so the model name,
reasoning setting, timeout, retry count and output-token cap are set in one
place (``backend/config.py``) instead of being repeated per module.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from backend.config import (
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    OPENAI_CHAT_MODEL,
    OPENAI_REASONING_EFFORT,
)


def get_chat_model(model: str | None = None, **overrides) -> ChatOpenAI:
    kwargs = {
        "model": model or OPENAI_CHAT_MODEL,
        "stream_usage": True,
        "timeout": LLM_TIMEOUT_SECONDS,
        "max_retries": LLM_MAX_RETRIES,
        "max_tokens": LLM_MAX_OUTPUT_TOKENS,
    }
    if OPENAI_REASONING_EFFORT:
        kwargs["reasoning_effort"] = OPENAI_REASONING_EFFORT
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)
