"""Tests for ingress guards input caps and root-trace metadata helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.ingress_guards_middleware import (
    MAX_MESSAGE_CHARS,
    IngressGuardsMiddleware,
)
from src.utils.trace_root_metadata import build_docs_agent_trace_metadata


def test_before_agent_truncates_oversized_human_message():
    middleware = IngressGuardsMiddleware()
    long_text = "x" * (MAX_MESSAGE_CHARS + 50)
    human = HumanMessage(content=long_text, id="h1")
    state = {"messages": [AIMessage(content="hi"), human]}

    update = middleware.before_agent(state, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].id == "h1"
    assert len(update["messages"][0].content) == MAX_MESSAGE_CHARS


def test_before_agent_noop_when_under_cap():
    middleware = IngressGuardsMiddleware()
    state = {"messages": [HumanMessage(content="Hello", id="h1")]}

    assert middleware.before_agent(state, runtime=SimpleNamespace()) is None


def test_before_agent_ends_run_when_latest_turn_already_answered():
    middleware = IngressGuardsMiddleware()
    state = {
        "messages": [
            HumanMessage(content="Hello", id="h1"),
            AIMessage(content="Hi there", id="a1"),
        ]
    }

    assert middleware.before_agent(state, runtime=SimpleNamespace()) == {
        "jump_to": "end"
    }


def test_before_agent_continues_after_tool_results():
    middleware = IngressGuardsMiddleware()
    state = {
        "messages": [
            HumanMessage(content="Hello", id="h1"),
            AIMessage(
                content="",
                id="a1",
                tool_calls=[{"name": "search", "args": {}, "id": "c1"}],
            ),
            ToolMessage(content="result", tool_call_id="c1", id="t1"),
        ]
    }

    assert middleware.before_agent(state, runtime=SimpleNamespace()) is None


def test_before_agent_continues_when_ai_has_pending_tool_calls():
    middleware = IngressGuardsMiddleware()
    state = {
        "messages": [
            HumanMessage(content="Hello", id="h1"),
            AIMessage(
                content="",
                id="a1",
                tool_calls=[{"name": "search", "args": {}, "id": "c1"}],
            ),
        ]
    }

    assert middleware.before_agent(state, runtime=SimpleNamespace()) is None


def test_before_agent_continues_for_new_human_turn():
    middleware = IngressGuardsMiddleware()
    state = {
        "messages": [
            HumanMessage(content="Hello", id="h1"),
            AIMessage(content="Hi there", id="a1"),
            HumanMessage(content="Follow-up", id="h2"),
        ]
    }

    assert middleware.before_agent(state, runtime=SimpleNamespace()) is None


def test_resent_answered_turn_skips_model_call():
    model = GenericFakeChatModel(messages=iter([AIMessage(content="second answer")]))
    agent = create_agent(model=model, middleware=[IngressGuardsMiddleware()])

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content="Hello", id="h1"),
                AIMessage(content="Hi there", id="a1"),
            ]
        }
    )

    assert [m.content for m in result["messages"]] == ["Hello", "Hi there"]


def test_build_docs_agent_trace_metadata_includes_provenance_and_version(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_REVISION_ID", "rev-a")
    monkeypatch.setenv("LANGSMITH_HOST_REVISION_ID", "rev-b")
    monkeypatch.setattr(
        "src.utils.prompt_provenance._USE_LOCAL_PROMPTS",
        True,
    )

    metadata = build_docs_agent_trace_metadata()

    assert metadata["source_type"] == "Chat-LangChain"
    assert metadata["prompt_source"] == "local:instructions.md"
    assert (
        metadata["guardrails_prompt_source"]
        == "local:src/prompts/guardrails_prompts.py"
    )
    assert metadata["LANGSMITH_AGENT_VERSION"] == "rev-a"


def test_build_docs_agent_trace_metadata_falls_back_to_host_revision(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_REVISION_ID", raising=False)
    monkeypatch.setenv("LANGSMITH_HOST_REVISION_ID", "host-rev")

    metadata = build_docs_agent_trace_metadata()
    assert metadata["LANGSMITH_AGENT_VERSION"] == "host-rev"
