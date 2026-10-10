"""Tests for ToolRetryMiddleware handling of tool-reported upstream failures."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from src.middleware.tool_retry_middleware import ToolRetryMiddleware

TOOL_NAME = "query_docs_filesystem_docs_by_lang_chain"


def _request():
    return SimpleNamespace(tool_call={"name": TOOL_NAME, "id": "call_1", "args": {}})


def _message(text: str, *, blocks: bool = False) -> ToolMessage:
    content = [{"type": "text", "text": text}] if blocks else text
    return ToolMessage(content=content, name=TOOL_NAME, tool_call_id="call_1")


def _run(middleware: ToolRetryMiddleware, results: list[ToolMessage]):
    calls: list[int] = []

    async def handler(request):
        calls.append(1)
        return results[len(calls) - 1]

    result = asyncio.run(middleware.awrap_tool_call(_request(), handler))
    return result, len(calls)


def test_content_level_504_retries_once_then_returns_unavailable():
    middleware = ToolRetryMiddleware(max_attempts=3, initial_delay=0)
    failure = "Docs filesystem query failed: 504 Gateway Time-out"

    result, calls = _run(
        middleware,
        [
            _message(failure, blocks=True),
            _message(failure, blocks=True),
            _message("ok"),
        ],
    )

    assert calls == 2
    payload = json.loads(result.content)
    assert payload["error"] == "Tool unavailable"
    assert payload["message"] == f"{TOOL_NAME} failed after 2 attempts."
    assert payload["details"] == failure
    assert result.tool_call_id == "call_1"


def test_content_level_failure_returns_successful_retry():
    middleware = ToolRetryMiddleware(max_attempts=3, initial_delay=0)
    success = _message("exit: 0\n--- stdout ---\nabac.mdx", blocks=True)

    result, calls = _run(
        middleware,
        [_message("Search failed: 504 Gateway Time-out"), success],
    )

    assert calls == 2
    assert result is success


def test_normal_content_is_returned_unchanged():
    middleware = ToolRetryMiddleware(max_attempts=3, initial_delay=0)
    success = _message(
        "Title: Timeouts\nContent: Requests failed: 504 Gateway Time-out when the "
        "gateway timeout is too low."
    )

    result, calls = _run(middleware, [success])

    assert calls == 1
    assert result is success
