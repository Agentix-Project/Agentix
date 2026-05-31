"""abridge as a `/trace` span producer.

Each LLM call routed through the proxy opens one span with OTel GenAI
attributes; tool calls in the response surface as a span event. abridge
only *produces* spans into core `/trace` — export to an OTLP backend is
the Processor's job, exercised here with a capture Processor.
"""

from __future__ import annotations

import agentix.bridge.proxy as proxy_mod
import pytest
from agentix.bridge import detect
from agentix.bridge.detection import ApiFamily
from agentix.bridge.storage import make_record

from agentix.utils import trace


def test_llm_request_attrs_follow_genai_conventions() -> None:
    attrs = proxy_mod._llm_request_attrs(
        ApiFamily.ANTHROPIC_MESSAGES,
        {"model": "claude-x", "max_tokens": 256, "temperature": 0.2},
        session_id="s1",
        record_id="r1",
    )
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.system"] == "anthropic"
    assert attrs["gen_ai.request.model"] == "claude-x"
    assert attrs["gen_ai.request.max_tokens"] == 256
    assert attrs["gen_ai.request.temperature"] == 0.2
    assert attrs["agentix.session_id"] == "s1"
    assert attrs["agentix.request_id"] == "r1"


def test_tool_call_names_from_both_families() -> None:
    openai_resp = {
        "choices": [
            {"message": {"tool_calls": [{"function": {"name": "search"}}, {"function": {"name": "open"}}]}}
        ]
    }
    assert proxy_mod._tool_call_names(openai_resp, family=ApiFamily.OPENAI_CHAT_COMPLETIONS) == [
        "search",
        "open",
    ]
    anthropic_resp = {"content": [{"type": "tool_use", "name": "bash"}, {"type": "text", "text": "hi"}]}
    assert proxy_mod._tool_call_names(anthropic_resp, family=ApiFamily.ANTHROPIC_MESSAGES) == ["bash"]


def test_apply_response_span_sets_usage_and_tool_event() -> None:
    record = make_record(
        request_id="r1",
        session_id="s1",
        family=ApiFamily.OPENAI_CHAT_COMPLETIONS,
        started_at=0.0,
        request_path="/v1/chat/completions",
        request_body={},
        upstream_body={},
        response_body={
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "choices": [{"message": {"tool_calls": [{"function": {"name": "grep"}}]}}],
        },
    )
    with trace.span("chat test") as sp:
        proxy_mod._apply_response_span(sp, record)

    assert sp.attrs["gen_ai.usage.input_tokens"] == 5
    assert sp.attrs["gen_ai.usage.output_tokens"] == 3
    assert sp.attrs["gen_ai.response.model"] == "gpt-4o"
    tool_events = [e for e in sp.events if e.name == "gen_ai.tool_calls"]
    assert len(tool_events) == 1
    assert tool_events[0].attributes["names"] == ["grep"]


class _CaptureProcessor(trace.Processor):
    def __init__(self) -> None:
        self.spans: list[trace.Span] = []

    def on_span_end(self, s: trace.Span) -> None:
        self.spans.append(s)


@pytest.fixture
def capture_spans():
    proc = _CaptureProcessor()
    trace.add_processor(proc)
    try:
        yield proc
    finally:
        trace.remove_processor(proc)


@pytest.mark.asyncio
async def test_request_through_proxy_emits_span(wired, capture_spans) -> None:
    import httpx

    handle = wired["handle"]
    async with httpx.AsyncClient(base_url=handle.openai_base_url, timeout=10) as c:
        await c.post(
            "/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        )

    llm_spans = [s for s in capture_spans.spans if s.attrs.get("gen_ai.operation.name") == "chat"]
    assert len(llm_spans) == 1
    sp = llm_spans[0]
    assert sp.attrs["gen_ai.request.model"] == "gpt-4o-mini"
    assert sp.attrs["agentix.session_id"] == "sess-test"
    assert sp.attrs["gen_ai.usage.input_tokens"] == 7
    assert sp.status != "error"


def test_detect_still_classifies_paths() -> None:
    assert detect("/v1/messages") is ApiFamily.ANTHROPIC_MESSAGES
