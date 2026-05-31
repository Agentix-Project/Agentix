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


def test_span_carries_prompt_and_completion_content():
    """Issue: LangSmith showed only tokens. Spans must also carry the prompt +
    completion text as gen_ai.prompt.*/completion.* (what backends render)."""
    attrs = proxy_mod._llm_request_attrs(
        ApiFamily.ANTHROPIC_MESSAGES,
        {"model": "m", "system": "sys", "messages": [{"role": "user", "content": "hello"}]},
        session_id="s", record_id="r",
    )
    assert attrs["gen_ai.prompt.0.role"] == "system"
    assert attrs["gen_ai.prompt.0.content"] == "sys"
    assert attrs["gen_ai.prompt.1.role"] == "user"
    assert attrs["gen_ai.prompt.1.content"] == "hello"

    rec = make_record(
        request_id="r", session_id="s", family=ApiFamily.ANTHROPIC_MESSAGES,
        started_at=0.0, request_path="/v1/messages", request_body={}, upstream_body={},
        response_body={"content": [{"type": "text", "text": "world"}],
                       "usage": {"input_tokens": 1, "output_tokens": 1}},
    )
    with trace.span("x") as sp:
        proxy_mod._apply_response_span(sp, rec)
    assert sp.attrs["gen_ai.completion.0.role"] == "assistant"
    assert sp.attrs["gen_ai.completion.0.content"] == "world"


def test_handle_request_span_nests_under_parent():
    """Issue: 11 separate traces. With parent ids, the LLM span shares the
    rollout's trace_id and parent_id -> one nested trace."""
    p = proxy_mod.trace.Span(span_id="rootspan", trace_id="roottrace", parent_id=None, name="rollout")
    with proxy_mod.trace.span("chat m", parent=p) as sp:
        pass
    assert sp.trace_id == "roottrace"
    assert sp.parent_id == "rootspan"


@pytest.mark.asyncio
async def test_bridge_cm_owns_root_span_and_propagates_to_proxy():
    """`async with bridge:` opens ONE rollout span; start_proxy captures it and
    forwards its ids to the sandbox proxy, so all LLM spans nest under it."""
    from agentix.bridge import Bridge, OpenAIClient

    class _FakeSandbox:
        def __init__(self) -> None:
            self.captured: dict[str, object] = {}

        def register_namespace(self, ns: object) -> None:
            pass

        async def remote(self, fn, **kwargs):  # noqa: ANN001
            self.captured = kwargs
            return proxy_mod.ProxyHandle(
                proxy_id="p", url="http://127.0.0.1:1", port=1, anthropic_base_url="x", openai_base_url="y"
            )

    bridge = Bridge(OpenAIClient(base_url="http://x", api_key="k", model="m"))
    sandbox = _FakeSandbox()
    async with bridge:
        root = trace.get_current_span()
        assert root is not None and root.name == "abridge"
        await bridge.start_proxy(sandbox, family="anthropic")

    assert sandbox.captured["parent_trace_id"] == root.trace_id
    assert sandbox.captured["parent_span_id"] == root.span_id
