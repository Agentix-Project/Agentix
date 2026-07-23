"""abridge clients as OTel `/trace` span producers.

abridge core does NO tracing — it never opens spans, never inspects
shapes. Each bundled client opens its OWN `trace.span(...)` inside its
`@on` method (caller-side `trace.span` doesn't propagate across the
HTTP/SIO boundary, so the client is the right scope owner). The
populate helpers stamp OTel GenAI attrs on that client-opened span.

Tests verify each bundled client emits a span with the expected attrs;
the "silent client" test confirms abridge core doesn't inject anything
when the client doesn't.
"""

from __future__ import annotations

import asyncio
from typing import Any

import agentix.bridge.proxy as proxy_mod
import httpx
import pytest
from agentix.bridge import ClientResponse, Proxy, Request, on

from agentix.utils import trace


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
async def test_anthropic_client_emits_genai_span(wired, capture_spans) -> None:
    """`AnthropicFromOpenAIClient.messages` opens its own span named
    `anthropic messages <model>` and `populate_anthropic_span` stamps
    OTel GenAI attrs onto it."""
    handle = wired["handle"]
    async with httpx.AsyncClient(base_url=handle.url, timeout=10) as c:
        await c.post(
            "/v1/messages",
            json={
                "model": "claude-3-haiku",
                "max_tokens": 32,
                "system": "be brief",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    host = [s for s in capture_spans.spans if s.name.startswith("anthropic messages")]
    assert len(host) == 1
    sp = host[0]
    assert sp.attrs["gen_ai.system"] == "anthropic"
    assert sp.attrs["gen_ai.request.model"] == "claude-3-haiku"
    assert sp.attrs["gen_ai.request.max_tokens"] == 32
    assert sp.attrs["gen_ai.prompt.0.role"] == "system"
    assert sp.attrs["gen_ai.prompt.0.content"] == "be brief"
    assert sp.attrs["gen_ai.prompt.1.role"] == "user"
    assert sp.attrs["gen_ai.prompt.1.content"] == "hi"
    assert sp.attrs["gen_ai.usage.input_tokens"] == 7
    assert sp.attrs["gen_ai.usage.output_tokens"] == 4
    assert sp.attrs["gen_ai.completion.0.content"] == "hello from upstream"


@pytest.mark.asyncio
async def test_openai_client_emits_genai_span(wired, capture_spans) -> None:
    """`OpenAIClient.chat` opens `openai chat <model>` span; OpenAI-shape
    attrs land on it."""
    handle = wired["handle"]
    async with httpx.AsyncClient(base_url=handle.url, timeout=10) as c:
        await c.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        )

    host = [s for s in capture_spans.spans if s.name.startswith("openai chat")]
    assert len(host) == 1
    sp = host[0]
    assert sp.attrs["gen_ai.system"] == "openai"
    assert sp.attrs["gen_ai.request.model"] == "gpt-4o-mini"
    assert sp.attrs["gen_ai.usage.input_tokens"] == 7
    assert sp.attrs["gen_ai.usage.output_tokens"] == 4
    assert sp.attrs["gen_ai.completion.0.content"] == "hello from upstream"


def test_genai_span_content_is_byte_faithful_for_long_prompts(capture_spans, tmp_path) -> None:
    """Byte-fidelity spike: the span-population path must NOT truncate
    gen_ai content attributes.

    A downstream consumer's captured-span fixture showed every
    `gen_ai.prompt.*.content` cut at exactly 120 chars, raising the
    question whether agentix's capture truncates attribute values.
    Verdict from source: it does not — `populate_*_span` stores the full
    string, `Span.set_attribute` keeps raw values, and `JsonlProcessor`
    writes `Span.export()` verbatim (no OTel attribute-length limit is
    involved on this path). That fixture was hand-trimmed. This test
    locks the property in: a >2000-char prompt survives population,
    in-memory capture, AND the JSONL sink byte-for-byte.
    """
    from agentix.bridge.clients import populate_anthropic_span

    # Deterministic, marker-rich content: mid-string and end markers would
    # be destroyed by any length-limited capture.
    prompt = ("0123456789" * 250) + "<END-OF-PROMPT sentinel 中文>"
    completion = ("abcdefghij" * 300) + "<END-OF-COMPLETION sentinel>"
    assert len(prompt) > 2000 and len(completion) > 2000

    sink_path = tmp_path / "spans.jsonl"
    sink = trace.JsonlProcessor(sink_path)
    trace.add_processor(sink)
    try:
        with trace.span("anthropic messages fidelity"):
            populate_anthropic_span(
                request={
                    "model": "claude-3-haiku",
                    "system": prompt,
                    "messages": [{"role": "user", "content": prompt}],
                },
                response={
                    "model": "m",
                    "content": [{"type": "text", "text": completion}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
    finally:
        trace.remove_processor(sink)
        sink.shutdown()

    # In-memory capture: full fidelity.
    (sp,) = capture_spans.spans
    assert sp.attrs["gen_ai.prompt.0.content"] == prompt
    assert sp.attrs["gen_ai.prompt.1.content"] == prompt
    assert sp.attrs["gen_ai.completion.0.content"] == completion

    # JSONL capture channel (`trace.collect` / `JsonlProcessor`): what is
    # read back from disk is byte-identical to what was populated.
    import json as _json

    (record,) = [_json.loads(line) for line in sink_path.read_text().splitlines()]
    assert record["attrs"]["gen_ai.prompt.0.content"] == prompt
    assert record["attrs"]["gen_ai.prompt.1.content"] == prompt
    assert record["attrs"]["gen_ai.completion.0.content"] == completion


@pytest.mark.asyncio
async def test_silent_client_emits_no_spans(capture_spans) -> None:
    """A custom client that opens no spans and calls no populate helper
    produces zero spans. Shape-blindness contract: abridge core never
    injects anything; observability is entirely the client's choice."""

    class _Silent:
        @on("/anything")
        async def handle(self, request: Request) -> ClientResponse:
            return ClientResponse.json({"ok": True})

    proxy = Proxy(_Silent())
    captured_emits: list[tuple[str, Any]] = []

    async def fake_emit(event: str, data: Any = None, **_: Any) -> None:
        captured_emits.append((event, data))

    proxy.emit = fake_emit  # type: ignore[method-assign]
    await proxy.trigger_event(
        "/anything",
        {"request_id": "rid-1", "data": {"body": {"x": 1}}},
    )
    await asyncio.sleep(0.05)  # let the detached handler task finish

    # Zero spans emitted — neither Proxy nor the silent client opened one.
    assert len(capture_spans.spans) == 0
    result_events = [(e, d) for e, d in captured_emits if e.endswith(":result")]
    assert len(result_events) == 1


@pytest.mark.asyncio
async def test_proxy_session_runs_start_stop_without_tracing():
    """`proxy.session(sandbox)` doesn't open any span on its own — it's
    just start + stop sugar. The caller wraps in `trace.span(...)` if
    they want rollout grouping."""

    class _Silent:
        @on("/_unused")
        async def _(self, request: Request) -> ClientResponse:
            return ClientResponse.json({})

    class _FakeSandbox:
        def __init__(self) -> None:
            self.remote_calls: list[tuple[Any, dict[str, Any]]] = []

        def register_namespace(self, ns: object) -> None:
            pass

        async def remote(self, fn, **kwargs):  # noqa: ANN001
            self.remote_calls.append((fn, kwargs))
            return proxy_mod.TunnelHandle(url="http://127.0.0.1:1", port=1)

    proxy = Proxy(_Silent())
    sandbox: Any = _FakeSandbox()
    async with proxy.session(sandbox) as handle:
        assert handle.url == "http://127.0.0.1:1"

    # Two remote calls: tunnel start, tunnel stop.
    assert len(sandbox.remote_calls) == 2
    start_kwargs = sandbox.remote_calls[0][1]
    assert "paths" in start_kwargs
    assert "session_id" not in start_kwargs  # session_id lives on the Client now
    assert sandbox.remote_calls[-1][1] == {"handle": handle}
