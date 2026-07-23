"""Verify `OTelTraceProcessor` maps `agentix.trace` spans into OTel.

Uses OTel's `InMemorySpanExporter` + `SimpleSpanProcessor` to avoid
any network — every exported span goes into a local list we can
assert against.
"""

from __future__ import annotations

import pytest
from agentix.utils.trace.otel import OTelTraceProcessor
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from agentix import trace


@pytest.fixture
def otel_recorder():
    exporter = InMemorySpanExporter()
    processor = OTelTraceProcessor(processor=SimpleSpanProcessor(exporter))
    trace.add_processor(processor)
    try:
        yield exporter, processor
    finally:
        trace.remove_processor(processor)
        processor.shutdown()


def _by_name(spans: list[ReadableSpan], name: str) -> ReadableSpan:
    matches = [s for s in spans if s.name == name]
    if not matches:
        raise AssertionError(f"no span named {name!r} in {[s.name for s in spans]}")
    return matches[-1]


def test_simple_span_exports_with_attrs_events_and_status(otel_recorder) -> None:
    exporter, _ = otel_recorder

    with trace.trace("workflow") as t:
        with trace.span("op", model="claude") as s:
            s.add_event("first_chunk")
            s.set_status("ok")
        agentix_trace_id = t.trace_id
        agentix_span_id = s.span_id

    spans = list(exporter.get_finished_spans())
    op = _by_name(spans, "op")
    assert op.status.status_code is StatusCode.OK
    # OTel keeps its own ids; ours ride along as attributes.
    assert op.attributes["agentix.trace_id"] == agentix_trace_id
    assert op.attributes["agentix.span_id"] == agentix_span_id
    assert op.attributes["agentix.parent_span_id"] == ""
    assert op.attributes["model"] == "claude"
    event_names = [e.name for e in op.events]
    assert event_names == ["first_chunk"]


def test_nested_spans_preserve_parent_linkage(otel_recorder) -> None:
    exporter, _ = otel_recorder

    parent_agentix_span_id: str | None = None
    child_agentix_span_id: str | None = None
    with trace.trace("workflow"):
        with trace.span("parent") as parent:
            parent_agentix_span_id = parent.span_id
            with trace.span("child") as child:
                child_agentix_span_id = child.span_id

    spans = list(exporter.get_finished_spans())
    parent_span = _by_name(spans, "parent")
    child_span = _by_name(spans, "child")

    # OTel parent linkage: child's parent SpanContext == parent's
    # SpanContext.
    assert child_span.parent is not None
    assert child_span.parent.span_id == parent_span.context.span_id
    assert child_span.parent.trace_id == parent_span.context.trace_id

    # Same trace_id at the OTel layer.
    assert child_span.context.trace_id == parent_span.context.trace_id

    # Agentix attributes carry the originating ids.
    assert child_span.attributes["agentix.parent_span_id"] == parent_agentix_span_id
    assert parent_span.attributes["agentix.span_id"] == parent_agentix_span_id
    assert child_span.attributes["agentix.span_id"] == child_agentix_span_id


def test_error_span_marked_error_with_recorded_exception(otel_recorder) -> None:
    exporter, _ = otel_recorder

    with pytest.raises(RuntimeError):
        with trace.trace("workflow"):
            with trace.span("failing"):
                raise RuntimeError("nope")

    spans = list(exporter.get_finished_spans())
    failing = _by_name(spans, "failing")
    assert failing.status.status_code is StatusCode.ERROR
    # `record_exception` adds an "exception" event.
    exception_events = [e for e in failing.events if e.name == "exception"]
    assert exception_events, [e.name for e in failing.events]


def test_resource_carries_service_name() -> None:
    exporter = InMemorySpanExporter()
    processor = OTelTraceProcessor(
        service_name="rollout-eval",
        extra_resource_attrs={"deployment.environment": "test"},
        processor=SimpleSpanProcessor(exporter),
    )
    trace.add_processor(processor)
    try:
        with trace.trace("workflow"):
            with trace.span("op"):
                pass
    finally:
        trace.remove_processor(processor)
        processor.shutdown()

    spans = list(exporter.get_finished_spans())
    assert spans, "no spans exported"
    resource = spans[0].resource
    assert resource.attributes["service.name"] == "rollout-eval"
    assert resource.attributes["deployment.environment"] == "test"


def test_long_attribute_values_export_untruncated(otel_recorder) -> None:
    """Byte-fidelity guard for the OTel leg: the exporter must not
    truncate long attribute values. The OTel Python SDK's attribute
    value-length limit defaults to None (unlimited) and
    `OTelTraceProcessor` sets no explicit limit; gen_ai content
    attributes (full prompts/completions) rely on that. If a future SDK
    or processor change introduces a limit, this fails loudly instead of
    silently clipping capture data."""
    exporter, _ = otel_recorder

    long_value = ("0123456789" * 250) + "<END sentinel>"
    assert len(long_value) > 2000

    with trace.trace("workflow"):
        with trace.span("llm.request") as s:
            s.set_attribute("gen_ai.prompt.0.content", long_value)

    span = _by_name(list(exporter.get_finished_spans()), "llm.request")
    assert span.attributes["gen_ai.prompt.0.content"] == long_value
