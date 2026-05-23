"""OpenTelemetry exporter for `agentix.trace` spans.

Plug-in to keep `opentelemetry-*` out of the core agentix
dependencies. Once installed, register a `OTelTraceProcessor` on the
agentix trace pipeline:

```python
from agentix import trace
from agentix.utils.trace.otel import OTelTraceProcessor

trace.add_processor(
    OTelTraceProcessor(
        endpoint="https://otlp.example.com:4318/v1/traces",
        headers={"x-api-key": "..."},
        service_name="my-rollout",
    )
)
```

Every span Agentix produces ends up on the configured OTLP backend
with the same name, attributes, events, and parent linkage as the
agentix-side span. Agentix's string `trace_id` / `span_id` /
`parent_id` are preserved as span attributes (`agentix.trace_id`,
`agentix.span_id`, `agentix.parent_span_id`) so consumers can join
agentix-side records against the exported trace.

Design choices:

* We map agentix spans onto **OTel SDK spans** via the public
  `Tracer.start_span()` / `Span.end()` API. The SDK owns OTel-style
  128/64-bit ids and propagation through `Context`; we keep the
  agentix-side ids only in attributes.
* Parent linkage is reconstructed by looking up the previously
  started OTel span by its agentix `span_id` and threading its
  `Context` into the child.
* Timestamps are converted from ISO-8601 UTC to nanoseconds since
  epoch, matching OTel's `start_time` / `end_time` units.
* By default we install a `BatchSpanProcessor`; one OTel
  `TracerProvider` and HTTP exporter are constructed per
  `OTelTraceProcessor` instance so multiple processors with different
  endpoints can coexist.
* `shutdown()` flushes and shuts down the OTel pipeline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace as otel_trace
from opentelemetry.context import Context, set_value
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.span import Span as OTelSpan

from agentix import trace as agentix_trace

_AGENTIX_PARENT_ATTR = "agentix.parent_span_id"
_AGENTIX_SPAN_ATTR = "agentix.span_id"
_AGENTIX_TRACE_ATTR = "agentix.trace_id"
_AGENTIX_KIND_ATTR = "agentix.span_kind"


def _iso_to_ns(iso: str | None) -> int | None:
    """Convert agentix's ISO-8601 UTC stamp to nanoseconds since epoch."""
    if not iso:
        return None
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1_000_000_000)


def _coerce_attr_value(value: Any) -> Any:
    """OTel attributes must be primitives or sequences thereof.

    Cast everything else to a string so the exporter accepts it
    without raising; loss is acceptable for high-cardinality custom
    values (callers can pre-coerce if they care).
    """
    if value is None:
        return ""
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_attr_value(v) for v in value]
    return str(value)


def _coerce_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    return {k: _coerce_attr_value(v) for k, v in attrs.items()}


class OTelTraceProcessor(agentix_trace.Processor):
    """`agentix.trace.Processor` that mirrors spans to an OTLP backend.

    Parameters mirror the `OTLPSpanExporter` shape:

    * `endpoint`     — full OTLP/HTTP traces URL, e.g.
                       `https://api.honeycomb.io/v1/traces`.
    * `headers`      — dict of header pairs (usually auth + dataset).
    * `service_name` — `service.name` resource attribute.
    * `extra_resource_attrs` — additional resource attributes.
    * `processor`    — override the default `BatchSpanProcessor` (a
                       caller can pass `SimpleSpanProcessor` for
                       tests, or a pre-configured batch processor).

    Provider construction is lazy: the OTel `TracerProvider` is
    created on first `on_span_start` so callers can register the
    processor at import time without immediately opening sockets.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        headers: dict[str, str] | None = None,
        service_name: str = "agentix",
        extra_resource_attrs: dict[str, Any] | None = None,
        processor: SpanProcessor | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._headers = headers or {}
        self._service_name = service_name
        self._extra_resource_attrs = extra_resource_attrs or {}
        self._processor_override = processor
        self._provider: TracerProvider | None = None
        self._tracer: otel_trace.Tracer | None = None
        self._otel_spans: dict[str, OTelSpan] = {}

    def _ensure_provider(self) -> TracerProvider:
        if self._provider is not None:
            return self._provider
        resource = Resource.create(
            {"service.name": self._service_name, **self._extra_resource_attrs}
        )
        provider = TracerProvider(resource=resource)
        processor = self._processor_override or BatchSpanProcessor(
            OTLPSpanExporter(endpoint=self._endpoint, headers=self._headers)
        )
        provider.add_span_processor(processor)
        self._provider = provider
        self._tracer = provider.get_tracer("agentix.utils.trace.otel")
        return provider

    # ── agentix.Processor protocol ────────────────────────────────────

    def on_trace_start(self, t: agentix_trace.Trace) -> None:  # noqa: D401
        return None

    def on_trace_end(self, t: agentix_trace.Trace) -> None:  # noqa: D401
        return None

    def on_span_start(self, span: agentix_trace.Span) -> None:
        self._ensure_provider()
        assert self._tracer is not None  # set by _ensure_provider

        parent_otel = (
            self._otel_spans.get(span.parent_id) if span.parent_id is not None else None
        )
        parent_context: Context | None = None
        if parent_otel is not None:
            parent_context = otel_trace.set_span_in_context(parent_otel)
        else:
            parent_context = set_value("agentix.trace_id", span.trace_id, Context())

        start_ns = _iso_to_ns(span.started_at)
        attributes = _coerce_attrs(
            {
                **span.attrs,
                _AGENTIX_SPAN_ATTR: span.span_id,
                _AGENTIX_TRACE_ATTR: span.trace_id,
                _AGENTIX_PARENT_ATTR: span.parent_id or "",
                _AGENTIX_KIND_ATTR: "agentix",
            }
        )
        otel_span = self._tracer.start_span(
            name=span.name,
            context=parent_context,
            start_time=start_ns,
            attributes=attributes,
        )
        self._otel_spans[span.span_id] = otel_span

    def on_span_end(self, span: agentix_trace.Span) -> None:
        otel_span = self._otel_spans.pop(span.span_id, None)
        if otel_span is None:
            # Race: the start hook didn't fire (processor was added
            # mid-span); skip silently — incomplete spans aren't
            # exportable.
            return

        # Final attributes — span.attrs may have been mutated between
        # start and end; copy the current snapshot.
        otel_span.set_attributes(_coerce_attrs(span.attrs))

        for event in span.events:
            otel_span.add_event(
                event.name,
                attributes=_coerce_attrs(event.attributes),
                timestamp=_iso_to_ns(event.timestamp),
            )

        otel_span.set_status(_status_from_span(span))
        if span.error is not None:
            otel_span.record_exception(
                _SpanErrorAsException(span.error.message),
                attributes=_coerce_attrs(span.error.data or {}),
            )

        otel_span.end(end_time=_iso_to_ns(span.ended_at))

    def force_flush(self) -> None:
        if self._provider is not None:
            self._provider.force_flush()

    def shutdown(self) -> None:
        if self._provider is not None:
            self._provider.force_flush()
            self._provider.shutdown()
            self._provider = None
            self._tracer = None
        self._otel_spans.clear()


def _status_from_span(span: agentix_trace.Span) -> Status:
    if span.status == "ok":
        return Status(StatusCode.OK)
    if span.status == "error":
        description = span.error.message if span.error is not None else None
        return Status(StatusCode.ERROR, description=description)
    return Status(StatusCode.UNSET)


class _SpanErrorAsException(Exception):
    """Adapter — `record_exception` wants an exception-shaped object."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


__all__ = ["OTelTraceProcessor"]
