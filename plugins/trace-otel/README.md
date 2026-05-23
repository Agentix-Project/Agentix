# agentix-trace-otel

OpenTelemetry exporter for `agentix.trace` spans. Plugs into the
existing trace pipeline as a `Processor` — no changes to the
`agentix.trace` public API; the `opentelemetry-*` dependencies stay
out of agentix core.

## Install

```bash
pip install agentix-trace-otel        # HTTP/OTLP exporter
pip install "agentix-trace-otel[grpc]"  # adds gRPC/OTLP exporter
```

## Usage

```python
from agentix import trace
from agentix.utils.trace.otel import OTelTraceProcessor

trace.add_processor(
    OTelTraceProcessor(
        endpoint="https://otlp.example.com:4318/v1/traces",
        headers={"x-api-key": "sk-..."},
        service_name="my-rollout",
        extra_resource_attrs={"deployment.environment": "prod"},
    )
)

with trace.trace("eval-cc-swe"):
    with trace.span("instance", id="django-11099") as s:
        s.add_event("first_chunk")
        s.set_status("ok")
```

Every Agentix span (`agentix.trace.span(...)`, including the
`llm.request` spans the abridge proxy emits) is mirrored to the
configured OTel-compatible backend. Datadog, Honeycomb, Jaeger,
Tempo, New Relic, signoz — any OTLP/HTTP collector works.

## Mapping

| `agentix.Span` | OTLP span                                    |
| -------------- | -------------------------------------------- |
| `name`         | `name`                                       |
| `attrs`        | `attributes`                                 |
| `events`       | `events`                                     |
| `started_at`   | `start_time` (ns since epoch)                |
| `ended_at`     | `end_time`   (ns since epoch)                |
| `status`       | `status` (`ok` -> OK; `error` -> ERROR)      |
| `error`        | `record_exception` (+ attributes)            |
| `trace_id`     | attribute `agentix.trace_id`                 |
| `span_id`      | attribute `agentix.span_id`                  |
| `parent_id`    | attribute `agentix.parent_span_id`           |

OTel SDK generates its own 128-bit `trace_id` / 64-bit `span_id`;
the original agentix string ids ride along as attributes so consumers
can correlate exported spans back to `CompletionRecord`s, log lines
that carry `record_id`, and other agentix-side artifacts.

## Sandbox- vs host-side export

Per the parent roadmap, the recommended placement is **host-side**:
sandboxes are ephemeral and the host owns the long-lived collector
connection. The `OTelTraceProcessor` is sandbox-safe too — install
it inside the sandbox if you have a collector reachable from there
and you don't want the trace events to hop through the
`/trace` SIO namespace first. Both modes are supported; mix-and-match
is fine.
