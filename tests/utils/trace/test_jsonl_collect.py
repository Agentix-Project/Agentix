"""`trace.collect(path)` is the one-call rollout-data sink: a JsonlProcessor
that writes every span/trace as JSON Lines. Plus the zero-processor warning,
so the marquee collection journey doesn't silently produce nothing.
"""

from __future__ import annotations

import json
import logging

from agentix.utils import trace


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_collect_writes_spans_and_trace_sharing_trace_id(tmp_path) -> None:
    out = tmp_path / "runs" / "spans.jsonl"  # parent dir auto-created
    sink = trace.collect(str(out))
    try:
        with trace.trace("rollout", rollout_id="inst-1") as t:
            with trace.span("step", n=1):
                pass
            with trace.span("step", n=2):
                pass
    finally:
        sink.shutdown()
        trace.remove_processor(sink)

    records = _read_jsonl(out)
    spans = [r for r in records if r["object"] == "span"]
    traces = [r for r in records if r["object"] == "trace"]
    assert len(spans) == 2
    assert len(traces) == 1
    # Every record shares the trace id, so a rollout's spans group by it.
    assert {s["trace_id"] for s in spans} == {t.trace_id}
    assert traces[0]["id"] == t.trace_id
    assert traces[0]["metadata"] == {"rollout_id": "inst-1"}


def test_collect_filters_can_drop_traces(tmp_path) -> None:
    out = tmp_path / "spans_only.jsonl"
    sink = trace.collect(str(out), traces=False)
    try:
        with trace.trace("r"):
            with trace.span("only-span"):
                pass
    finally:
        sink.shutdown()
        trace.remove_processor(sink)
    records = _read_jsonl(out)
    assert records and all(r["object"] == "span" for r in records)


def test_span_end_with_no_processor_warns_once(caplog) -> None:
    # Save/restore global provider state so this is deterministic and doesn't
    # leak into other tests.
    saved = trace.get_processors()
    trace.set_processors([])
    trace.set_tracing_disabled(False)
    trace._provider._warned_no_processor = False
    try:
        with caplog.at_level(logging.WARNING, logger="agentix.utils.trace"):
            with trace.span("dropped-1"):
                pass
            with trace.span("dropped-2"):
                pass
        hits = [r for r in caplog.records if "no trace Processor is registered" in r.getMessage()]
        assert len(hits) == 1  # warned exactly once, not per span
    finally:
        trace._provider._warned_no_processor = False
        trace.set_processors(saved)
