"""Importable target for the cross-process context-propagation test.

Runs inside the sandbox worker and reports what it observed of the
host's propagated ambient context: the baggage values, the restored
trace id, and the trace/parent ids of a span it opens. The host asserts
these match the trace + baggage it had active when it called `remote`.
"""

from __future__ import annotations

from agentix import trace
from agentix.utils import context


def observe_context() -> dict:
    with trace.span("worker.child") as child:
        return {
            "tenant": context.get("tenant"),
            "attempt": context.get("attempt"),
            "observed_trace_id": trace.current_trace_id(),
            "child_trace_id": child.trace_id,
            "child_parent_id": child.parent_id,
        }
