"""Explicit `parent=` for spans opened off the originating task's context.

A `ContextVar` does not cross thread/executor boundaries, so a span opened in a
worker thread can't see a parent opened on the originating task. Passing the
captured parent explicitly keeps the trace tree intact.
"""

from __future__ import annotations

import threading

from agentix.utils import trace


def test_default_parenting_via_contextvar() -> None:
    with trace.span("root") as root, trace.span("child") as child:
        assert child.parent_id == root.span_id
        assert child.trace_id == root.trace_id


def test_explicit_parent_overrides_contextvar() -> None:
    with trace.span("a") as a, trace.span("b") as b:
        # Without parent=, `c` would attach to `b` (the contextvar parent).
        with trace.span("c", parent=a) as c:
            assert c.parent_id == a.span_id
            assert c.parent_id != b.span_id
            assert c.trace_id == a.trace_id


def test_explicit_parent_with_no_current_span() -> None:
    with trace.trace("t"), trace.span("root") as root:
        captured = root
    # Outside any open span (fresh context), the captured parent still attaches
    # the child correctly.
    with trace.span("detached-child", parent=captured) as child:
        assert child.parent_id == captured.span_id
        assert child.trace_id == captured.trace_id


def test_explicit_parent_across_a_real_thread() -> None:
    results: dict[str, str | None] = {}

    with trace.trace("t"), trace.span("parent") as parent:

        def work() -> None:
            # A new thread starts with a fresh context — no contextvar parent.
            with trace.span("worker", parent=parent) as w:
                results["parent_id"] = w.parent_id
                results["trace_id"] = w.trace_id

        th = threading.Thread(target=work)
        th.start()
        th.join()

    assert results["parent_id"] == parent.span_id
    assert results["trace_id"] == parent.trace_id
